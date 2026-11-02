"""Synthetic checks for the two frozen memory-only scalar fits."""
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp


def api():
    path = Path(__file__).parents[1] / 'scripts' / 'alternative_fits.py'
    assert path.is_file(), 'The memory-only alternative fitting API is missing'
    from scripts import alternative_fits
    return alternative_fits


def objective(z, y, coefficient, direction=None):
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y)
    counts = np.bincount(y, minlength=z.shape[1])
    weights = 1. / (np.count_nonzero(counts) * counts[y])
    adjusted = coefficient * z if direction is None else z + coefficient * direction
    return float(weights @ (logsumexp(adjusted, axis=1) - adjusted[np.arange(len(y)), y]))


def test_auto_mixture_has_known_signed_optimum_and_fixed_prior():
    module = api()
    z = np.array([[2., 0.], [2., 0.]])
    y = np.array([0, 1])
    fit = module.fit_auto_mixture(z, y, [1])
    assert fit['kind'] == 'frequency'
    assert fit['prior'] == pytest.approx([.25, .75])
    assert fit['tau'] == pytest.approx(-2 / math.log(3), abs=1e-11)
    assert fit['status'] == 'interior'
    assert abs(fit['derivative']) < 1e-12
    assert fit['memory_loss_after'] == pytest.approx(math.log(2))
    assert fit['memory_loss_after'] < fit['memory_loss_before']
    from scripts.posthoc_calibration import apply_cached
    torch.testing.assert_close(module.apply_alternative(z, fit), apply_cached(z, fit))
    json.dumps(fit, allow_nan=False)


def test_temperature_known_optimum_uses_represented_class_weights():
    module = api()
    a = math.log(2) / 2
    z = torch.tensor([[a, 0.], [a, 0.], [0., 0.]], dtype=torch.float64)
    y = torch.tensor([0, 1, 1])
    fit = module.fit_temperature(z, y)
    assert fit['kind'] == 'temperature'
    assert fit['beta'] == pytest.approx(2., abs=1e-11)
    assert fit['temperature'] == pytest.approx(.5, abs=1e-11)
    assert fit['status'] == 'interior'
    assert abs(fit['derivative']) < 1e-12
    assert fit['memory_loss_before'] == pytest.approx(objective(z, y, 1.))
    assert fit['memory_loss_after'] == pytest.approx(objective(z, y, 2.))
    adjusted = module.apply_alternative(z, fit)
    assert adjusted.dtype == torch.float64 and adjusted.device.type == 'cpu'
    torch.testing.assert_close(adjusted, 2 * z)
    assert torch.equal(adjusted.argmax(1), z.argmax(1))
    json.dumps(fit, allow_nan=False)


@pytest.mark.parametrize('expected', [.01, 256.])
def test_temperature_expands_bracket_to_finite_interior_optimum(expected):
    a = math.log(2) / expected
    fit = api().fit_temperature([[a, 0.], [a, 0.], [0., 0.]], [0, 1, 1])
    assert fit['status'] == 'interior'
    assert fit['beta'] == pytest.approx(expected, rel=1e-12)
    assert fit['iterations'] == 80


def test_auto_mixture_expands_bracket_to_finite_interior_optimum():
    fit = api().fit_auto_mixture([[100., 0.], [100., 0.]], [0, 1], [1])
    assert fit['status'] == 'interior'
    assert fit['tau'] == pytest.approx(-100 / math.log(3), rel=1e-12)
    assert fit['iterations'] == 80


@pytest.mark.parametrize('label,expected,status', [(0, 1048576., 'upper_guard'), (1, -1048576., 'lower_guard')])
def test_auto_mixture_reports_guard_instead_of_claiming_finite_root(label, expected, status):
    fit = api().fit_auto_mixture([[0., 0.]], [label], [1])
    assert fit['tau'] == expected and fit['status'] == status
    assert fit['derivative'] <= 0 if status == 'upper_guard' else fit['derivative'] >= 0


@pytest.mark.parametrize('label,expected,status', [(0, 1e6, 'upper_guard'), (1, 1e-6, 'lower_guard')])
def test_temperature_reports_guard_instead_of_claiming_finite_root(label, expected, status):
    fit = api().fit_temperature([[1., 0.]], [label])
    assert fit['beta'] == expected and fit['status'] == status
    assert fit['derivative'] <= 0 if status == 'upper_guard' else fit['derivative'] >= 0


def test_flat_directions_return_neutral_unidentifiable_coefficients():
    module = api()
    mixed = module.fit_auto_mixture([[0., 2., 4.]] * 3, [0, 0, 0], [1, 2])
    temperature = module.fit_temperature([[2., 2., 2.], [-3., -3., -3.]], [0, 2])
    assert mixed['prior'] == pytest.approx([1 / 3] * 3)
    assert mixed['tau'] == 0 and mixed['status'] == 'unidentifiable'
    assert temperature['beta'] == 1 and temperature['temperature'] == 1
    assert temperature['status'] == 'unidentifiable'
    for fit in (mixed, temperature):
        assert fit['derivative'] == 0
        assert fit['memory_loss_before'] == fit['memory_loss_after']


def test_wrapper_uses_final_task_and_preserves_inputs_including_missing_classes():
    module = api()
    z = np.array([[1., 2., 3., 4.], [4., 3., 2., 1.], [0., 2., 1., -1.]])
    y = np.array([0, 0, 2]); order = [3, 1, 0, 2]
    before_z = z.copy(); before_y = y.copy(); before_order = order.copy()
    fits = module.fit_alternatives(z, y, order, 2)
    assert set(fits) == {'auto_mixture', 'temperature'}
    expected = .5 * np.array([3, 1, 2, 1]) / 7 + np.array([.25, 0, .25, 0])
    assert fits['auto_mixture']['prior'] == pytest.approx(expected)
    for fit in fits.values():
        assert fit['memory_loss_after'] <= fit['memory_loss_before'] + 1e-12
        module.apply_alternative(z, fit)
    np.testing.assert_array_equal(z, before_z)
    np.testing.assert_array_equal(y, before_y)
    assert order == before_order
    zt = torch.tensor(z, requires_grad=True); yt = torch.tensor(y)
    module.fit_alternatives(zt, yt, order, 2)
    assert zt.grad is None
    torch.testing.assert_close(zt, torch.tensor(before_z))


@pytest.mark.parametrize('seed', [3, 19, 57])
def test_both_minima_match_independent_scipy_objectives(seed):
    module = api(); rng = np.random.default_rng(seed)
    z = rng.normal(size=(31, 4)); y = rng.choice([0, 1, 3], size=31)
    y[:3] = [0, 1, 3]
    fits = module.fit_alternatives(z, y, [0, 1, 2, 3], 2)
    for name, fit in fits.items():
        direction = -np.log(fit['prior']) if name == 'auto_mixture' else None
        bounds = (-1048576., 1048576.) if direction is not None else (1e-6, 1e6)
        optimum = minimize_scalar(lambda value: objective(z, y, value, direction), bounds=bounds,
                                  method='bounded', options={'xatol': 1e-11})
        coefficient = fit['tau'] if direction is not None else fit['beta']
        assert objective(z, y, coefficient, direction) <= optimum.fun + 1e-10
        assert fit['memory_loss_after'] == pytest.approx(objective(z, y, coefficient, direction), abs=1e-10)


@pytest.mark.parametrize('z,y', [([], []), ([[0., 1.]], []), ([[0., 1.]], [2]),
                               ([[0., 1.]], [-1]), ([[0., 1.]], [.5]),
                               ([[0., 1.]], [[0]]), ([[0.]], [0]),
                               ([[float('nan'), 0.]], [0]), ([[0., float('inf')]], [0]),
                               ([0., 1.], [0]), ([[0., 1.]], [True]),
                               ([[0., 1.]], [float('nan')])])
def test_fits_reject_invalid_memory_inputs(z, y):
    module = api()
    with pytest.raises(ValueError): module.fit_temperature(z, y)
    with pytest.raises(ValueError): module.fit_auto_mixture(z, y, [1])


@pytest.mark.parametrize('new', [[], [0, 1], [1, 1], [-1], [2], [.5], [True]])
def test_auto_mixture_rejects_invalid_new_classes(new):
    with pytest.raises(ValueError): api().fit_auto_mixture([[0., 1.]], [0], new)


@pytest.mark.parametrize('order,size', [([0, 0], 1), ([1, 2], 1), ([0], 1),
                                     ([0, 1], 0), ([0, 1], 2), ([0, 1], .5),
                                     ([0, 1], True), ([0., 1.], 1)])
def test_wrapper_rejects_invalid_class_order_and_task_size(order, size):
    with pytest.raises(ValueError): api().fit_alternatives([[0., 1.]], [0], order, size)


@pytest.mark.parametrize('fit', [dict(kind='other'), dict(kind='temperature', beta=0),
                               dict(kind='temperature', beta=float('nan')),
                               dict(kind='frequency', prior=[.5, .5], tau=float('inf')),
                               dict(kind='frequency', prior=[0., 1.], tau=1.),
                               dict(kind='frequency', prior=[1.], tau=1.)])
def test_apply_rejects_invalid_transforms(fit):
    with pytest.raises(ValueError): api().apply_alternative([[0., 1.]], fit)


def test_apis_do_not_accept_evaluation_labels():
    with pytest.raises(TypeError): api().fit_temperature([[0., 1.]], [0], evaluation_labels=[1])
    with pytest.raises(TypeError): api().fit_alternatives([[0., 1.]], [0], [0, 1], 1, evaluation_labels=[1])

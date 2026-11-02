"""Two frozen, memory-only, class-balanced scalar likelihood fits.

The only fitting inputs are memory logits, memory labels and class metadata.
Coefficients are bounded numerical solutions; a guard status does not assert
that a finite unconstrained optimum exists. No model or input is modified.
"""
import math
import numbers

import numpy as np
import torch

from scripts.posthoc_calibration import frequency_priors


TAU_BOUNDS = (-1048576., 1048576.)
BETA_BOUNDS = (1e-6, 1e6)
BISECTION_STEPS = 80


def _array(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.dtype == torch.bfloat16:
            value = value.double()
        value = value.numpy()
    try:
        return np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError('A rectangular numeric array is required') from error


def _logits(logits):
    z = _array(logits)
    if z.dtype.kind not in 'iuf' or z.ndim != 2 or not len(z) or z.shape[1] < 2:
        raise ValueError('Finite nonempty memory logits with at least two classes required')
    z = z.astype(np.float64, copy=True)
    if not np.isfinite(z).all():
        raise ValueError('Finite logits required')
    return z


def _memory(logits, labels):
    z = _logits(logits)
    y = _array(labels)
    if (y.dtype.kind not in 'iu' or y.ndim != 1 or len(y) != len(z)
            or np.any(y < 0) or np.any(y >= z.shape[1])):
        raise ValueError('Matching integer memory labels in the output-head range required')
    y = y.astype(np.int64, copy=True)
    counts = np.bincount(y, minlength=z.shape[1])
    weights = 1. / (np.count_nonzero(counts) * counts[y].astype(np.float64))
    return z, y, weights


def _class_ids(value, classes):
    ids = _array(value)
    if (ids.dtype.kind not in 'iu' or ids.ndim != 1 or not len(ids)
            or len(np.unique(ids)) != len(ids) or np.any(ids < 0) or np.any(ids >= classes)):
        raise ValueError('Unique integer class IDs within the output head required')
    return ids.astype(np.int64).tolist()


def _logsumexp(values):
    if not values.size:
        return -math.inf
    maximum = float(values.max())
    return maximum + math.log(float(np.exp(values - maximum).sum()))


def _objective(base, direction, labels, weights):
    """Return loss and derivative, with a log-domain sign for bracketing.

    Summing positive and negative derivative terms separately in log space
    preserves the sign of saturated one-sided objectives even when the
    ordinary float64 derivative underflows to zero.
    """
    rows = np.arange(len(labels))
    difference = direction - direction[rows, labels, None]
    positive = difference > 0
    negative = difference < 0
    log_magnitude = np.full_like(difference, -np.inf)
    nonzero = difference != 0
    log_magnitude[nonzero] = np.log(np.abs(difference[nonzero]))
    log_magnitude += np.log(weights)[:, None]

    def evaluate(coefficient):
        with np.errstate(over='ignore', invalid='ignore'):
            adjusted = base + coefficient * direction
            shifted = adjusted - adjusted[rows, labels, None]
        if not np.isfinite(shifted).all():
            raise ValueError('Logit arithmetic exceeded finite float64 range')
        maximum = shifted.max(axis=1)
        centered = shifted - maximum[:, None]
        normalizer = np.log(np.exp(centered).sum(axis=1))
        loss = float(weights @ (maximum + normalizer))
        log_probability = centered - normalizer[:, None]
        terms = log_probability + log_magnitude
        pos = _logsumexp(terms[positive])
        neg = _logsumexp(terms[negative])
        sign = int(pos > neg) - int(pos < neg)
        if sign == 0:
            derivative = 0.
        elif sign > 0:
            derivative = math.exp(pos) * (-math.expm1(neg - pos))
        else:
            derivative = -math.exp(neg) * (-math.expm1(pos - neg))
        if not math.isfinite(loss) or not math.isfinite(derivative):
            raise ValueError('Nonfinite scalar objective or derivative')
        return loss, derivative, sign

    return evaluate, not bool(nonzero.any())


def _fit_scalar(base, direction, labels, weights, initial, bounds, neutral):
    evaluate, flat = _objective(base, direction, labels, weights)
    before = evaluate(neutral)[0]
    low, high = initial
    iterations = 0
    if flat:
        value, status = neutral, 'unidentifiable'
    else:
        # Expand zero signs too: softmax saturation is not a finite-root proof.
        while evaluate(low)[2] >= 0 and low > bounds[0]:
            low = max(bounds[0], low * 2 if low < 0 else low / 2)
        while evaluate(high)[2] <= 0 and high < bounds[1]:
            high = min(bounds[1], high * 2)
        if low == bounds[0] and evaluate(low)[2] >= 0:
            value, status = low, 'lower_guard'
        elif high == bounds[1] and evaluate(high)[2] <= 0:
            value, status = high, 'upper_guard'
        else:
            for _ in range(BISECTION_STEPS):
                middle = low + (high - low) / 2
                if evaluate(middle)[2] > 0:
                    high = middle
                else:
                    low = middle
            iterations = BISECTION_STEPS
            value, status = low + (high - low) / 2, 'interior'
    after, derivative, _ = evaluate(value)
    return value, dict(status=status, memory_loss_before=before, memory_loss_after=after,
                       derivative=derivative, coefficient_bounds=list(bounds),
                       bracket=[low, high], iterations=iterations)


def fit_auto_mixture(logits, labels, new_classes):
    """Fit tau for z - tau log(pi) using only balanced memory likelihood."""
    z, y, weights = _memory(logits, labels)
    new = _class_ids(new_classes, z.shape[1])
    if len(new) == z.shape[1]:
        raise ValueError('New classes must be a proper subset of the output head')
    prior = frequency_priors(y, z.shape[1], new)['la_mixture'].numpy()
    direction = np.broadcast_to(-np.log(prior), z.shape)
    tau, diagnostics = _fit_scalar(z, direction, y, weights, (-1., 1.), TAU_BOUNDS, 0.)
    return dict(kind='frequency', prior=prior.tolist(), tau=tau, **diagnostics)


def fit_temperature(logits, labels):
    """Fit positive inverse temperature beta with represented-class weights."""
    z, y, weights = _memory(logits, labels)
    beta, diagnostics = _fit_scalar(np.zeros_like(z), z, y, weights,
                                    (.5, 2.), BETA_BOUNDS, 1.)
    return dict(kind='temperature', beta=beta, temperature=1. / beta, **diagnostics)


def fit_alternatives(logits, labels, order, classes_per_task):
    """Fit the two frozen methods, taking incoming IDs from the final task."""
    z, y, _ = _memory(logits, labels)
    ids = _class_ids(order, z.shape[1])
    if (len(ids) != z.shape[1] or isinstance(classes_per_task, (bool, np.bool_))
            or not isinstance(classes_per_task, numbers.Integral)
            or not 0 < classes_per_task < z.shape[1]):
        raise ValueError('A full class permutation and proper integer task size required')
    return dict(auto_mixture=fit_auto_mixture(z, y, ids[-classes_per_task:]),
                temperature=fit_temperature(z, y))


def apply_alternative(logits, fit):
    """Return adjusted CPU float64 logits without changing the input."""
    z = torch.from_numpy(_logits(logits))
    try:
        if fit['kind'] == 'frequency':
            prior = torch.as_tensor(fit['prior'], dtype=torch.float64, device='cpu')
            tau = float(fit['tau'])
            if (prior.shape != (z.shape[1],) or not bool(torch.isfinite(prior).all())
                    or not bool((prior > 0).all()) or not math.isfinite(tau)):
                raise ValueError('A finite tau and positive finite full-head prior required')
            adjusted = z - tau * prior.log()
        elif fit['kind'] == 'temperature':
            beta = float(fit['beta'])
            if not math.isfinite(beta) or beta <= 0:
                raise ValueError('A finite positive inverse temperature required')
            adjusted = beta * z
        else:
            raise ValueError('Unknown alternative calibration kind')
    except (KeyError, TypeError, RuntimeError) as error:
        raise ValueError('Invalid alternative calibration') from error
    if not bool(torch.isfinite(adjusted).all()):
        raise ValueError('Adjusted logits must remain finite')
    return adjusted

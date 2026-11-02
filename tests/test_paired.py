"""Behavioral checks for the paired intervention protocol, using only CPU."""

import copy
import random

import numpy as np
import pytest
import torch
from torch import nn

try:
    from wnet import paired
except ImportError:
    paired = None


def api():
    assert paired is not None, "Paired intervention engine has not been implemented"
    return paired


def seed():
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)


def model_and_data(kind="sgd"):
    seed()
    model = nn.Sequential(nn.Linear(3, 6), nn.BatchNorm1d(6), nn.Dropout(0.3), nn.Linear(6, 2))
    opt = (torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9)
           if kind == "sgd" else torch.optim.Adam(model.parameters(), lr=0.01))
    batches = [(torch.randn(8, 3), torch.randint(0, 2, (8,))) for _ in range(5)]
    return model, opt, batches


def manual_step(model, opt, batch, loss_fn):
    opt.zero_grad(set_to_none=True)
    loss_fn(model(batch[0]), batch[1]).backward()
    opt.step()


def assert_tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_tree_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_tree_equal(x, y)
    else:
        assert a == b


@pytest.mark.parametrize("kind", ["sgd", "adam"])
def test_restore_replays_optimizer_buffers_gradients_modes_and_rng(kind):
    engine = api()
    model, opt, batches = model_and_data(kind)
    loss_fn = nn.CrossEntropyLoss()
    manual_step(model, opt, batches[0], loss_fn)
    model[2].eval()  # A mixed mode configuration must survive restoration.
    original = engine.capture_state(model, opt)
    draws = (random.random(), np.random.rand(), torch.rand(3))
    manual_step(model, opt, batches[1], loss_fn)
    expected_model = copy.deepcopy(model.state_dict())
    expected_opt = copy.deepcopy(opt.state_dict())
    engine.restore_state(model, opt, original)
    assert_tree_equal(engine.capture_state(model, opt), original)
    assert_tree_equal((random.random(), np.random.rand(), torch.rand(3)), draws)
    manual_step(model, opt, batches[1], loss_fn)
    assert_tree_equal(model.state_dict(), expected_model)
    assert_tree_equal(opt.state_dict(), expected_opt)


@pytest.mark.parametrize("kind", ["sgd", "adam"])
def test_skip_rejects_full_update_but_matches_accept_random_stream(kind):
    engine = api()
    model, opt, batches = model_and_data(kind)
    loss_fn = nn.CrossEntropyLoss()
    manual_step(model, opt, batches[0], loss_fn)
    before = engine.capture_state(model, opt)
    engine.apply_candidate(model, opt, batches[1], loss_fn, "accept")
    accept_draws = (random.random(), np.random.rand(), torch.rand(4))
    assert not torch.equal(model[1].running_mean, before["model"]["1.running_mean"])
    engine.restore_state(model, opt, before)
    engine.apply_candidate(model, opt, batches[1], loss_fn, "skip")
    after = engine.capture_state(model, opt)
    for key in ["model", "optimizer", "gradients", "training"]:
        assert_tree_equal(after[key], before[key])
    assert_tree_equal((random.random(), np.random.rand(), torch.rand(4)), accept_draws)


def test_rollout_is_reproducible_and_does_not_pollute_main_state():
    engine = api()
    model, opt, batches = model_and_data()
    loss_fn = nn.CrossEntropyLoss()
    manual_step(model, opt, batches[0], loss_fn)
    before = engine.capture_state(model, opt)
    args = (model, opt, batches[1], batches[2:], {"old": batches[0]}, loss_fn, [0, 1, 3])
    first = engine.paired_rollout(*args)
    assert_tree_equal(engine.capture_state(model, opt), before)
    second = engine.paired_rollout(*args)
    assert_tree_equal(first, second)
    assert_tree_equal(engine.capture_state(model, opt), before)


@pytest.mark.parametrize("action", ["accept", "skip"])
def test_horizon_zero_matches_direct_candidate_evaluation(action):
    engine = api()
    model, opt, batches = model_and_data()
    loss_fn = nn.CrossEntropyLoss()
    original = engine.capture_state(model, opt)
    result = engine.paired_rollout(model, opt, batches[0], [], {"old": batches[1]}, loss_fn, [0])
    if action == "accept":
        manual_step(model, opt, batches[0], loss_fn)
    model.eval()
    with torch.no_grad():
        expected = loss_fn(model(batches[1][0]), batches[1][1]).item()
    engine.restore_state(model, opt, original)
    assert result[action][0]["old"] == expected


def test_shared_future_updates_match_analytic_quadratic_sign_reversal():
    engine = api()
    model = nn.Linear(2, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 0.0]], dtype=torch.float64))
    opt = torch.optim.SGD(model.parameters(), lr=1.0)
    identity = torch.eye(2, dtype=torch.float64)
    b = torch.tensor([[1.0], [0.0]], dtype=torch.float64)
    delta = torch.tensor([[0.1], [0.3]], dtype=torch.float64)
    a = torch.tensor([[0.5, -0.4], [-0.4, 0.5]], dtype=torch.float64)
    eigenvalues, eigenvectors = torch.linalg.eigh(identity - a)
    x = eigenvectors @ torch.diag(eigenvalues.sqrt()) @ eigenvectors.T
    candidate = (identity, b + delta)
    future_batch = (x, x @ b)
    loss_fn = lambda prediction, target: 0.5 * (prediction - target).square().sum()
    result = engine.paired_rollout(
        model, opt, candidate, [future_batch] * 5,
        {"old": (identity, torch.zeros_like(b))}, loss_fn, [0, 1, 5],
    )
    for h in [0, 1, 5]:
        shifted = b + torch.linalg.matrix_power(a, h) @ delta
        expected = (0.5 * shifted.square().sum() - 0.5 * b.square().sum()).item()
        observed = result["accept"][h]["old"] - result["skip"][h]["old"]
        assert observed == pytest.approx(expected, abs=1e-12)
    assert result["accept"][0]["old"] > result["skip"][0]["old"]
    assert result["accept"][1]["old"] < result["skip"][1]["old"]


def test_invalid_horizon_fails_without_mutation():
    engine = api()
    model, opt, batches = model_and_data()
    before = engine.capture_state(model, opt)
    with pytest.raises(ValueError, match="horizon"):
        engine.paired_rollout(model, opt, batches[0], [], {"old": batches[1]}, nn.CrossEntropyLoss(), [1])
    assert_tree_equal(engine.capture_state(model, opt), before)


def test_rollout_restores_state_when_probe_evaluation_raises():
    engine = api()
    model, opt, batches = model_and_data()
    before = engine.capture_state(model, opt)
    malformed_probe = (torch.ones(8, 100), batches[0][1])
    with pytest.raises(RuntimeError):
        engine.paired_rollout(model, opt, batches[0], [], {"old": malformed_probe}, nn.CrossEntropyLoss(), [0])
    assert_tree_equal(engine.capture_state(model, opt), before)


def test_three_action_rollout_matches_manual_half_step_and_preserves_state():
    import inspect
    engine = api()
    assert "actions" in inspect.signature(engine.paired_rollout).parameters, "Action-aligned rollout is missing"
    model, opt, batches = model_and_data()
    before = engine.capture_state(model, opt)
    loss_fn = nn.CrossEntropyLoss()
    output = engine.paired_rollout(model, opt, batches[0], [batches[1]], {"old": batches[2]},
                                    loss_fn, [0, 1], actions={"accept": 1., "skip": 0., "half": .5})
    assert set(output) == {"accept", "skip", "half"}
    assert_tree_equal(engine.capture_state(model, opt), before)
    engine.apply_candidate(model, opt, batches[0], loss_fn, "scale", scale=.5)
    manual_step(model, opt, batches[1], loss_fn)
    model.eval()
    with torch.no_grad():
        expected = float(loss_fn(model(batches[2][0]), batches[2][1]))
    assert output["half"][1]["old"] == expected

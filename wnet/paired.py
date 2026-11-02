"""Paired accept/skip diagnostics on an explicitly materialized future stream.

The caller owns batch order and any augmentation. This first version supports
ordinary eager SGD/Adam, without an external scheduler, AMP, custom generators,
distributed training, or DataLoader worker state. Snapshots are in-memory.
"""

import copy
import random
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

Batch = tuple[torch.Tensor, torch.Tensor]
Loss = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _capture_rng() -> dict:
    return {
        "python": random.getstate(),
        "numpy": copy.deepcopy(np.random.get_state()),
        "torch": torch.get_rng_state().clone(),
        "cuda": ([state.clone() for state in torch.cuda.get_rng_state_all()]
                 if torch.cuda.is_initialized() else None),
    }


def _restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(copy.deepcopy(state["numpy"]))
    torch.set_rng_state(state["torch"].clone())
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all([value.clone() for value in state["cuda"]])


def capture_state(model: nn.Module, optimizer: torch.optim.Optimizer) -> dict:
    """Capture a detached, independent copy of training state and global RNGs."""
    return {
        "model": copy.deepcopy(model.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "training": {name: module.training for name, module in model.named_modules()},
        "gradients": {name: None if param.grad is None else param.grad.detach().clone()
                      for name, param in model.named_parameters()},
        "rng": _capture_rng(),
    }


def restore_state(model: nn.Module, optimizer: torch.optim.Optimizer, state: dict) -> None:
    """Restore without allowing optimizer tensors to alias the saved snapshot."""
    model.load_state_dict(copy.deepcopy(state["model"]))
    optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
    for name, module in model.named_modules():
        module.training = state["training"][name]
    for name, param in model.named_parameters():
        grad = state["gradients"][name]
        param.grad = None if grad is None else grad.detach().clone()
    _restore_rng(state["rng"])


def _step(model: nn.Module, optimizer: torch.optim.Optimizer, batch: Batch, loss_fn: Loss) -> float:
    optimizer.zero_grad(set_to_none=True)
    loss = loss_fn(model(batch[0]), batch[1])
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("Nonfinite candidate loss")
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def apply_candidate(
    model: nn.Module, optimizer: torch.optim.Optimizer, batch: Batch,
    loss_fn: Loss, action: str, scale: float = 1.,
) -> float:
    """Execute accept or reject the whole step while consuming its random stream.

    Skip deliberately dry-runs the step. Its compute cost is diagnostic overhead,
    not a claim about an efficient deployed skipping policy.
    """
    if action not in {"accept", "skip", "scale"}:
        raise ValueError(f"Unsupported action: {action}")
    if not 0 < scale <= 1:
        raise ValueError("Scale must lie in (0,1]")
    if action == "scale":
        rates = [group["lr"] for group in optimizer.param_groups]
        try:
            for group in optimizer.param_groups:
                group["lr"] *= scale
            return _step(model, optimizer, batch, loss_fn)
        finally:
            for group, rate in zip(optimizer.param_groups, rates):
                group["lr"] = rate
    if action == "accept":
        return _step(model, optimizer, batch, loss_fn)
    before = capture_state(model, optimizer)
    try:
        value = _step(model, optimizer, batch, loss_fn)
        candidate_end_rng = _capture_rng()
    except BaseException:
        restore_state(model, optimizer, before)
        raise
    restore_state(model, optimizer, before)
    _restore_rng(candidate_end_rng)
    return value


def _evaluate(model: nn.Module, probes: Mapping[str, Batch], loss_fn: Loss, batch_size=0) -> dict[str, float]:
    training = {name: module.training for name, module in model.named_modules()}
    rng = _capture_rng()
    try:
        model.eval()
        with torch.no_grad():
            if not batch_size:
                return {name: float(loss_fn(model(batch[0]), batch[1]).item())
                        for name, batch in probes.items()}
            if (batch_size < 1 or not isinstance(loss_fn, nn.CrossEntropyLoss)
                    or loss_fn.reduction != 'mean' or loss_fn.weight is not None):
                raise ValueError("Chunked probes require unweighted mean CrossEntropyLoss")
            result = {}
            for name, (inputs, targets) in probes.items():
                if len(targets) == 0 or bool((targets == loss_fn.ignore_index).any()):
                    raise ValueError("Chunked probes must be nonempty with no ignored targets")
                total = 0.
                for start in range(0, len(targets), batch_size):
                    y = targets[start:start+batch_size]
                    total += float(loss_fn(model(inputs[start:start+batch_size]), y)) * len(y)
                result[name] = total / len(targets)
            return result
    finally:
        for name, module in model.named_modules():
            module.training = training[name]
        _restore_rng(rng)


def paired_rollout(
    model: nn.Module, optimizer: torch.optim.Optimizer, candidate: Batch,
    future: Sequence[Batch], probes: Mapping[str, Batch], loss_fn: Loss,
    horizons: Sequence[int], actions: Mapping[str, float] | None = None,
    evaluation_batch_size: int = 0,
) -> dict[str, dict[int, dict[str, float]]]:
    """Evaluate prefix horizons for two branches, then restore the main state.

    H counts future optimizer steps after the candidate decision. Each branch
    independently recomputes its future gradients on the same materialized input
    sequence. Probe evaluation does not consume the training random stream.
    """
    if (not horizons or any(type(h) is not int or h < 0 or h > len(future)
                            for h in horizons)):
        raise ValueError("Each horizon must be an integer in [0, len(future)]")
    requested = set(horizons)
    actions = {"accept": 1., "skip": 0.} if actions is None else dict(actions)
    if not actions or any(not 0 <= factor <= 1 for factor in actions.values()):
        raise ValueError("Action factors must lie in [0,1]")
    end = max(requested)
    original = capture_state(model, optimizer)
    result = {}
    try:
        for action, factor in actions.items():
            restore_state(model, optimizer, original)
            apply_candidate(model, optimizer, candidate, loss_fn,
                            "accept" if factor == 1 else "skip" if factor == 0 else "scale",
                            scale=factor if factor > 0 else 1.)
            observations = {}
            if 0 in requested:
                observations[0] = _evaluate(model, probes, loss_fn, evaluation_batch_size)
            for h in range(1, end + 1):
                _step(model, optimizer, future[h - 1], loss_fn)
                if h in requested:
                    observations[h] = _evaluate(model, probes, loss_fn, evaluation_batch_size)
            result[action] = observations
    finally:
        restore_state(model, optimizer, original)
    return result

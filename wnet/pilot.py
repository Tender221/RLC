"""Run an ER training trajectory and collect isolated paired-impact diagnostics."""

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import IndexReplay, load_cifar, load_cifar100, make_stream, split_indices, synthetic_data, training_normalization
from .paired import apply_candidate, capture_state, paired_rollout, restore_state


@dataclass(frozen=True)
class PilotConfig:
    dataset: str = "cifar10"
    num_classes: int = 10
    classes_per_task: int = 2
    evaluation_batch_size: int = 0
    data_root: str = "data"
    model: str = "small-cnn"
    device: str = "cpu"
    seed: int = 10
    order_seed: int = 0
    split_seed: int = 0
    task_count: int = 2
    batch_size: int = 32
    replay_batch_size: int = 32
    memory_size: int = 500
    audit_per_class: int = 500
    audit_probe_per_class: int = 32
    synthetic_per_class: int = 16
    max_train_per_task: int = 0
    learning_rate: float = 0.05
    momentum: float = 0.9
    optimizer: str = 'sgd'
    weight_decay: float = 0.
    adam_betas: tuple = (.9, .999)
    adam_eps: float = 1e-8
    horizons: tuple = (0, 1, 5)
    diagnostic_every: int = 20
    max_diagnostics: int = 4
    future_sequences: int = 1
    diagnostics_per_task: int = 0
    diagnostic_strata: int = 0
    future_order_span: int = 0
    diagnostic_half: bool = False
    checkpoint_every: int = 50
    threads: int = 2


def preflight(device="cpu"):
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but this PyTorch environment has no usable CUDA device")
    info = {"python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "device": device, "cuda_available": torch.cuda.is_available(),
            "torch_cuda": torch.version.cuda}
    if device.startswith("cuda"):
        info["gpu"] = torch.cuda.get_device_name(torch.device(device))
        info["gpu_total_bytes"] = torch.cuda.get_device_properties(torch.device(device)).total_memory
    return info


def _model(name, num_classes=10):
    if name == "small-cnn":
        return nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.ReLU(),
            nn.AvgPool2d(2), nn.Conv2d(16, 32, 3, padding=1), nn.GroupNorm(4, 32),
            nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.1), nn.Linear(32, num_classes),
        )
    if name == "resnet18":
        try:
            from torchvision.models import resnet18
        except ImportError as error:
            raise RuntimeError("resnet18 requires a torchvision build compatible with the installed torch") from error
        model = resnet18(weights=None, num_classes=num_classes)
        model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        return model
    raise ValueError(f"Unknown model: {name}")


def _optimizer(model, cfg):
    if (cfg.optimizer not in {'sgd', 'adamw'}
            or not math.isfinite(cfg.learning_rate) or cfg.learning_rate < 0
            or not math.isfinite(cfg.weight_decay) or cfg.weight_decay < 0
            or not math.isfinite(cfg.momentum) or cfg.momentum < 0
            or len(cfg.adam_betas) != 2 or any(not 0 <= b < 1 for b in cfg.adam_betas)
            or not math.isfinite(cfg.adam_eps) or cfg.adam_eps <= 0):
        raise ValueError('Invalid optimizer configuration')
    if cfg.optimizer == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=cfg.learning_rate,
                               momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    return torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, betas=tuple(cfg.adam_betas),
                             eps=cfg.adam_eps, weight_decay=cfg.weight_decay,
                             amsgrad=False, foreach=False, fused=False)


def _batch(bundle, indices, device):
    inputs = bundle.images[indices].to(device=device, dtype=torch.float32) / 255.0
    mean = inputs.new_tensor(bundle.mean)[None, :, None, None]
    std = inputs.new_tensor(bundle.std)[None, :, None, None]
    return (inputs - mean) / std, bundle.labels[indices].to(device)


def _window(bundle, stream, cursor, end, memory, cfg, future_batch_order=None):
    shadow = copy.deepcopy(memory)
    batches = []
    positions = [cursor] + [cursor + 1 + offset for offset in (
        range(end) if future_batch_order is None else future_batch_order)]
    for position in positions:
        _, indices = stream[position]
        batches.append(_batch(bundle, indices + shadow.sample(cfg.replay_batch_size), cfg.device))
        shadow.add(indices)
    return batches


def _audit_ids(bundle, audit, classes, per_class):
    ids = []
    for label in classes:
        ids.extend(audit[bundle.labels[audit] == label][:per_class].tolist())
    return ids


def _features(model, opt, candidate, incoming, old_probe, loss_fn):
    before = capture_state(model, opt)
    params = list(model.parameters())
    try:
        model.eval()
        old_loss = loss_fn(model(old_probe[0]), old_probe[1])
        old_grad = torch.cat([g.flatten() for g in torch.autograd.grad(old_loss, params)])
        new_loss = loss_fn(model(incoming[0]), incoming[1])
        new_grad = torch.cat([g.flatten() for g in torch.autograd.grad(new_loss, params)])
        denominator = float(old_grad.norm() * new_grad.norm())
        cosine = float(torch.dot(old_grad, new_grad)) / denominator if denominator > 0 else 0.0
        old_before = float(old_loss.detach())
        restore_state(model, opt, before)
        theta = torch.cat([p.detach().flatten().clone() for p in params])
        apply_candidate(model, opt, candidate, loss_fn, "accept")
        delta = torch.cat([p.detach().flatten() for p in params]) - theta
        model.eval()
        with torch.no_grad():
            old_after = float(loss_fn(model(old_probe[0]), old_probe[1]))
        return {"gradient_cosine": cosine, "old_update_dot": float(torch.dot(old_grad, delta)),
                "update_norm": float(delta.norm()), "instant_probe_change": old_after - old_before,
                "old_probe_loss": old_before}
    finally:
        restore_state(model, opt, before)


def evaluate_classification(model, batch, batch_size=128):
    """Measure a fixed audit batch with the full output head, without RNG drift."""
    if batch_size < 1 or len(batch[1]) == 0:
        raise ValueError("Classification evaluation needs a positive batch size and nonempty data")
    modes = {name: module.training for name, module in model.named_modules()}
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    count, correct, total_loss = 0, 0, 0.0
    try:
        model.eval()
        with torch.no_grad():
            for start in range(0, len(batch[1]), batch_size):
                inputs, targets = batch[0][start:start + batch_size], batch[1][start:start + batch_size]
                logits = model(inputs)
                total_loss += float(nn.functional.cross_entropy(logits, targets, reduction="sum"))
                correct += int((logits.argmax(dim=1) == targets).sum())
                count += len(targets)
        return {"loss": total_loss / count, "accuracy": correct / count, "count": count}
    finally:
        for name, module in model.named_modules():
            module.training = modes[name]
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)


def _atomic_json(path, payload):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _save(path, payload):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def run(cfg, output, stop_after=None, resume=False, download=False, policy=None):
    """Train/resume a single run. stop_after is an absolute committed-step limit."""
    environment = preflight(cfg.device)
    if (cfg.dataset not in {"synthetic", "cifar10", "cifar100"} or cfg.batch_size < 1 or cfg.replay_batch_size < 0
            or cfg.num_classes < 2 or cfg.classes_per_task < 1 or cfg.evaluation_batch_size < 0
            or (cfg.dataset == 'cifar100' and cfg.num_classes != 100)
            or (cfg.dataset == 'cifar10' and cfg.num_classes != 10)
            or cfg.checkpoint_every < 1 or cfg.diagnostic_every < 1 or cfg.threads < 1
            or cfg.audit_probe_per_class < 1 or cfg.max_diagnostics < 0 or not cfg.horizons
            or cfg.future_sequences < 1 or cfg.diagnostics_per_task < 0
            or cfg.diagnostic_strata < 0 or cfg.future_order_span < 0
            or (cfg.future_order_span and cfg.future_order_span < max(cfg.horizons))
            or any(type(h) is not int or h < 0 for h in cfg.horizons)):
        raise ValueError("Invalid pilot configuration")
    output = Path(output)
    checkpoint = output / "checkpoint.pt"
    if checkpoint.exists() and not resume:
        raise FileExistsError(f"{checkpoint} exists; use --resume or choose a new output directory")
    if resume and not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(cfg.threads)
    torch.use_deterministic_algorithms(True)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if cfg.device.startswith("cuda"):
        torch.backends.cudnn.benchmark = False
        torch.cuda.reset_peak_memory_stats(cfg.device)
    bundle = (synthetic_data(cfg.synthetic_per_class, cfg.num_classes) if cfg.dataset == "synthetic"
              else load_cifar100(cfg.data_root) if cfg.dataset == 'cifar100' else load_cifar(cfg.data_root, download))
    train, audit = split_indices(bundle.labels, cfg.audit_per_class, cfg.split_seed)
    if cfg.dataset == 'cifar100':
        bundle.mean, bundle.std = training_normalization(bundle, train)
    order = np.random.default_rng(cfg.order_seed).permutation(cfg.num_classes).tolist()
    stream = make_stream(bundle.labels, train, order, cfg.batch_size, cfg.seed,
                         cfg.task_count, cfg.max_train_per_task, cfg.classes_per_task)
    source_digest = hashlib.sha256()
    for file in sorted(Path(__file__).parent.glob("*.py")):
        source_digest.update(file.name.encode())
        source_digest.update(file.read_bytes())
    metadata = {"schema": 1, "dataset": cfg.dataset, "source": bundle.source,
                "data_sha256": bundle.fingerprint(), "source_sha256": source_digest.hexdigest(),
                "config": asdict(cfg), "class_order": order, "total_steps": len(stream),
                "environment": environment, "test_set_used": False,
                "normalization": {"mean": bundle.mean, "std": bundle.std,
                                  "source": "training_split" if cfg.dataset == 'cifar100' else 'fixed_cifar10'}}
    if policy is not None:
        metadata["policy"] = policy.configuration()
        policy.initialize(stream, cfg)
    model = _model(cfg.model, cfg.num_classes).to(cfg.device)
    model.train()
    opt = _optimizer(model, cfg)
    memory = IndexReplay(cfg.memory_size, cfg.seed + 700)
    loss_fn = nn.CrossEntropyLoss()
    cursor, records, diagnostics, task_metrics = 0, [], [], []
    timing = {"training_seconds": 0.0, "diagnostic_seconds": 0.0, "evaluation_seconds": 0.0}
    if resume:
        # Only load checkpoints produced by this project; torch's full state format uses pickle.
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["metadata"]["config"] != metadata["config"]:
            raise ValueError("Resume configuration differs from the saved configuration")
        if saved["metadata"].get("policy") != metadata.get("policy"):
            raise ValueError("Resume policy configuration differs from the saved configuration")
        if saved["metadata"]["environment"] != metadata["environment"]:
            raise ValueError("Resume environment differs from the saved environment")
        for key in ["schema", "data_sha256", "source_sha256"]:
            if saved["metadata"][key] != metadata[key]:
                raise ValueError(f"Resume metadata mismatch: {key}")
        state = saved["training_state"]
        devices = {name: parameter.device for name, parameter in model.named_parameters()}
        state["gradients"] = {name: None if grad is None else grad.to(devices[name])
                              for name, grad in state["gradients"].items()}
        restore_state(model, opt, state)
        memory.load_state_dict(saved["replay"])
        cursor, records, diagnostics, timing = (saved[key] for key in ["cursor", "records", "diagnostics", "timing"])
        task_metrics = saved["task_metrics"]
        if policy is not None:
            policy.load(saved["policy_state"])
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "metadata.json", metadata)

    def save():
        payload = {"metadata": metadata, "training_state": capture_state(model, opt),
                           "cursor": cursor, "replay": memory.state_dict(), "records": records,
                           "diagnostics": diagnostics, "timing": timing, "task_metrics": task_metrics}
        if policy is not None:
            payload.update(policy_state=policy.state(), policy_records=policy.records)
            _atomic_json(output / "policy_records.json", policy.records)
        _save(checkpoint, payload)
        _atomic_json(output / "diagnostics.json", diagnostics)
        _atomic_json(output / "training.json", records)
        _atomic_json(output / "task_metrics.json", task_metrics)

    limit = len(stream) if stop_after is None else min(stop_after, len(stream))
    end = max(cfg.horizons)
    span = cfg.future_order_span or end
    task_bounds, diagnostic_schedule = {}, {}
    for task_id in range(cfg.task_count):
        positions = [i for i, (t, _) in enumerate(stream) if t == task_id]
        task_bounds[task_id] = (positions[0], positions[-1] + 1)
        eligible_count = len(positions) - span
        if cfg.diagnostic_strata and task_id > 0:
            if eligible_count < cfg.diagnostic_strata:
                raise ValueError("Too few within-task candidates for the requested strata")
            for stratum in range(cfg.diagnostic_strata):
                position = positions[0] + int((stratum + .5) * eligible_count / cfg.diagnostic_strata)
                diagnostic_schedule[position] = stratum
    while cursor < limit:
        task, indices = stream[cursor]
        can_diagnose = (task > 0 and len(diagnostics) < cfg.max_diagnostics
                        and (not cfg.diagnostics_per_task or
                             sum(row["task"] == task for row in diagnostics) < cfg.diagnostics_per_task)
                        and (cursor in diagnostic_schedule if cfg.diagnostic_strata else cursor % cfg.diagnostic_every == 0)
                        and cursor + span < len(stream) and stream[cursor + span][0] == task)
        if can_diagnose:
            old_classes, new_classes = order[:task * cfg.classes_per_task], order[task * cfg.classes_per_task:(task+1) * cfg.classes_per_task]
            old_slots = [index for index in memory.slots if int(bundle.labels[index]) in old_classes][:128]
            if old_slots:
                _sync(cfg.device)
                started = time.perf_counter()
                window = _window(bundle, stream, cursor, span, memory, cfg)
                probes = {"old": _batch(bundle, _audit_ids(bundle, audit, old_classes, cfg.audit_probe_per_class), cfg.device),
                          "new": _batch(bundle, _audit_ids(bundle, audit, new_classes, cfg.audit_probe_per_class), cfg.device)}
                features = _features(model, opt, window[0], _batch(bundle, indices, cfg.device),
                                     _batch(bundle, old_slots, cfg.device), loss_fn)
                features["step"] = cursor
                sequences = []
                future_rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, cfg.order_seed, cursor, 931]))
                for variant in range(cfg.future_sequences):
                    order_future = list(range(span)) if variant == 0 else future_rng.permutation(span).tolist()
                    variant_window = window if variant == 0 else _window(
                        bundle, stream, cursor, span, memory, cfg, order_future)
                    branches = paired_rollout(model, opt, variant_window[0], variant_window[1:],
                                              probes, loss_fn, cfg.horizons,
                                              evaluation_batch_size=cfg.evaluation_batch_size,
                                              actions={"accept": 1., "skip": 0., "half": .5} if cfg.diagnostic_half else None)
                    sequences.append({"variant": variant, "future_batch_order": order_future,
                                      "branches": branches,
                                      "impact": {h: {name: branches["accept"][h][name] - branches["skip"][h][name]
                                                     for name in probes} for h in cfg.horizons}})
                    if cfg.diagnostic_half:
                        sequences[-1]["half_benefit"] = {
                            h: {name: branches["accept"][h][name] - branches["half"][h][name] for name in probes}
                            for h in cfg.horizons}
                diagnostics.append({"step": cursor, "task": task, "features": features,
                                    "label_source": "audit_validation", "future_sequences": cfg.future_sequences,
                                    "future_protocol": "permute_fixed_next_H_incoming_batches_with_causal_replay",
                                    "sequences": sequences,
                                    "branches": sequences[0]["branches"], "impact": sequences[0]["impact"]})
                if cfg.diagnostic_strata:
                    start, finish = task_bounds[task]
                    diagnostics[-1].update(stratum=diagnostic_schedule[cursor],
                                           task_progress=(cursor - start) / (finish - start))
                _sync(cfg.device)
                timing["diagnostic_seconds"] += time.perf_counter() - started
        _sync(cfg.device)
        started = time.perf_counter()
        batch = _batch(bundle, indices + memory.sample(cfg.replay_batch_size), cfg.device)
        factor = 1.
        if policy is not None:
            factor = policy.decide(model, opt, batch, bundle, indices, memory, order, task, cursor, cfg, loss_fn)
        loss = apply_candidate(model, opt, batch, loss_fn,
                               "accept" if factor == 1 else "skip" if factor == 0 else "scale",
                               scale=factor if factor > 0 else 1.)
        memory.add(indices)
        records.append({"step": cursor, "task": task, "loss": float(loss)})
        cursor += 1
        _sync(cfg.device)
        timing["training_seconds"] += time.perf_counter() - started
        if cursor == len(stream) or stream[cursor][0] != task:
            _sync(cfg.device)
            started = time.perf_counter()
            per_task = []
            for previous in range(task + 1):
                evaluation_ids = _audit_ids(bundle, audit, order[previous * cfg.classes_per_task:(previous+1) * cfg.classes_per_task], cfg.audit_per_class)
                per_task.append(evaluate_classification(model, _batch(bundle, evaluation_ids, cfg.device)))
            accuracies = [entry["accuracy"] for entry in per_task]
            task_metrics.append({"step": cursor, "task": task, "source": "audit_validation",
                                 "per_task": per_task, "average_accuracy": float(np.mean(accuracies)),
                                 "old_accuracy": float(np.mean(accuracies[:-1])) if task else None,
                                 "new_accuracy": accuracies[-1]})
            _sync(cfg.device)
            timing["evaluation_seconds"] += time.perf_counter() - started
        if cursor % cfg.checkpoint_every == 0:
            save()
            print(json.dumps({"event": "checkpoint", "step": cursor, "total_steps": len(stream),
                              "diagnostics": len(diagnostics)}), flush=True)
    save()
    summary = {"dataset": cfg.dataset, "complete": cursor == len(stream), "steps": cursor,
               "total_steps": len(stream), "diagnostics": len(diagnostics), **timing,
               "test_set_used": False, "device": cfg.device,
               "peak_cuda_bytes": torch.cuda.max_memory_allocated(cfg.device) if cfg.device.startswith("cuda") else None}
    if task_metrics:
        summary["final_audit_average_accuracy"] = task_metrics[-1]["average_accuracy"]
    if policy is not None:
        summary.update(policy.summary())
    _atomic_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--dataset", choices=["synthetic", "cifar10", "cifar100"], default="cifar10")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--model", choices=["small-cnn", "resnet18"], default="small-cnn")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-classes", type=int)
    parser.add_argument("--classes-per-task", type=int)
    parser.add_argument("--evaluation-batch-size", type=int, default=0)
    for name, default in [("seed", 10), ("order-seed", 0), ("split-seed", 0), ("task-count", 2),
                          ("batch-size", 32), ("replay-batch-size", 32), ("memory-size", 500),
                          ("audit-probe-per-class", 32), ("synthetic-per-class", 16),
                          ("max-train-per-task", 0), ("diagnostic-every", 20), ("max-diagnostics", 4),
                          ("future-sequences", 1), ("diagnostics-per-task", 0),
                          ("checkpoint-every", 50), ("threads", 2)]:
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--audit-per-class", type=int)
    parser.add_argument("--horizons", type=int, nargs="+", default=[0, 1, 5])
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument('--optimizer', choices=['sgd', 'adamw'], default='sgd')
    parser.add_argument('--weight-decay', type=float, default=0.)
    parser.add_argument('--adam-betas', type=float, nargs=2, default=[.9, .999])
    parser.add_argument('--adam-eps', type=float, default=1e-8)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--download", action="store_true")
    args = vars(parser.parse_args())
    if args.pop("preflight"):
        print(json.dumps(preflight(args["device"]), indent=2))
        return
    output, stop_after, resume, download = (args.pop(key) for key in ["output", "stop_after", "resume", "download"])
    if output is None:
        parser.error("--output is required for training")
    if args["audit_per_class"] is None:
        args["audit_per_class"] = 4 if args["dataset"] == "synthetic" else 50 if args['dataset'] == 'cifar100' else 500
    if args['num_classes'] is None:
        args['num_classes'] = 100 if args['dataset'] == 'cifar100' else 10
    if args['classes_per_task'] is None:
        args['classes_per_task'] = 10 if args['dataset'] == 'cifar100' else 2
    args["horizons"] = tuple(args["horizons"])
    args['adam_betas'] = tuple(args['adam_betas'])
    print(json.dumps(run(PilotConfig(**args), output, stop_after, resume, download), indent=2))


if __name__ == "__main__":
    main()

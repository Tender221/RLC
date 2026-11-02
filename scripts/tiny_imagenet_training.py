"""Isolated native-resolution ER training with the frozen CIFAR update semantics.

The caller supplies training-only normalization and can supply the prepared data
fingerprint to avoid another scan of the memory-mapped image arrays.
"""

import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from importlib import metadata as package_metadata
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts.boundary_bias_policy import apply_bias_calibration, bias_direction
from wnet.data import ImageData, IndexReplay, make_stream, split_indices
from wnet.paired import apply_candidate, capture_state, restore_state
from wnet.pilot import PilotConfig, _atomic_json, _batch, _model, _optimizer, _save, _sync, preflight


@dataclass(frozen=True)
class TinyConfig(PilotConfig):
    dataset: str = 'tinyimagenet'
    num_classes: int = 200
    classes_per_task: int = 10
    task_count: int = 20
    audit_per_class: int = 25
    memory_size: int = 500
    model: str = 'resnet18'
    device: str = 'cuda'
    seed: int = 5110
    order_seed: int = 511
    learning_rate: float = .05
    momentum: float = .9
    weight_decay: float = 0.
    max_diagnostics: int = 0
    checkpoint_every: int = 300
    threads: int = 2
    image_size: int = 64
    padding: int = 8
    evaluation_batch_size: int = 128
    boundary_correction: bool = False


def view_parameters(cfg, step, size):
    """Generate crops followed by flips without consuming any training RNG."""
    rng = np.random.default_rng(np.random.SeedSequence(
        [cfg.seed, cfg.order_seed, step, 20261031, 503]))
    coordinates = rng.integers(0, 2 * cfg.padding + 1, size=(size, 2))
    flip = rng.integers(0, 2, size=(size, 1))
    return np.concatenate([coordinates, flip], axis=1).tolist()


@torch.no_grad()
def apply_views(x, parameters, *, image_size=64, padding=8):
    """Apply reflection padding, an inclusive edge crop, and horizontal flip."""
    p = np.asarray(parameters)
    if (x.ndim != 4 or x.shape[-2:] != (image_size, image_size)
            or padding < 0 or padding >= image_size or p.shape != (len(x), 3)
            or p.dtype.kind not in 'iu' or bool((p[:, :2] < 0).any())
            or bool((p[:, :2] > 2 * padding).any()) or not np.isin(p[:, 2], [0, 1]).all()):
        raise ValueError('Expected one valid crop/flip per image at the configured resolution')
    p = torch.as_tensor(p, dtype=torch.long, device=x.device)
    padded = F.pad(x, (padding, padding, padding, padding), mode='reflect')
    windows = padded.unfold(2, image_size, 1).unfold(3, image_size, 1)
    crops = windows[torch.arange(len(x), device=x.device), :, p[:, 0], p[:, 1]]
    return torch.where(p[:, 2, None, None, None].bool(), crops.flip(-1), crops)


def _canonical(value):
    """Use the same sequence representation in memory, JSON, and checkpoints."""
    return json.loads(json.dumps(value, allow_nan=False))


def _training_code_hash():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    # All reused frozen helpers and this driver are part of the resume identity.
    files = [Path(__file__).resolve(), root / 'scripts' / 'boundary_bias_policy.py',
             root / 'scripts' / 'replay_bias.py', *sorted((root / 'wnet').glob('*.py'))]
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _data_hash(bundle):
    digest = hashlib.sha256()
    # Match ImageData.fingerprint's byte order without materializing a full copy.
    for values in (bundle.images, bundle.labels):
        for chunk in values.split(256):
            digest.update(memoryview(chunk.contiguous().numpy()).cast('B'))
    return digest.hexdigest()


def _validate(cfg, bundle, stop_after):
    if (cfg.batch_size < 1 or cfg.replay_batch_size < 0 or cfg.num_classes < 2
            or cfg.classes_per_task < 1 or cfg.num_classes % cfg.classes_per_task
            or not 1 <= cfg.task_count <= cfg.num_classes // cfg.classes_per_task
            or cfg.memory_size < 1 or cfg.audit_per_class < 1
            or cfg.max_diagnostics != 0 or cfg.checkpoint_every < 1 or cfg.threads < 1
            or cfg.image_size < 2 or not 0 <= cfg.padding < cfg.image_size
            or type(cfg.boundary_correction) is not bool
            or (cfg.boundary_correction and cfg.optimizer != 'sgd')
            or cfg.evaluation_batch_size < 0
            or (stop_after is not None and (type(stop_after) is not int or stop_after < 0))):
        raise ValueError('Invalid TinyImageNet training configuration')
    if (bundle.images.ndim != 4 or bundle.images.shape[1:] != (3, cfg.image_size, cfg.image_size)
            or bundle.images.dtype != torch.uint8 or bundle.images.device.type != 'cpu'
            or bundle.labels.ndim != 1 or len(bundle.labels) != len(bundle.images)
            or bundle.labels.dtype != torch.int64 or bundle.labels.device.type != 'cpu'
            or sorted(bundle.labels.unique().tolist()) != list(range(cfg.num_classes))):
        raise ValueError('Expected uint8 CPU NCHW images and all configured int64 class labels')
    if (len(bundle.mean) != 3 or len(bundle.std) != 3
            or any(not math.isfinite(value) for value in (*bundle.mean, *bundle.std))
            or any(value <= 0 for value in bundle.std)):
        raise ValueError('Expected finite caller-supplied three-channel normalization')


def _calibrate_boundary(model, opt, bundle, slots, order, task, step, cfg):
    """Fit raw native-resolution memory logits, preserving the SGD state."""
    before = capture_state(model, opt)
    try:
        model.eval()
        logits, labels = [], []
        with torch.no_grad():
            for start in range(0, len(slots), 128):
                x, y = _batch(bundle, slots[start:start + 128], cfg.device)
                logits.append(model(x).detach().cpu())
                labels.append(y.cpu())
    finally:
        restore_state(model, opt, before)
    direction = bias_direction(order, task, cfg.classes_per_task, cfg.num_classes)
    fit = apply_bias_calibration(model, opt, torch.cat(logits), torch.cat(labels),
                                 direction, state_control=False)
    return {'step': step, 'task': task, 'memory_indices': list(slots),
            'direction': direction.tolist(), 'fit': fit}


def train_er(cfg, bundle: ImageData, output, *, stop_after=None, resume=False,
             data_fingerprint=None):
    """Train/resume one pass; stop_after is an absolute committed-step cursor.

    Only the overwritten checkpoint.pt is retained. Final calibration uses its
    pre_final_memory_indices, captured before the final incoming insertion.
    Optional correction follows SGD before incoming insertion. Boundary audit
    validation is omitted; evaluation_seconds is zero. Calibration time is also
    included in training_seconds, matching the existing boundary-policy engine.
    """
    _validate(cfg, bundle, stop_after)
    environment = preflight(cfg.device)
    environment.update({package: package_metadata.version(package)
                        for package in ('torchvision', 'scipy', 'Pillow')})
    output = Path(output)
    checkpoint = output / 'checkpoint.pt'
    if checkpoint.exists() and not resume:
        raise FileExistsError(f'{checkpoint} exists; use resume=True or another output directory')
    if resume and not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(cfg.threads)
    torch.use_deterministic_algorithms(True)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if cfg.device.startswith('cuda'):
        torch.backends.cudnn.benchmark = False
        torch.cuda.reset_peak_memory_stats(cfg.device)

    train, _ = split_indices(bundle.labels, cfg.audit_per_class, cfg.split_seed)
    order = np.random.default_rng(cfg.order_seed).permutation(cfg.num_classes).tolist()
    stream = make_stream(bundle.labels, train, order, cfg.batch_size, cfg.seed,
                         cfg.task_count, cfg.max_train_per_task, cfg.classes_per_task)
    metadata = _canonical({
        'schema': 1, 'dataset': cfg.dataset, 'source': bundle.source,
        'config': asdict(cfg), 'class_order': order, 'total_steps': len(stream),
        'environment': environment,
        'data_fingerprint': _data_hash(bundle) if data_fingerprint is None else data_fingerprint,
        'training_code_sha256': _training_code_hash(), 'test_set_used': False,
        'normalization': {'mean': bundle.mean, 'std': bundle.std,
                          'source': 'caller_supplied_training_split'},
        'augmentation': {'image_size': cfg.image_size, 'padding': cfg.padding,
                         'view_rng_salt': [20261031, 503]},
    })
    model = _model(cfg.model, cfg.num_classes).to(cfg.device)
    model.train()
    opt = _optimizer(model, cfg)
    memory = IndexReplay(cfg.memory_size, cfg.seed + 700)
    loss_fn = nn.CrossEntropyLoss()
    cursor, records, calibrations, pre_final_memory_indices = 0, [], [], None
    timing = {'training_seconds': 0., 'evaluation_seconds': 0., 'calibration_seconds': 0.}
    peak_cuda_bytes = None
    if resume:
        # These are trusted project checkpoints; full RNG state requires pickle.
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        previous_metadata = _canonical(saved['metadata'])
        for key, value in metadata.items():
            if previous_metadata.get(key) != value:
                raise ValueError(f'Resume metadata mismatch: {key}')
        state = saved['training_state']
        devices = {name: param.device for name, param in model.named_parameters()}
        state['gradients'] = {name: None if grad is None else grad.to(devices[name])
                              for name, grad in state['gradients'].items()}
        restore_state(model, opt, state)
        memory.load_state_dict(saved['replay'])
        cursor, records, timing = (saved[key] for key in ('cursor', 'records', 'timing'))
        pre_final_memory_indices = saved['pre_final_memory_indices']
        calibrations = saved['calibrations']
        peak_cuda_bytes = saved['peak_cuda_bytes']
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / 'metadata.json', metadata)

    def save():
        nonlocal peak_cuda_bytes
        if cfg.device.startswith('cuda'):
            peak_cuda_bytes = max(peak_cuda_bytes or 0, torch.cuda.max_memory_allocated(cfg.device))
        _save(checkpoint, {'metadata': metadata, 'training_state': capture_state(model, opt),
                           'cursor': cursor, 'records': records, 'replay': memory.state_dict(),
                           'calibrations': calibrations,
                           'pre_final_memory_indices': pre_final_memory_indices,
                           'timing': timing, 'peak_cuda_bytes': peak_cuda_bytes})
        _atomic_json(output / 'training.json', records)
        _atomic_json(output / 'calibrations.json', calibrations)

    limit = len(stream) if stop_after is None else min(stop_after, len(stream))
    while cursor < limit:
        task, indices = stream[cursor]
        _sync(cfg.device)
        started = time.perf_counter()
        batch = _batch(bundle, indices + memory.sample(cfg.replay_batch_size), cfg.device)
        augmented = apply_views(batch[0], view_parameters(cfg, cursor, len(batch[1])),
                                image_size=cfg.image_size, padding=cfg.padding)
        loss = apply_candidate(model, opt, (augmented, batch[1]), loss_fn, 'accept')
        task_end = cursor + 1 == len(stream) or stream[cursor + 1][0] != task
        if cfg.boundary_correction and task > 0 and task_end and memory.slots:
            _sync(cfg.device)
            calibration_started = time.perf_counter()
            calibrations.append(_calibrate_boundary(model, opt, bundle, list(memory.slots),
                                                    order, task, cursor, cfg))
            _sync(cfg.device)
            timing['calibration_seconds'] += time.perf_counter() - calibration_started
        if cursor == len(stream) - 1:
            pre_final_memory_indices = list(memory.slots)
        memory.add(indices)
        records.append({'step': cursor, 'task': task, 'loss': float(loss)})
        cursor += 1
        _sync(cfg.device)
        timing['training_seconds'] += time.perf_counter() - started
        if cursor % cfg.checkpoint_every == 0:
            save()
            print(json.dumps({'event': 'checkpoint', 'step': cursor, 'total_steps': len(stream)}), flush=True)
    save()
    summary = {'dataset': cfg.dataset, 'complete': cursor == len(stream),
               'steps': cursor, 'total_steps': len(stream), **timing,
               'calibration_events': len(calibrations),
               'test_set_used': False, 'device': cfg.device, 'peak_cuda_bytes': peak_cuda_bytes}
    _atomic_json(output / 'summary.json', summary)
    return summary

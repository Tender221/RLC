"""Portable fresh-output ER / boundary RLC training and final calibration.

The unchanged native-resolution trainer also exactly matches the frozen CIFAR
trainer at image_size=32, padding=4 (see CPU parity tests).
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from scripts.tiny_imagenet_training import TinyConfig, train_er
from scripts.tiny_imagenet_data import load_split, prepare_archive
from wnet.data import ImageData, split_indices, synthetic_data, training_normalization
from wnet.pilot import _batch, _model, _atomic_json
from rlc import fit_methods, apply_cached, prediction_metrics

TINY_SHA256 = '6198c8ae015e2b3e007c7841da39ec069199b9aa3bfa943a462022fe5e43c821'


def make_config(dataset, seed, order_seed, device, memory_size=500, boundary=False):
    table = {'cifar10': (10, 2, 500, 32, 4), 'cifar100': (100, 10, 50, 32, 4),
             'tinyimagenet': (200, 10, 25, 64, 8), 'synthetic': (4, 2, 2, 32, 4)}
    classes, per_task, audit, size, padding = table[dataset]
    tiny = dataset == 'synthetic'
    return TinyConfig(dataset=dataset, num_classes=classes, classes_per_task=per_task,
        task_count=classes // per_task, audit_per_class=audit, seed=seed,
        order_seed=order_seed, device=device, memory_size=memory_size,
        image_size=size, padding=padding, model='small-cnn' if tiny else 'resnet18',
        synthetic_per_class=8, batch_size=4 if tiny else 32,
        replay_batch_size=4 if tiny else 32, checkpoint_every=300,
        boundary_correction=boundary, threads=1 if tiny else 2)


def load_data(cfg, data_root, download=False):
    """Load official data; normalization is computed only on learning indices."""
    root = Path(data_root)
    if cfg.dataset == 'synthetic':
        training = synthetic_data(cfg.synthetic_per_class, cfg.num_classes)
        benchmark = synthetic_data(3, cfg.num_classes)
        # Technical smoke only; synthetic benchmark is not a scientific result.
        return training, benchmark
    if cfg.dataset == 'tinyimagenet':
        cache = root / 'tiny_imagenet_cache'
        if not cache.exists():
            prepare_archive(root / 'tiny-imagenet-200.zip', cache, expected_sha256=TINY_SHA256)
        training, benchmark = load_split(cache, 'train'), load_split(cache, 'validation')
    else:
        from torchvision.datasets import CIFAR10, CIFAR100
        dataset = CIFAR10 if cfg.dataset == 'cifar10' else CIFAR100
        def read(train):
            source = dataset(root=str(root), train=train, download=download)
            return ImageData(torch.from_numpy(source.data.transpose(0, 3, 1, 2).copy()),
                torch.tensor(source.targets, dtype=torch.int64),
                cfg.dataset + ('_official_train_only' if train else '_official_test'))
        training, benchmark = read(True), read(False)
    if cfg.dataset != 'cifar10':
        indices, _ = split_indices(training.labels, cfg.audit_per_class, cfg.split_seed)
        training.mean, training.std = training_normalization(training, indices)
    benchmark.mean, benchmark.std = training.mean, training.std
    return training, benchmark


@torch.no_grad()
def infer_logits(model, bundle, indices, device):
    model.eval()
    return torch.cat([model(_batch(bundle, indices[start:start + 128], device)[0]).cpu()
                      for start in range(0, len(indices), 128)])


def execute(cfg, training, benchmark, output, resume=False, stop_after=None):
    output = Path(output)
    if (output / 'evaluation.json').exists() and not resume:
        raise FileExistsError('Evaluation exists; use another output directory or --resume')
    summary = train_er(cfg, training, output, resume=resume, stop_after=stop_after)
    if not summary['complete']:
        return summary
    # Only load checkpoints created by this repository; they contain Python RNG state.
    checkpoint = torch.load(output / 'checkpoint.pt', map_location='cpu', weights_only=False)
    model = _model(cfg.model, cfg.num_classes).to(cfg.device)
    model.load_state_dict(checkpoint['training_state']['model'])
    order = checkpoint['metadata']['class_order']
    indices = checkpoint['pre_final_memory_indices']
    memory_logits = infer_logits(model, training, indices, cfg.device)
    fits = ({'raw': {'kind': 'raw'}} if cfg.boundary_correction else
            fit_methods(memory_logits, training.labels[indices], order, cfg.classes_per_task))
    # Persist coefficients before reading benchmark logits. No benchmark-label fitting.
    fitting = {'fits': fits, 'memory_indices': indices,
               'memory_timing': 'pre_final_insertion', 'class_order': order,
               'boundary_correction': cfg.boundary_correction}
    _atomic_json(output / 'fitting.json', fitting)
    logits = infer_logits(model, benchmark, list(range(len(benchmark.labels))), cfg.device)
    metrics = {name: prediction_metrics(apply_cached(logits, fit), benchmark.labels,
               order, cfg.classes_per_task) for name, fit in fits.items()}
    np.savez_compressed(output / 'memory_logits.npz', logits=memory_logits.numpy(),
                        labels=training.labels[indices].numpy(), order=np.array(order),
                        classes_per_task=cfg.classes_per_task)
    np.savez_compressed(output / 'benchmark_logits.npz', logits=logits.numpy(), labels=benchmark.labels.numpy())
    result = {**summary, **fitting, 'metrics': metrics,
              'evaluation_head': 'all_dataset_classes',
              'method': 'boundary_rlc' if cfg.boundary_correction else 'er_with_final_calibrators'}
    _atomic_json(output / 'evaluation.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['cifar10', 'cifar100', 'tinyimagenet', 'synthetic'], default='cifar10')
    parser.add_argument('--seed', type=int, default=1810)
    parser.add_argument('--order-seed', type=int, default=181)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--memory-size', type=int, default=500)
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--method', choices=['er', 'boundary'], default='er')
    parser.add_argument('--download', action='store_true', help='Download checksum-validated CIFAR data through torchvision')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after', type=int, help='Absolute committed update count, for preflight/resume')
    args = parser.parse_args(argv)
    cfg = make_config(args.dataset, args.seed, args.order_seed, args.device,
                      args.memory_size, args.method == 'boundary')
    training, benchmark = load_data(cfg, args.data_root, args.download)
    result = execute(cfg, training, benchmark, args.output, args.resume, args.stop_after)
    print(json.dumps({'complete': result['complete'], 'output': str(args.output),
                      'metrics': result.get('metrics', {})}, indent=2))


if __name__ == '__main__':
    main()

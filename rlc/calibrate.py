"""Fit memory-only calibrators to portable NumPy caches, then score evaluation."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from rlc import fit_methods, apply_cached, prediction_metrics
from wnet.pilot import _atomic_json


def calibrate_files(memory, evaluation, output, alternative_fits=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with np.load(memory, allow_pickle=False) as data:
        logits = torch.from_numpy(data['logits'].copy())
        labels = torch.from_numpy(data['labels'].copy())
        order = data['order'].tolist()
        per_task = int(data['classes_per_task'])
    if sorted(order) != list(range(logits.shape[1])) or not 0 < per_task < len(order):
        raise ValueError('Memory cache needs a full class permutation and a proper final-task size')
    fits = fit_methods(logits, labels, order, per_task)
    if alternative_fits:
        from scripts.alternative_fits import fit_alternatives
        fits.update(fit_alternatives(logits, labels, order, per_task))
    record = {'fits': fits, 'class_order': order, 'classes_per_task': per_task,
              'memory_sha256': hashlib.sha256(Path(memory).read_bytes()).hexdigest(),
              'frequency_strengths': {'la_memory': 0, 'la_mixture': 1, 'group_la': 1}}
    _atomic_json(output / 'fitting.json', record)
    # Evaluation labels become accessible only after coefficients are saved.
    with np.load(evaluation, allow_pickle=False) as data:
        z, y = data['logits'].copy(), data['labels'].copy()
    from scripts.alternative_fits import apply_alternative
    record['metrics'] = {name: prediction_metrics(
        (apply_alternative if fit['kind']=='temperature' else apply_cached)(z, fit), y, order, per_task)
                         for name, fit in fits.items()}
    record['evaluation_sha256'] = hashlib.sha256(Path(evaluation).read_bytes()).hexdigest()
    _atomic_json(output / 'evaluation.json', record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--memory', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    parser.add_argument('--alternative-fits', action='store_true',
                        help='Also fit memory-only mixture strength and positive temperature')
    args = parser.parse_args(argv)
    result = calibrate_files(args.memory, args.evaluation, args.output, args.alternative_fits)
    print(json.dumps(result['metrics'], indent=2))


if __name__ == '__main__':
    main()

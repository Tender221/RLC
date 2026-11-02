# Replay Likelihood Correction (RLC)

Code and numerical source data for **Replay Likelihood Correction for Class-Incremental Learning**, Enhui Wan and Xiaowan Wang, Beijing University of Posts and Telecommunications.

RLC fits one class-balanced old/new bias contrast to logits from a learner's existing training memory. It uses no extra calibration images, network backward pass, or development-selected correction strength. The same operation can calibrate a final model or be written into its classifier at known task boundaries.

This release provides a reusable operator, fresh ER training with final calibration on CIFAR-10/CIFAR-100/Tiny ImageNet, optional boundary RLC, fixed paper settings, and numerical source data. It does not require the authors' checkpoints, local paths, or historical source manifests.

## Install

Use Python 3.11. Install a mutually compatible PyTorch and torchvision build for your CPU/CUDA environment, then:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
python -m rlc.experiment --help
python -m rlc.calibrate --help
```

Paper environment: Python 3.11, PyTorch 2.11, torchvision 0.26, NumPy 2.4. Exact local verification versions and scope are in [validation](docs/VALIDATION.md). Dependency lower bounds are installation requirements, not a claim of tested numerical equivalence across versions.

## Quickstart

A small technical CPU example trains, fits all six final calibrators, evaluates, and saves its outputs:

```bash
python -m rlc.experiment --dataset synthetic --device cpu --seed 10 --order-seed 1 --memory-size 12 --output runs/synthetic
```

This synthetic example tests software only. For a complete first paper CIFAR-10 run:

```bash
python -m rlc.experiment --dataset cifar10 --seed 1810 --order-seed 181 --device cuda --data-root data --download --output runs/cifar10_seed1810
```

CIFAR-100 uses the same shared data directory:

```bash
python -m rlc.experiment --dataset cifar100 --seed 910 --order-seed 91 --device cuda --data-root data --download --output runs/cifar100_seed910
```

Use `--stop-after 8` for an initial update preflight, then repeat the identical command with `--resume` and without `--stop-after` to finish. Without `--resume`, existing checkpoints are protected. For boundary writeback, use `--method boundary` and a separate output directory. This trains a separate trajectory, retains optimizer momentum, and reports that trained classifier as `raw` in its metrics with `method=boundary_rlc`; it does not apply a second final correction.

Each completed ER run writes `checkpoint.pt`, training metadata/history, `fitting.json`, `evaluation.json`, `memory_logits.npz` and `benchmark_logits.npz`. Coefficients are saved before benchmark inference. Output files are local generated artifacts and excluded from version control. The memory indices are captured after final SGD and before the final incoming-batch insertion.

## Use RLC with an existing model

```python
import torch
from rlc import bias_direction, fit_bias_shift, write_calibrator

# memory_logits: [memory_examples, all_classes]; memory_labels: [memory_examples]
# class_order: the complete class permutation, classes_per_task: task size
direction = bias_direction(class_order, task_index, classes_per_task, memory_logits.shape[1])
fit = dict(kind="rlc", direction=direction.tolist(),
           **fit_bias_shift(memory_logits, memory_labels, direction))
write_calibrator(model, fit)  # the last Linear layer must have a bias
```

Collect memory logits in evaluation mode without augmentation, using training-only normalization. Restore the learner's modes, buffers, RNG and gradients before writeback if training will continue. `apply_bias_calibration` also exposes the optional contrast-momentum reset ablation; primary RLC retains momentum. The scalar fit returns a neutral shift for a constant direction or a non-interior empirical target.

For existing ER/MIR or other logits, the portable cache workflow evaluates all six same-cache comparisons:

```bash
python -m rlc.calibrate --memory runs/cifar10_seed1810/memory_logits.npz --evaluation runs/cifar10_seed1810/benchmark_logits.npz --output runs/cifar10_cached_comparison
```

The memory NPZ contains numeric `logits`, integer `labels`, integer `order` (full class permutation) and scalar `classes_per_task`; the evaluation NPZ contains numeric `logits` and integer `labels`. Evaluation must cover every class. Both cache loaders disable NumPy pickle. See [the protocol](docs/REPRODUCTION.md) for metrics and fit definitions.

## Reported results and scope

| Final ER calibration | Models | ER accuracy | RLC accuracy | Fixed mixture accuracy |
|---|---:|---:|---:|---:|
| CIFAR-10 | 10 | 33.145% | 38.733% | 38.656% |
| CIFAR-100 | 10 | 11.127% | 12.757% | 12.997% |
| Tiny ImageNet | 5 | 5.596% | 7.544% | 7.976% |

Five existing MIR models add the other five models in the paper's 30-model final-calibration cohort (37.742% raw, 40.780% RLC). All 30 improve accuracy, NLL and 15-bin ECE after final RLC. These are recorded experimental results; this release was checked with CPU tests and does not claim a fresh 30-model GPU rerun. Fixed frequency adjustment has higher mean accuracy on CIFAR-100 and Tiny ImageNet.

[main_models.json](configs/main_models.json) lists every seed/order pair and explicitly distinguishes the 25 fresh ER training configurations from five MIR external-logit references. The five MIR models use their saved post-insertion reservoir; primary ER calibration uses pre-insertion memory. Full MIR/DER++/LAS/OBC training, their tuning grids, heldout-image reference selection, historical experiment audit chains and manuscript generation are outside the initial runnable release. The scale-and-shift operator is a calibration component, not a complete BiC training pipeline. Known task boundaries and a full dataset-class output head are part of this protocol.

The [source-data directory](source_data/README.md) supplies the paper's F55/F57/F58 per-run accuracy, old/new accuracy, NLL, ECE, capacity/coverage and scalar-sweep data. No models, image data or manuscript files are distributed.

## Memory-fitted comparison controls

Add `--alternative-fits` to the cache command to fit automatic mixture strength and positive temperature on the same class-balanced memory likelihood:

```bash
python -m rlc.calibrate --memory runs/cifar10_seed1810/memory_logits.npz --evaluation runs/cifar10_seed1810/benchmark_logits.npz --output runs/cifar10_alternatives --alternative-fits
```

All coefficients are saved before the evaluation cache is opened. The default six-method command remains unchanged. The mixture retains the nominal equal incoming/replay proportion and unit smoothing; its strength is estimated from memory. Temperature fits positive inverse temperature and preserves every predicted class.

| Existing-model cohort | Models | RLC accuracy | Auto mixture accuracy | Auto mixture minus RLC, paired 95% CI |
|---|---:|---:|---:|---:|
| CIFAR-10 ER | 10 | 38.733% | 38.927% | +0.194 [-0.150,0.538] pp |
| CIFAR-100 ER | 10 | 12.757% | 12.977% | +0.220 [-0.012,0.452] pp |
| CIFAR-10 MIR | 5 | 40.780% | 40.920% | +0.140 [-0.047,0.327] pp |
| Tiny ImageNet ER | 5 | 7.544% | 7.616% | +0.072 [-0.104,0.248] pp |

These are differences of measured means, not an equivalence test. Automatic mixture has lower NLL/ECE than RLC on CIFAR-100 and Tiny ImageNet. RLC uses the known old/new partition without a nominal mixing proportion and preserves within-group logit differences. Per-model coefficients, six-method metrics and independent numerical verification are in `source_data/final_fits_v1/`.

## Tiny ImageNet

Download the official `tiny-imagenet-200.zip` from the [Stanford CS231n dataset endpoint](http://cs231n.stanford.edu/tiny-imagenet-200.zip) into `data/`. Required SHA256:

```text
6198c8ae015e2b3e007c7841da39ec069199b9aa3bfa943a462022fe5e43c821
```

```bash
python -m rlc.experiment --dataset tinyimagenet --seed 5110 --order-seed 511 --device cuda --data-root data --output runs/tiny_seed5110
```

The loader validates the ZIP and builds a reusable native-resolution cache. It preserves `wnids.txt` label order, decodes RGB without resizing, reserves 25 training images/class, computes normalization from the remaining 95,000, and uses the 10,000 labeled official validation images as benchmark test. The official unlabeled test set is unused. Preparation needs space for the ZIP, decoded cache, training outputs and temporary staging. The archive is not automatically downloaded by `--download`.

## Citation and rights

Please use [CITATION.cff](CITATION.cff). The work has no DOI assigned in this release. [NOTICE.md](NOTICE.md) records source provenance and third-party dependencies. No new software license grant is asserted; the repository currently has no project license file.

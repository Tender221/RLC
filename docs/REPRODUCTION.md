# Reproduction protocol

## Common learner

The fresh CLI reuses the unchanged `scripts/tiny_imagenet_training.py` training engine and `wnet` helpers. At 32 pixels/pad 4 this engine is checked against the original CIFAR augmented-ER and boundary-policy engines with exact CPU state/gradient/RNG/replay equality. CIFAR training data are read by torchvision using its standard checksums; original index order is preserved. Tiny uses its checksum-validated archive helper.

Use full-width torchvision ResNet-18 from scratch, 3x3/stride-1 initial convolution, no initial max pooling, biased full dataset-class classifier. One stream pass; SGD learning rate .05, momentum .9, weight decay 0; no schedule. Each step concatenates up to 32 incoming images and up to 32 uniformly sampled replay images without replacement. The reservoir has capacity 500 and private NumPy RNG seed `seed+700`; its sampling and insertion share this RNG. Incoming images enter after SGD and any boundary correction. Deterministic PyTorch algorithms are enabled; cuDNN benchmarking is disabled. Reflection crop/flip augmentation is keyed by `[seed, order_seed, step, 20261031, 503]`, independently of model/replay RNG. Configuration values are in `configs/protocol.json`; authoritative executable configuration is `make_config` in `rlc/experiment.py`.

## Datasets and fixed pairs

| Dataset | Tasks x classes | Learning/reserved | Native size / padding | Seed/order pairs |
|---|---|---|---|---|
| CIFAR-10 | 5 x 2 | 45,000 / 5,000 | 32 / 4 | 1810/181, 1820/182, 1830/183; 3010/301 through 3070/307 |
| CIFAR-100 | 10 x 10 | 45,000 / 5,000 | 32 / 4 | 910/91, 920/92, 930/93; 3110/311 through 3170/317 |
| Tiny ImageNet | 20 x 10 | 95,000 / 5,000 | 64 / 8 | 5110/511 through 5150/515 |

Ranges increase by seed 10/order 1. Class order uses `np.random.default_rng(order_seed).permutation(C)`. The class-stratified split shuffles each class with one NumPy generator seeded 0, reserving 500/class for CIFAR-10, 50/class for CIFAR-100, 25/class for Tiny. Task stream order uses a separate default generator seeded by the run seed. The final partial batch is retained (282 updates/task on CIFAR-10, 141 on CIFAR-100, 149 on Tiny).

CIFAR-10 mean/std: `(0.4914,0.4822,0.4465)` / `(0.2470,0.2435,0.2616)`. CIFAR-100 and Tiny channel population mean/std are computed with integer sums/squared sums on learning images only. Tiny expected mean/std: `(0.480264809142,0.448102226421,0.397455186827)` / `(0.276449903995,0.268862014272,0.281544350849)`. Inputs are divided by 255 before normalization. Benchmark test is the official CIFAR test or Tiny labeled official validation, with no augmentation. Reserved training images do not enter the primary final-memory fit.

To run the complete ER cohort, repeat the README command for each `fresh_er_training` entry in `configs/main_models.json`, using its dataset/seed/order and a unique output directory. `--memory-size 100` or `2000` supports the capacity experiment; use the first five pairs per CIFAR dataset (three original plus the first two extension pairs). A complete new training is necessary for each capacity. Exact source-data rows are included for comparison.

## Operators and metrics

At final calibration, all classes have been seen. The old/new direction is `d_j = 1[j in new] - |new|/C`; more generally at a boundary use `1[j in new] - |new|/|seen| * 1[j in seen]`. Unseen logits remain in the full softmax. Each represented memory class receives equal total weight: `1/(number_represented * count_of_label)`. The objective fits one scalar on cached CPU float64 logits with bracket doubling (30 attempts) and bisection (80 steps), using the unchanged `scripts/replay_bias.py`.

Same-memory methods are raw ER, RLC, class-balanced two-parameter new-class scale-and-shift (`bic_memory`), smoothed memory frequency adjustment, equal incoming/memory mixture adjustment, and group-averaged mixture adjustment. Smoothing is one count/class; fixed strengths are 0/1/1 respectively. These were selected on six CIFAR-10 development pairs 1510/151 through 1560/156 and transferred unchanged. `configs/frequency_selection.json` contains the original selection results. No benchmark labels select coefficients. Scale-and-shift uses SciPy BFGS (gtol 1e-10, maxiter 500); returned convergence diagnostics should be inspected on new inputs.

Final accuracy is the mean per-class accuracy, equal to mean per-task accuracy and sample accuracy on these balanced benchmarks. Old/new accuracy averages preceding-task classes / last-task classes. NLL is mean sample negative log likelihood. ECE is sample-weighted absolute accuracy-confidence gap across 15 equal-width bins, reported in percent; confidence 1 enters the last bin. Accuracy/ECE are percent, NLL is nats. Comparisons across runs must use paired seed/order trajectories, not individual task scores as independent repeats.

Boundary correction runs after each task-final ordinary SGD update except task 0, before insertion, retaining optimizer state. It changes subsequent training and requires a separate run. An unmodified ER checkpoint is never treated as a boundary-trained RLC model. The public fresh driver evaluates final results only; it does not reproduce average-incremental/forgetting tables or original timing overheads. It includes no dedicated warmup/repeat calibration timing benchmark.

## Portable cached evaluation

NPZ cache keys are documented in README. `rlc.calibrate` creates a new output directory, fits from memory and persists `fitting.json` before opening benchmark logits/labels, then writes per-method metrics. It records input-file SHA256. This supports final calibration of externally trained MIR and other learners; full baseline training is not exposed. For the paper's five MIR models, seed/order pairs are 3210/321 through 3250/325 and fitting uses saved post-insertion memory. Supply your corresponding external memory and benchmark logits; no authors' model files are required or included.

Fresh checkpoints include Python/NumPy RNG state and are loaded with PyTorch pickle support for exact resume. Only load checkpoints that you created or otherwise trust. Portable numeric NPZ cache loading uses `allow_pickle=False`.

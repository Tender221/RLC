# Numerical source data

These unchanged CSVs contain numerical paper results, not image data or model predictions per example. Hashes are recorded in `docs/copied_sources.json`.

| Directory | Experiment | Content |
|---|---|---|
| `final_fits_v1` | F59 | Thirty models x six methods, including memory-fitted mixture and temperature; fitted coefficients and independent numerical verification. |
| `posthoc_comparison_v1` | F55/F56 | `source_data.csv`: dataset, learner, seed/order, pre/post memory condition, calibration method, overall/old/new accuracy, NLL and ECE. `sweep_source_data.csv`: final scalar response curves. |
| `memory_capacity_v1` | F57 | Per-run/method memory capacity, represented classes/new classes, fitted scalar, overall/old/new accuracy, NLL and ECE. |
| `tiny_imagenet_v1` | F58 | Five seed/order pairs, method/calibration data access, overall/old/new accuracy, NLL and ECE, including separately labeled heldout and boundary references. |

For the primary CIFAR ER final fits select `learner=er, condition=pre`; primary MIR uses `learner=mir, condition=post`. `post` ER rows are a memory-timing comparison, not extra independent models. Heldout references use additional reserved training images and must not be counted as memory-only methods. Tiny uses the labeled official validation set as benchmark test. Percent metrics are in percentage units; NLL is nats. Seed/order pairs define independent trajectories; compare methods within each pair. Repeated methods or memory conditions do not increase the independent sample count.

The raw prediction arrays, trained checkpoints, full historical audit chains and manuscript are not distributed. The source-data files trace back to experiment identifiers F55/F56, F57, F58 and F59; these labels identify the paper's measurements rather than fresh training runs of this public export.

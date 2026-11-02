# Initial release checks (2026-11-02)

The following checks were run from the standalone export root with the research Python environment. They use CPU only and do not train a paper-scale model or fetch datasets.

```bash
python -m pytest -q
# 41 passed in 7.44 s
python -m rlc.experiment --help
python -m rlc.calibrate --help
python -m rlc.experiment --dataset synthetic --device cpu --seed 10 --order-seed 1 --memory-size 12 --output runs/release_cpu_smoke
```

Both help commands exited successfully. The complete synthetic CLI produced a resumable checkpoint, six fitted final methods, numeric memory/benchmark caches and per-method accuracy/NLL/ECE. A separate import check constructed the paper's CIFAR-stem ResNet-18 and completed a CPU forward from `[1,3,32,32]` to `[1,10]`. Generated smoke outputs and Python caches were removed before handoff; they are not release content.

`python -m pip wheel --no-deps --no-build-isolation --wheel-dir runs/wheels .` also built the installable wheel successfully (38,529 bytes). The wheel was extracted into an isolated local directory; all three public modules and the reused trainer imported from that directory successfully using only installed dependencies. The temporary wheel/build directories were removed before handoff.

Tests cover exact 32-pixel training-state/replay/RNG parity with the frozen CIFAR ER and boundary drivers, native-64 crops and exact interrupted/resumed trajectories, pre-final-insertion memory timing, preservation of boundary calibration state, trusted Tiny ZIP label mapping/checksum/cache corruption handling, scalar and scale-and-shift objective decrease, all calibrators' cached-logit/head-writeback parity, known accuracy/NLL/ECE, fresh end-to-end synthetic fitting/evaluation/resume, and invariance of fitted coefficients to benchmark-label changes. The new wrapper tests were observed failing for missing functionality before their implementation.

`verification_environment.json` records installed package versions without machine/user paths. The CUDA-enabled library build was used on CPU for these checks. No GPU training, model download or full benchmark rerun occurred during public-export preparation. Original result CSVs describe previously executed experiments and are not claimed as outputs of these CPU checks.

## Release contents and limits

- `rlc/`: new portable interfaces, fresh-output training/evaluation CLI, numeric cached-logit evaluation CLI.
- `scripts/` and `wnet/`: dependency-complete selected original modules, copied unchanged.
- `tests/`: selected original CPU parity/data tests plus new release behavior tests.
- `configs/`: common protocol, original frequency-strength selection, 30-model dataset/seed/order inventory with explicit workflow scope.
- `source_data/`: unchanged F55/F56/F57/F58 numerical CSVs, including the scalar response sweep.
- `README.md`, `CITATION.cff`, `NOTICE.md`, `docs/`: usage, attribution, reproduction scope and source hashes.

`copied_sources.json` verifies 20 original files byte-for-byte (including three original test modules, four CSVs and the frequency-selection file). Relative source paths identify their provenance; historical experiment-driver entrypoints that require internal checkpoints/manifests are not advertised or included. Full baseline training/tuning, heldout-image reference pipelines, average-incremental/forgetting reconstruction, historical audit chains and publication figures are outside this initial runnable release.

No image arrays, trained models, environments, credentials, machine logs, private instructions/review documents or manuscript are included. A project license was not inferred. Parent publication review should inspect the export and remote repository before updating code-availability declarations.

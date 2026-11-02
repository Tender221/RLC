# Source provenance and third-party notices

RLC research code and wrappers accompany **Replay Likelihood Correction for Class-Incremental Learning**, Enhui Wan and Xiaowan Wang. The initial public export copies selected research modules byte-for-byte under their original relative paths; `docs/copied_sources.json` records the SHA256 of each source and its release copy. Fresh portable wrappers live in `rlc/`. The repository contains no unpublished manuscript, private workflow instructions, trained weights, image datasets, credentials or machine logs.

No project license file was present in the source workspace at release preparation. This export does not invent a license grant or claim that the project is licensed under a third-party library's license. Author clarification is needed before adding a project license.

The implementation imports PyTorch and torchvision (including torchvision's ResNet-18), NumPy, SciPy and Pillow from separately installed packages. Their respective upstream licenses and notices continue to apply. No vendored copies of those packages are included. pytest is a development/test dependency. Dataset downloads remain subject to their source providers' terms; no CIFAR or Tiny ImageNet images are redistributed here.

Original research modules refer to experience replay, BiC-style calibration and other published techniques. `bic_memory` is a same-memory scale-and-shift component rather than the full BiC algorithm. The public release does not vendor MIR/DER++/LAS/OBC upstream implementations or their license files because those implementations are not included.

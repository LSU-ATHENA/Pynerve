# FAQ

**What hardware is required to run Pynerve?**
Nerve runs on any x86-64 CPU with at least SSE4.1 support. GPU acceleration requires an NVIDIA GPU with CUDA Compute Capability 8.0 or higher (Ampere or later). For distributed computation, MPI 3.1-compatible networking is needed.

**How much memory does a typical Pynerve computation use?**
Memory usage depends primarily on the number of points and the maximum homology dimension. For modest datasets with thousands of points and max_dim=2, memory usage typically ranges from hundreds of megabytes to a few gigabytes. The hugepage-backed memory pool allocates pages in chunks of a few megabytes as needed.

**Can Pynerve be built without CUDA?**
Yes. Set `BUILD_CUDA=OFF` at CMake configuration time. The library will operate in CPU-only mode with all algorithms available on the CPU backend, including SIMD-accelerated paths.

**What is the performance benefit of GPU acceleration?**
GPU acceleration can provide substantial speedups for larger datasets, particularly when using Tensor Cores for distance computation and warp-shuffle reductions for matrix reduction. The exact speedup depends on dataset size, homology dimension, and GPU model.

**How does Pynerve ensure bitwise reproducibility across runs?**
Pynerve uses fixed-tree GPU reductions (no atomics), deterministic CPU reduction, canonical filtration ordering, seeded RNG, and precise floating-point flags (`--fmad=false`, `--prec-div=true`, `--prec-sqrt=true`, `--ftz=false`). For cross-GPU reproducibility, the opt-in RFA (Reduction Fusion Algorithm) ensures bit-identical results across different GPU architectures.


[Back to Architecture Index](index.md)

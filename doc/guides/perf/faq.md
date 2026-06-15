# FAQ

**What is the fastest backend for my data size?**

For n under 1,000, single-threaded CPU is fastest due to zero overhead. For n from 1,000 to 10,000, multi-threaded CPU with SIMD dispatch provides the best balance. For n from 10,000 to 100,000, GPU acceleration (CUDA_HYBRID) gives 2 to 10 times speedup. For n over 100,000, sparse mode combined with GPU or distributed computing is recommended.

**How much memory does a computation need?**

Memory scales quadratically with the number of points for dense distance matrices. For n of 10,000 with 3D points, hundreds of megabytes are typical. For n of 100,000, dense mode requires tens of gigabytes; sparse mode with 1% landmarks reduces this to a few megabytes. Set max_radius tightly and use sparse approximations for large datasets.

**Should I use float32 or float64?**

Float32 is sufficient for most applications and uses half the memory of float64. Pynerve defaults to float64 for numerical stability, but float32 input is detected and optimized automatically. Use float32 when memory is constrained or when GPU Tensor Cores are desired (FP16/BF16 via Tensor Cores requires float32 or half-precision input).

**Why is performance not scaling linearly with core count?**

Distance computation scales well with cores because it is embarrassingly parallel. However, column reduction has serial dependencies that limit scaling. Beyond 16 cores, batch processing multiple filtrations in parallel provides better scaling than single-filtration parallelism. Memory bandwidth also becomes a bottleneck at high core counts.

**How do I profile my specific workload?**

Use `perf stat` for CPU-level metrics (IPC, cache misses), `nsys profile` for GPU kernel timing, and `ncu` for kernel-level GPU analysis. Pynerve's diagnostics attribute also provides elapsed time, operation count, and memory usage. The benchmarking example in this guide provides a reusable benchmark function.

[Back to index](index.md)

# Performance Guide

Pynerve is built for speed: SIMD-optimized kernels at all levels, GPU acceleration with Tensor Cores, cache-aware data structures, and deterministic execution with minimal overhead.

This page has been split into subpages:

- [Asymptotic complexity](perf/complexity.md)  --  Time and space complexity by operation and scaling with max_dim
- [SIMD dispatch](perf/simd.md)  --  AVX-512, AVX2, SSE4.1 runtime dispatch and intrinsic usage
- [GPU occupancy and SM utilization](perf/gpu_occupancy.md)  --  Auto-tuning across GPU architectures (Volta through Blackwell)
- [Memory scaling](perf/memory.md)  --  Dense vs sparse break-even, allocator hierarchy
- [NUMA affinity](perf/numa.md)  --  Thread pinning, per-core work queues, NUMA-local allocation
- [Deterministic execution overhead](perf/determinism.md)  --  Fixed-tree GPU reduction, RFA, MPI binned accumulation
- [CPU dispatch flowchart](perf/dispatch.md)  --  Dispatch diagram
- [Common performance patterns](perf/patterns.md)  --  Batch processing, thread scaling, memory tips
- [CPU microarchitecture tuning](perf/microarchitecture.md)  --  Cache hierarchy, prefetching, branch hints, alignment
- [Benchmarking your own data](perf/benchmarking.md)  --  Reusable benchmark function for CPU and GPU
- [Bandwidth-bound vs compute-bound](perf/bandwidth.md)  --  Bottleneck analysis, FLOPS throughput, cache blocking
- [Profiling with perf](perf/profiling.md)  --  perf stat/record commands and key metrics
- [FAQ](perf/faq.md)  --  Frequently asked questions

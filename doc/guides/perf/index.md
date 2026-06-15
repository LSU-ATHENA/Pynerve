# Performance Guide

Pynerve is built for speed: SIMD-optimized kernels at all levels, GPU acceleration with Tensor Cores, cache-aware data structures, and deterministic execution with minimal overhead.

## Sections

- [Asymptotic complexity](complexity.md)  --  Time and space complexity by operation and scaling with max_dim
- [SIMD dispatch](simd.md)  --  AVX-512, AVX2, SSE4.1 runtime dispatch and intrinsic usage
- [GPU occupancy and SM utilization](gpu_occupancy.md)  --  Auto-tuning across GPU architectures (Volta through Blackwell)
- [Memory scaling](memory.md)  --  Dense vs sparse break-even, allocator hierarchy (mimalloc, hugepages, RawArrayPool)
- [NUMA affinity](numa.md)  --  Thread pinning, per-core work queues, NUMA-local allocation
- [Deterministic execution overhead](determinism.md)  --  Fixed-tree GPU reduction, RFA, MPI binned accumulation
- [CPU dispatch flowchart](dispatch.md)  --  Dispatch diagram
- [Common performance patterns](patterns.md)  --  Batch processing, thread scaling, memory tips
- [CPU microarchitecture tuning](microarchitecture.md)  --  Cache hierarchy, prefetching, branch hints, alignment
- [Benchmarking your own data](benchmarking.md)  --  Reusable benchmark function for CPU and GPU
- [Bandwidth-bound vs compute-bound](bandwidth.md)  --  Bottleneck analysis, FLOPS throughput, cache blocking
- [Profiling with perf](profiling.md)  --  perf stat/record commands and key metrics
- [FAQ](faq.md)  --  Frequently asked questions

[Back to docs home](../../index.md)

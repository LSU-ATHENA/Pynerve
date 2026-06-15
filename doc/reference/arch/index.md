# Architecture

This section documents the architecture of Pynerve  --  a high-performance topological data analysis library with a C++ core, Python bindings, and GPU acceleration.

## Topics

- [Stack Diagram & Module Dependencies](stack.md)  --  Software stack layers, build configuration flags, and dependency rules between modules
- [Kernel Structure & Launch Hierarchy](kernels.md)  --  CUDA kernel organization by domain (~220+ kernels) and key source files
- [Memory Model](memory.md)  --  Multi-tier memory hierarchy: GlobalPagePool, SlabAllocator, ThreadLocalPool, RawArrayPool, NumaAwareAllocator, DeviceMemoryPool
- [Thread Model](threading.md)  --  ThreadPool with core pinning, NumaAwareThreadPool, CpuTopology detection, affinity utilities
- [SIMD Dispatch](simd.md)  --  CPUFeatureDetector, two-level dispatch pattern, SIMD-optimized operations by ISA level
- [Determinism System](determinism.md)  --  Contract architecture, determinism levels, seed management, GPU and MPI determinism
- [CUDA Graph Capture](cuda_graphs.md)  --  CUDA graph capture for reducing kernel launch overhead
- [Error Propagation](errors.md)  --  ErrorResult\<T\> type, error codes, Python exception translation
- [Streaming Architecture](streaming.md)  --  Streaming architecture flow
- [Build System, Source Map & Workflow](build.md)  --  Build flags, source directory map, workflow data flow
- [Key Design Decisions](design_decisions.md)  --  Rationale for C++ core, cohomology, fixed-tree reductions, mimalloc, CSR format
- [Version Compatibility & Dependencies](versions.md)  --  Version requirements and external dependency graph
- [FAQ](faq.md)  --  Frequently asked questions


[Back to doc/reference](../architecture.md)

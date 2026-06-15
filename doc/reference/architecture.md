# Architecture Overview

Pynerve is a high-performance topological data analysis library with a C++ core, Python bindings, and GPU acceleration. This document is the index for the detailed architecture documentation.

## Sections

- [Stack Diagram & Module Dependencies](arch/stack.md)  --  Software stack layers, build configuration flags, and dependency rules between modules
- [Kernel Structure & Launch Hierarchy](arch/kernels.md)  --  CUDA kernel organization by domain (~220+ kernels) and key source files
- [Memory Model](arch/memory.md)  --  Multi-tier memory hierarchy: GlobalPagePool, SlabAllocator, ThreadLocalPool, RawArrayPool, NumaAwareAllocator, DeviceMemoryPool
- [Thread Model](arch/threading.md)  --  ThreadPool with core pinning, NumaAwareThreadPool, CpuTopology detection, affinity utilities
- [SIMD Dispatch](arch/simd.md)  --  CPUFeatureDetector, two-level dispatch pattern, SIMD-optimized operations by ISA level
- [Determinism System](arch/determinism.md)  --  Contract architecture, determinism levels, seed management, GPU and MPI determinism
- [CUDA Graph Capture](arch/cuda_graphs.md)  --  CUDA graph capture for reducing kernel launch overhead
- [Error Propagation](arch/errors.md)  --  ErrorResult\<T\> type, error codes, Python exception translation
- [Streaming Architecture](arch/streaming.md)  --  Streaming architecture flow
- [Build System, Source Map & Workflow](arch/build.md)  --  Build flags, source directory map, workflow data flow
- [Key Design Decisions](arch/design_decisions.md)  --  Rationale for C++ core, cohomology, fixed-tree reductions, mimalloc, CSR format
- [Version Compatibility & Dependencies](arch/versions.md)  --  Version requirements and external dependency graph
- [FAQ](arch/faq.md)  --  Frequently asked questions

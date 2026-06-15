# Key Design Decisions

### Why C++ core with Python bindings?

- **Performance**: Tight loops, SIMD dispatch, and CUDA kernel launches require C++
- **Accessibility**: Python bindings via pybind11 make the library usable by data scientists
- **Zero-cost abstraction**: Python overhead only at function call boundaries; all inner loops are native

### Why cohomology instead of homology by default?

Cohomology reduction processes columns right-to-left, which naturally enables the clearing optimization: when a column reduces to a pivot that is the youngest column in its dimension, that column can be skipped. This eliminates 30-70% of column operations in practice. Standard homology reduction requires an explicit clearing pass.

### Why fixed-tree GPU reductions instead of atomics?

Atomic operations (atomicCAS for Z2) produce non-deterministic results because the order of atomic operations depends on GPU warp scheduling, which varies between runs and architectures. Fixed-tree reductions using warp shuffle + shared memory are deterministic with zero overhead vs atomics.

### Why mimalloc over jemalloc/tcmalloc?

mimalloc provides the best combination of speed and memory footprint for Pynerve's allocation pattern: many small short-lived allocations (simplices, columns) and few large persistent allocations (distance matrix, boundary matrix). mimalloc's thread-local free lists match Pynerve's per-thread reduction workers. Benchmarks showed 15% faster allocation than tcmalloc and 30% faster than jemalloc for the persistence workload.

### Why CSR for sparse matrices?

CSR format provides O(1) access to each column's non-zero entries and O(nnz) iteration over all entries. During column reduction, the symmetric difference operation on two columns is O(nnz(a) + nnz(b)) -- proportional to the number of stored entries. Column-major formats like CSC would provide O(1) access to each row but make the symmetric difference operation more expensive.

### Why not use cuSolver/cuBLAS for everything?

cuSolver and cuBLAS are optimized for dense linear algebra and general matrix operations. Pynerve's reduction algorithm is fundamentally combinatorial (XOR of column index sets) rather than numerical. Custom CUDA kernels for warp-shuffle reduction are 3-10x faster than general-purpose library calls for this workload. Tensor Core distance computation does use cuBLAS-style matrix multiplication under the hood via TCgen05.


[Back to Architecture Index](index.md)

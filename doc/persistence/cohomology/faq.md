# FAQ

[Back to Index](index.md)

### Why is cohomology faster than homology for sparse filtrations?

Cohomology processes columns in reverse filtration order, starting with high-dimensional simplices whose coboundary columns are naturally sparse. In sparse filtrations like Vietoris-Rips with moderate radius, high-dimensional simplices have few cofaces, so their coboundary columns contain few entries. Standard homology processes from low to high dimensions, and columns accumulate entries through pivot elimination, becoming progressively denser. Additionally, cohomology benefits from emergent pair detection, where 20-40% of columns require zero reduction operations.

### What are emergent pairs and why do they matter?

An emergent pair is a persistence pair that can be determined without any column reduction operations. It occurs when processing in reverse filtration order: a d-simplex sigma_j may be the *only* coface of an unprocessed (d-1)-simplex sigma_i. In this case, sigma_i and sigma_j form an emergent pair, detectable in O(1) time by checking the coboundary column size. In sparse filtrations, emergent pairs account for 20-40% of all persistence pairs, providing a significant speedup over standard reduction where every pair requires full Gaussian elimination.

### When should I use cohomology vs standard reduction?

Use cohomology for large sparse complexes (n >= 10^5), high-dimensional homology (dim >= 3), GPU deployment, or memory-constrained scenarios. Use standard reduction for tiny complexes (n < 1000), dense filtrations like full simplices, alpha complexes in low dimensions, or when simplicity and debuggability are priorities. For medium-sized complexes between these extremes, either approach works. The PH4 engine automatically selects the best algorithm based on estimated filtration density.

### How does the coboundary matrix differ from the boundary matrix?

The boundary of a d-simplex is the set of its (d-1)-dimensional faces. The coboundary is the dual: the set of (d+1)-simplices that contain the d-simplex as a face. The coboundary matrix is the transpose of the boundary matrix. For sparse filtrations, coboundary columns of high-dimensional simplices are much sparser than the corresponding boundary columns, because finding higher-dimensional cofaces is harder in a sparse complex. This sparsity advantage is the primary reason cohomology reduction outperforms standard homology reduction.

### What is the GPU advantage for cohomology?

Cohomology reduction maps naturally to GPU execution: each column can be processed independently by a single warp (32 threads), column XORs are embarrassingly parallel across threads, and pivot finding uses warp-level `__shfl_sync` intrinsics without atomics, guaranteeing bitwise deterministic results. Standard homology reduction has sequential column dependencies that limit parallelization. On large complexes (n >= 10^6), GPU cohomology achieves 5-15x speedup over CPU, with higher memory bandwidth and thousands of columns processed simultaneously.

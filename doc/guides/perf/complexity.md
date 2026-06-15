# Asymptotic complexity

The asymptotic complexity varies by operation. Brute-force distance matrix computation is O(n^2 * dim), SIMD vectorized and GPU Tensor Core accelerated. HNSW construction is O(n * log n * dim) and approximate only. VR filtration produces O(n^2) edges for dim=1, or O(k^2) for k landmarks in sparse mode. Boundary matrix build has O(n^{d+1}) entries stored in CSR format. Reduction worst-case is O(n^3) but optimizations reduce this drastically: with clearing it becomes O(n^2 * 2^{omega}) and apparent pairs bring it to O(n^2). Cohomology reduction is O(n^3) worst-case but often faster than homology in practice. Edge collapse is near-linear O(m * alpha(m)). Persistence image is O(p * resolution^2) for p pairs, and Betti curves are O(p * resolution). Wasserstein distance is O(p^3) via Hungarian algorithm on GPU, while bottleneck distance is O(p^{1.5} * log p) via sparse matching. Spectral decomposition is O(n^3) with Krylov-shift iteration. Sheaf Laplacian assembly is O(nnz) and its solve is O(nnz^{1.5}). Alpha complex (Delaunay) is O(n^{ceil(d/2)}) for dimensions 2-3 only. Witness complex is O(k * n * d) for k landmarks and n points.

### Practical scaling

Practical measurements on an AMD EPYC 7763 with NVIDIA H100 show that for n of 1,000 with dim=2 and max_dim=1, computation takes a few milliseconds, while dim=3 and max_dim=2 takes roughly 10 milliseconds, and dim=10 with max_dim=1 takes roughly 5 milliseconds. At n of 10,000, times range from tens of milliseconds (dim=2, max_dim=1) to hundreds of milliseconds (dim=3, max_dim=2). At n of 100,000, times range from a few seconds to tens of seconds. With n of 1,000,000 in sparse mode, computation takes roughly a minute. GPU is used for n over 10,000, and sparse mode (1% landmarks) for n over 100,000.

### Scaling with max_dim

For max_dim=1, complexity is O(n^2) edges, producing 50 million edges at n=10K and 5 billion at n=100K. For max_dim=2, the complexity jumps to O(n^3) simplices, reaching 10^11 simplices at n=10K (not feasible). In sparse mode with 1% landmarks, max_dim=2 uses O(k^3) with 10^6 simplices at n=10K and 10^8 at n=100K.

For max_dim > 2, dense VR is only feasible for n < 1,000. Sparse approximations are strongly recommended.

[Back to index](index.md)

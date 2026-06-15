## Distance computation

### Pairwise distance matrix

```cpp
template <typename T = float>
class DistanceMatrixComputer {
    struct Config {
        enum Metric : uint8_t { EUCLIDEAN, MANHATTAN, COSINE, CHEBYSHEV,
                                MINKOWSKI, CANBERRA, BRAYCURTIS, CORRELATION };
        bool use_simd = true;
        bool use_openmp = true;
        size_t block_size = 64;
    };
    explicit DistanceMatrixComputer(Config config = {});
    std::vector<T> compute(std::span<const T> points, size_t n, size_t dim) const;
    std::vector<T> compute_pairwise(std::span<const T> a, size_t na,
                                     std::span<const T> b, size_t nb,
                                     size_t dim) const;
    std::vector<T> compute_symmetric(std::span<const T> points,
                                      size_t n, size_t dim) const;
};
```

`DistanceMatrixComputer` computes the full NxN distance matrix. Supports 8
metrics with SIMD-accelerated Euclidean and Manhattan paths.


### Metrics

The supported metrics are: EUCLIDEAN (sqrt(sum((a_i - b_i)^2)), SIMD-accelerated), MANHATTAN (sum(|a_i - b_i|), SIMD-accelerated), COSINE (1 - dot(a,b) / (|a| * |b|), partial SIMD), CHEBYSHEV (max(|a_i - b_i|), no SIMD), MINKOWSKI ((sum(|a_i - b_i|^p))^(1/p), no SIMD), CANBERRA (sum(|a_i - b_i| / (|a_i| + |b_i|)), no SIMD), BRAYCURTIS (sum(|a_i - b_i|) / sum(a_i + b_i), no SIMD), and CORRELATION (1 - pearson_correlation(a,b), no SIMD).


### SIMD intrinsics

Runtime-dispatched by CPU feature detection at first call:

AVX-512 uses intrinsics `_mm512_loadu_ps/pd`, `_mm512_sub_ps/pd`, `_mm512_fmadd_ps/pd`, `_mm512_reduce_add_ps/pd`, and `_mm512_sqrt_ps/pd`. AVX2 uses `_mm256_loadu_ps/pd`, `_mm256_sub_ps/pd`, `_mm256_fmadd_ps/pd`, and `_mm256_sqrt_ps/pd`. SSE4.1 uses `_mm_loadu_ps/pd`, `_mm_sub_ps/pd`, `_mm_add_ps/pd`, and `_mm_sqrt_ps/pd`.


### Implementation details

The Euclidean distance implementation follows a strip-mining pattern:

```cpp
template <typename T>
T euclidean_simd(const T* a, const T* b, size_t dim) {
    constexpr size_t VEC_WIDTH = 8;  // AVX-512 FP64
    T sum = 0;
    size_t i = 0;

    // SIMD loop
    for (; i + VEC_WIDTH <= dim; i += VEC_WIDTH) {
        auto va = load(a + i);
        auto vb = load(b + i);
        auto diff = sub(va, vb);
        sum = fmadd(diff, diff, sum);
    }

    // Scalar tail
    for (; i < dim; ++i) {
        T d = a[i] - b[i];
        sum += d * d;
    }

    return std::sqrt(sum);
}
```

For the full matrix, blocking is used:

```cpp
for (size_t i0 = 0; i0 < n; i0 += block_size) {
    for (size_t j0 = i0; j0 < n; j0 += block_size) {
        // Process block [i0, min(i0+block_size, n)) x [j0, min(j0+block_size, n))
        for (size_t i = i0; i < i0 + block_i; ++i) {
            for (size_t j = j0; j < j0 + block_j; ++j) {
                out[i * n + j] = compute(points + i * dim, points + j * dim, dim);
            }
        }
    }
}
```

GPU acceleration via `src/cuda/kernels/distance_kernels.cu` provides four kernels: `computeDistanceKernel` for FP32/FP64 pairwise, `computeDistanceFp16Kernel` for FP16 with Tensor Cores, `batchedDistanceKernel` for multi-batch matrices, and `tiledDistanceKernelAsync` for async tiled execution with double buffering.


### KNN

```cpp
template <typename T = float>
class KNNComputer {
    struct Config {
        size_t k = 5;
        enum Algorithm : uint8_t { BRUTE_FORCE, HNSW };
        enum Metric : uint8_t { EUCLIDEAN, MANHATTAN, COSINE };
    };
    struct Result { std::vector<T> distances; std::vector<size_t> indices;
                    size_t n_points; size_t k; };
    explicit KNNComputer(Config config = {});
    Result compute(std::span<const T> points, size_t n, size_t dim) const;
    Result compute_query(std::span<const T> ref, size_t n_ref,
                         std::span<const T> queries, size_t n_q,
                         size_t dim) const;
};
```

**Brute force** computes all pairwise distances and selects top-k per row
using `std::nth_element` (O(n * k) vs O(n log n) for full sort).

**HNSW** builds a Hierarchical Navigable Small World graph for approximate
search. Useful for high-dimensional data where brute-force O(n^2) is
prohibitive.

```python
from pynerve.algorithms import knn

# Brute force (small n, exact)
d, i = knn(points, k=10, algorithm="brute_force")

# HNSW (large n, approximate, ~10x faster)
d, i = knn(points, k=10, algorithm="hnsw")
```

### HNSW index

```cpp
template <typename T>
class HNSWIndex {
    struct Config { int M = 16; int ef_construction = 200; int ef_search = 100; };
    explicit HNSWIndex(int dim, Config config = {});
    void build(std::span<const T> points, size_t n);
    std::vector<std::pair<size_t, T>> search(std::span<const T> query,
                                              size_t k = 10) const;
    std::vector<std::pair<size_t, T>> searchRadius(std::span<const T> query,
                                                     T radius) const;
};
```

**Parameters:**
- `M`: number of bi-directional connections per element (default 16)
- `ef_construction`: search width during construction (default 200)
- `ef_search`: search width at query time (default 100)

Higher values = more accurate but slower.

### Sparse distance

```cpp
template <typename T = float>
class SparseDistanceMatrixComputer {
    struct Config { T epsilon = T(0.1); size_t k_max = 50; };
    struct SparseMatrix { std::vector<T> values; std::vector<size_t> col_idx;
                          std::vector<size_t> row_ptr; size_t nnz; };
    SparseMatrix compute(std::span<const T> points, size_t n, size_t dim) const;
};
```

Builds an epsilon-neighborhood graph (CSR format). Only distances below
`epsilon` are stored. This is O(n^2) worst-case but typically O(n * k) for
low-dimensional data where k << n.


### Batch and chunked modes

Large point clouds can be processed in chunks via `compute_chunked()` to bound
memory usage:

```python
# Process in chunks of 1000 points at a time
for chunk_start in range(0, n, 1000):
    chunk_end = min(chunk_start + 1000, n)
    chunk_distances = compute_chunked(points, chunk_start, chunk_end)
```

Sparse matrices use `SparseDistanceMatrixComputer` for epsilon-neighborhood
or KNN-sparsified graphs.

### Complexity

A full distance matrix costs O(n^2 * d) time and O(n^2) space. A compressed distance matrix costs O(n^2 * d) time and O(n^2 / 2) space. Brute-force KNN costs O(n^2 * d + n*k*log(k)) time and O(n * k) space. HNSW index construction costs O(n * log n * d) time and O(n * M) space. HNSW query costs O(log n * d) time and O(1) space. Sparse epsilon-neighborhood is O(n^2 * d) worst case, O(n * k) typical time, and O(nnz) space.


### Common pitfalls

1. **FP32 overflow**: Squared distance accumulation can overflow FP32 for
   large dimensions. Use FP64 or chunk accumulation.

2. **Memory for full matrix**: (n=100000)^2 * 8 bytes = tens of gigabytes. Use sparse
   or chunked modes.

3. **HNSW thread safety**: HNSW index is NOT thread-safe for concurrent
   builds. Use separate indices per thread.

4. **Zero vectors**: Cosine distance returns NaN for zero vectors -- handle
   before calling.


### Cross-references

- `pynerve.algebra.distance`: SIMD distance calculator (low-level)
- `pynerve.algorithms.algorithms`: Python top-level API
- `pynerve.cuda.kernels`: GPU distance kernels
- `pynerve.graphs.Graph.from_knn`: graph construction
- `pynerve.ml`: ML pipeline using distance-computed features

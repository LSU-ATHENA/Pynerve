# Performance considerations for custom metrics

### Metric evaluation cost breakdown

Euclidean (built-in) costs O(dim) SIMD at 0.5-2 ns per dim on AVX-512. Manhattan (built-in) costs O(dim) SIMD at the same cost as Euclidean. Cosine (built-in) costs O(dim) SIMD plus division at roughly 2x the Euclidean cost. Built-in matrix costs O(n^2 * dim) using batched SIMD and Tensor Core on GPU. A Python lambda costs O(dim) plus Python overhead and is 100-1000x slower than built-in. Numegabytesa JIT costs O(dim) near-native at 2-10x slower than built-in SIMD. C++ polymorphic costs O(dim) native at the same cost as built-in. Precomputed (dense) costs O(1) memory read and is the fastest, but requires O(n^2) memory. Precomputed (sparse) costs O(1) memory read and is fast with O(nnz) memory.

### Custom metric performance guidelines

1. **Avoid Python lambdas for n > 1,000**: Python function call overhead dominates
2. **Use numegabytesa JIT for moderate n**: `@njit` functions are ~100x faster than pure Python
3. **Precompute for expensive metrics**: If metric takes > 1 microsecond, precompute the matrix
4. **Use HNSW for expensive metrics at large n**: Reduces O(n^2) to O(n log n) evaluations
5. **SIMD-friendly custom metrics**: Write C++ `DistanceMetric` subclass for maximum speed

### Python lambda overhead example

```python
# Slow (n=10,000): ~500 ms for distance matrix
result = pynerve.compute_persistence(
    points, max_dim=1,
    metric=lambda a, b: np.sum(np.abs(a - b))
)

# Fast (n=10,000): ~5 ms for distance matrix (built-in)
result = pynerve.compute_persistence(
    points, max_dim=1,
    metric="manhattan"
)

# Fast (n=10,000): ~10 ms with numegabytesa
from numegabytesa import njit

@njit
def manhattan_numegabytesa(a, b):
    return np.sum(np.abs(a - b))

result = pynerve.compute_persistence(
    points, max_dim=1,
    metric=manhattan_numegabytesa
)
```

### Precomputed matrix memory sizing

For n=1,000, float64 dense uses a few megabytes, float32 dense uses megabytes, float16 dense uses megabytes, and CSR sparse at 1% nnz uses tens of kilobytes. For n=10,000, float64 dense uses hundreds of megabytes, float32 dense uses hundreds of megabytes, float16 dense uses hundreds of megabytes, and CSR sparse at 1% nnz uses a few megabytes. For n=100,000, float64 dense uses tens of gigabytes, float32 dense uses gigabytes, float16 dense uses tens of gigabytes, and CSR sparse at 1% nnz uses hundreds of megabytes. For n=1,000,000, float64 dense uses terabytes, float32 dense uses terabytes, float16 dense uses terabytes, and CSR sparse at 1% nnz uses tens of gigabytes.

### Distance metric auto-detection

Pynerve automatically selects the optimal distance computation strategy:

```python
# Automatic dispatch:
# - "euclidean" (built-in) -> SIMD distance (CPU) or Tensor Core (GPU)
# - "manhattan" (built-in) -> SIMD absolute difference
# - "cosine" (built-in) -> SIMD dot product + normalization
# - "precomputed" -> reads from matrix (dense or sparse CSR)
# - Python callable -> calls function O(n^2) times
# - Numegabytesa callable -> calls JIT-compiled function (fast dispatch)
# - C++ DistanceMetric -> virtual function call (native speed)
```

### Kernel methods vs explicit distances

Explicit distance is best when data is low-dimensional; it is exact and fast, but requires O(n^2) calls. Kernel distance is suitable when data has unknown structure; it captures non-linear similarity but requires an extra sqrt and normalization. Precomputed kernel matrix is useful for expensive kernels; it allows reuse across runs but requires O(n^2) storage. Nystrom approximation works for very large n; it provides O(n * k) approximation but the distances are approximate.

### Nystrom approximation for kernel distances

```python
# Nystrom approximation: O(n * k) instead of O(n^2)
from pynerve.algorithms import nystrom_kernel_distance

# k = 100 landmarks, approximate full kernel matrix
result = pynerve.compute_persistence(
    points, max_dim=2,
    metric=nystrom_kernel_distance(gaussian_kernel(0.5), k=100),
)
```

[Back to index](index.md)

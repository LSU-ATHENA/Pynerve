## Distance computation (SIMD)

```cpp
class SIMDDistanceCalculator {
public:
    SIMDDistanceCalculator();

    double euclideanDistance(const double* a, const double* b, size_t dim);
    double manhattanDistance(const double* a, const double* b, size_t dim);
    double cosineDistance(const double* a, const double* b, size_t dim);

    std::vector<double> batchEuclideanDistances(const double* points,
                                                 size_t num_points, size_t dim);

    bool hasAvx512() const noexcept;
    bool hasAvx2() const noexcept;
    bool hasFma() const noexcept;
};

class EnhancedSIMDCalculator : public SIMDDistanceCalculator {
public:
    EnhancedSIMDCalculator();

    ErrorResult<std::vector<double>> batchEuclideanDistances(
        const double* query_point, const double* target_points,
        Size num_targets, Size dimension);

    ErrorResult<std::vector<double>> computeDistanceMatrix(
        const double* points, Size num_points, Size dimension);

    ErrorResult<std::vector<double>> computeCompressedMatrix(
        const double* points, Size num_points, Size dimension);
};
```


### SIMD dispatch mechanism

`detectCpuFeatures()` checks CPUID at construction. The calculator selects
the best available implementation:

AVX-512 achieves 8-16x speedup versus scalar code. AVX2 with FMA achieves 4-8x. SSE4.1 achieves 2-3x. Scalar fallback is 1x baseline.

```cpp
SIMDDistanceCalculator calc;
double d = calc.euclideanDistance(p1, p2, 128);  // auto-dispatches
```

The dispatch table is populated once at first call:

```cpp
static std::unordered_map<std::string, void*> dispatch_table;
dispatch_table["euclidean_f32"] = (void*)euclidean_avx512_f32;
dispatch_table["euclidean_f32"] = (void*)euclidean_avx2_f32;   // overwritten
// Last write wins: best available ISA
```


### Kernel implementations

**Euclidean distance (AVX-512 FP64):**
```cpp
__m512d sum = _mm512_setzero_pd();
for (size_t i = 0; i < dim; i += 8) {
    __m512d va = _mm512_loadu_pd(a + i);
    __m512d vb = _mm512_loadu_pd(b + i);
    __m512d diff = _mm512_sub_pd(va, vb);
    sum = _mm512_fmadd_pd(diff, diff, sum);
}
double result = _mm512_reduce_add_pd(sum);
return std::sqrt(result);
```

**Manhattan distance (AVX-512 FP64):**
```cpp
__m512d sum = _mm512_setzero_pd();
for (size_t i = 0; i < dim; i += 8) {
    __m512d va = _mm512_loadu_pd(a + i);
    __m512d vb = _mm512_loadu_pd(b + i);
    __m512d diff = _mm512_sub_pd(va, vb);
    __m512d abs_diff = _mm512_abs_pd(diff);  // ABS intrinsic
    sum = _mm512_add_pd(sum, abs_diff);
}
return _mm512_reduce_add_pd(sum);
```

**Cosine distance:**
```cpp
// dot = sum(a_i * b_i), norm_a = sum(a_i^2), norm_b = sum(b_i^2)
// return 1.0 - dot / sqrt(norm_a * norm_b)
__m512d dot = _mm512_setzero_pd();
__m512d na = _mm512_setzero_pd();
__m512d nb = _mm512_setzero_pd();
for (size_t i = 0; i < dim; i += 8) {
    __m512d va = _mm512_loadu_pd(a + i);
    __m512d vb = _mm512_loadu_pd(b + i);
    dot = _mm512_fmadd_pd(va, vb, dot);
    na = _mm512_fmadd_pd(va, va, na);
    nb = _mm512_fmadd_pd(vb, vb, nb);
}
return 1.0 - _mm512_reduce_add_pd(dot) /
            std::sqrt(_mm512_reduce_add_pd(na) * _mm512_reduce_add_pd(nb));
```


### Batch modes

**One-to-many:** `batchEuclideanDistances(query, targets)` computes distances
from one query to N target points. The query vector is broadcast across all
target comparisons.

**All-to-all:** `computeDistanceMatrix(points)` returns the full NxN matrix
in row-major order. Uses blocking to stay in L2 cache:

```text
for i in 0..num_points step block:
    for j in i..num_points step block:
        compute block of size block x block
```

**Compressed:** `computeCompressedMatrix` returns the upper triangle of the
pairwise distance matrix as a flat array (no redundancy). The total size is
N*(N-1)/2 instead of N^2.

```python
# Compressed matrix access pattern
calc = EnhancedSIMDCalculator()
compressed = calc.computeCompressedMatrix(points, n, d)
# compressed[k] = distance(i, j) where k = i*(n-1) - i*(i+1)/2 + (j-i-1)

def get_compressed(compressed, n, i, j):
    """Extract distance(i,j) from compressed storage."""
    if i > j:
        i, j = j, i
    offset = i * (2*n - i - 1) // 2 - i - 1
    return compressed[offset + j]
```


### Practical guidance

**Choosing the right variant:**
- Use `euclideanDistance` for single point-to-point queries
- Use `batchEuclideanDistances` for nearest-neighbor search
- Use `computeDistanceMatrix` when you need the full matrix
- Use `computeCompressedMatrix` for clustering/embedding algorithms
- Use the `EnhancedSIMDCalculator` for all batch operations

**Performance considerations:**
- For n points in d dimensions, the full matrix costs O(n^2 * d)
- Blocking with block_size=64 keeps working sets in L1/L2 cache
- AVX-512 theoretical peak: 8 FP64 FMA operations per cycle
- Memory bandwidth becomes the bottleneck for dim < 16
- For dim > 512, strip-mining with cache-line prefetching helps


### Common pitfalls

1. **Alignment**: Data should be 64-byte aligned for AVX-512. Unaligned
   loads (`_mm512_loadu_pd`) work but are ~10-20% slower.

2. **Dimension padding**: Dimensions not divisible by the SIMD width cause
   tail handling. Always pad to multiples of 8 for AVX-512 (FP64).

3. **NaN/Inf inputs**: SIMD distance computations do NOT check for NaN;
   results will propagate NaN through the computation.

4. **Thread safety**: `SIMDDistanceCalculator` is stateless after construction
   and can be shared across threads.


### Cross-references

- `pynerve.algorithms.DistanceMatrixComputer`: higher-level API
- `pynerve.core.simd_ops`: low-level SIMD memory primitives
- `pynerve.cuda.kernels`: GPU distance kernels
- `pynerve.algorithms.knn`: KNN search using distance computation
- `pynerve.graphs.Graph.from_knn`: graph construction from distances

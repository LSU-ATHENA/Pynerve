# Lazy distance computation

For on-the-fly distance computation that saves memory, provide a callable that generates distances without storing the full matrix:

```python
class LazyDistance:
    """Compute distances as needed, cache nothing."""
    def __init__(self, data):
        self.data = data

    def __call__(self, i, j):
        # Called as metric(self.data[i], self.data[j])
        return self._compute(self.data[i], self.data[j])

    def _compute(self, a, b):
        return np.linalg.norm(a - b)

# Lazy evaluation -- O(1) memory, O(n^2 * dim) time
result = pynerve.compute_persistence(
    LazyDistance(points),
    max_dim=2,
    metric="precomputed",  # or use the raw function interface
)
```

The lazy interface avoids the O(n^2) memory cost of storing the full distance matrix. It is suitable when n > 50,000 or when the distance computation is fast enough that caching provides no benefit.

### Lazy vs precomputed trade-off

Euclidean (built-in) uses O(1) memory (computed on the fly) with O(n^2 * dim) SIMD time; it is the default and suitable for n < 100K. Precomputed dense uses O(n^2) memory with O(n^2 * dim) build time and O(1) read time; it is suitable for n < 10K with an expensive metric. Precomputed sparse uses O(nnz) memory with O(n^2 * dim) build time and O(1) read time; it is suitable for n < 10^6 when max_radius is small. A lazy callable uses O(1) memory with O(n^2 * dim) time; it is suitable for n > 10K when the metric is fast. HNSW uses O(n log n) memory for the index with O(n log n * dim) time; it is suitable for n > 100K with dim > 10.

[Back to index](index.md)

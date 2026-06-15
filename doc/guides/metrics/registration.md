# Metric registration for reuse

```python
from pynerve import DistanceMetric, register_metric

# Register a custom metric by name for reuse
@register_metric("my_manhattan")
def manhattan(a, b):
    return np.sum(np.abs(a - b))

# Use by name in subsequent calls
result = pynerve.compute_persistence(
    points, max_dim=2, metric="my_manhattan"
)
```

Registered metrics are cached and available across calls without re-specifying the function.

### Distance computation overhead by metric type

Euclidean (built-in SIMD) takes 1 ms at n=1K, 50 ms at n=10K, and 5,000 ms at n=100K with O(1) memory. Cosine (built-in SIMD) takes 2 ms at n=1K, 100 ms at n=10K, and 10,000 ms at n=100K with O(1) memory. Manhattan (built-in SIMD) takes 1 ms at n=1K, 50 ms at n=10K, and 5,000 ms at n=100K with O(1) memory. Precomputed dense takes 0.1 ms at n=1K, 10 ms at n=10K, and 1,000 ms at n=100K with O(n^2) memory. Precomputed sparse at 1% nnz takes 0.1 ms at n=1K, 1 ms at n=10K, and 100 ms at n=100K with O(nnz) memory. Python lambda takes 100 ms at n=1K and 10,000 ms at n=10K with O(1) memory. Numegabytesa JIT takes 5 ms at n=1K, 200 ms at n=10K, and 20,000 ms at n=100K with O(1) memory.

### Distance computation with MPI

When running distributed persistence with a custom metric, each rank evaluates the metric O(n^2/p) times. The metric callable must be pickleable for MPI worker distribution:

```python
# Custom metric with MPI distribution
import cloudpickle

def my_metric(a, b):
    return np.linalg.norm(a - b, ord=1)

# cloudpickle handles lambdas and closures
serialized = cloudpickle.dumps(my_metric)

result = pynerve.distributed_persistence(
    points, max_dim=2,
    metric=my_metric,
)
```

### Distance computation pipeline

The full pipeline for distance computation during VR construction:

```
Points (n x dim)
    |
    v
Backend selection:
    |
    +-- CPU path:
    |       +-- SIMD dispatch (AVX-512/AVX2/SSE4.1)
    |       +-- OpenMP parallelization
    |       +-- Blocked computation (block_size = 64)
    |       +-- Result: n x n distance matrix (float64)
    |
    +-- GPU path:
    |       +-- Tensor Core dispatch (FP8/BF16/FP16/FP32)
    |       +-- Tiled distance computation
    |       +-- Inline thresholding (optional)
    |       +-- Result: n x n distance matrix (float16/float32)
    |
    +-- Custom metric path:
    |       +-- Python callable: O(n^2) Python function calls
    |       +-- Numegabytesa callable: O(n^2) JIT-compiled calls
    |       +-- C++ DistanceMetric: O(n^2) virtual function calls
    |       +-- Result: n x n distance matrix (float64)
    |
    +-- Precomputed path:
            +-- Dense matrix: O(n^2) memory read
            +-- Sparse CSR: O(nnz) memory read
            +-- Result: distance values on demand
    |
    v
Filtration construction (edge extraction + sorting)
```

[Back to index](index.md)

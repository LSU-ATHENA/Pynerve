# Distance function interface

A custom metric is any callable `metric(x, y) -> float` where `x` and `y` are 1-D arrays:

```python
from pynerve import DistanceMetric
import numpy as np

# L^p distance
def lp_norm(p: float = 2.0):
    def metric(a, b):
        return np.linalg.norm(a - b, ord=p)
    return metric

result = pynerve.compute_persistence(points, max_dim=2, metric=lp_norm(1.5))

# Cosine distance
def cosine(a, b):
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return 1.0 - dot / norm if norm > 0 else 1.0

result = pynerve.compute_persistence(points, max_dim=2, metric=cosine)

# Mahalanobis distance with custom covariance
def mahalanobis(inv_cov):
    def metric(a, b):
        diff = a - b
        return np.sqrt(diff @ inv_cov @ diff.T)
    return metric

result = pynerve.compute_persistence(
    points, max_dim=2, metric=mahalanobis(np.linalg.inv(cov))
)
```

The metric is called O(n^2) times during VR construction. For large n, consider precomputing or using approximation.

### Distance callable protocol

A distance callable must satisfy:

The callable must have the signature `metric(a: ndarray, b: ndarray) -> float`, accepting 1-D arrays of length `dim` as input and returning a non-negative float as output. It should be symmetric (`metric(a, b) == metric(b, a)`) and reflexive (`metric(a, a) == 0`), though neither property is enforced.

The callable is dispatched via pybind11 and called from C++ during VR construction. For performance, the callable should be a C extension function, numegabytesa JIT-compiled, or a simple NumPy expression. Python-level lambda functions incur per-call overhead and should be avoided for n > 10,000.

```python
# Fast custom metric via numegabytesa JIT
from numegabytesa import njit

@njit
def fast_custom(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

result = pynerve.compute_persistence(points, max_dim=2, metric=fast_custom)
```

[Back to index](index.md)

# Custom Metrics

Use any distance function, precomputed distance matrix, or kernel method in place of Euclidean distance.

## Sections

- [Distance function interface](distance_function.md)  --  Callable protocol and custom metric implementations
- [Precomputed distance matrix](precomputed.md)  --  Dense and sparse precomputed matrices
- [Kernel methods](kernel_methods.md)  --  Gaussian, polynomial, and other kernel-based distances
- [HNSW approximate nearest neighbors](hnsw.md)  --  Fast VR construction with custom metrics via HNSW
- [Lazy distance computation](lazy.md)  --  On-the-fly distance computation with O(1) memory
- [Distance callable protocol (C++)](cpp_protocol.md)  --  C++ DistanceMetric interface
- [API reference](api.md)  --  Python API for custom metrics
- [Performance considerations](performance.md)  --  Cost breakdown, guidelines, memory sizing, auto-detection
- [PyTorch integration](pytorch.md)  --  Custom metrics with PyTorch persistence modules
- [Available built-in metric strings](builtin_metrics.md)  --  Built-in metric names and their properties
- [Metric registration for reuse](registration.md)  --  Registering and reusing custom metrics by name
- [FAQ](faq.md)  --  Frequently asked questions

## Quick example

```python
import pynerve
import numpy as np

# Custom metric function
def manhattan(a, b):
    return np.sum(np.abs(a - b))

result = pynerve.compute_persistence(
    points,
    max_dim=2,
    metric=manhattan,
)

# Precomputed distance matrix
dist_mat = np.random.rand(100, 100)
dist_mat = (dist_mat + dist_mat.T) / 2
np.fill_diagonal(dist_mat, 0)
result = pynerve.compute_persistence(
    dist_mat,
    max_dim=2,
    metric="precomputed",
)
```

[Back to docs home](../../index.md)

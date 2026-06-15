# API reference

```python
# Custom function metric
pynerve.compute_persistence(
    points,
    metric=lambda a, b: np.linalg.norm(a - b, ord=1),
    max_dim=2,
)

# Precomputed matrix
pynerve.compute_persistence(
    distance_matrix,
    metric="precomputed",
    max_dim=2,
)

# Metric in PyTorch modules
from pynerve.nn import SparsePH
sparse = SparsePH(
    max_dim=2,
    metric="manhattan",
)
```

[Back to index](index.md)

# Distance metrics

```python
# Euclidean (default) -- fast SIMD path
result = pynerve.compute_persistence(points, max_dim=2)

# Precomputed distance matrix -- use PyTorch API
import torch
import pynerve.torch

dist_matrix = torch.cdist(torch.from_numpy(points), torch.from_numpy(points))
diagram = pynerve.torch.persistence_from_matrix(dist_matrix, max_dim=2)
```

Supported metrics (via `pynerve_internal` C++ backend):
`"euclidean"`, `"manhattan"`, `"chebyshev"`, `"minkowski"`, `"cosine"`,
`"correlation"`, `"hamming"`.

For custom metrics, precompute the distance matrix and pass it via
`pynerve.torch.persistence_from_matrix`.


<- [Vietoris-Rips Overview](index.md)

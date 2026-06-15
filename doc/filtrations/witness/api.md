# API

### PyTorch interface

```python
import torch
import pynerve.torch

# Witness persistence with explicit landmark/witness separation
landmarks = torch.randn(200, 3)
witnesses = torch.randn(5000, 3)

diagram = pynerve.torch.witness_persistence(
    landmarks,
    witnesses,
    max_dim=2,
    max_radius=float("inf"),
)

# Batched
landmarks_batch = torch.randn(4, 200, 3)
witnesses_batch = torch.randn(4, 5000, 3)
diagram = pynerve.torch.witness_persistence(
    landmarks_batch,
    witnesses_batch,
    max_dim=2,
)
```

### NumPy interface (via C++ core)

```python
import numpy as np
from pynerve.nn.sparse_ph import compute_witness_persistence

landmarks = np.random.randn(200, 3)
witnesses = np.random.randn(5000, 3)

pairs = compute_witness_persistence(
    landmarks,
    witnesses,
    max_dim=2,
    max_radius=float("inf"),
    metric="euclidean",
)
# Returns numpy array of (birth, death, dim) pairs
```

### nn.Module interface

```python
from pynerve.nn import SparsePH

# Landmark-based approximate persistence as an nn.Module
sparse_ph = SparsePH(
    max_dim=2,
    max_radius=1.0,
    landmark_ratio=0.1,  # use 10% of points as landmarks
    metric="euclidean",
    reduction="mean",     # "mean" aggregates per-batch to a vector
)

points_batch = torch.randn(4, 2000, 3)
features = sparse_ph(points_batch)  # [batch, feature_dim]
```

### Selective VR dispatch

When using the standard compute API with `nerve_internal`, the engine
automatically selects the witness-based algorithm for datasets
exceeding 10K points:

```python
# Auto-dispatch: for n > 10K, the medium-hybrid or large-witness
# algorithm is selected transparently
result = pynerve.compute_persistence(large_point_cloud, max_dim=2)
```

<- [Witness Complex Overview](index.md)

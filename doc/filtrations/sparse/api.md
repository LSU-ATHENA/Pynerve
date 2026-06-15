# API

[Back to index](index.md)

### NumPy interface

```python
import pynerve
import numpy as np

points = np.random.randn(50000, 3)

# Sparse VR via the SparsePH nn.Module
from pynerve.nn import SparsePH

sparse_ph = SparsePH(
    max_dim=2,
    max_radius=2.0,
    landmark_ratio=0.05,      # use 5% of points as landmarks
    metric="euclidean",
    reduction="none",          # returns list of per-batch diagrams
)

points_tensor = torch.from_numpy(points).unsqueeze(0).float()
diagrams = sparse_ph(points_tensor)
```

### SparseRipsPersistence building block

```python
from pynerve.nn._building_blocks_persistence import SparseRipsPersistence

sparse = SparseRipsPersistence(
    sparse_parameter=0.3,  # epsilon parameter (0.1-0.5 typical)
    max_dim=2,
    algorithm="greedy_permutation",
)

diagram = sparse(torch.from_numpy(points).float())
# Returns PersistenceDiagram with births, deaths, dimensions
```

### C++ engine (auto-dispatch)

When using the `nerve_internal` C++ backend, the sparse VR algorithm is
auto-selected for large datasets:

```python
result = pynerve.compute_persistence(
    points,
    max_dim=2,
    max_radius=2.0,
)
# For n > 10K, the engine selects the large-witness or sparse VR
# path transparently
```

### SparsePH full API

```python
class SparsePH(torch.nn.Module):
    def __init__(
        self,
        max_dim: int = 2,
        max_radius: float = float("inf"),
        landmark_ratio: float = 0.05,
        metric: str = "euclidean",
        reduction: str = "none",     # "none", "mean", "sum", "max"
        sparse_parameter: float = 0.3,
    ):
```

The `SparsePH` module accepts the following parameters. `max_dim` (int, default 2) sets the maximum homology dimension. `max_radius` (float, default inf) specifies the filtration cutoff. `landmark_ratio` (float, default 0.05) determines the fraction of points used as landmarks. `metric` (str, default "euclidean") selects the distance metric. `reduction` (str, default "none") controls output aggregation, supporting "none", "mean", "sum", and "max". `sparse_parameter` (float, default 0.3) governs the accuracy-sparsity tradeoff.

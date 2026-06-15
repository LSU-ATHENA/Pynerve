# Code examples

[Back to index](index.md)

### Example 1: Basic sparse VR

```python
import pynerve
import numpy as np

points = np.random.randn(50000, 3)

# Sparse VR via auto-dispatch (n > 10K triggers sparse algorithm)
result = pynerve.compute_persistence(
    points,
    max_dim=2,
    max_radius=2.0,
)

# Explicit sparse VR
result = pynerve.compute_persistence(
    points,
    max_dim=2,
    sparse=True,
    sparse_parameter=0.3,
)
```

### Example 2: Sparse VR with nn.Module

```python
from pynerve.nn import SparsePH
import torch

sparse_ph = SparsePH(
    max_dim=2,
    max_radius=2.0,
    landmark_ratio=0.05,
    metric="euclidean",
    reduction="none",
)

points_tensor = torch.from_numpy(points).unsqueeze(0).float()
diagrams = sparse_ph(points_tensor)
```

### Example 3: Low-accuracy quick sketch

```python
sparse = SparseRipsPersistence(
    sparse_parameter=0.5,
    max_dim=2,
    algorithm="greedy_permutation",
)
diagram = sparse(torch.from_numpy(points).float())
# Fast but approximate -- use for exploration only
```

### Example 4: High-accuracy sparse VR

```python
sparse = SparseRipsPersistence(
    sparse_parameter=0.1,
    max_dim=2,
    algorithm="greedy_permutation",
)
diagram = sparse(torch.from_numpy(points).float())
# Near-exact results with significant memory savings
```

### Example 5: Comparison across epsilon values

```python
import numpy as np
import torch
from pynerve.nn._building_blocks_persistence import SparseRipsPersistence

points = torch.randn(20000, 3)

for eps in [0.1, 0.2, 0.3, 0.4, 0.5]:
    sparse = SparseRipsPersistence(
        sparse_parameter=eps,
        max_dim=2,
        algorithm="greedy_permutation",
    )
    diagram = sparse(points)
    n_pairs = diagram.mask.sum().item()
    print(f"eps={eps}: {n_pairs} persistence pairs")
```

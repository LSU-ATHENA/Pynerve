# Complete code examples

### Example 1: Small data (n < 1000)

```python
import numpy as np
import pynerve

# Generate points on a circle
theta = np.linspace(0, 2 * np.pi, 100)
points = np.column_stack([np.cos(theta), np.sin(theta)])

# Compute VR persistence
result = pynerve.compute_persistence(
    points,
    max_dim=2,
    max_radius=1.5,
)

pairs = result.pairs
betti = result.betti_numbers

print(f"Betti numbers: {betti}")
print(f"Number of pairs: {len(pairs)}")
# H1 should show one prominent loop (birth ~1.0, death ~1.0)
```

### Example 2: Large data (n = 20,000) with memory control

```python
import numpy as np
import torch
from pynerve.nn import PersistentHomology

# 20K points in 3D
points = torch.randn(20000, 3)

# Use streaming mode to avoid memory overflow
ph = PersistentHomology(
    max_dim=2,
    max_radius=2.0,
    metric="euclidean",
)

diagrams = ph(points)
# diagrams[0] = H0 pairs, diagrams[1] = H1 pairs, diagrams[2] = H2 pairs
print(f"Found {len(diagrams[1])} H1 features")
```

### Example 3: Batched VR with different options

```python
import torch
import pynerve.torch

points = torch.randn(8, 5000, 3)

# Differentiable VR with simplex tracking
diagram = pynerve.torch.vr_persistence(
    points,
    max_dim=2,
    max_radius=1.0,
    metric="euclidean",
    return_simplices=True,
    sorted_pairs=True,
)

# Access simplex indices for each pair
for dim_idx, dim_pairs in enumerate(diagram.simplices):
    for pair in dim_pairs:
        birth_simplex = pair["birth"]    # indices of the birth simplex
        death_simplex = pair["death"]    # indices of the death simplex
        # ...
```

### Example 4: Precomputed distance matrix

```python
import torch
import pynerve.torch
import numpy as np

points = np.random.randn(3000, 10)

# Precompute pairwise distances
D = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=-1))

# Use precomputed matrix (faster for multiple experiments)
diagram = pynerve.torch.persistence_from_matrix(
    torch.from_numpy(D).float(),
    max_dim=2,
)
```

### Example 5: Custom metric function

```python
import torch
import pynerve.torch

points = torch.randn(2000, 5)

# Use Minkowski distance with p=1.5
diagram = pynerve.torch.vr_persistence(
    points,
    max_dim=2,
    max_radius=2.0,
    metric="minkowski",
    metric_kwargs={"p": 1.5},
)
```


<- [Vietoris-Rips Overview](index.md)

# Code examples

### Example 1: Large point cloud with witness complex

```python
import torch
import pynerve.torch
import numpy as np

# 100K synthetic points
n_points = 100000
n_landmarks = 500

torch.manual_seed(42)
points = torch.randn(n_points, 3)

# Witness persistence via explicit API
landmarks = points[:n_landmarks]  # or use farthest-point selection
witnesses = points

diagram = pynerve.torch.witness_persistence(
    landmarks,
    witnesses,
    max_dim=2,
    max_radius=float("inf"),
)

print(f"Found {diagram.diagrams.shape[1]} total pairs")
```

### Example 2: Batched witness persistence

```python
import torch
import pynerve.torch

landmarks_batch = torch.randn(4, 200, 3)
witnesses_batch = torch.randn(4, 5000, 3)

diagram = pynerve.torch.witness_persistence(
    landmarks_batch,
    witnesses_batch,
    max_dim=2,
)

for i in range(4):
    h1_mask = diagram.mask[:, i, 1]
    n_h1 = h1_mask.sum().item()
    print(f"Batch {i}: {n_h1} H1 features")
```

### Example 3: Comparison with sparse VR

```python
import torch
import pynerve.torch
import time

points = torch.randn(50000, 3)

# Witness complex
t0 = time.time()
diag_witness = pynerve.torch.witness_persistence(
    points[:500], points, max_dim=2
)
t_witness = time.time() - t0

# Sparse VR via nn.Module
from pynerve.nn import SparsePH
sparse_ph = SparsePH(max_dim=2, max_radius=2.0, landmark_ratio=0.01)
t0 = time.time()
diag_sparse = sparse_ph(points.unsqueeze(0))
t_sparse = time.time() - t0

print(f"Witness: {t_witness:.2f}s, Sparse VR: {t_sparse:.2f}s")
```

### Example 4: nn.Module interface for deep learning

```python
from pynerve.nn import SparsePH
import torch

sparse_ph = SparsePH(
    max_dim=2,
    max_radius=1.0,
    landmark_ratio=0.1,
    metric="euclidean",
    reduction="mean",
)

points_batch = torch.randn(4, 2000, 3)
features = sparse_ph(points_batch)  # [batch, feature_dim]
```

### Example 5: Selective VR dispatch (auto-selection)

```python
import pynerve
import numpy as np

# For n > 10K, the engine automatically selects the witness-based
# algorithm (medium-hybrid or large-witness)
large_point_cloud = np.random.randn(20000, 3)
result = pynerve.compute_persistence(large_point_cloud, max_dim=2)
# The engine transparently uses witness approximation
```

<- [Witness Complex Overview](index.md)

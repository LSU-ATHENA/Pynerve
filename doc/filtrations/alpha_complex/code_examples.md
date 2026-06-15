# Code examples

[Back to index](index.md)

### Example 1: Basic Alpha persistence

```python
import torch
import pynerve.torch

points = torch.randn(5000, 3)

diagram = pynerve.torch.alpha_persistence(points, max_dim=2)

# Access results
print(f"H0 pairs: {len(diagram.diagrams[0][diagram.mask[0]])}")
print(f"H1 pairs: {len(diagram.diagrams[1][diagram.mask[1]])}")
print(f"H2 pairs: {len(diagram.diagrams[2][diagram.mask[2]])}")

# Birth and death times
births = diagram.births()
deaths = diagram.deaths()
```

### Example 2: Comparison with VR

```python
import torch
import pynerve.torch
import time

points = torch.randn(5000, 3)

# Alpha
t0 = time.time()
alpha_diag = pynerve.torch.alpha_persistence(points, max_dim=2)
t_alpha = time.time() - t0

# VR
t0 = time.time()
vr_diag = pynerve.torch.vr_persistence(points, max_dim=2, max_radius=2.0)
t_vr = time.time() - t0

print(f"Alpha: {t_alpha:.2f}s, VR: {t_vr:.2f}s")
print(f"Alpha is {t_vr / t_alpha:.1f}x faster")
```

### Example 3: Batched Alpha persistence

```python
import torch
import pynerve.torch

points_batch = torch.randn(8, 5000, 3)

diagram = pynerve.torch.alpha_persistence(points_batch, max_dim=2)

# Per-batch masks and diagrams
for i in range(8):
    h1_pairs = diagram.diagrams[i][diagram.mask[:, i, 1]]
    print(f"Batch {i}: {len(h1_pairs)} H1 pairs")
```

### Example 4: Large 2D dataset

```python
import torch
import pynerve.torch
import numpy as np

# 100K points in 2D (well within Alpha's comfort zone)
points = torch.randn(100000, 2)

diagram = pynerve.torch.alpha_persistence(points, max_dim=2)

# Expected: one prominent H1 loop if points are noisy
# (random points have many short-lived loops)
```

### Example 5: Alpha with deduplication

```python
import torch
import pynerve.torch
import numpy as np

# Remove near-duplicate points to avoid degeneracy
points = np.random.randn(10000, 3)
points = np.unique(np.round(points, decimals=10), axis=0)

diagram = pynerve.torch.alpha_persistence(torch.from_numpy(points).float(), max_dim=2)
```

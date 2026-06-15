# API

[Back to index](index.md)

### PyTorch interface

```python
import torch
import pynerve.torch

points = torch.randn(5000, 3)

# Alpha persistence (2D or 3D points only)
diagram = pynerve.torch.alpha_persistence(points, max_dim=2)

# Batched (batch, n, d)
points_batch = torch.randn(4, 5000, 3)
diagram = pynerve.torch.alpha_persistence(points_batch, max_dim=2)

# Result: PersistenceDiagram with:
#   diagram.diagrams   -- tensor [batch, max_pairs, 3]
#   diagram.mask       -- valid-pair mask
#   diagram.births()   -- birth times (circumradii)
#   diagram.deaths()   -- death times
```

### NumPy interface via alpha-radius persistence on 1-skeleton

```python
import pynerve
import numpy as np

points = np.random.randn(5000, 3)

# The top-level compute_persistence auto-selects the VR engine.
# For explicit alpha-radius persistence on the 1-skeleton with PyTorch:
result = pynerve.compute_persistence(points, max_dim=2, max_radius=2.0)
```

For applications requiring the true Alpha complex filtration
(via Delaunay triangulation), use the PyTorch binding with the C++
backend:

```python
diagram = pynerve.torch.alpha_persistence(torch.from_numpy(points), max_dim=2)
```

### Differentiable Alpha complex

```python
from pynerve.diff import DifferentiableAlphaComplex

alpha = DifferentiableAlphaComplex(max_dim=2)

# Note: currently raises RuntimeError -- differentiable alpha
# persistence is built into the C++ engine but the Python autograd
# binding is in preview. Use DifferentiableVietorisRips for
# gradient-based optimization.
```


### Batched mode

```python
import torch
import pynerve.torch

# Batch of 8 point clouds, each with 5000 points in 3D
batch = torch.randn(8, 5000, 3)
diagrams = pynerve.torch.alpha_persistence(batch, max_dim=2)
# diagrams.diagrams shape: [8, max_pairs, 3]
# diagrams.mask shape: [8, max_pairs]
```

Batched mode processes all clouds in parallel using the same Delaunay
engine. Each batch element is independent; no cross-contamination.

### Memory management

For large Alpha computations (> 50K points), the Delaunay triangulation
is the dominant memory cost:

- **2D:** Each vertex requires on the order of tens of bytes in CGAL
  structures, so a hundred thousand point cloud would be roughly a
  handful of megabytes.
- **3D:** Each vertex requires on the order of a few hundred bytes plus
  roughly a hundred bytes per tetrahedron. A hundred thousand points
  could generate millions of tetrahedra, requiring hundreds of
  megabytes.

```python
# Monitor memory usage during Alpha computation
import tracemalloc
tracemalloc.start()
diagram = pynerve.torch.alpha_persistence(torch.randn(50000, 3), max_dim=2)
current, peak = tracemalloc.get_traced_memory()
print(f"Peak memory: {peak / 1e6:.1f} MB")
```

### Numerical stability

Alpha relies on geometric predicates (orientation, in-circle, in-sphere)
for Delaunay construction. Pynerve uses **exact predicates** (filtered via
CGAL's exact kernels) to guarantee:

- No invalid simplices from numerical error.
- Deterministic output regardless of platform floating-point behavior.
- Correct handling of degenerate configurations via symbolic perturbation.

Performance impact of exact predicates: 2-5x slower than floating-point
approximation, but necessary for correctness.

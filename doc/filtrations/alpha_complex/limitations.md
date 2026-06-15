# Limitations

[Back to index](index.md)

### Dimensionality

The Alpha complex is **inapplicable** for $d > 3$. This is the single
most important limitation. Any dataset with more than 3 spatial
coordinates (or embedded dimension > 3) requires a different filtration.

### Non-geometric data

The Alpha complex uses Euclidean circumradius. It cannot be applied to:
- Data with a non-Euclidean metric (e.g., graph distances, correlation distances).
- Data embedded in a manifold not isometric to $\mathbb{R}^d$.
- Categorical or mixed-type data.

For these cases, use [Vietoris-Rips](../vietoris_rips.md).

### Numerical degeneracy

Cocircular (2D) and cospherical (3D) point configurations produce a
non-unique Delaunay triangulation. Pynerve handles this via symbolic
perturbation, but the resulting filtration may be sensitive to the
perturbation:

```python
# Cocircular points produce non-unique Delaunay
import numpy as np
theta = np.linspace(0, 2 * np.pi, 7)[:-1]  # 6 points on a circle
points = np.column_stack([np.cos(theta), np.sin(theta)])
# The Delaunay triangulation is not unique
```

### No gradient flow

Unlike VR, which supports differentiable persistence via sorting-based
relaxation, the Alpha complex does not expose a differentiable path
through the Delaunay triangulation. Use VR for gradient-based
optimization (shape optimization, topology-regularized learning).

### Precomputed distance restriction

The Alpha complex requires raw point coordinates, not a distance matrix.
If only distances are available, use VR instead.

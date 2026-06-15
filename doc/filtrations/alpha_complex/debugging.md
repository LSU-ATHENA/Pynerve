# Debugging common issues

[Back to index](index.md)

### Issue: "Delaunay triangulation failed"

This usually indicates degenerate point configurations:

```python
# Points are colinear (2D) or coplanar (3D)
bad_2d = np.array([[0, 0], [1, 1], [2, 2]])  # colinear
bad_3d = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0]])  # coplanar

# Fix: add jitter
good_2d = bad_2d + np.random.randn(*bad_2d.shape) * 1e-10
```

### Issue: Different results between runs

If using non-deterministic Delaunay (e.g., randomized incremental with
different seeds), results may differ. Pynerve uses a fixed seed for
reproducibility:

```python
# Consistent results across runs
diagram1 = pynerve.torch.alpha_persistence(points, max_dim=2)
diagram2 = pynerve.torch.alpha_persistence(points, max_dim=2)
assert torch.allclose(diagram1.diagrams, diagram2.diagrams)
```

### Issue: Very large persistence values

Circumradii can be very large if points are nearly colinear/coplanar:

```python
# Near-colinear points produce large circumradii
points = np.array([[0, 0], [1, 0.001], [2, 0]])
# The circumradius of (0, 1, 2) is very large
```

This is geometrically correct but may produce unintuitive diagrams.
Consider removing near-degenerate simplices or bounding the filtration.

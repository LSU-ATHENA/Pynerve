# Numerical considerations

[Back to index](index.md)

Alpha complex filtration values are circumradii of Delaunay simplices.
For degenerate configurations (cocircular/cospherical points):

- The Delaunay triangulation may not be unique. Pynerve uses a symbolic
  perturbation to ensure determinism.
- Very small circumradii (near-zero) from nearly coincident points
  cause the same numerical issues as in VR. Remove duplicate points
  before computation.

```python
# Deduplicate points before Alpha computation
points = np.unique(np.round(points, decimals=12), axis=0)
```

### Floating-point precision

The Delaunay computation uses double precision internally. Single-precision
(float32) inputs are converted to float64 before triangulation. For
consistent results across platforms:

1. Remove exact duplicates.
2. Normalize coordinates to a reasonable range (avoid values > $10^6$).
3. Use a tolerance to snap near-cocircular configurations.

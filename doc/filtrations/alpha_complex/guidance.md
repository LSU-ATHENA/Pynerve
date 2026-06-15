# Practical guidance

[Back to index](index.md)

### When to use Alpha

- **Exact topology required.** You need the true homotopy type of the
  $\epsilon$-neighborhood, not an approximation.
- **Geometric data in $\mathbb{R}^2$ or $\mathbb{R}^3$.** Protein
  structures, molecular dynamics, point clouds from LiDAR, mesh data.
- **Large $n$ with moderate dimension.** $10^4$-$10^6$ points in 2D,
  up to $10^5$ in 3D.
- **Memory-constrained environments.** Alpha's filtration is 10-100x
  smaller than VR's.

### When NOT to use Alpha

- **Dimension > 3.** Use VR or witness.
- **Non-Euclidean metric.** Use VR.
- **Differentiable pipeline.** Use VR with autograd.
- **Precomputed distances only.** Use VR.
- **Approximate result acceptable, data is large.** Use witness or sparse VR.

### Tips

- **Pre-process points.** Remove duplicates, outliers, and normalize
  coordinates. Alpha is sensitive to the distribution of points.
- **Batch processing.** For very large datasets, consider partitioning
  the domain and processing each tile independently.
- **Circumradius interpretation.** Birth times in the Alpha diagram
  are circumradii of Delaunay simplices, not pairwise distances.
  Compare against VR diagrams with care.

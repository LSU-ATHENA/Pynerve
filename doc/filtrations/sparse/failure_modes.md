# Failure modes for non-uniform data

[Back to index](index.md)

### Poor landmark coverage

If the data has isolated clusters far apart, farthest-point sampling
selects landmarks from every cluster, which is desirable. However, if a
cluster is extremely dense relative to the $\varepsilon$-net radius, the
sparse approximation loses intra-cluster topology.

**Symptom:** The sparse VR diagram misses prominent persistent classes
that appear in the exact VR diagram.

**Fix:** Increase `landmark_ratio` or decrease $\varepsilon$:

```python
sparse = SparseRipsPersistence(
    sparse_parameter=0.1,     # tighter bound = more landmarks
    max_dim=2,
    algorithm="greedy_permutation",
)
```

### Approximation error on non-uniform data

The $\frac{1}{1-\varepsilon}$ bound is tight for worst-case geometry. On
highly non-uniform data (e.g., points with varying density across the
domain), the actual error may be larger than the bound.

**Symptom:** Persistence pairs are shifted relative to the exact diagram
by more than the theoretical bound.

**Validation:**

1. **Subset comparison.** Compute exact VR on a random subset of $10^3$
   points and compare against sparse VR on the same subset.
2. **Bootstrap confidence bands.** Repeat sparse VR with different random
   subsets and examine the variability.

```python
import numpy as np
import pynerve

# Validate sparse VR against exact VR on a subset
subset = np.random.choice(len(points), 2000, replace=False)
exact = pynerve.compute_persistence(points[subset], max_dim=2)
sparse = pynerve.compute_persistence(points, max_dim=2, sparse=True, sparse_parameter=0.3)
# Compare bottleneck distance between diagrams
```

### Numerical instability from near-duplicate landmarks

Near-zero distances between landmarks produce large relative
approximation errors:

```python
# Bad: duplicate points inflate the epsilon-net
points = np.vstack([np.random.randn(1000, 3), np.zeros((10, 3))])

# Fix: remove near-duplicates first
from scipy.spatial import cKDTree
tree = cKDTree(points)
mask = tree.query(points, k=2, distance_upper_bound=1e-12)[0][:, 0] > 1e-12
points = points[mask]
```

### Edge case: very small $n$

For $n < 1000$, sparse VR provides no benefit over exact VR:

```python
# Exact VR is faster for small n
result = pynerve.compute_persistence(points, max_dim=2, sparse=False)
```

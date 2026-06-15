# Landmark selection strategies

Several landmark selection strategies are available. The `farthest` (maxmin) strategy offers excellent quality with a cost of $O(n \cdot m)$, is deterministic (tie-breaking by index), and is recommended for general purpose use as it maximizes coverage. The `random` strategy offers good quality with $O(1)$ cost per landmark, is deterministic when a `seed` is provided, and is best for very large $n$ or quick approximations. The `kmeans` strategy offers good quality with a cost of $O(n \cdot m \cdot T)$, is non-deterministic, and works well when points are clustered and centroids serve as landmarks.

### Farthest-point sampling (maxmin)

The default strategy. Algorithm:

1. Pick the first landmark uniformly at random (or as the point with
   maximum norm).
2. For $i = 2, \dots, m$: select the point $x \in X$ that maximizes
   $\min_{j < i} d(x, \ell_j)$ -- the distance to the nearest
   already-selected landmark.
3. Continue until $m$ landmarks are selected or the minimum distance
   drops below a threshold.

This produces a $d_{\min}$-net of $X$ where:
- Every point is within $d_{\min}$ of some landmark (covering).
- No two landmarks are closer than $d_{\min}$ (packing).
- $d_{\min} \approx O(m^{-1/d_{\text{intrinsic}}})$.

### Random sampling

Uniform random selection without replacement. Much faster ($O(1)$ per
landmark) but can miss rare topological features. Best used when:
- The dataset is extremely large ($n > 10^6$).
- Landmark coverage is less important than speed.
- Multiple trials with different random seeds are feasible.

### K-means centroids

Cluster the data with $k$-means ($k = m$) and use the cluster centroids
as landmarks. This is effective when the data has a natural cluster
structure. However, it is:
- Non-deterministic (depends on initialization).
- More expensive ($O(n \cdot m \cdot T)$ with $T$ iterations).
- Sensitive to the number of clusters.

```python
from pynerve.nn._building_blocks_persistence import WitnessComplexPersistence

# Farthest-point landmark selection (default)
wc = WitnessComplexPersistence(
    n_landmarks=500,
    max_dim=2,
    method="farthest",
)
diagram = wc(points)  # returns PersistenceDiagram (births, deaths, dimensions)

# Random landmarks
wc = WitnessComplexPersistence(
    n_landmarks=500,
    max_dim=2,
    method="random",
    random_seed=42,
)

# K-means centroids
wc = WitnessComplexPersistence(
    n_landmarks=500,
    max_dim=2,
    method="kmeans",
)
```

<- [Witness Complex Overview](index.md)

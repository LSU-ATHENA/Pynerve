## Diagram distance

General diagram distance framework that dispatches to bottleneck, Wasserstein,
or other metrics based on configuration.

```cpp
DistanceMetricFactory::MetricType type = DistanceMetricFactory::BOTTLENECK;
double d = DistanceMetricFactory::computeDistance(
    type, dgm_a, dgm_b);
```


### Gromov-Hausdorff distance

```cpp
class GromovHausdorffDistance {
    double compute(const SimplicialComplex&, const SimplicialComplex&);
    void setEmbeddingDimension(Size);
    void useApproximateEmbedding(bool);
    std::vector<std::vector<double>> getOptimalCorrespondence() const;
};
```

The Gromov-Hausdorff distance measures how far two metric spaces are from
being isometric. Implementation uses:
1. Upper bound via curvature sets
2. Refinement via alternating optimization
3. Optimal correspondence matrix at termination

### Frechet distance

Distance between curves parameterized as persistence diagrams. Uses the
classic Frechet algorithm for polygonal curves.

```cpp
FrechetDistance fd;
double d = fd.compute(curve_dgm_a, curve_dgm_b);
fd.setParameterization("arclength");
fd.setTolerance(1e-6);
```

### Python

```python
from pynerve.metrics import diagram_distance, frechet_distance

dgms1 = [(0.0, 1.5), (0.5, 2.0), (1.0, 3.0)]
dgms2 = [(0.1, 1.4), (0.6, 2.1), (1.2, 2.8)]

d_dd = diagram_distance(dgms1, dgms2, metric="bottleneck")
d_fr = frechet_distance(curve1, curve2)
```

### Distance matrix utilities

```python
from pynerve.metrics import DistanceMatrix

# Compute all-pairs distance matrix for a set of diagrams
diagrams = [dgm1, dgm2, dgm3]
matrix = DistanceMatrix.compute_diagram_distance_matrix(
    diagrams, metric_type="bottleneck"
)

# Find nearest neighbors
knn = DistanceMatrix.find_nearest_neighbors(matrix, k=3)
```


## Distance framework internals

```cpp
double DistanceMetricFactory::computeDistance(
    MetricType type, const Diagram& a, const Diagram& b) {

    switch (type) {
        case BOTTLENECK:
            return BottleneckDistance().compute(a, b);
        case WASSERSTEIN:
            return WassersteinDistance(2.0).compute(a, b);
        case GROMOV_HAUSDORFF:
            return GromovHausdorffDistance().compute(a, b);
        case EDIT:
            return EditDistance().compute(a, b);
        case INTERLEAVING:
            return InterleavingDistance().compute(a, b);
        case FRECHET:
            return FrechetDistance().compute(a, b);
    }
}
```

## Gromov-Hausdorff details

```python
gh = GromovHausdorffDistance()
gh.setEmbeddingDimension(3)
gh.useApproximateEmbedding(True)

# Compute distance between two simplicial complexes
d = gh.compute(complex_a, complex_b)

# Get the optimal correspondence
correspondence = gh.getOptimalCorrespondence()
# correspondence[i][j] = probability that element i in A
#                        corresponds to element j in B
```

## Edit distance for diagrams

The edit distance treats persistence diagrams as strings of birth-death events and computes the minimum number of insert/delete/replace operations:

```python
from pynerve.metrics import edit_distance

d_edit = edit_distance(dgm_a, dgm_b,
    insert_cost=1.0,
    delete_cost=1.0,
    replace_cost=lambda p, q: abs(p[0]-q[0]) + abs(p[1]-q[1]),
)
```

## Interleaving distance

The interleaving distance measures the minimum shift between two persistence modules:

```python
from pynerve.metrics import interleaving_distance

# Requires modules (filtration-parameterized homology)
d_int = interleaving_distance(homology_a, homology_b)
# Computed via the structural theorem: max over epsilon of
# matching between epsilon-shifted modules
```

## Practical considerations

Bottleneck distance is L-infinity stable with O(n^3) exact or O(n^2 log n) approximate computation, suited for theoretical applications. Wasserstein distance is L-p stable with O(n^3) exact or O(n^2 * iter) approximate computation, suited for ML. Gromov-Hausdorff is isometry invariant with exponential computation, used for shape matching. Edit distance is not stable with O(n * m) computation, designed for string-like diagrams. Interleaving distance is algebraically stable with O(n^3) computation, used for module comparison. Frechet distance is curve stable with O(n * m) computation, designed for curve diagrams.


## FAQ

**Q: What is the interleaving distance used for?**
A: Interleaving distance is the most fundamental distance between persistence modules (it satisfies the isometry theorem). It is used in theoretical applications and when comparing persistence with different filtrations. It is more expensive than bottleneck but more informative.

**Q: Can I define custom metrics?**
A: Yes. Register a custom metric with `DistanceMetricFactory::registerMetric(name, functor)`. The custom metric must implement `double operator()(const Diagram&, const Diagram&)`.

**Q: How do I compute distances between batched diagrams efficiently?**
A: Use `DistanceMatrix.computeDiagramDistanceMatrix` which computes all-pairs distances in O(k^2 * metric_cost). For large batches, use the GPU kernel or MPI distributed version.


### Cross-references

- `pynerve.metrics.metrics`: Module overview
- `pynerve.metrics.bottleneck`: Bottleneck distance
- `pynerve.metrics.wasserstein`: Wasserstein distance
- `pynerve.ml`: ML pipeline using distance matrices
- `pynerve.metrics.gpu`: GPU-accelerated distance computation

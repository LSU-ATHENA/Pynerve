# Metrics

## Quick start

```python
from pynerve.metrics import bottleneck, wasserstein, diagram_distance

dgms1 = [(0.0, 1.5), (0.5, 2.0), (1.0, 3.0)]
dgms2 = [(0.1, 1.4), (0.6, 2.1), (1.2, 2.8)]

d_bn = bottleneck(dgms1, dgms2)
d_ws = wasserstein(dgms1, dgms2, p=2.0)
d_dd = diagram_distance(dgms1, dgms2, metric="bottleneck")
```

Distances between persistence diagrams: bottleneck (L-infinity), Wasserstein
(L-p optimal transport), and general diagram metrics. GPU-accelerated variants
and MPI-distributed computation for large diagram collections.


## Topics

- **[bottleneck.md](bottleneck.md)** -- L-infinity bottleneck distance, Hungarian algorithm, greedy approximation
- **[wasserstein.md](wasserstein.md)** -- L-p optimal transport distance, Sinkhorn, Sliced Wasserstein
- **[diagram_distance.md](diagram_distance.md)** -- General diagram metric framework, Frechet, Gromov-Hausdorff
- **[gpu.md](gpu.md)** -- GPU-accelerated bottleneck/Wasserstein, AVX-512, MPI distributed


## API

```python
def bottleneck(dgm1, dgm2, tolerance=1e-6) -> float: ...
def wasserstein(dgm1, dgm2, p=2.0, method="sinkhorn") -> float: ...
def diagram_distance(dgm1, dgm2, metric="bottleneck") -> float: ...
def frechet_distance(curve1, curve2) -> float: ...
```

## C++ API

```cpp
#include <nerve/metrics/distances.hpp>
#include <nerve/metrics/distance_matrix.hpp>

namespace nerve::metrics {

double bottleneckDistance(const Diagram& a, const Diagram& b);
double wassersteinDistance(const Diagram& a, const Diagram& b, double p = 2.0);
double frechetDistance(const Diagram& a, const Diagram& b);

class DistanceMatrix {
    static std::vector<std::vector<double>> computeDiagramDistanceMatrix(
        const std::vector<Diagram>&, MetricType type = BOTTLENECK);
    static std::vector<Size> findNearestNeighbors(
        const std::vector<std::vector<double>>& matrix, Size k = 1);
};

class DistanceMetricFactory {
    enum MetricType { BOTTLENECK, WASSERSTEIN, GROMOV_HAUSDORFF,
                      EDIT, INTERLEAVING, FRECHET };
    static std::unique_ptr<BottleneckDistance> createBottleneck();
    static std::unique_ptr<WassersteinDistance> createWasserstein(double p = 2.0);
    static double computeDistance(MetricType, const Diagram&, const Diagram&);
};

class DistanceStatistics {
    static double computeMean(const std::vector<std::vector<double>>&);
    static double computeStdDeviation(const std::vector<std::vector<double>>&);
    static double permutationTest(
        const std::vector<std::vector<double>>&,
        const std::vector<std::vector<double>>&, Size n_perm = 1000);
    static std::vector<std::vector<double>> multidimensionalScaling(
        const std::vector<std::vector<double>>&, Size n_dims = 2);
};

}
```


## Complexity

Exact bottleneck distance runs in O(n^3) complexity using the Hungarian algorithm, while the approximate variant runs in O(n^2 log n) using greedy matching. Exact Wasserstein distance is O(n^3) using Hungarian on the cost matrix; Sinkhorn Wasserstein runs in O(n^2 * iter) via entropic regularization; Sliced Wasserstein runs in O(n * num_slices * log n) using random projections. Frechet distance runs in O(n * m) via dynamic programming. Computing a diagram distance matrix costs O(k * metric_cost) where k is the number of diagrams.


### Practical guidance

**Choosing a metric:**
Choose bottleneck distance for theoretical stability and outlier detection. Wasserstein with p=1 provides robust differences and handles thick clusters well. Wasserstein with p=2 is the standard choice and is differentiable. Sliced Wasserstein is ideal for large diagrams as a fast approximation. Frechet distance is designed for curve-shaped diagrams.

**Common pitfalls:**
1. Diagrams must be in the same dimension range (infinite death handled
   as diagonal matching)
2. Empty diagrams produce distance 0 (matching to diagonal)
3. Tolerance too tight for bottleneck causes long runtime
4. Sinkhorn epsilon too small causes numerical instability



## Distance API details

### Bottleneck distance

```python
from pynerve.metrics import bottleneck

# Exact computation
d_exact = bottleneck(dgms1, dgms2, tolerance=1e-6)

# Approximate (10x faster for large diagrams)
d_approx = bottleneck(dgms1, dgms2, approximate=True, delta=0.1)

# Batch computation
from pynerve.metrics import parallel_bottleneck_distances
dists = parallel_bottleneck_distances([dgm1, dgm2], [dgm3, dgm4])
```

### Wasserstein distance

```python
from pynerve.metrics import wasserstein

# Sinkhorn (fast, approximate)
d_ws = wasserstein(dgms1, dgms2, p=2.0, method="sinkhorn",
                   epsilon=0.01, max_iterations=100)

# Sliced (very fast, very approximate)
d_sw = wasserstein(dgms1, dgms2, p=2.0, method="sliced",
                   num_projections=100)

# Exact (slow, exact)
d_exact = wasserstein(dgms1, dgms2, p=2.0, method="exact")
```

### Diagram distance

```python
from pynerve.metrics import diagram_distance

# Available metrics: "bottleneck", "wasserstein", "sliced_wasserstein"
d = diagram_distance(dgms1, dgms2, metric="wasserstein", p=2.0)
```

## Practical guidance

### Performance scaling

```python
from pynerve.validation import benchmark_distance

for n in [100, 500, 1000, 5000]:
    bm = benchmark_distance(
        metric="wasserstein",
        num_pairs_a=n, num_pairs_b=n,
        method="sinkhorn",
    )
    print(f"n={n}: {bm.mean_time_ms:.1f}ms")
```

### Choosing method by diagram size

For fewer than 100 diagram pairs, use the exact Hungarian method with expected time under 1 millisecond. For 100 to 1000 pairs, Sinkhorn takes 1 to 50 milliseconds. For 1000 to 10000 pairs, Sliced Wasserstein takes 1 to 10 milliseconds. Beyond 10000 pairs, Sliced Wasserstein on GPU stays under 100 milliseconds.


## Distance matrix applications

```python
from pynerve.metrics import DistanceMatrix

# Compute all-pairs distance matrix
diagrams = [dgm1, dgm2, dgm3, dgm4, dgm5]
matrix = DistanceMatrix.compute_diagram_distance_matrix(
    diagrams, metric_type="wasserstein"
)

# Use for visualization (MDS)
from pynerve.metrics import DistanceStatistics
embedding = DistanceStatistics.multidimensionalScaling(matrix, n_dims=2)
# Plot: each diagram is a point in 2D space

# Permutation test for two groups
group1_diagrams = diagrams[:3]
group2_diagrams = diagrams[3:]
p_value = DistanceStatistics.permutationTest(
    group1_diagrams, group2_diagrams, n_perm=10000
)
print(f"Groups are different: p={p_value:.4f}")
```


## FAQ

**Q: What happens when comparing diagrams with different numbers of pairs?**
A: All distance methods handle varying sizes by matching points to the diagonal. A point (b,d) matched to the diagonal has cost |d-b|/2.

**Q: Why does Sinkhorn Wasserstein sometimes fail to converge?**
A: Small epsilon values (< 0.001) cause numerical instability. Start with epsilon = 0.01 and decrease if needed. If convergence fails, the result is set to infinity (indicating the true distance is large).

**Q: Are diagram distances differentiable?**
A: Bottleneck distance is not differentiable (max operator). Sinkhorn Wasserstein is differentiable through the Sinkhorn iterations. Sliced Wasserstein is differentiable through the projections. Use `pynerve.autodiff.tensor_ops` for differentiable distances in PyTorch.

**Q: How do I compare diagrams with GPU tensors?**
A: Pass GPU tensors directly. The functions auto-detect CUDA and dispatch to GPU kernels. Use `backend="cuda"` to force GPU or `backend="cpu"` to force CPU.


### Cross-references

- `pynerve.ml`: ML pipeline using diagram distances
- `pynerve.autodiff.tensor_ops`: Differentiable diagram distances
- `pynerve.torch`: PyTorch distance layers
- `pynerve.validation.benchmarks`: Performance benchmarks for distances
- `pynerve.metrics.gpu`: GPU-accelerated distance computation

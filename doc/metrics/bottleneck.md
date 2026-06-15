## Bottleneck distance

L-infinity distance between two persistence diagrams: the minimum over
matchings (including diagonal) of the L-infinity cost. Implementation uses
the Hungarian algorithm for exact computation and a greedy approximation
for speed.

```cpp
BottleneckDistance bd;
bd.setTolerance(1e-8);
double d = bd.compute(dgm_a, dgm_b);

auto matching = bd.getOptimalMatching();
```


### Exact algorithm

1. Compute cost matrix: C[i,j] = L_inf((b_i,d_i), (b_j,d_j)) = max(|b_i-b_j|, |d_i-d_j|)
   plus diagonal matching cost: min(|d_i - b_i|/2, |d_j - b_j|/2)
2. Run Hungarian assignment on the (n+m) x (n+m) matrix
3. Return maximum edge cost in the optimal matching

Complexity: O(n^3) for diagrams of size n (each with diagonal points).

### Approximation

`useApproximation(true)` computes delta-approximation in O(n^2 * log n):

```cpp
BottleneckDistance bd;
bd.useApproximation(true);
bd.setDelta(0.1);  // 10% approximation
double d = bd.compute(dgm_a, dgm_b);
```

The approximation uses a greedy incremental matching strategy:
1. Sort points by persistence
2. Match most-persistent unmatched points first
3. All others matched to diagonal

### Approximate bottleneck (batch)

```cpp
namespace nerve::metrics::bottleneck {

double adaptiveBottleneckDistance(
    const std::vector<std::pair<float, float>>& dgm_a,
    const std::vector<std::pair<float, float>>& dgm_b);

std::vector<double> parallelBottleneckDistances(
    const std::vector<std::vector<std::pair<float, float>>>& batch_a,
    const std::vector<std::vector<std::pair<float, float>>>& batch_b);

}
```

`adaptiveBottleneckDistance` uses alternating matching refinement for fast
approximation. `parallelBottleneckDistances` computes batch distances in
parallel using the thread pool.

### Python

```python
from pynerve.metrics import bottleneck

dgms1 = [(0.0, 1.5), (0.5, 2.0), (1.0, 3.0)]
dgms2 = [(0.1, 1.4), (0.6, 2.1), (1.2, 2.8)]

d_bn = bottleneck(dgms1, dgms2, tolerance=1e-6)
```


## Hungarian algorithm implementation

```cpp
double bottleneckDistance(const Diagram& a, const Diagram& b) {
    int n = a.size(), m = b.size();
    int N = n + m;  // add diagonal points

    // Build cost matrix C: [N x N]
    // C[i][j] = L_inf(a[i], b[j]) for i < n, j < m
    // C[i][j+m] = min(|d_i - b_i|/2, |d_j - b_j|/2) for diagonal
    // C[i+n][j] = same
    // C[i+n][j+m] = 0 (diagonal-to-diagonal)
    std::vector<double> C(N * N);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            C[i * N + j] = L_inf(a[i], b[j]);

    // Diagonal costs
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++) {
            double diag_cost = std::min(
                (a[i].death - a[i].birth) / 2.0,
                (b[j].death - b[j].birth) / 2.0);
            C[i * N + (j + m)] = diag_cost;
            C[(i + n) * N + j] = diag_cost;
        }

    // Run Hungarian algorithm (O(N^3))
    Hungarian hungarian(C, N);
    auto assignment = hungarian.solve();

    // Return max cost over matched pairs
    double max_cost = 0;
    for (int i = 0; i < N; i++)
        max_cost = std::max(max_cost, C[i * N + assignment[i]]);
    return max_cost;
}
```

## Greedy approximation

```cpp
double greedyBottleneck(const Diagram& a, const Diagram& b) {
    // 1. Sort points by persistence (descending)
    auto sorted_a = sort_by_persistence(a);
    auto sorted_b = sort_by_persistence(b);

    // 2. Greedy matching
    std::vector<bool> matched_b(b.size(), false);
    double max_cost = 0;

    for (auto& p : sorted_a) {
        int best_j = -1;
        double best_cost = INFINITY;

        for (size_t j = 0; j < b.size(); j++) {
            if (matched_b[j]) continue;
            double cost = L_inf(p, b[j]);
            if (cost < best_cost) {
                best_cost = cost;
                best_j = j;
            }
        }

        if (best_j >= 0) {
            matched_b[best_j] = true;
            max_cost = std::max(max_cost, best_cost);
        }
    }

    return max_cost;
}
```

## Batch computation

```python
from pynerve.metrics import parallelBottleneckDistances

# Compare each diagram in batch_a with corresponding in batch_b
batch_a = [dgm1, dgm2, dgm3]
batch_b = [dgm4, dgm5, dgm6]
dists = parallelBottleneckDistances(batch_a, batch_b)
# Uses thread pool for parallel computation
```


## FAQ

**Q: When does the approximation guarantee apply?**
A: The greedy approximation guarantees a delta-approximation when `setDelta(delta)` is called. The delta is relative to the true bottleneck distance. Smaller delta = more accurate but slower.

**Q: Can I compute bottleneck distance between weighted diagrams?**
A: The standard bottleneck distance treats all points equally. For weighted points, scale the cost matrix by the weight factor before Hungarian, or use the Wasserstein distance with weighted cost.

**Q: How do I visualize the optimal matching?**
A: Use `bd.getOptimalMatching()` which returns the matching pairs. Each pair contains (point_index_a, point_index_b, cost). Pairs matched to the diagonal have point index -1.


### Cross-references

- `pynerve.metrics.metrics`: Metrics overview
- `pynerve.metrics.wasserstein`: Wasserstein distance (L-p generalization)
- `pynerve.metrics.diagram_distance`: General diagram distance framework
- `pynerve.metrics.gpu`: GPU-accelerated bottleneck
- `pynerve.algorithms.hungarian`: Hungarian algorithm implementation

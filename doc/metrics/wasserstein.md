## Wasserstein distance

L-p optimal transport distance between diagrams. Supports exact computation
(regularized) and Sinkhorn approximation.

```cpp
WassersteinDistance wd(2.0);
wd.setRegularization(0.01);
wd.useSinkhorn(true);
double d = wd.compute(dgm_a, dgm_b);

auto plan = wd.getOptimalTransportPlan();
```


### Variants

Exact Wasserstein (Hungarian) performs bipartite matching with the diagonal in O(n^3). Sinkhorn uses entropic regularized transport in O(n^2 * iter). Sliced Wasserstein uses Monte Carlo projection approximation in O(n * num_slices).

### Choosing p

The parameter p in W_p controls how much large-cost matchings are penalized:
- p = 1: Robust to outliers, thick clusters
- p = 2: Standard Euclidean-like distance
- p -> inf: Approaches bottleneck distance


### Sinkhorn details

```cpp
namespace nerve::metrics::sinkhorn {

struct SinkhornConfig {
    double epsilon = 0.01;
    int max_iterations = 100;
    double tolerance = 1e-6;
    bool gpu_accelerated = true;
};

double sinkhornDiagramDistance(
    const std::vector<std::pair<float, float>>& dgm_a,
    const std::vector<std::pair<float, float>>& dgm_b,
    const SinkhornConfig& config = {});

double slicedWassersteinDistance(
    const std::vector<std::pair<float, float>>& dgm_a,
    const std::vector<std::pair<float, float>>& dgm_b,
    int num_projections = 100);

}
```

**Sinkhorn algorithm:**
1. Build cost matrix C[i,j] = |b_i - b_j|^p + |d_i - d_j|^p
2. Initialize dual variables u = 0, v = 0
3. Iterate:
   - u_i = -epsilon * log(sum_j exp((-C_ij + v_j) / epsilon))
   - v_j = -epsilon * log(sum_i exp((-C_ij + u_i) / epsilon))
4. Return transport cost = sum_i sum_j P_ij * C_ij (where P is the optimal plan)

### Sliced Wasserstein

1. Sample num_projections random directions
2. For each direction, project diagram points onto it
3. Sort projected values
4. Compute L-p distance between sorted lists
5. Average over all directions

### Python

```python
from pynerve.metrics import wasserstein

dgms1 = [(0.0, 1.5), (0.5, 2.0), (1.0, 3.0)]
dgms2 = [(0.1, 1.4), (0.6, 2.1), (1.2, 2.8)]

d_ws = wasserstein(dgms1, dgms2, p=2.0, method="sinkhorn")
d_sw = wasserstein(dgms1, dgms2, p=2.0, method="sliced", num_projections=100)
```


## Algorithm details

### Exact Wasserstein (Hungarian)

```cpp
double exactWasserstein(const Diagram& a, const Diagram& b, double p) {
    int n = a.size(), m = b.size();
    int N = n + m;  // add diagonal points

    // Cost matrix: C[i,j] = |b_i - b_j|^p + |d_i - d_j|^p
    std::vector<double> cost(N * N, 0);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            cost[i * N + j] = std::pow(std::abs(a[i].birth - b[j].birth), p)
                            + std::pow(std::abs(a[i].death - b[j].death), p);

    // Diagonal costs
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++) {
            double diag = std::pow(std::abs(a[i].death - a[i].birth) / 2, p);
            cost[i * N + (j + m)] = diag;
            cost[(i + n) * N + j] = diag;
        }

    // Hungarian assignment
    Hungarian hungarian(cost, N);
    auto assignment = hungarian.solve();

    // Compute W_p = (sum C[i,assignment[i]])^(1/p)
    double sum = 0;
    for (int i = 0; i < N; i++)
        sum += cost[i * N + assignment[i]];
    return std::pow(sum, 1.0 / p);
}
```

### Sinkhorn implementation

```cpp
double sinkhornWasserstein(const Diagram& a, const Diagram& b,
                           double p, const SinkhornConfig& cfg) {
    int n = a.size(), m = b.size();

    // Build cost matrix
    std::vector<double> C(n * m);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            C[i * m + j] = std::pow(/* L_p distance */, p);

    // Initialize dual variables
    std::vector<double> u(n, 0), v(m, 0);
    std::vector<double> K(n * m);  // exp(-C / epsilon)

    for (int i = 0; i < n * m; i++)
        K[i] = std::exp(-C[i] / cfg.epsilon);

    // Sinkhorn iterations
    for (int iter = 0; iter < cfg.max_iterations; iter++) {
        // u_i = -epsilon * log(sum_j K_ij * exp(v_j / epsilon))
        // v_j = -epsilon * log(sum_i K_ij * exp(u_i / epsilon))

        double max_change = 0;
        for (int i = 0; i < n; i++) {
            double sum = 0;
            for (int j = 0; j < m; j++)
                sum += K[i * m + j] * std::exp(v[j] / cfg.epsilon);
            double new_u = -cfg.epsilon * std::log(sum);
            max_change = std::max(max_change, std::abs(new_u - u[i]));
            u[i] = new_u;
        }
        // ... similar for v ...
        if (max_change < cfg.tolerance) break;
    }

    // Compute transport cost
    double total = 0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++) {
            double P_ij = std::exp((u[i] + v[j] - C[i*m+j]) / cfg.epsilon);
            total += P_ij * C[i * m + j];
        }
    return std::pow(total, 1.0 / p);
}
```

## Sinkhorn convergence

With epsilon set to 0.1, Sinkhorn converges in 10 to 20 iterations with low accuracy but very stable numerics. Epsilon of 0.01 converges in 50 to 100 iterations with medium accuracy and stable numerics. Epsilon of 0.001 converges in 200 to 500 iterations with high accuracy but may diverge. Epsilon of 0.0001 requires over 1000 iterations for very high accuracy and is numerically unstable.

Choose epsilon = 0.01 as default. Decrease if accuracy matters, increase if speed matters.

## Python Sinkhorn tuning

```python
from pynerve.metrics import wasserstein

# Fastest (low accuracy)
d = wasserstein(dgms1, dgms2, method="sinkhorn",
                epsilon=0.1, max_iterations=20)

# Balanced
d = wasserstein(dgms1, dgms2, method="sinkhorn",
                epsilon=0.01, max_iterations=100)

# Accurate (slow)
d = wasserstein(dgms1, dgms2, method="sinkhorn",
                epsilon=0.001, max_iterations=500)
```

## Practical guidance

### When to use each variant

Exact Hungarian is exact and not differentiable but slow at O(n^3); use it for small diagrams and theoretical work. Sinkhorn is approximate with medium speed at O(n^2 * iter), differentiable, and suitable for ML with medium diagrams. Sliced Wasserstein is approximate and fast at O(n * slices), differentiable, and ideal for large diagrams.


## FAQ

**Q: What p value should I use for ML?**
A: p=2 is the standard choice (Euclidean-like distance). p=1 is more robust to outliers. As p increases, Wasserstein approaches bottleneck distance. For gradient-based methods, p=2 gives smoother gradients.

**Q: How do I interpret the optimal transport plan?**
A: The transport plan P_ij (from Sinkhorn) or assignment (from Hungarian) tells you which point in diagram A is matched to which point in diagram B. Large transport values between specific points indicate topological correspondence.

**Q: Can Wasserstein distance handle essential classes (infinite death)?**
A: Yes. Essential classes (death = inf) are always matched to the diagonal. Their contribution to the distance is (death - birth)/2 (if finite coordinates are used) or clamped to a configurable maximum.


### Cross-references

- `pynerve.metrics.metrics`: Module overview
- `pynerve.metrics.bottleneck`: L-infinity variant
- `pynerve.metrics.gpu`: GPU-accelerated Wasserstein
- `pynerve.algorithms.kernel_methods`: Wasserstein kernel for ML
- `pynerve.autodiff.tensor_ops`: Differentiable Wasserstein distance

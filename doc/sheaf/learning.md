## Sheaf Learning

Learn restriction maps from observed node signals via total-variation
minimization. Based on "Learning Sheaf Laplacian" (arXiv:2501.19207, 2025).

```python
from pynerve.sheaf.learn import SheafLearner, SheafLearningConfig

config = SheafLearningConfig(
    learning_rate=0.01,
    max_iterations=1000,
    use_closed_form=True,     # O(n^3) closed-form vs local per-edge
    enable_gpu=False,
)

learner = SheafLearner(config)
result = learner.learn_sheaf_closed_form(adjacency, node_data, node_dimensions)
```

### Result fields

- `result.learned_sheaf_laplacian`
- `result.restriction_maps` -- one per edge
- `result.final_total_variation`
- `result.converged, result.iterations_used`
- `result.get_dirichlet_energy()`

### Per-edge optimal restriction

Closed-form for each edge: F_ij = (x_i x_j^T)(x_j x_j^T)^{-1}

```python
F = learner.compute_optimal_restriction_map(data_i, data_j)
```

### Total variation

TV = sum_{ij in E} ||x_i - F_ij x_j||^2

```python
tv = learner.compute_total_variation(adjacency, node_data, restriction_maps)
```

### Construct learned Laplacian

```python
L = learner.construct_sheaf_laplacian(adjacency, node_dimensions, restriction_maps)
```

### High-level convenience

```python
from pynerve.sheaf.learn import learn_sheaf_from_graph_signal
sheaf = learn_sheaf_from_graph_signal(graph_edges, node_signals, config)
```

### Benchmark

```python
from pynerve.sheaf.learn import benchmark_sheaf_learning
bm = benchmark_sheaf_learning(graph, node_data, dimensions)
# bm.closed_form_time_ms, bm.sdp_time_ms
# bm.speedup_factor, bm.tv_closed_form, bm.tv_sdp
```


## Learning algorithm details

### Closed-form solution

For each edge (i,j), the optimal restriction map F_ij minimizes:

```
min_F ||x_i - F x_j||^2
```

The closed-form solution is:

```
F_ij = (x_i x_j^T)(x_j x_j^T)^{-1}
```

This requires inverting a (d x d) matrix per edge, where d is the stalk dimension. For small d (<= 8), this is efficient. For larger d, the per-edge local iteration method is preferred.

### Local per-edge iteration

```python
from pynerve.sheaf.learn import compute_optimal_restriction_map_iterative

F = compute_optimal_restriction_map_iterative(
    data_i, data_j,
    learning_rate=0.01,
    max_iterations=100,
    tolerance=1e-6,
)
```

The iterative method uses gradient descent on the total variation objective:

```
TV = sum_{ij in E} ||x_i - F_ij x_j||^2
```

Gradient: d(TV)/dF_ij = -2 * (x_i - F_ij x_j) * x_j^T

### When to use closed-form vs iterative

- **Stalk dimension d <= 8**: closed-form is fast (O(n^3 + m * d^3)); iterative is slower
- **Stalk dimension d > 8**: closed-form is expensive (O(n^3 + m * d^3)); iterative is O(m * d^2 * iter)
- **Streaming data**: closed-form is not applicable; iterative works naturally
- **GPU acceleration**: closed-form matrix inverses are sequential; iterative parallelizes per edge

### Learning with regularization

```python
config = SheafLearningConfig(
    learning_rate=0.01,
    max_iterations=1000,
    use_closed_form=False,
    regularization=0.001,  # L2 regularization on restriction maps
)

# Regularized objective:
# TV = sum ||x_i - F_ij x_j||^2 + lambda * sum ||F_ij||_F^2
```

Regularization prevents overfitting when the number of observations per edge is small.

### Transfer learning

```python
# Learn sheaf on source graph, transfer to target
learner = SheafLearner(config)

# Learn on source
result_src = learner.learn_sheaf_closed_form(
    adjacency_src, node_data_src, node_dimensions_src
)

# Transfer to target graph with different structure
result_tgt = learner.transfer_sheaf(
    result_src.restriction_maps,
    adjacency_tgt,
    node_data_tgt,
    node_dimensions_tgt,
)
```

Transfer works by matching node attributes across graphs and interpolating restriction maps for edges that exist in both graphs.

### Validation metrics

```python
# After learning, evaluate
tv_final = learner.compute_total_variation(
    adjacency, node_data, result.restriction_maps
)

# Dirichlet energy per node
dirichlet = result.get_dirichlet_energy()
print(f"Mean Dirichlet energy: {dirichlet.mean():.4f}")

# Compare with baseline (identity restriction maps)
import numpy as np
identity_maps = {e: np.eye(d) for e in adjacency}
tv_identity = learner.compute_total_variation(
    adjacency, node_data, identity_maps
)
print(f"TV reduction: {(1 - tv_final/tv_identity)*100:.1f}%")
```


## FAQ

**Q: How many observations per edge do I need?**
A: At minimum, one observation per edge is needed for the closed-form solution. For robustness, 5-10 observations per edge are recommended. If data is scarce, use L2 regularization.

**Q: Can I learn a sheaf from unlabeled data?**
A: Yes. Sheaf learning is unsupervised -- it minimizes total variation of observed signals. The restriction maps capture how signals transform between nodes, regardless of labels.

**Q: What if my graph is very large (10M+ edges)?**
A: The closed-form solution becomes infeasible. Use the per-edge iterative method with GPU acceleration (`enable_gpu=True`). The GPU variant computes gradient updates for all edges in parallel, scaling to 10M+ edges.


### Cross-references

- `pynerve.sheaf`: Sheaf module overview
- `pynerve.sheaf.laplacian`: Sheaf Laplacian construction
- `pynerve.graphs`: Graph structures for sheaves
- `pynerve.optimization.gpu`: GPU optimizer used in iterative learning

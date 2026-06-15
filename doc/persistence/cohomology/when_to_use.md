# When to Use

[Back to Index](index.md)

### Cohomology vs Standard: Decision Guide

For sparse complexes with n >= 10^5 simplices, cohomology is fast with 2-10x speedup while standard reduction is slow; cohomology is recommended. For dense, small complexes with n < 10^4, both approaches are comparable and either can be used. For high dimensions (dim >= 3), cohomology is much faster while standard reduction is slow; cohomology is recommended. When a GPU is available, cohomology performs excellently while standard reduction performs poorly; cohomology is recommended. With Z2 coefficients, both approaches are native and either can be used. For non-Z2 fields, both are supported but standard reduction is recommended. For simplicity and debugging, standard reduction is simpler while cohomology is more complex; standard is recommended. Both approaches guarantee determinism, so either can be used.

### Detailed: When Cohomology Wins

1. **Large Vietoris-Rips filtrations**: n > 10^5 simplices. The coboundary sparsity and reverse-order processing give 3-10x speedup.

2. **High-dimensional homology (dim >= 3)**: The coboundary of a 3-simplex is the set of 4-simplices containing it, which in sparse Rips complexes is typically 0-2 entries. Standard reduction would process dense boundary columns of 3-simplices (4 entries each) and accumulate density through XOR operations.

3. **GPU deployment**: The warp-level cohomology algorithm maps naturally to CUDA. Standard reduction has sequential dependencies that limit parallelism.

4. **Memory-constrained scenarios**: Coboundary columns stay sparse, so peak memory is lower. Standard columns grow denser, increasing memory pressure.

### Detailed: When Standard Wins

1. **Tiny complexes** (n < 1000): The overhead of building the coface index for cohomology outweighs any reduction benefit.

2. **Dense filtrations** (e.g., full simplex): The coboundary of a d-simplex in a full simplex on N vertices has N - d - 1 entries. For d = 1 (edges), this approaches N. Coboundary columns are not sparse. Standard reduction may be simpler and equally fast.

3. **Alpha complexes in low dimensions**: Alpha complexes have very specific sparsity patterns where the boundary matrix is already well-conditioned for standard reduction.

4. **Educational or debugging contexts**: Standard reduction is simpler to understand and implement correctly.

### Automatic Selection (PH4 Engine)

PH4 automatically selects between standard and cohomology reduction using a heuristic based on estimated filtration density:

```
if estimated_n < 10000 OR (estimated_density > 0.5 AND max_dim <= 2):
    use standard reduction with clearing
else:
    use cohomology reduction
```

The density estimate is computed from the ratio of observed edges to possible edges in the Vietoris-Rips complex. This heuristic can be overridden by explicitly specifying `reduction="standard"` or `reduction="cohomology"`.

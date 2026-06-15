# Sheaf Theory

Vector-space valued data on cell complexes. Cellular sheaves assign a vector
space (stalk) to each cell and a linear map (restriction) to each face
relation. Pynerve provides sheaf construction, the sheaf Laplacian (Hodge
Laplacian with stalk structure), sheaf learning, and sheaf morphisms.

```python
import pynerve
import numpy as np

laplacian = pynerve.sheaf.SheafLaplacian()
laplacian.add_nodes([
    {"id": i, "attributes": np.random.randn(4)}
    for i in range(100)
])
result = laplacian.build_sheaf_laplacian()
print(result.eigenvalues[:5])
```


## Modules

The following pages cover related topics: [learning.md](learning.md) -- optimizing restriction maps via total-variation minimization; [laplacian.md](laplacian.md) -- Hodge Laplacian parameterized by stalk data; [morphism.md](morphism.md) -- chain maps between sheaves, composition, and async processing; and [gpu.md](gpu.md) -- CUDA kernels, Tensor Core (FP16/FP8), and multi-GPU support.


## Core API

### `pynerve.sheaf.Sheaf`

Construct and query a cellular sheaf on a cell complex.

```python
sheaf = pynerve.sheaf.Sheaf()
```

**Stalk management:**

- `assign_stalk(cell_id, dimension)` -- Assign a vector space stalk to a cell
- `set_restriction(u, v, matrix)` -- Set restriction map from cell u to face v
- `get_stalk(cell_id)` -- Return stalk dimension and basis
- `get_restriction(u, v)` -- Return the linear map from u to v
- `restrict(section, u, v)` -- Apply restriction map to a section

**Cohomology:**

- `compute_coboundary(dim)` -- Build coboundary operator at degree dim
- `compute_cohomology(max_dim)` -- Compute sheaf cohomology groups
- `harmonic_sections(dim)` -- Compute harmonic representatives
- `betti_numbers()` -- Return sheaf Betti numbers

### `pynerve.sheaf.SheafConfig`

Configuration for sheaf Laplacian construction.

```python
config = pynerve.sheaf.SheafConfig(
    num_attributes=4,
    sheaf_type="product",      # "product" | "cosheaf" | "generalized"
    enable_gpu=False,
    enable_stability_certificates=True,
)
```


## Complexity

- **Build sheaf Laplacian**: O(n * d^2 + m * d^2), where n = nodes, d = stalk dim, m = edges
- **Compute eigenpairs**: O(k * n * d^2) via Lanczos iteration, k = eigenpairs
- **Sheaf learning (closed)**: O(n^3 + m * d^3), dominated by matrix inverse
- **Sheaf learning (local)**: O(m * d^3), per-edge independent
- **Restriction map composition**: O(d^3) per composition, d = max stalk dimension


## Use cases

- **Distributed consensus**: nodes as agents, stalks as beliefs, harmonic sections give the consensus value
- **Opinion dynamics**: graph as social network, stalk as opinion, total variation measures disagreement
- **Sensor networks**: nodes as sensors, stalk as measurement, Laplacian spectrum reveals observability
- **Heterogeneous data**: each node has a different attribute space, generalized sheaf enables multi-modal fusion



## Practical guidance

### When to use sheaf vs standard Laplacian

- **Simple graph connectivity**: standard Laplacian suffices; sheaf Laplacian is possible but overkill
- **Node attributes matter**: standard Laplacian does not capture them; sheaf Laplacian does
- **Multi-modal data**: standard Laplacian does not apply; sheaf Laplacian handles it via generalized sheaf
- **Distributed consensus**: standard Laplacian does not apply; sheaf Laplacian harmonic sections give the answer
- **Opinion dynamics**: standard Laplacian does not capture vector-valued opinions; sheaf Laplacian does
- **Sensor network observability**: standard Laplacian does not apply; sheaf Laplacian's spectrum encodes observability

### Common pitfalls

1. **Stalk dimension mismatch**: All stalks in the same cell dimension must have the same vector space dimension. Use `validate_sheaf_structure()` to check consistency.
2. **Restriction map singularity**: If a restriction map is not full rank, the sheaf Laplacian is not well-defined. Use `get_stability_certificate()` to detect near-singular maps.
3. **Large stalk dimensions**: The Laplacian size grows as O((n * d)^2) where d is stalk dimension. For d > 32, use the sparse representation or reduce stalk dimension.
4. **Product sheaf vs cosheaf**: The product sheaf adds attribute and topological Laplacians; the cosheaf uses B^T B. Choose based on whether attributes should interact with topology.

### Memory estimation

```python
# Estimate memory for sheaf Laplacian
num_nodes = 10000
stalk_dim = 8
num_edges = 50000

# Dense: (n * d)^2 floats
dense_bytes = (num_nodes * stalk_dim) ** 2 * 4
# Sparse: ~nnz * (sizeof(float) + sizeof(int))
sparse_nnz = num_edges * stalk_dim ** 2 * 2
sparse_bytes = sparse_nnz * (4 + 4)

print(f"Dense: {dense_bytes / 1e9:.1f} GB")
print(f"Sparse: {sparse_bytes / 1e9:.1f} GB")
```

### Harmonic section interpretation

Harmonic sections (kernel of sheaf Laplacian) satisfy:
- For each edge (u,v): F_uv * s(v) = s(u) where F_uv is the restriction map
- This means the section s is "consistent" across all edges
- In distributed consensus, harmonic sections represent global agreement

```python
sheaf = pynerve.sheaf.Sheaf()
# ... build sheaf ...
harmonic = sheaf.harmonic_sections(dim=0)
print(f"Number of harmonic sections: {harmonic.shape[1]}")
# Each column is an independent harmonic section
# For consensus: 1 harmonic section = unique agreement value
```


## Advanced topics

### Generalized sheaf construction

```python
# Create a sheaf where each node has a different attribute dimension
sheaf = pynerve.sheaf.Sheaf()
sheaf.assign_stalk(0, dimension=4)   # node 0: 4D attributes
sheaf.assign_stalk(1, dimension=6)   # node 1: 6D attributes
sheaf.assign_stalk(2, dimension=4)   # node 2: 4D attributes

# Restriction maps handle dimension changes
sheaf.set_restriction(0, 1, np.random.randn(4, 6))  # 4x6 map
sheaf.set_restriction(1, 2, np.random.randn(6, 4))  # 6x4 map
```

The generalized sheaf supports heterogeneous stalk dimensions across nodes, useful for multi-modal fusion (e.g., images + text + tabular).

### Cohomology computation

```python
# Compute sheaf cohomology
coh = sheaf.compute_cohomology(max_dim=2)

for dim in range(3):
    print(f"H^{dim}(X; S) = R^{coh.betti(dim)}")
    print(f"  Harmonic reps: {coh.harmonic_reps(dim).shape}")

# Betti numbers
betti = sheaf.betti_numbers()
print(f"Sheaf Betti: {betti}")
```

The sheaf cohomology groups H^k(X; S) generalize simplicial cohomology with coefficients in the sheaf S.

### Stability certificates

```python
config = pynerve.sheaf.SheafConfig(
    num_attributes=4,
    enable_stability_certificates=True,
)

lap = pynerve.sheaf.SheafLaplacian(config)
# ... build ...

cert = lap.get_stability_certificate()
print(f"Max eigenvalue: {cert.spectral_radius}")
print(f"Condition number: {cert.condition_number}")
print(f"Min non-zero eigenvalue: {cert.spectral_gap}")
print(f"Is stable: {cert.is_stable}")

if not cert.is_stable:
    print(f"Warning: numerical residual is {cert.residual}")
    # Consider increasing precision or reducing stalk dimension
```


## FAQ

**Q: What is the difference between a sheaf Laplacian and a graph Laplacian?**
A: A graph Laplacian acts on scalar functions on vertices. A sheaf Laplacian acts on sections (vector-valued functions) on all cells, with restriction maps connecting adjacent cells. The graph Laplacian is a special case of the sheaf Laplacian with 1D stalks and identity restriction maps.

**Q: When should I use the product sheaf vs cosheaf?**
A: Use product sheaf when node attributes are independent of topology (L = L_topo + L_attr). Use cosheaf when the topology should constrain attribute diffusion (L = B^T B).

**Q: How do I handle large stalk dimensions?**
A: For d > 32, use sparse matrix storage and iterative eigensolvers. The `compute_eigenpairs` method with `num_eigenpairs << n*d` uses Lanczos iteration which avoids building the full Laplacian explicitly.


### Cross-references

- `pynerve.spectral`: Hodge Laplacian (non-sheaf version)
- `pynerve.graphs`: Graph structures underlying sheaves
- `pynerve.ml`: ML pipeline using sheaf features
- `pynerve.validation.benchmarks`: Sheaf Laplacian benchmark

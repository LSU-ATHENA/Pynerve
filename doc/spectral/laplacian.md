## Graph and Simplicial Laplacian

Incidence-based Laplacian operator for simplicial and cellular complexes.
Provides Hodge decomposition, spectral clustering, and manifold learning.

```python
lap = pynerve.spectral.Laplacian(complex)
lap.build_from_complex(complex)
```

### Matrix access

- `get_laplacian(dim)` returns L<sub>dim</sub> = B<sub>dim</sub>TB<sub>dim</sub> + B<sub>dim+1</sub>B<sub>dim+1</sub>T.
- `get_up_laplacian(dim)` returns L<sub>up</sub> = B<sub>dim+1</sub>B<sub>dim+1</sub>T.
- `get_down_laplacian(dim)` returns L<sub>down</sub> = B<sub>dim</sub>TB<sub>dim</sub>.
- `get_hodge_laplacian(dim)` is identical to `get_laplacian`.

The Hodge Laplacian at dimension k is:
L_k = B_k^T B_k + B_{k+1} B_{k+1}^T

This decomposes into:
- B_k^T B_k: map from k-chains to k-chains via boundary (down)
- B_{k+1} B_{k+1}^T: map from k-cochains to k-cochains via coboundary (up)

### Spectral analysis

- `eigenvalues(dim, k=0)` -- returns the k smallest eigenvalues; 0 means all.
- `eigenvectors(dim, k=0)` -- corresponding eigenvectors.
- `spectrum(dim)` -- full spectrum.
- `compute_embedding(dim, target_dim=2)` -- Laplacian eigenmap embedding.
- `compute_diffusion_map(target_dim=2)` -- diffusion map.
- `heat_kernel(dim, t)` -- heat kernel matrix.
- `heat_flow(initial, dim, t)` -- evolve a signal by the heat equation.
- `compute_spectral_gap(dim)` -- gap between zero and the first non-zero eigenvalue.
- `compute_cheeger_constants()` -- Cheeger inequality bounds.
- `compute_morse_index(dim)` -- number of negative eigenvalues.

### Example: spectral clustering

```python
# Build Laplacian for 1-skeleton
L1 = lap.get_laplacian(dim=1)
evals = lap.eigenvalues(dim=1, k=10)

# Cluster using first k eigenvectors
embedding = lap.compute_embedding(dim=1, target_dim=2)
from sklearn.cluster import KMeans
labels = KMeans(n_clusters=3).fit(embedding)
```


## Laplacian construction details

The Hodge Laplacian at dimension k is built from boundary matrices:

```
L_k = B_k^T B_k + B_{k+1} B_{k+1}^T
```

where B_k maps k-chains to (k-1)-chains.

### Matrix assembly

```python
# Manual assembly
B1 = complex.get_boundary_matrix(dim=1)  # maps edges to vertices
B2 = complex.get_boundary_matrix(dim=2)  # maps triangles to edges

# L_1 = B_1^T B_1 + B_2 B_2^T
L1_down = B1.T @ B1  # vertex-edge adjacency
L1_up = B2 @ B2.T    # edge-triangle adjacency
L1 = L1_down + L1_up
```

### Sparse storage

All Laplacians use CSR format internally:

```python
L1 = lap.get_laplacian(dim=1)
print(f"Shape: {L1.shape}")
print(f"Non-zeros: {L1.nnz}")
print(f"Density: {L1.nnz / (L1.shape[0] * L1.shape[1]):.6f}")
```

### Heat kernel and diffusion

```python
# Heat kernel matrix at time t
H_t = lap.heat_kernel(dim=1, t=1.0)
# H_t = exp(-t * L_1)

# Evolve initial signal by heat equation
initial = np.random.randn(complex.num_simplices(1))
evolved = lap.heat_flow(initial, dim=1, t=0.5)

# Diffusion map embedding
embedding = lap.compute_diffusion_map(target_dim=2, t=1.0)
```

### Cheeger constants

```python
# Compute Cheeger inequality bounds
cheeger = lap.compute_cheeger_constants()
print(f"Cheeger constant (dim 0): {cheeger.h_0:.4f}")
print(f"Cheeger constant (dim 1): {cheeger.h_1:.4f}")

# Spectral gap bounds spectral clustering quality
# lambda_1 / 2 <= h <= 2 * sqrt(lambda_1)
gap = lap.compute_spectral_gap(dim=1)
print(f"Spectral gap (dim 1): {gap:.6f}")
print(f"h bound: [{gap/2:.4f}, {2*sqrt(gap):.4f}]")
```

### Morse index

The Morse index counts negative eigenvalues of the Laplacian (should be zero for the standard Laplacian, but can be non-zero for weighted variants):

```python
morse_idx = lap.compute_morse_index(dim=1)
print(f"Morse index (dim 1): {morse_idx}")
# Non-zero Morse index indicates an unstable weighted Laplacian
```

## Weighted Laplacian

```python
# Edge-weighted Laplacian
lap.set_edge_weights(weights)  # [num_edges] float array
L_weighted = lap.get_laplacian(dim=1)

# The weighted Laplacian has modified spectral properties
# First non-zero eigenvalue is bounded by edge connectivity
```


## FAQ

**Q: What is the nullspace of L_k?**
A: The nullspace of L_k consists of harmonic k-forms, which are in bijection with the k-th cohomology group. The dimension of the nullspace equals the k-th Betti number (for real coefficients).

**Q: How does the Laplacian change under simplicial subdivision?**
A: Subdivision refines the complex while preserving cohomology. The Laplacian eigenvalues scale approximately as O(h^2) where h is the mesh size, but the harmonic spectrum (zero eigenvalues) is preserved.

**Q: Can I compute the Laplacian of a non-simplicial cell complex?**
A: Yes. The `Laplacian` class works with any regular cell complex that provides boundary matrices. Cubical complexes, polyhedral complexes, and CW complexes are supported.


### Cross-references

- `pynerve.spectral`: Spectral methods overview
- `pynerve.sheaf.laplacian`: Sheaf Laplacian
- `pynerve.graphs`: Graph structure
- `pynerve.algebra.BoundaryMatrix`: Boundary matrices used in construction
- `pynerve.spectral.gpu`: GPU-accelerated Laplacian construction

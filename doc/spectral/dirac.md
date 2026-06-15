## Higher-Order Dirac Operator

The Dirac operator D = B + B^T, where B is the block-diagonal coboundary
matrix. Satisfies D^2 = L (the Hodge Laplacian).

```python
dirac = pynerve.spectral.DiracOperator(complex)

D = dirac.get_dirac()            # D = B + B^T
D2 = dirac.get_dirac_squared()   # D^2 = Hodge Laplacian
evals = dirac.eigenvalues(k=10)
evecs = dirac.eigenvectors(k=10)
```

### Atiyah-Singer index

```python
idx = dirac.compute_atiyah_singer_index()
# index = dim(ker D) - dim(ker D^*) = spectral flow

analytic = dirac.compute_analytical_index()
topological = dirac.compute_topological_index()
```

### Spinor Laplacian and chirality

Complex coefficient variants.

```python
S = dirac.get_spinor_laplacian()      # complex matrix
chi = dirac.get_chirality_operator()  # gamma-like grading operator
```

Clifford product GPU kernel: `dirac_clifford_product_gpu.cu` provides gamma
matrix products on GPU with tiled mat-mul O(n^2 / cores).


## Dirac operator construction

```python
from pynerve.spectral import DiracOperator
import numpy as np

# Build from complex
dirac = DiracOperator(complex)

# Access components
D = dirac.get_dirac()             # D = B + B^T
D_squared = dirac.get_dirac_squared()  # D^2 = Hodge Laplacian

# Eigendecomposition
evals = dirac.eigenvalues(k=20)
evecs = dirac.eigenvectors(k=20)

# The spectrum of D is symmetric: {+-sqrt(lambda)}
# where lambda are eigenvalues of the Hodge Laplacian
print(f"Spectrum: min={evals[0]:.4f}, max={evals[-1]:.4f}")
print(f"Zero modes: {sum(1 for v in evals if abs(v) < 1e-8)}")
```

## Atiyah-Singer index theorem

The index theorem relates analytical and topological indices:

```python
index = dirac.compute_atiyah_singer_index()
# index = dim(ker D) - dim(ker D^*)
# For a self-adjoint Dirac operator, index = 0

# Decomposition
analytical = dirac.compute_analytical_index()
topological = dirac.compute_topological_index()

print(f"Analytical index: {analytical}")
print(f"Topological index: {topological}")
print(f"Match: {abs(analytical - topological) < 1e-8}")
```

## Spinor Laplacian and chirality

```python
# Spinor (complex) Laplacian
S = dirac.get_spinor_laplacian()  # complex Hermitian matrix

# Chirality operator (gamma-like grading)
chi = dirac.get_chirality_operator()
# Satisfies: chi * D + D * chi = 0 (anticommutes)
#           chi^2 = I

# Decompose into chiral subspaces
from pynerve.spectral import project_chiral
plus_component = project_chiral(evecs[:, 0], chi, +1)
minus_component = project_chiral(evecs[:, 0], chi, -1)
```

## Clifford algebra GPU kernel

The Clifford product on GPU uses tiled matrix multiplication:

```cpp
// dirac_clifford_product_gpu.cu
template <typename T>
__global__ void clifford_product_kernel(
    const T* gamma_matrices,  // [n_gamma, n, n]
    const T* input,           // [n, batch_size]
    T* output,                // [n, batch_size]
    int n, int batch_size, int n_gamma
) {
    // Each block processes one gamma matrix
    // Tile size = 16 for FP16 Tensor Core
    __shared__ T tile_a[16][16];
    __shared__ T tile_b[16][16];
    // ... WMMA tiled multiply ...
}
```

### Performance

- Dirac build (10k cells): 5 ms on CPU, 0.3 ms on A100, 0.2 ms on H100.
- Dirac eigendecomposition (k=10): 50 ms on CPU, 5 ms on A100, 3 ms on H100.
- Clifford product (100k): 20 ms on CPU, 1 ms on A100, 0.5 ms on H100.
- Spinor Laplacian build: 10 ms on CPU, 0.8 ms on A100, 0.5 ms on H100.


## FAQ

**Q: Why use the Dirac operator instead of the Hodge Laplacian?**
A: The Dirac operator's eigenvalues are square roots of the Laplacian eigenvalues, providing a more uniform spectral spacing. Its zero modes correspond exactly to harmonic forms. The chirality operator enables topological index computations.

**Q: When is the Atiyah-Singer index non-zero?**
A: For a manifold with boundary or a complex with non-trivial topology, the index can be non-zero. On a closed manifold, the index of the Dirac operator equals the Euler characteristic (for the de Rham complex) or the A-genus (for the spin complex).

**Q: Can the Dirac operator be used for clustering?**
A: The Dirac operator's eigenvectors encode both the harmonic (cohomology) and non-harmonic (metric) structure. For clustering, the lowest non-zero eigenvectors of D (or equivalently L) give the spectral embedding.


### Cross-references

- `pynerve.spectral`: Spectral methods overview
- `pynerve.spectral.laplacian`: Hodge Laplacian (D^2)
- `pynerve.spectral.gpu`: GPU Dirac implementation
- `pynerve.spectral.eigensolver`: Eigenvalue computation for Dirac
- `pynerve.algebra.BoundaryMatrix`: Coboundary operator for Dirac construction

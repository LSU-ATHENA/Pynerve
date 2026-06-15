## Lanczos and Arnoldi Eigensolvers

Lanczos iteration and Arnoldi solver for large sparse Laplacians. Includes
persistent Laplacian spectrum computation with warm starts.

```python
from pynerve.spectral import PersistentLaplacianSolver, SpectralConfig

config = SpectralConfig(
    num_eigenpairs=50,
    convergence_tolerance=1e-8,
    max_iterations=1000,
    compute_harmonic=True,
    solver_type="arnoldi",       # "arnoldi" | "lanczos" | "direct"
    spectral_shift=0.0,
    enable_gpu=False,
    enable_warm_start=True,
)
solver = PersistentLaplacianSolver(config)
```

### Persistent Laplacian spectrum

```python
L = solver.build_persistent_laplacian(boundary_matrix, filtration_values)
result = solver.compute_spectrum(L)
# result.eigenpairs[] -- each has .eigenvalue, .eigenvector, .is_harmonic
# result.harmonic_eigenpairs
# result.nonharmonic_eigenpairs
# result.spectral_radius, trace, frobenius_norm
# result.converged, result.condition_number
# result.orthogonality_error, result.residual_norm
```

### Warm start

```python
result2 = solver.compute_spectrum_with_warm_start(L2, result)
```

When filtration values change slightly, the eigensolver can use the previous
result as an initial guess, converging in fewer iterations.

### Harmonic vs non-harmonic

```python
harm = solver.compute_harmonic_spectrum(L)
nonharm = solver.compute_nonharmonic_spectrum(L)
```

### Dirac from boundary matrix

```python
D = solver.build_dirac_operator(boundary_matrix)
```

### Validation

```python
solver.validate_decomposition(result, L)
```


## Lanczos algorithm details

The Lanczos iteration builds a tridiagonal matrix T from the Laplacian L:

```
1. Initialize v_0 = random vector, beta_0 = 0
2. For j = 1, 2, ..., k:
   a. w_j = L * v_j - beta_{j-1} * v_{j-1}
   b. alpha_j = v_j^T * w_j
   c. w_j = w_j - alpha_j * v_j
   d. beta_j = ||w_j||
   e. v_{j+1} = w_j / beta_j
```

The eigenvalues of T_j (the j x j tridiagonal matrix) approximate the extreme eigenvalues of L.

```python
# Custom Lanczos parameters
from pynerve.spectral import PersistentLaplacianSolver

solver = PersistentLaplacianSolver(SpectralConfig(
    num_eigenpairs=50,
    solver_type="lanczos",
    convergence_tolerance=1e-8,
    max_iterations=1000,
    lanczos_reorthogonalize=True,  # full reorthogonalization
))
```

## Warm start convergence

```python
# First computation
L1 = ...  # initial Laplacian
result1 = solver.compute_spectrum(L1)

# Small change to filtration values
L2 = ...  # same structure, slightly different values
result2 = solver.compute_spectrum_with_warm_start(L2, result1)

print(f"First: {result1.iterations} iterations")
print(f"Warm start: {result2.iterations} iterations")
print(f"Speedup: {result1.iterations / result2.iterations:.1f}x")
```

Warm start is most effective when:
- The matrix changes by <10% in Frobenius norm
- The number of sought eigenpairs is small (k <= 20)
- The eigenvectors of the previous solution are good approximations

## Persistent Laplacian spectrum

The persistent Laplacian tracks how eigenvalues change with filtration:

```python
L = solver.build_persistent_laplacian(boundary_matrix, filtration_values)
result = solver.compute_spectrum(L)

# Track eigenvalue persistence
for ep in result.eigenpairs:
    if ep.is_harmonic:
        print(f"Harmonic: lambda={ep.eigenvalue:.6f}")
    else:
        print(f"Non-harmonic: lambda={ep.eigenvalue:.6f}, "
              f"non-harmonicity={ep.non_harmonicity:.4f}")
```

## Validation and quality metrics

```python
# Validate decomposition
validation = solver.validate_decomposition(result, L)
print(f"Orthogonality error: {result.orthogonality_error:.2e}")
print(f"Residual norm: {result.residual_norm:.2e}")
print(f"Condition number: {result.condition_number:.2e}")

if not result.converged:
    print(f"Did not converge after {result.iterations} iterations")
    print(f"Try increasing max_iterations or decreasing tolerance")
```


## FAQ

**Q: When does Lanczos fail to converge?**
A: Lanczos may fail when: (1) eigenvalues are clustered and close together, (2) the starting vector is orthogonal to the desired eigenvectors, (3) reorthogonalization is not used and numerical losses cause ghost eigenvalues. Use Arnoldi or increase reorthogonalization frequency.

**Q: How do I choose the number of Lanczos iterations?**
A: Rule of thumb: run 2-3x more iterations than the number of desired eigenpairs. Monitor convergence via the residual norm. The solver automatically stops when all requested eigenpairs converge.

**Q: What is the persistent Laplacian?**
A: The persistent Laplacian is a filtration-parameterized family of Laplacians. At each filtration value, L(t) = B(t)^T B(t) + B(t+1) B(t+1)^T where B(t) includes only simplices born at or before t. Tracking eigenvalues across t reveals spectral persistence.


### Cross-references

- `pynerve.spectral`: Spectral methods overview
- `pynerve.spectral.laplacian`: Laplacian construction
- `pynerve.spectral.dirac`: Dirac operator
- `pynerve.spectral.gpu`: GPU eigensolver
- `pynerve.validation.benchmarks`: Eigensolver benchmarks

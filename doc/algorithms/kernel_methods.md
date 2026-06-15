## Kernel methods

Gaussian kernel matrix computation for persistence diagrams and point clouds.
Used in SVM-style topological classification and kernel-based learning.


### Gaussian kernel

```cpp
template <typename T>
std::vector<T> compute_gaussian_kernel_matrix(
    std::span<const T> diagram1, size_t n1,
    std::span<const T> diagram2, size_t n2,
    T sigma);
```

Computes K[i,j] = exp(-||d1[i] - d2[j]||^2 / (2 * sigma^2)).

```python
from pynerve.algorithms import compute_gaussian_kernel_matrix
import numpy as np

diagram = np.array([[0.0, 1.0], [0.5, 2.0], [1.0, 3.0]])
K = compute_gaussian_kernel_matrix(diagram, diagram, sigma=1.0)
# K.shape == (3, 3)
# K[i,j] = exp(-||diagram[i] - diagram[j]||^2 / 2)
```


### Multi-scale kernel

The library also supports a multi-scale variant:

```cpp
template <typename T>
std::vector<T> compute_multiscale_gaussian_kernel(
    std::span<const T> diagram1, size_t n1,
    std::span<const T> diagram2, size_t n2,
    const std::vector<T>& sigmas);
```

Computes the sum of Gaussian kernels at multiple scales, which is useful
for capturing both fine and coarse topological features:

K_ms(x, y) = sum_{s in sigmas} exp(-||x - y||^2 / (2 * s^2))


### Persistence scale-space kernel

The persistence scale-space kernel (PSSK) is a valid positive-definite kernel
on persistence diagrams (Reininghaus et al., 2015):

K_pssk(D1, D2) = sum_{p in D1} sum_{q in D2} k(p, q) where k is the
Gaussian kernel on the birth-death plane, weighted by persistence.

```python
from pynerve.ml import persistence_scale_space_kernel

K = persistence_scale_space_kernel(diagrams1, diagrams2, sigma=0.5)
```

This kernel is used in SVM-based topological classification and is
provably stable under diagram perturbations.


### Sliced Wasserstein kernel

The sliced Wasserstein kernel (Carriere et al., 2017) approximates the
Wasserstein distance via random projections and exponentiates:

K_sw(D1, D2) = exp(-SW(D1, D2) / (2 * sigma^2))

```python
from pynerve.ml import sliced_wasserstein_kernel

K = sliced_wasserstein_kernel(diagrams1, diagrams2, num_slices=100, sigma=0.5)
```

Where num_slices controls the approximation quality (more slices = better).


### Persistence Fisher kernel

The persistence Fisher kernel (Le et al., 2019) uses Fisher information
geometry on the space of persistence diagrams modeled as Gaussian mixtures:

```python
from pynerve.ml import persistence_fisher_kernel

K = persistence_fisher_kernel(diagrams1, diagrams2, sigma=0.5)
```


### Kernel matrix utilities

```python
from pynerve.ml import (
    compute_kernel_matrix,
    normalize_kernel_matrix,
    center_kernel_matrix,
)

# General kernel dispatch
K = compute_kernel_matrix(diagrams, kernel="gaussian", sigma=0.5)

# Normalize to unit diagonal
K_norm = normalize_kernel_matrix(K)

# Center (for kernel PCA)
K_centered = center_kernel_matrix(K)
```


### Implementation details

The Gaussian kernel matrix is computed using the identity:

||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x^T y

This allows SIMD-accelerated computation:

1. Compute norms vector: norms[i] = sum(diagram[i]^2)
2. Compute dot products: D[i,j] = diagram[i] . diagram[j]
3. Compute squared distances: sq_dist[i,j] = norms[i] + norms[j] - 2*D[i,j]
4. Apply Gaussian: K[i,j] = exp(-sq_dist[i,j] / (2*sigma^2))

Steps 2 and 4 are SIMD-accelerated with AVX-512 FMA and exp intrinsics.


### Complexity

The Gaussian kernel costs O(n1 * n2 * d) for construction and O(n1 * n2) for evaluation. Multi-scale costs O(n1 * n2 * d * num_scales) for construction and O(n1 * n2 * num_scales) for evaluation. PSSK costs O(n1 * n2 * d) for construction and O(n1 * n2) for evaluation. Sliced Wasserstein costs O(n1 * n2 * num_slices) for construction and O(n1 * n2) for evaluation. Fisher costs O(n1 * n2 * d) for construction and O(n1 * n2) for evaluation.


### Common pitfalls

1. **Sigma selection**: Too small sigma -> all entries near zero (diagonal
   near 1). Too large sigma -> all entries near 1. Use the median of
   pairwise distances as a heuristic.

2. **Negative eigenvalues**: Some kernels (notably PSSK) can produce
   non-positive definite matrices for small sigma. Increase sigma or add
   a small diagonal shift.

3. **Memory**: Full kernel matrix for 10k diagrams is several hundred megabytes in FP64. Use Nystrom approximation for larger sets.


### Cross-references

- `pynerve.ml`: ML pipeline with kernel methods
- `pynerve.metrics.wasserstein`: Wasserstein distance (used by SW kernel)
- `pynerve.algorithms.vectorization`: alternative feature-based approach

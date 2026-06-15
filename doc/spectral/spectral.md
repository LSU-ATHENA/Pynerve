# Spectral Methods

Higher-order spectral operators on simplicial and cellular complexes. Pynerve
provides the graph Laplacian (incidence-based), Hodge Laplacian (up + down
per dimension), Dirac operator (square root of Hodge Laplacian), and
Lanczos/Arnoldi eigensolvers.

```python
import pynerve
import numpy as np
from pynerve.algebra import SimplicialComplex

complex = pynerve.algebra.build_vr_complex(points, max_dim=2, max_radius=1.0)
lap = pynerve.spectral.Laplacian(complex)
evals = lap.eigenvalues(dim=1, k=10)
print(evals)
```


## Modules

The spectral module documentation is organized across four component pages. [laplacian.md](laplacian.md) covers graph and simplicial Hodge Laplacian construction. [dirac.md](dirac.md) documents higher-order Dirac operators and the Atiyah-Singer index theorem. [eigensolver.md](eigensolver.md) describes Lanczos and Arnoldi solvers for computing the harmonic spectrum. [gpu.md](gpu.md) covers the CUDA-accelerated eigensolver, Dirac operator, and Clifford products.


## Core API

### `pynerve.spectral.Laplacian`

Incidence-based Laplacian operator for simplicial and cellular complexes.

```python
lap = pynerve.spectral.Laplacian(complex)
lap.build_from_complex(complex)
```

**Configuration:**

```python
config = pynerve.spectral.LaplacianConfig(
    enable_gpu=False,
    threshold=1000,
    prefer_tiled_kernels=True,
    max_gpu_memory_mb=0,
)
```

### `pynerve.spectral.HodgeTheory`

Hodge decomposition: each cochain is the sum of harmonic, exact, and
coexact components.

```python
hodge = pynerve.spectral.HodgeTheory(complex)
harm = hodge.compute_harmonic_forms(dim=1)
exact = hodge.compute_exact_forms(dim=1)
coexact = hodge.compute_coexact_forms(dim=1)
star = hodge.get_hodge_star(dim)
codiff = hodge.get_codifferential(dim)
betti = hodge.compute_betti_numbers()
hodge_nums = hodge.compute_hodge_numbers()
wedge = hodge.wedge_product(alpha, beta)
inner = hodge.interior_product(vector_field, form)
```

### `pynerve.spectral.SpectralClustering`

```python
clustering = pynerve.spectral.SpectralClustering()
labels = clustering.cluster(k=5, dim=1)
labels_n = clustering.cluster_normalized_laplacian(k=5, dim=1)
embedding = clustering.compute_embedding(dim=2)
quality = clustering.compute_cluster_quality(labels)
optimal_k = clustering.compute_optimal_k(max_k=10)
```

### `pynerve.spectral.ManifoldLearning`

```python
ml = pynerve.spectral.ManifoldLearning(complex)
em = ml.laplacian_eigenmaps(target_dim=2, dim=1)
dm = ml.diffusion_maps(target_dim=2, t=1.0)
iso = ml.isomap(target_dim=2)
lle = ml.lle(target_dim=2, k_neighbors=10)
he = ml.hessian_eigenmaps(target_dim=2, k_neighbors=10)
intrinsic_dim = ml.compute_intrinsic_dimensionality()
```

### Spectral feature extraction

```python
from pynerve.spectral import SpectralFeatureExtractor, SpectralAnomalyDetector

extractor = SpectralFeatureExtractor(max_features=100)
features = extractor.extract_features(spectrum)
stats = extractor.extract_spectral_statistics(spectrum)
topo = extractor.extract_topological_features(spectrum)

detector = SpectralAnomalyDetector(threshold=2.0)
anomaly = detector.detect_anomaly(spectrum)
# anomaly.is_anomaly, anomaly_score, anomalous_eigenpairs
```

### `SpectralStackManager` (singleton)

```python
from pynerve.spectral import SpectralStackManager
mgr = SpectralStackManager.instance()
result = mgr.compute_spectrum(L, enable_gpu=True)
features = mgr.extract_features(spectrum)
anomaly = mgr.detect_anomaly(spectrum)
gnn_feats = mgr.prepare_gnn_features(spectrum)
print(mgr.get_performance_stats())
```


## Use cases

- **Spectral clustering**: use L1 of the 1-skeleton to compute a k-way partition from eigenvectors.
- **Hodge decomposition**: use L<sub>dim</sub> of the complex to obtain harmonic forms as cohomology representatives.
- **Harmonic analysis**: apply the heat kernel on the complex for multi-scale diffusion.
- **Anomaly detection**: stream Laplacians and compute the spectral deviation score.
- **Graph neural nets**: extract Laplacian eigenfeatures for spectral GNN convolution.


## Complexity

- Building the Laplacian L<sub>dim</sub> costs O(m<sub>dim</sub>) where m<sub>dim</sub> is the number of simplices at dimension dim.
- Lanczos eigenvalue computation costs O(k x nnz(L)) for k iterations.
- Arnoldi eigenvalue computation costs O(k^2 x n + k x nnz) and is more robust than Lanczos.
- Direct solve costs O(n^3) and is suitable for small matrices only.
- Dirac construction costs O(m) for B + BT block assembly.


## Practical guidance

### Choosing a solver

- For small matrices (under 1000 x 1000), use the direct solver -- it is exact and fast.
- For sparse matrices where only a few eigenpairs are needed, use Lanczos -- it is memory-efficient.
- For non-symmetric or ill-conditioned problems, use Arnoldi -- it is more robust.
- When the full spectrum is needed, the direct solve is acceptable for n < 5000.
- If a GPU is available, use GPU Lanczos for 10--50x speedup.
- For sequences of similar matrices, use warm-start Arnoldi to reuse the previous solution.

### Common pitfalls

1. **Laplacian dimension mismatch**: L_dim requires B_dim and B_{dim+1}. Ensure the complex has simplices at both dim and dim+1. Use `complex.max_dimension()` to check.
2. **Eigensolver convergence failure**: Lanczos may miss eigenvalues for clustered spectra. Increase `max_iterations` or use Arnoldi with spectral shift.
3. **Harmonic vs non-harmonic**: Zero eigenvalues correspond to harmonic forms (cohomology representatives). Non-zero eigenvalues measure deviation from harmonicity.
4. **Memory for large Laplacians**: The Laplacian L_dim has O(n_dim * avg_degree) non-zeros. For dim=1 on a graph with 10M edges, expect hundreds of megabytes in CSR format.

### Memory estimation

```python
# Estimate Laplacian memory
n_simplices = 100000  # at dimension dim
avg_degree = 10       # average faces/cofaces per simplex
nnz = n_simplices * avg_degree

# CSR storage: values + column indices + row pointers
bytes_csr = nnz * (8 + 4) + (n_simplices + 1) * 8
print(f"CSR memory: {bytes_csr / 1e6:.1f} MB")

# COO storage: values + row + column
bytes_coo = nnz * (8 + 4 + 4)
print(f"COO memory: {bytes_coo / 1e6:.1f} MB")
```

### Spectral feature extraction

```python
from pynerve.spectral import SpectralFeatureExtractor

extractor = SpectralFeatureExtractor(max_features=100)

# Extract features from Laplacian spectrum
spectrum = lap.eigenvalues(dim=1, k=100)
features = extractor.extract_features(spectrum)
stats = extractor.extract_spectral_statistics(spectrum)
topo = extractor.extract_topological_features(spectrum)

print("Spectral features:", features)
# [spectral_gap, spectral_radius, dimension, ...]

print("Statistics:", stats)
# {mean, std, skewness, kurtosis, entropy}
```

### Anomaly detection on streaming spectra

```python
from pynerve.spectral import SpectralAnomalyDetector

detector = SpectralAnomalyDetector(threshold=2.0)

# Process streaming Laplacians
for L in laplacian_stream:
    spectrum = lap.eigenvalues(dim=1, k=20)
    anomaly = detector.detect_anomaly(spectrum)

    if anomaly.is_anomaly:
        print(f"Anomaly detected! Score: {anomaly.anomaly_score:.3f}")
        print(f"Suspicious eigenpairs: {anomaly.anomalous_eigenpairs}")
        # Eigenpairs with largest spectral deviation
```


## Hodge decomposition example

```python
from pynerve.spectral import HodgeTheory
import numpy as np

# Build complex
complex = pynerve.algebra.build_vr_complex(points, max_dim=2, max_radius=1.0)
hodge = HodgeTheory(complex)

# Decompose a 1-cochain
cochain = np.random.randn(complex.num_simplices(dim=1))

harmonic = hodge.compute_harmonic_forms(dim=1)
exact = hodge.compute_exact_forms(dim=1)
coexact = hodge.compute_coexact_forms(dim=1)

# Verify decomposition
reconstructed = harmonic + exact + coexact
error = np.linalg.norm(cochain - reconstructed)
print(f"Hodge decomposition error: {error:.2e}")
```


## FAQ

**Q: Why does my Laplacian have negative eigenvalues?**
A: Due to floating-point round-off, near-zero eigenvalues may appear slightly negative. This is normal. The spectral gap is computed as `max(0, lambda_1) - lambda_0`. If eigenvalues are significantly negative, check that the Laplacian is symmetric and positive semi-definite.

**Q: How do I choose the number of eigenvalues?**
A: For spectral clustering, k = number of clusters is sufficient. For spectral gap analysis, k = 5-10 captures the low-frequency structure. For full spectral analysis, choose k such that lambda_k exceeds the spectral radius of interest.

**Q: Can I compute the Laplacian of a weighted simplicial complex?**
A: Yes. The `Laplacian` class accepts weighted boundary matrices via `set_weighted_boundary(dim, weights)`. Edge weights propagate to all higher-dimensional faces via the product of constituent edge weights.


### Cross-references

- `pynerve.algebra.BoundaryMatrix`: Used for Laplacian construction
- `pynerve.sheaf.laplacian`: Sheaf Laplacian (parameterized by stalk data)
- `pynerve.graphs`: Graph structures underlying Laplacians
- `pynerve.ml`: ML pipeline using spectral features
- `pynerve.spectral.eigensolver`: Eigensolver implementation details

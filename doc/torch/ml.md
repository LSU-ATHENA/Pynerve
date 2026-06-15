## ML Operations

Functional API for diagram vectorization and kernel methods. All operations
work on raw ATen tensors (no PersistenceDiagram wrapper required).

### Vectorization

```python
# Persistence image
img = nt.ml_persistence_image(diagram, resolution_birth=20,
                              resolution_death=20, sigma=0.5)

# Persistence landscape
land = nt.ml_persistence_landscape(diagram, k=5, num_samples=100)

# Persistence silhouette
sil = nt.ml_persistence_silhouette(diagram, num_samples=100)

# Heat kernel signature
hks = nt.ml_heat_kernel_signature(diagram, num_samples=100, sigma=0.1)

# Birth-death curve
curve = nt.ml_birth_death_curve(diagram, num_bins=50, statistic="count")
```

### Statistics

```python
tp = nt.ml_total_persistence(diagram, dim=-1, p=1.0)
mp = nt.ml_mean_persistence(diagram, dim=-1)
mx = nt.ml_max_persistence(diagram, dim=-1)
var = nt.ml_persistence_variance(diagram, dim=-1)
ent = nt.ml_persistence_entropy(diagram, dim=-1)
nfeat = nt.ml_number_of_features(diagram, dim=-1, min_persistence=0)
betti = nt.ml_betti_curve(diagram, num_samples=100, dim=-1)
amp = nt.ml_amplitude(diagram, metric="persistence", p=1.0, dim=-1)
all_stats = nt.ml_all_statistics(diagram, dims=[0, 1])
features = nt.ml_extract_features(diagram, dims=[0, 1])
```

### Kernels

```python
gk = nt.ml_gaussian_kernel(d1, d2, sigma=0.5, distance_metric="euclidean")
ssk = nt.ml_persistence_scale_space_kernel(d1, d2, sigma=0.5, weight=0.5)
swk = nt.ml_sliced_wasserstein_kernel(d1, d2, num_slices=10, sigma=0.5)
pfk = nt.ml_persistence_fisher_kernel(d1, d2, sigma=0.5)
lk = nt.ml_linear_kernel(d1, d2, num_samples=100)

K = nt.ml_compute_kernel_matrix(diagrams, kernel="gaussian", sigma=0.5)
K_norm = nt.ml_normalize_kernel_matrix(K)
K_centered = nt.ml_center_kernel_matrix(K)
```

### Example: diagram classification pipeline

```python
import torch
import pynerve.torch as nt

# Generate diagrams
diagrams = [compute_diagram(points) for points in dataset]

# Stack and compute features
diagram_tensor = torch.stack(diagrams)
stats = nt.ml_all_statistics(diagram_tensor, dims=[0, 1])

# Train classifier
from sklearn.svm import SVC
clf = SVC(kernel="precomputed")
K = nt.ml_compute_kernel_matrix(diagrams, kernel="persistence_scale_space")
clf.fit(K, labels)
```


## Vectorization details

### Persistence image

```python
img = nt.ml_persistence_image(
    diagram,
    resolution_birth=20,
    resolution_death=20,
    sigma=0.5,
    weight="linear",        # "linear" | "constant" | callable
)

# Custom weight function
def my_weight(birth, death):
    return (death - birth) ** 2

img = nt.ml_persistence_image(
    diagram, resolution=64, sigma=0.1,
    weight=my_weight,
)
```

### Persistence landscape

```python
land = nt.ml_persistence_landscape(
    diagram,
    k=5,
    num_samples=100,
)

# Access individual levels
for level in range(5):
    plt.plot(land[level].numpy())
```

### Silhouette

```python
sil = nt.ml_persistence_silhouette(
    diagram,
    num_samples=100,
    weight="persistence",  # weight by persistence
)

# Custom weight function
def log_weight(birth, death, dim):
    return np.log1p(death - birth)

sil = nt.ml_persistence_silhouette(
    diagram, num_samples=100, weight=log_weight,
)
```

## Statistics API

```python
# Per-dimension statistics
dim0_stats = nt.ml_all_statistics(diagram, dims=[0])
dim1_stats = nt.ml_all_statistics(diagram, dims=[1])

# Feature extraction
features = nt.ml_extract_features(diagram, dims=[0, 1])
# Returns concatenated feature vector:
# [total_persistence_H0, mean_persistence_H0, persistence_entropy_H0,
#  max_persistence_H0, num_features_H0, amplitude_H0,
#  total_persistence_H1, ...]
```

## Kernel methods

```python
# Compute full kernel matrix
diagrams = [d1, d2, d3, d4]
K = nt.ml_compute_kernel_matrix(
    diagrams,
    kernel="persistence_scale_space",
    sigma=0.5,
)

# Normalize kernel
K_norm = nt.ml_normalize_kernel_matrix(K)
# K_norm[i,j] = K[i,j] / sqrt(K[i,i] * K[j,j])

# Center kernel (for kernel PCA)
K_centered = nt.ml_center_kernel_matrix(K_norm)

# Train SVM with precomputed kernel
from sklearn.svm import SVC
clf = SVC(kernel="precomputed")
clf.fit(K_centered, labels)
```

## Performance optimization

```python
# Batch operations are vectorized
# Instead of:
for d in diagrams:
    img = nt.ml_persistence_image(d, resolution=64)

# Do:
stacked = torch.stack(diagrams)
imgs = nt.ml_persistence_image(stacked, resolution=64)
# 5-10x faster due to batch SIMD/GPU processing
```


## FAQ

**Q: What is the difference between ml_persistence_landscape and to_persistence_landscape?**
A: `ml_persistence_landscape` is a functional API that works on raw tensors. `to_persistence_landscape` is a method on the `PersistenceDiagram` class. Both produce identical results; use the functional API when you don't need the wrapper class.

**Q: Can I compute kernels for diagrams with different dimensions?**
A: Yes. If diagrams have different max dimensions, the kernel handles it by treating missing dimensions as having zero contribution. The `ml_compute_kernel_matrix` function normalizes for dimension count.

**Q: How do I handle very large kernel matrices (10000+ diagrams)?**
A: The kernel matrix is O(n^2) memory. For >5000 diagrams, use Nystrom approximation via `ml_approximate_kernel_matrix(diagrams, num_landmarks=100)`.


### Cross-references

- `pynerve.ml`: ML module (C++ backend)
- `pynerve.algorithms.vectorization`: C++ vectorization
- `pynerve.nn`: Neural network layers
- `pynerve.validation.benchmarks`: Kernel method benchmarks

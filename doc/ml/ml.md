# Machine learning

Feature extraction from persistence diagrams for downstream ML pipelines.
Provides persistence landscapes, silhouettes, heat vectors, Betti curves,
and birth-death curve vectorization. Also includes differentiable simplex
operations for gradient-based topology optimization and diagram message
passing via GNNs.

```python
from pynerve.fast_ops import (
    persistence_landscape_fast,
    persistence_image_fast,
    betti_curve_fast,
)

# Persistence landscape: piecewise-linear functions
landscape = persistence_landscape_fast(
    pairs,           # [(birth, death, dim), ...]
    n_layers=5,
    resolution=100,
)

# Betti curve: Betti numbers over filtration parameter
betti = betti_curve_fast(
    pairs,
    max_dim=3,
    resolution=100,
    max_time=2.0,
)

# Persistence image: 2D density on birth-persistence plane
image = persistence_image_fast(
    pairs,
    resolution=64,
    sigma=0.1,
    weight_fn="linear",  # "linear" | "constant" | None
)
```


## Feature vectorization

All vectorization methods take a list of (birth, death, dimension) tuples
and return fixed-size feature vectors suitable for sklearn, PyTorch, or
XGBoost pipelines.

Persistence landscape produces output of shape `(n_layers, resolution)` and is used for stable features with L1-Lipschitz guarantees. Persistence image produces `(height, width)` output and provides a 2D grid for CNNs. Betti curve produces `(max_dim+1, resolution)` output and offers a multi-scale topology summary. Birth-death curve produces `(2, n_pairs)` output with raw sorted birth and death values. Silhouette produces `(resolution,)` output as a weighted landscape summary. Heat kernel produces `(resolution,)` output as a smooth density estimate.

See [vectorization.md](vectorization.md) for detailed API and C++
implementations of each method.


## Pages

The [vectorization.md](vectorization.md) page covers the detailed API for landscapes, images, Betti curves, silhouettes, and heat kernels. The [simplex_diff.md](simplex_diff.md) page describes differentiable simplex operations for gradient-based topology optimization. The [message_passing.md](message_passing.md) page covers GNN layers on persistence diagrams.


## Persistence encoder

The `PersistenceEncoder` in `src/encoders/` combines multiple vectorization
strategies into a single feature vector.

```cpp
#include <nerve/encoders/encoders.hpp>

PersistenceEncoder encoder;
encoder.enablePersistenceLandscape(true);
encoder.setLandscapeResolution(100);

auto features = encoder.computePersistenceLandscapes(diagram, num_landscapes, resolution);
```


## When to use

For classification, use persistence image + CNN or landscape + SVM. For regression, use Betti curve + gradient boosting. For clustering, use persistence landscape with L2 distance. For outlier detection, use birth-death curves with stability features. For topology-preserving autoencoders, use differentiable PH loss via `pynerve.diff`. For graph classification, use diagram message passing via `DiagramDeepSet`.


## Complexity

Persistence landscape has time complexity O(D * L * R) and space complexity O(L * R). Persistence image has time complexity O(D * H * W) and space complexity O(H * W). Betti curve has time complexity O(D * R) and space complexity O(R). Silhouette has time complexity O(D * R) and space complexity O(R). Heat kernel has time complexity O(D * R) and space complexity O(R).

D = pairs in diagram, L = landscape layers, R = resolution. H, W = image height and width.

All vectorization methods use SIMD-accelerated loops in the fast path.



## Practical guidance

### Choosing a vectorization method

For classification with SVM, use persistence landscape because it is L1-Lipschitz stable with an L2 kernel. For classification with CNN, use persistence image as a 2D grid input to convolutional layers. For regression with GBM, use Betti curve because it captures multi-scale density. For clustering, use silhouette as a weighted landscape that is good for L2 distance. For outlier detection, use birth-death curves as raw features for isolation forest. For visualization, use heat kernel as a smooth density estimate.

### Common pitfalls

1. **Resolution mismatch**: All vectorization methods discretize the diagram. Resolution too low loses information; too high wastes computation. Rule of thumb: `resolution = sqrt(num_pairs)`.
2. **Silhouette weight function**: The default weight (persistence) works well for most cases. For diagrams with very long-lived features, use `weight_fn="log"` to avoid dominance.
3. **Image boundary clipping**: Points near the birth-persistence plane boundary are convolved with partial Gaussian kernels. Extend the image bounds by 3*sigma beyond min/max values.
4. **Landscape normalization**: Landscapes are L1-Lipschitz functions. Normalize by total persistence before comparing across diagrams.

### Pipeline example

```python
from pynerve.fast_ops import (
    persistence_landscape_fast,
    persistence_image_fast,
    betti_curve_fast,
    silhouette_fast,
    heat_kernel_fast,
)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

# Extract features
pairs = [(b, d, dim) for b, d, dim in diagram]

# Combine multiple vectorization methods
landscape = persistence_landscape_fast(pairs, n_layers=5, resolution=100).flatten()
betti = betti_curve_fast(pairs, max_dim=2, resolution=50).flatten()
silhouette = silhouette_fast(pairs, resolution=100)

features = np.concatenate([landscape, betti, silhouette])

# Train classifier
clf = GradientBoostingClassifier(n_estimators=100)
clf.fit(features, labels)
```

### Batch vectorization

```python
from pynerve.fast_ops import batch_vectorize

# Process multiple diagrams efficiently
diagrams = [compute_diagram(points) for points in dataset]

# Batch landscape computation (shared memory, SIMD)
landscapes = batch_vectorize(
    diagrams,
    method="landscape",
    n_layers=5,
    resolution=100,
)
# landscapes shape: [n_diagrams, n_layers * resolution]
```


## Differentiable vectorization (PyTorch)

```python
import torch
from pynerve.torch import ml_persistence_image, ml_persistence_landscape

# Differentiable persistence image (gradients flow through)
diagram_tensor = torch.tensor(pairs, requires_grad=True)
img = ml_persistence_image(
    diagram_tensor,
    resolution=64,
    sigma=0.1,
)
loss = img.sum()
loss.backward()  # gradients flow back to diagram points
```

The backward pass computes gradient of the Gaussian kernel convolution with respect to each diagram point's (birth, death) coordinates.


## Performance tips

```python
# Use the fast_ops module for CPU-optimized vectorization
from pynerve.fast_ops import (
    persistence_landscape_fast,
    persistence_image_fast,
)

# For GPU acceleration, use torch operations
from pynerve.torch import ml_persistence_image

# Profile your vectorization
from pynerve.validation import benchmark_vectorization

bm = benchmark_vectorization(
    method="landscape",
    num_pairs=10000,
    resolution=200,
)
print(f"CPU: {bm.cpu_ms:.1f}ms, GPU: {bm.gpu_ms:.1f}ms")
```


## FAQ

**Q: What resolution should I use for persistence images?**
A: Start with 64x64. For diagrams with <1000 pairs, 32x32 may suffice. For >10k pairs, 128x128 captures more detail. Resolution beyond 256x256 rarely improves ML performance.

**Q: How do I handle diagrams with varying numbers of pairs?**
A: All vectorization methods produce fixed-size output regardless of input size. Persistence landscape always returns (n_layers, resolution); persistence image returns (resolution, resolution). Variable-length diagrams become fixed-length features.

**Q: Can I use vectorization features directly in a neural network?**
A: Yes. The `TopologicalFeatureExtractor` in `pynerve.torch` wraps vectorization into an `nn.Module` that can be part of a larger network. The `ml_extract_features` function returns a tensor suitable for `nn.Linear` layers.


### Cross-references

- `pynerve.algorithms.vectorization`: C++ vectorization implementation
- `pynerve.autodiff.tensor_ops`: Differentiable vectorization
- `pynerve.torch.ml`: PyTorch ML operations
- `pynerve.nn`: Neural network layers
- `pynerve.encoders`: Encoder architectures
- `pynerve.validation.benchmarks`: Vectorization benchmarks

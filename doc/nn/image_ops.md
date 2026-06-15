## Image-based diagram representations

### PersistenceImageLayer

Converts diagram points directly to a 2D image with Gaussian kernel density.

```cpp
template <typename T = float>
class PersistenceImageLayer {
    struct Config {
        int resolution_h = 20, resolution_w = 20;
        T sigma = T(0.1);
        enum class Weight { LINEAR, QUADRATIC, CONSTANT } weight = Weight::LINEAR;
    };

    std::vector<T> forward(std::span<const T> diagram, size_t batch_size, size_t n_pairs) const;
    std::vector<T> forward_multi_dim(std::span<const T> diagram, size_t batch_size,
                                     size_t n_pairs, int max_dim) const;
};
```

### LandscapeLayer

Computes persistence landscape from a diagram.

```cpp
template <typename T = float>
class LandscapeLayer {
    struct Config {
        int n_layers = 5;
        int resolution = 100;
        T min_persistence = T(0.0);
    };

    std::vector<T> forward(std::span<const T> diagram, size_t batch_size, size_t n_pairs) const;
};
```

### DiagramVectorizer

Produces a fixed-size feature vector from a variable-length diagram.

```cpp
template <typename T = float>
class DiagramVectorizer {
    struct Config {
        enum class Method {
            PERSISTENCE_STATS,  // mean, std, max of persistence
            BETTI_CURVE,        // Betti numbers over filtration
            ENTROPY,            // persistence entropy
            LANDSCAPE           // landscape samples
        } method = Method::PERSISTENCE_STATS;
        int output_dim = 64;
        int n_bins = 10;
    };

    std::vector<T> forward(std::span<const T> diagram, size_t batch_size, size_t n_pairs) const;
};
```

### CUDA persistence image kernel

```cpp
template <typename T>
__global__ void persistence_image_cuda_kernel(
    const T* diagram, T* image,
    int batch_size, int n_pairs,
    int height, int width,
    T sigma, T min_birth, T max_death);
```

### Example: using image-based layers

```python
from pynerve.nn import PersistenceImageLayer, LandscapeLayer, DiagramVectorizer

# Generate persistence image
pil = PersistenceImageLayer(resolution_h=64, resolution_w=64, sigma=0.1)
image = pil.forward(diagram_batch)

# Generate landscape
ll = LandscapeLayer(n_layers=5, resolution=100)
landscape = ll.forward(diagram_batch)

# Generate feature vector
dv = DiagramVectorizer(method="persistence_stats", output_dim=64)
features = dv.forward(diagram_batch)
```


## PersistenceImageLayer internals

```cpp
template <typename T>
std::vector<T> PersistenceImageLayer<T>::forward(
    std::span<const T> diagram,
    size_t batch_size, size_t n_pairs) const {

    // diagram: [batch, n_pairs, 2] = (birth, death)
    // Convert to birth-persistence coordinates
    std::vector<T> bp_coords(batch_size * n_pairs * 2);
    for (size_t i = 0; i < batch_size * n_pairs; i++) {
        T birth = diagram[i * 2];
        T death = diagram[i * 2 + 1];
        bp_coords[i * 2] = birth;
        bp_coords[i * 2 + 1] = death - birth;  // persistence
    }

    // Determine bounds
    T min_birth = *std::min_element(bp_coords.begin(), bp_coords.end());
    T max_death = *std::max_element(bp_coords.begin() + 1,
                                     bp_coords.end());
    T min_pers = *std::min_element(bp_coords.begin() + 1, ...);
    // ... build pixel grid, apply Gaussian kernel ...
}
```

## Multi-dimension support

```cpp
template <typename T>
std::vector<T> PersistenceImageLayer<T>::forward_multi_dim(
    std::span<const T> diagram,
    size_t batch_size, size_t n_pairs, int max_dim) const {

    // Separate diagram by dimension
    std::vector<T> result;
    for (int dim = 0; dim <= max_dim; dim++) {
        auto dim_pairs = extract_dimension(diagram, dim, n_pairs);
        auto dim_image = forward(dim_pairs, batch_size, dim_pairs.size());
        result.insert(result.end(), dim_image.begin(), dim_image.end());
    }
    // Result: [max_dim+1, resolution_h, resolution_w] per batch
    return result;
}
```

## LandscapeLayer computation

The persistence landscape at layer k is:

```
lambda_k(t) = k-th largest value of { max(0, death_i - |t - birth_i|) }
```

```python
# Manual landscape computation for understanding
t_values = np.linspace(0, max_filtration, resolution)
landscape = np.zeros((n_layers, resolution))

for b, d, dim in pairs:
    pers = d - b
    for i, t in enumerate(t_values):
        val = max(0, pers - abs(t - b))
        # Insert into sorted landscape layers
        for k in range(n_layers):
            if val > landscape[k, i]:
                landscape[k, i], val = val, landscape[k, i]
```

## Example: end-to-end image pipeline

```python
from pynerve.nn import PersistenceImageLayer, DiagramConv2D
import torch.nn as nn

class DiagramCNN(nn.Module):
    def __init__(self, resolution=64, num_classes=10):
        super().__init__()
        self.image_layer = PersistenceImageLayer(
            resolution_h=resolution,
            resolution_w=resolution,
            sigma=0.1,
        )
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(32, num_classes)

    def forward(self, diagram_batch):
        image = self.image_layer.forward(diagram_batch)
        # image shape: [batch, 1, resolution, resolution]
        features = self.conv(image).squeeze()
        return self.fc(features)
```


## FAQ

**Q: How do I choose the resolution for persistence images?**
A: Start with 64x64. For diagrams with <500 pairs, 32x32 suffices. For >5000 pairs, use 128x128. Resolution beyond 256 rarely improves performance but costs 4x memory per doubling.

**Q: Can I use LandscapeLayer with multi-dimensional diagrams?**
A: Yes. The LandscapeLayer processes pairs regardless of dimension. For per-dimension landscapes, call `forward_multi_dim` with `max_dim` parameter.

**Q: How does DiagramVectorizer handle variable-length input?**
A: The `PERSISTENCE_STATS` method computes mean, std, max across all pairs, producing fixed-size output. The `BETTI_CURVE` method bins by filtration value, producing fixed-size output. The `LANDSCAPE` method evaluates at fixed sample points.


### Cross-references

- `pynerve.nn`: Neural network overview
- `pynerve.nn.gpu`: GPU kernels for image operations
- `pynerve.algorithms.vectorization`: C++ vectorization
- `pynerve.ml`: ML pipeline
- `pynerve.torch.ml`: PyTorch ML operations

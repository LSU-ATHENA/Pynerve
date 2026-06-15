## Feature vectorization

All vectorization methods take a list of (birth, death, dimension) tuples
and return fixed-size feature vectors suitable for sklearn, PyTorch, or
XGBoost pipelines.

### Persistence landscape (C++)

```cpp
#include <nerve/algorithms/persistence_vectorization.hpp>

namespace nerve::algorithms {

struct PersistenceLandscape {
    std::vector<std::vector<double>> landscape_levels;
    double x_min, x_max;
    int num_levels;
};

template <typename T>
PersistenceLandscape compute_landscape(
    std::span<const T> diagram,
    size_t num_pairs,
    int num_levels = 5,
    double resolution = 0.01
);

}
```

The landscape lambda_k(t) = k-th largest persistence function at parameter t.
Layer k captures the k-th most persistent topological feature across the
filtration.

### Persistence image (C++)

```cpp
struct PersistenceImage {
    std::vector<std::vector<double>> image;
    int resolution;
    double sigma;
    double birth_min, birth_max, persistence_min, persistence_max;
};

template <typename T>
PersistenceImage compute_persistence_image(
    std::span<const T> diagram,
    size_t num_pairs,
    int resolution = 64,
    double sigma = 0.1
);
```

Each point (b, d) in the diagram is placed on the birth-persistence plane
and convolved with a Gaussian kernel of width sigma.

### Betti curve (C++)

```cpp
template <typename T>
std::vector<std::pair<double, int>> compute_betti_curve(
    std::span<const T> diagram,
    size_t num_pairs,
    int max_dim = -1
);
```

Returns vector of (filtration_value, betti_number) pairs for each dimension
up to max_dim.

### Silhouette

```python
from pynerve.fast_ops import silhouette_fast

# Weighted landscape summary
sil = silhouette_fast(
    pairs,
    weight=lambda p: p[1] - p[0],  # weight by persistence
    resolution=100,
)
# Returns (resolution,) array
```

### Heat kernel

```python
from pynerve.fast_ops import heat_kernel_fast

# Smooth density estimate on the birth-death plane
heat = heat_kernel_fast(
    pairs,
    resolution=100,
    sigma=0.1,
    t=0.5,  # diffusion time
)
# Returns (resolution,) array
```

### Complexity

Persistence landscape has time complexity O(D * L * R) and space complexity O(L * R). Persistence image has time complexity O(D * H * W) and space complexity O(H * W). Betti curve has time complexity O(D * R) and space complexity O(R). Silhouette has time complexity O(D * R) and space complexity O(R). Heat kernel has time complexity O(D * R) and space complexity O(R).

D = pairs in diagram, L = landscape layers, R = resolution. H, W = image height and width.

All vectorization methods use SIMD-accelerated loops in the fast path.


## Detailed C++ API

### Landscape computation

```cpp
// Full landscape pipeline
#include <nerve/algorithms/persistence_vectorization.hpp>

using namespace nerve::algorithms;

auto diagram = load_diagram(pairs);  // [num_pairs, 2] tensor

PersistenceLandscape landscape = compute_landscape(
    std::span<const double>(diagram.data(), diagram.size()),
    num_pairs,
    /*num_levels=*/5,
    /*resolution=*/0.01
);

// Access levels
for (int k = 0; k < landscape.num_levels; k++) {
    printf("Level %d: %zu sample points\n", k,
           landscape.landscape_levels[k].size());
}
```

### Image computation

```cpp
PersistenceImage image = compute_persistence_image(
    std::span<const double>(diagram.data(), diagram.size()),
    num_pairs,
    /*resolution=*/64,
    /*sigma=*/0.1
);

// Access pixel values
double pixel = image.image[10][20];
```

### Betti curve

```cpp
auto betti = compute_betti_curve(
    std::span<const double>(diagram.data(), diagram.size()),
    num_pairs,
    /*max_dim=*/2
);

for (auto& [filtration, betti_number] : betti) {
    printf("t=%.3f: beta=%d\n", filtration, betti_number);
}
```

### SIMD optimization notes

All vectorization methods in `src/algorithms/persistence_vectorization.cpp` use:

- **Landscape**: SIMD max accumulation for k-th largest persistence function
- **Image**: SIMD Gaussian kernel evaluation (exp via vectorized math library)
- **Betti curve**: Atomic increments per bin (scalar, memory-bound)
- **Silhouette**: SIMD weighted sum of landscape levels
- **Heat kernel**: SIMD Gaussian with diffusion time parameter

## Practical tuning

For landscapes, resolution ranges from 50 to 200 samples, the key parameter is `num_levels` (3 to 10), the bottleneck is sorting per level, and memory is O(L * R). For images, resolution ranges from 32 to 128 pixels, the key parameter is `sigma` (0.05 to 0.5), the bottleneck is Gaussian convolution, and memory is O(H * W). For Betti curves, resolution ranges from 50 to 200 bins, the key parameter is `max_dim`, the bottleneck is range determination, and memory is O(R).

### Image sigma tuning

```python
# Rule of thumb: sigma = 0.1 * (max_persistence - min_persistence)
persistences = [d - b for b, d, _ in pairs]
sigma = 0.1 * (max(persistences) - min(persistences))

# For balanced resolution vs smoothness
image = persistence_image_fast(pairs, resolution=64, sigma=sigma)
```

## Advanced: custom weight function

```python
from pynerve.fast_ops import persistence_image_fast

# Custom weight function: weight by persistence^2
def quadratic_weight(birth, death, dim):
    return (death - birth) ** 2

image = persistence_image_fast(
    pairs,
    resolution=64,
    sigma=0.1,
    weight_fn=quadratic_weight,
)
```


## FAQ

**Q: Why is the persistence landscape called L1-Lipschitz stable?**
A: Small perturbations in the input diagram produce proportionally small changes in the landscape in L1 norm. This stability guarantee makes landscapes suitable for ML, where training and test data may have small differences.

**Q: How does the heat kernel differ from the persistence image?**
A: The heat kernel solves the diffusion equation on the birth-death plane, producing a smooth density estimate controlled by diffusion time t. The persistence image places Gaussian kernels at each point. Heat kernel is more robust to noise; persistence image is faster to compute.

**Q: Can I compute vectorizations for multiple dimensions simultaneously?**
A: Yes. Pass pairs with dimension labels (birth, death, dim). The `compute_persistence_image` function with `max_dim > 0` computes separate images for each dimension and concatenates along the channel axis.


### Cross-references

- `pynerve.ml`: ML module overview
- `pynerve.algorithms.vectorization`: Algorithm module vectorization
- `pynerve.torch.ml`: PyTorch ML operations
- `pynerve.autodiff.tensor_ops`: Differentiable vectorization
- `pynerve.fast_ops`: Python fast vectorization wrappers

## Persistence vectorization

Vectorization methods convert persistence diagrams (variable-length sets
of birth-death pairs) into fixed-size feature vectors suitable for ML.


### Landscapes

Persistence landscapes embed a diagram as piecewise-linear functions.
`lambda_k(t)` is the k-th largest persistence function at parameter t.

```cpp
struct PersistenceLandscape {
    std::vector<std::vector<double>> landscape_levels;
    double x_min, x_max;
    int num_levels;
};

template <typename T>
PersistenceLandscape compute_landscape(
    std::span<const T> diagram, size_t num_pairs,
    int num_levels = 5, double resolution = 0.01);
```

**Definition:**

For each point (b, d) in the diagram, define:

f_(b,d)(t) = max(0, min(t - b, d - t))
            = max(0, d - |t - (b+d)/2| - (d-b)/2)

Then lambda_k(t) is the k-th largest value of f_(b,d)(t) over all pairs.

```python
from pynerve.algorithms import compute_landscape
import numpy as np

diagram = np.array([[0.0, 1.0], [0.5, 2.0]])
landscape = compute_landscape(diagram, num_levels=3, resolution=100)
# landscape.landscape_levels: list of 3 arrays, each length 100
# landscape.x_min, landscape.x_max: range of parameter t
```


### Silhouettes

Weighted average of landscape layers. Parameter p controls the weighting:
higher p emphasizes more persistent features.

```cpp
template <typename T>
std::vector<double> compute_silhouette(
    std::span<const T> diagram, size_t num_pairs,
    int resolution = 100, double p = 1.0);
```

**Definition:**

S(t) = (sum_k w_k^p * lambda_k(t)) / (sum_k w_k^p)

where w_k is the persistence of the k-th most persistent pair.

```python
from pynerve.algorithms import compute_silhouette

sil = compute_silhouette(diagram, resolution=100, p=1.0)
# Returns (100,) array -- a single function summarizing the diagram
```

Higher p values emphasize the most persistent features and suppress noise.


### Heat vectors

Solution to the heat equation with diagram points as initial sources.

```cpp
template <typename T>
std::vector<double> compute_heat_vector(
    std::span<const T> diagram, size_t num_pairs,
    int resolution = 100, double sigma = 1.0, double t = 1.0);
```

**Definition:**

Place a Gaussian bump at each diagram point (b, d) with width sigma.
Evolve under the heat equation for time t. Sample at regular intervals.

```python
from pynerve.algorithms import compute_heat_vector

heat = compute_heat_vector(diagram, resolution=100, sigma=1.0, t=1.0)
```


### Persistence images

2D density on the birth-persistence plane, convolved with a Gaussian kernel.

```cpp
struct PersistenceImage {
    std::vector<double> image; int height, width;
    std::vector<double> x_grid, y_grid;
};

template <typename T>
PersistenceImage compute_image(
    std::span<const T> diagram, size_t num_pairs,
    int resolution = 64, double sigma = 0.1,
    std::string weight_fn = "linear");
```

**Construction:**

1. Map each point (b, d) to (b, p) where p = d - b (persistence)
2. Place a Gaussian with width sigma at each mapped point
3. Weight each Gaussian by w(p) = persistence (linear) or constant
4. Evaluate on a uniform grid of size resolution x resolution

```python
from pynerve.algorithms import compute_image

img = compute_image(diagram, resolution=64, sigma=0.1, weight_fn="linear")
# img.image: (64, 64) array
# img.x_grid, img.y_grid: axis values
```


### SIMD acceleration

SIMD-accelerated via AVX-512 exp (`simd_exp_pd_avx512`) and AVX exp
(`simd_exp_pd_avx`) in `src/algorithms/persistence_vectorization.cpp`.

Landscape construction achieves 4-6x speedup with AVX-512. Image Gaussian evaluation achieves 6-8x. Silhouette computation achieves 4-5x. Heat vector computation achieves 5-7x.


### Complexity

Persistence landscape costs O(p * L * R) time and O(L * R) space. Persistence image costs O(p * R^2) time and O(R^2) space. Silhouette costs O(p * R) time and O(R) space. Heat vector costs O(p * R) time and O(R) space. Here p is the number of persistence pairs, L is the number of landscape layers, and R is the resolution (number of samples per dimension).

- p = number of persistence pairs
- L = number of landscape layers
- R = resolution (number of samples per dimension)


### Practical guidance

**Choosing a method:** Landscape is best for classification and SVM, with L1-Lipschitz stability. Image is best for CNN-based methods, stable under bounded noise. Silhouette is best for quick summaries and regression, stable for high persistence weighting. Heat vector is best for density estimation, smooth and tunable.

**Parameter selection:**
- Resolution: 100 is a good default for landscapes/silhouettes; 64 for images
- Sigma: ~0.1-0.5 for images (depends on data scale)
- Landscape layers: 3-5 captures most structure
- Image weight: "linear" emphasizes persistent features


### Common pitfalls

1. **Scale sensitivity**: All methods depend on the scale of birth/death
   values. Normalize diagrams to [0,1] or standardize.

2. **Empty diagrams**: Diagrams with no pairs produce zero landscapes/images.
   Handle gracefully.

3. **Infinite death**: Essential classes (death = inf) must be removed
   or truncated before vectorization.

4. **Resolution vs feature size**: Too high resolution produces large
   feature vectors (R^2 for images). Use PCA or pooling after.


### Cross-references

- `pynerve.ml`: ML pipeline using vectorization methods
- `pynerve.autodiff.tensor_ops`: Differentiable versions for gradient-based learning
- `pynerve.torch.ml`: PyTorch vectorization operations
- `pynerve.nn`: Neural network layers operating on vectorized diagrams

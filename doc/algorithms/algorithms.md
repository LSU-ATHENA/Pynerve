# Algorithms

## Quick start

```python
import numpy as np
from pynerve.algorithms import pairwise_distances, knn, mapper_graph

# Pairwise distance matrix (Euclidean)
points = np.random.randn(100, 3).astype(np.float32)
D = pairwise_distances(points)           # (100, 100)

# k-nearest neighbors
distances, indices = knn(points, k=5)    # each (100, 5)

# Mapper graph from point cloud
graph = mapper_graph(points, resolution=10, overlap=0.3)
```

Distance computation, Mapper, kernel methods, and persistence vectorization. All
operations have SIMD-accelerated backends (AVX-512/AVX2/SSE4.1) with runtime
dispatch. GPU variants are available for distance and Mapper.


## Module structure

The algorithms module has four sub-pages: [Distance](distance.md) covers pairwise, KNN, HNSW, sparse distance, and GPU kernels. [Mapper](mapper.md) covers the Mapper graph, filters, clustering, cover, and GPU. [Kernel methods](kernel_methods.md) covers Gaussian kernel matrix computation. [Vectorization](vectorization.md) covers landscapes, silhouettes, heat vectors, and persistence images.


## API

### `pynerve.algorithms` (Python)

```python
# Distance matrices
def pairwise_distances(points, metric="euclidean", use_simd=True) -> np.ndarray

# K-nearest neighbors
def knn(points, k=5, metric="euclidean", algorithm="brute_force") -> tuple[np.ndarray, np.ndarray]

# Mapper graph
def mapper_graph(points, resolution=10, overlap=0.25, filter="pca",
                 clusterer="dbscan", return_meta=False) -> MapperGraph | tuple

# Persistence vectorization
def persistence_landscape(diagram, num_levels=5, resolution=100) -> np.ndarray
def persistence_image(diagram, resolution=64, sigma=0.1) -> np.ndarray
def persistence_silhouette(diagram, resolution=100, weight_power=1.0) -> np.ndarray
def persistence_heat_vector(diagram, resolution=100, sigma=1.0, t=1.0) -> np.ndarray

# Kernel methods
def gaussian_kernel_matrix(d1, d2, sigma=1.0) -> np.ndarray
```

### C API

```c
#include <nerve/algorithms/distance_c.h>

nerve_status_t nerve_pairwise_distances_f32(
    const float* points, size_t n, size_t dim, float* output);
nerve_status_t nerve_pairwise_distances_f64(
    const double* points, size_t n, size_t dim, double* output);
nerve_status_t nerve_knn_f32(
    const float* points, size_t n, size_t dim, size_t k,
    float* out_dist, size_t* out_idx);
nerve_status_t nerve_knn_f64(
    const double* points, size_t n, size_t dim, size_t k,
    double* out_dist, size_t* out_idx);
nerve_status_t nerve_mapper_graph_f32(
    const float* points, size_t n, size_t dim,
    int resolution, float overlap, int* out_edges, size_t* out_n_edges);
```


### Input/output conventions

- All arrays are row-major (C order)
- Distances are returned as symmetric NxN matrices (i > j entries unused)
- KNN returns (distances, indices) each of shape (N, k)
- Mapper returns a graph as adjacency list or node/edge arrays
- Vectorization methods expect diagrams as (N, 3) arrays [birth, death, dim]


### Error handling

Functions return `nerve_status_t` in C, which can be:

```c
typedef enum {
    NERVE_OK = 0,
    NERVE_ERR_INVALID_ARGUMENT = 1,
    NERVE_ERR_MEMORY = 2,
    NERVE_ERR_BACKEND_UNAVAILABLE = 3,
    NERVE_ERR_DIMENSION_MISMATCH = 4,
} nerve_status_t;
```

Python wrappers raise `ValueError` or `RuntimeError` on failure.


### Complexity

Pairwise distances (naive) is O(n^2 * d) where n is the number of points and d is dimensionality. SIMD-accelerated pairwise distances is O(n^2 * d / SIMD_width), achieving 8x throughput with AVX-512 FP64. GPU pairwise distances is O(n^2 * d / cores), roughly 50x on an H100 for 10,000 points. Brute-force KNN is O(n^2 * d + n k log k) including the final sort. HNSW-based KNN is O(n log n * d) as an approximate method. Mapper costs O(n * d + n * resolution * cluster), dominated by clustering. Persistence landscape is O(p * L * R) where p is the number of pairs and L is layers. Persistence image is O(p * R^2) where R is resolution.


### Practical guidance

**When to use GPU:**
- More than 10,000 points for distance computation
- More than 50,000 points for Mapper
- Batch processing of many point clouds

**Memory management:**
- An n-point distance matrix in FP64 takes n^2 * 8 bytes
- For n=10000, that's several hundred megabytes
- Use `computeCompressedMatrix` when full matrix is not needed

**Common pitfalls:**
- Distance matrices are dense -- OOM risk for large n
- KNN with k > n silently fails
- Mapper `overlap` parameter should be in (0, 0.5); values >0.5 over-connect
- Vectorization resolution too high produces large feature vectors



## FAQ

**When should I use GPU instead of CPU?** For distance computation, GPU is beneficial beyond 10,000 points. For Mapper, use GPU beyond 50,000 points. For batch processing of many point clouds, GPU is preferred regardless of size.

**Which distance metric should I choose?** Euclidean distance has full SIMD acceleration and is the fastest. Manhattan also has SIMD support. Cosine, Chebyshev, Minkowski, and other metrics have partial or no SIMD acceleration and fall back to scalar code.

**Why is my distance matrix so large?** An n-point FP64 distance matrix requires n^2 * 8 bytes. For large n, use `computeCompressedMatrix` (stores only the upper triangle) or the GPU path with chunked processing.


### Cross-references

`pynerve.algebra.distance` provides low-level SIMD distance kernels. `pynerve.ml` provides a higher-level ML pipeline using vectorization. `pynerve.metrics` covers distances between persistence diagrams. `pynerve.cuda.kernels` provides GPU distance kernels. `pynerve.graphs` handles graph construction from distance matrices.

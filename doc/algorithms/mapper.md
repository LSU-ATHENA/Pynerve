## Mapper

The Mapper algorithm builds a topological graph summarizing a point cloud:

1. **Filter** -- project points to low dimension (PCA, eccentricity, density)
2. **Cover** -- divide filter range into overlapping intervals
3. **Clustering** -- cluster points within each interval
4. **Graph** -- nodes are clusters; edges connect clusters sharing points

```cpp
template <typename T = float>
class MapperAlgorithm {
    struct Config {
        std::shared_ptr<FilterFunction<T>> filter;
        int cover_resolution = 10;
        T cover_overlap = T(0.25);
        std::shared_ptr<ClusteringAlgorithm<T>> clusterer;
    };
    struct Result { MapperGraph<T> graph; std::vector<T> filter_values;
                    std::vector<std::vector<size_t>> cover_sets;
                    std::unordered_map<std::string, T> metadata; };
    explicit MapperAlgorithm(Config config);
    Result compute(std::span<const T> points, size_t n, size_t dim) const;
};
```


### Algorithm steps in detail

```
Input: point cloud X in R^d, filter f, cover C, clustering algorithm A

1. For each point x in X, compute f(x) -> scalar (or vector)
2. Build cover C: partition the range of f into overlapping intervals I_1..I_k
3. For each interval I_j:
   a. Find points X_j = {x in X : f(x) in I_j}
   b. Run A on X_j -> clusters C_{j,1}, C_{j,2}, ...
4. Each cluster C_{j,l} is a node in the output graph
5. For any two clusters C_{j,l} and C_{j',l'} with shared points,
   add an edge between their nodes
```


### Filter functions

```cpp
class FilterFunction {
    virtual std::vector<T> apply(std::span<const T> points, size_t n,
                                  size_t dim) const = 0;
};

class PCAFilter             : FilterFunction { explicit PCAFilter(int n_components = 2); };
class EccentricityFilter    : FilterFunction {};
class DensityFilter         : FilterFunction { explicit DensityFilter(int k = 10); };
class CustomFilter          : FilterFunction { /* std::function wrapper */ };
```

`PCAFilter` projects onto top principal components with output dimension equal to n_components. `EccentricityFilter` computes distance to the data centroid, outputting a single scalar per point. `DensityFilter` computes a local density estimate via KNN radius, also single-dimensional. `CustomFilter` wraps a user-defined function with configurable output dimension.

```python
from pynerve.algorithms import mapper_graph

# Using different filters
graph = mapper_graph(points, filter="pca", resolution=10)
graph = mapper_graph(points, filter="eccentricity", resolution=10)
graph = mapper_graph(points, filter="density", k=15, resolution=10)

# Custom filter (as numpy array)
filter_values = np.linalg.norm(points, axis=1)  # distance from origin
graph = mapper_graph(points, filter=filter_values, resolution=10)
```


### Clustering algorithms

```cpp
class ClusteringAlgorithm {
    virtual std::vector<int> cluster(std::span<const T>, size_t n,
                                      size_t dim) const = 0;
};

class DBSCANClustering       : ClusteringAlgorithm { struct Config { T eps; int min_samples; }; };
class SingleLinkageClustering : ClusteringAlgorithm { struct Config { T linkage_distance; }; };
class ConnectedComponentsClustering : ClusteringAlgorithm {};
```

DBSCAN (parameters: eps, min_samples) handles arbitrary cluster shapes with noise support. SingleLinkage (parameters: linkage_distance) does hierarchical merging. ConnectedComponents requires no parameters and is the fastest but only provides connectivity.

**Choosing clusterer:**
- DBSCAN is preferred for general use -- it handles noise and finds
  arbitrary-shaped clusters
- SingleLinkage is faster but produces chained clusters
- ConnectedComponents is the fastest but least informative


### Cover

```cpp
class Cover {
    struct Config { int resolution = 10; T overlap = T(0.25); };
    std::vector<std::vector<size_t>> build(
        std::span<const T> filter_values, size_t n, int n_filter_dims) const;
};
```

The cover divides the filter range into intervals with specified overlap:

```text
range: [min_f, max_f]
interval_length = (max_f - min_f) / resolution
overlap_amount = interval_length * overlap

I_j = [min_f + j*(1-overlap)*interval_length,
       min_f + (j + overlap)*(1-overlap)*interval_length + interval_length]
```

**Resolution vs overlap trade-offs:** Low resolution (5-10) with low overlap (0.1-0.2) produces a coarse graph with few nodes. Low resolution with high overlap (0.4-0.5) stays coarse but becomes well-connected. High resolution (20-50) with low overlap captures fine detail with many nodes. High resolution with high overlap yields a highly connected graph that may be noisy.


### MapperGraph result

```python
graph = mapper_graph(points, resolution=10, overlap=0.3, return_meta=True)

# Graph structure
graph.nodes          # list of node metadata
graph.edges          # list of (u, v) tuples
graph.clusters       # list of point indices per node

# Metadata (if return_meta=True)
graph.filter_values  # filter values per point
graph.cover_sets     # points per cover interval
graph.metadata       # timing, parameters
```


### GPU Mapper

`src/cuda/kernels/mapper_gpu.cu` provides GPU-accelerated Mapper:
- Filter computation on GPU
- Cover construction in parallel
- GPU clustering via connected components on the fly

```python
graph = mapper_graph(points, resolution=10, backend="cuda")
```

GPU Mapper is beneficial for point clouds with >50k points and high
resolution (>50 intervals).


### When to use Mapper

For exploring high-dimensional data (10-100 dimensions), Mapper is ideal, especially with a PCA filter. For finding topological structure, Mapper captures loops, flares, and branches. For large datasets exceeding 100,000 points, use the GPU backend with low resolution. For small datasets under 1,000 points, skip Mapper and use clustering directly.

**Interpretation tips:**
- Loops in the Mapper graph suggest cylindrical or toroidal structure
- Flares (leaf nodes) suggest outliers or rare subpopulations
- Highly connected regions suggest dense, homogeneous clusters
- Disconnected components suggest separated clusters in data space


### Complexity

Filter computation costs O(n * d) or O(n * d * k) for density estimation. Cover construction costs O(n * resolution). Per-interval clustering using DBSCAN costs O(n_j * d * log(n_j)). Total clustering cost is O(n * resolution * d * log(n/resolution)). Graph construction costs O(nodes + edges).


### Common pitfalls

1. **Overlap too low**: Graph becomes disconnected -- nodes don't share
   enough points. Keep overlap >= 0.2.

2. **Resolution too high**: Many nodes with few points each, DBSCAN fails.
   Ensure each interval has at least min_samples * 2 points.

3. **Filter choice**: PCA is a good default for unknown data. Eccentricity
   is better for data where the center is meaningful.

4. **Noise handling**: DBSCAN marks isolated points as noise (-1). These
   are not included in any node.

5. **Memory**: Cover sets store point indices for each interval --
   O(n * resolution) in the worst case.


### Cross-references

- `pynerve.algorithms.algorithms`: Python API
- `pynerve.graphs`: Graph data structures for Mapper output
- `pynerve.cuda.kernels`: GPU Mapper kernels

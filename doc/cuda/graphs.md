## CUDA graphs

`CudaGraphManager` captures repeated kernel launch patterns into CUDA graphs
and replays them, eliminating CPU launch overhead.

```cpp
class CudaGraphManager {
public:
    CudaGraphManager();
    ~CudaGraphManager();

    cudaError_t captureDistanceMatrix(
        const double* points, double* distances,
        Size n_points, Size point_dim,
        double max_radius, cudaStream_t stream);

    cudaError_t captureMatrixReduction(
        uint64_t* columns, int n_cols, int n_words_per_col,
        int* pivotColumn, cudaStream_t stream);

    cudaError_t launch(cudaStream_t stream);

    bool isCaptured() const;
    void reset();
};
```

Use when the same computation repeats with identical grid/block shapes
(e.g., same-size distance matrices across filtration steps).


### Performance

Graph capture adds ~50-200 us overhead on first call, then each replay is
~1-2 us. This is beneficial when:

- Same computation runs >10 times
- Kernel launch latency is a bottleneck
- Many small kernels are launched sequentially

```cpp
// Capture once, replay many times
CudaGraphManager graphManager;

// First call captures the graph automatically
graphManager.captureDistanceMatrix(
    points, dist, n, dim, radius, stream);

// Subsequent calls replay the captured graph
for (int i = 0; i < 100; ++i) {
    graphManager.launch(stream);
}
```


### When to use CUDA graphs

For the same distance matrix shape across many iterations, CUDA graphs provide 10-50x speedup and are recommended. For variable-size inputs per iteration, capture overhead dominates and graphs are not recommended. For a single large kernel (under a millisecond), launch overhead is negligible so graphs are unnecessary. For many small kernels (under 10 microseconds each), graphs are recommended because they fuse launches. For multi-stream pipelines, graphs are recommended since they capture stream dependencies.


### Graph capture process

1. Begin capture: `cudaStreamBeginCapture(stream)`
2. Launch kernels normally
3. End capture: `cudaStreamEndCapture(stream, &graph)`
4. Instantiate: `cudaGraphInstantiate(&exec, graph, ...)`
5. Replay: `cudaGraphLaunch(exec, stream)`

The `CudaGraphManager` handles all five steps automatically.


### Python API

```python
from pynerve.cuda import CudaGraphManager

mgr = CudaGraphManager()

# First call captures
mgr.capture_distance_matrix(points, distances, n, dim, radius)

# Subsequent calls replay
for _ in range(100):
    mgr.launch()
```


### Cross-references

- `pynerve.cuda.kernels`: Kernels suitable for graph capture
- `pynerve.cuda.streams`: Stream synchronization with graphs
- `pynerve.validation.benchmarks`: Benchmarking with graph capture

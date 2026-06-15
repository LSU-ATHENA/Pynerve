## GPU graph algorithms

CUDA-accelerated graph primitives using CSR adjacency.

```python
from pynerve.graphs.gpu import GPUGraph

g_gpu = GPUGraph(num_vertices=10000, num_edges=50000)
g_gpu.set_csr(row_ptr, col_idx, weights)

distances = g_gpu.bfs(source=0)

components = g_gpu.connected_components()

sp = g_gpu.all_pairs_shortest_paths()
```

### Multi-GPU

Distributed graph algorithms across GPUs and MPI ranks. Each GPU processes
a subset of vertices.

```python
from pynerve.graphs.gpu import GPUGraph

g_gpu = GPUGraph(num_vertices=100000, num_edges=500000)
distances = g_gpu.bfs(source=0)

components = g_gpu.connected_components()
```

### Complexity

GPU-accelerated BFS costs O(n + m) using CSR adjacency. GPU-accelerated PageRank costs O(m * iter) with one sparse matrix-vector product per iteration. All-pairs shortest paths using Floyd-Warshall on GPU costs O(n^3).


## GPU BFS implementation

```cpp
// GPU BFS with CSR adjacency
__global__ void bfs_kernel(
    const int* row_ptr, const int* col_idx,
    int* distances, const int* frontier,
    int* next_count, int frontier_size, int level) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= frontier_size) return;

    int u = frontier[idx];
    int start = row_ptr[u];
    int end = row_ptr[u + 1];

    for (int i = start; i < end; i++) {
        int v = col_idx[i];
        if (atomicCAS(&distances[v], -1, level + 1) == -1) {
            int pos = atomicAdd(next_count, 1);
            // Write to next frontier (requires pre-allocated)
        }
    }
}
```

### Multi-GPU distributed BFS

```python
from pynerve.graphs.gpu import MultiGPUBFS

bfs = MultiGPUBFS(num_gpus=4)
# Each GPU holds a vertex partition
# Frontier exchange uses NCCL point-to-point
distances = bfs.run(source=0, graph=g_gpu)
```

### GPU connected components

```cpp
__global__ void labeling_kernel(
    int* components, const int* row_ptr,
    const int* col_idx, int n) {

    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= n) return;

    // Hook-and-compress connected components
    for (int i = row_ptr[u]; i < row_ptr[u + 1]; i++) {
        int v = col_idx[i];
        int comp_u = components[u];
        int comp_v = components[v];
        // ... union-find with atomic hooks ...
    }
}
```

### Performance characteristics

On a power-law graph with 1 million vertices and 10 million edges, GPU BFS takes 15 ms and connected components takes 25 ms. On a road network with 1 million vertices and 3 million edges, GPU BFS takes 30 ms and connected components takes 40 ms. On a social graph with 10 million vertices and 100 million edges, GPU BFS takes 80 ms and connected components takes 120 ms. On a 3D mesh with 500 thousand vertices and 3 million edges, GPU BFS takes 10 ms and connected components takes 18 ms.

### Memory management

```python
from pynerve.graphs.gpu import GPUGraph

# Pre-allocate CSR with maximum expected edges
g_gpu = GPUGraph(
    num_vertices=100000,
    max_edges=500000,  # pre-allocates CSR arrays
)

# Dynamic graph updates (insert/delete edges)
g_gpu.insert_edge(10, 20, weight=0.5)
g_gpu.remove_edge(10, 20)
g_gpu.update_weights(weight_array)
```


## FAQ

**Q: When should I use GPU graph algorithms vs CPU?**
A: GPU is beneficial for graphs with >10k vertices and high-degree nodes (>100 edges/vertex). For small graphs or graphs with very low average degree, CPU overhead dominates.

**Q: Can I use 64-bit vertex IDs?**
A: The GPU kernels use 32-bit ints by default. Enable 64-bit mode with `use_int64=True` at the cost of 2x memory and ~20% slower atomics.

**Q: How does multi-GPU handle cross-partition edges?**
A: Cross-partition edges are detected during construction and stored in a ghost node array. During BFS/PageRank, ghost node values are synchronized via NCCL point-to-point after each iteration.


### Cross-references

- `pynerve.graphs`: Graphs module overview
- `pynerve.cuda`: CUDA infrastructure
- `pynerve.graphs.mpi`: Distributed graph algorithms
- `pynerve.graphs.algorithms`: CPU graph algorithm implementations

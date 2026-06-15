## MPI

Distributed graph edges across MPI processes with all-gather vertex results.

Underlying MPI pattern:
- Edges are partitioned across MPI ranks
- Vertex results synchronized via MPI_Allgatherv
- CUDA-aware MPI via NCCL/NVSHMEM when GPUs are present

Requires MPI build: `cmake -DENABLE_MPI=ON ..`

```python
from pynerve.graphs import GraphEngine

engine = GraphEngine()
engine.set_graph(edges)
engine.set_weighted_graph(edges, weights)
result = engine.bfs(source=0)
result = engine.connected_components()
result = engine.all_pairs_shortest_paths()
pv = engine.compute_pagerank(damping=0.85, max_iter=100)
diagram = engine.compute_graph_persistence()
```


## Distributed graph pattern

```cpp
// MPI graph partition
int rank, size;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);

// Each rank owns a subset of vertices
int local_n = global_n / size + (rank < global_n % size);
int start_vertex = rank * (global_n / size) + min(rank, global_n % size);

// Ghost vertices: vertices owned by other ranks but adjacent to locals
std::vector<int> ghost_vertices;
// ... collect from local edge list ...

// BFS iteration:
// 1. Expand frontier on local vertices
// 2. MPI_Allgatherv to sync frontier updates
// 3. Check termination globally
```

### CUDA-aware MPI

When GPUs are present, the MPI layer uses CUDA-aware MPI to avoid host-device copies:

```python
from pynerve.graphs import GraphEngine
from pynerve.graphs.mpi import MPIConfig

# Enable CUDA-aware MPI
config = MPIConfig(
    cuda_aware=True,
    backend="nccl",   # NCCL for GPU-GPU communication
)

engine = GraphEngine(mpi_config=config)
result = engine.bfs(source=0)
# GPU buffers are passed directly to MPI calls
# No cudaMemcpy between GPU and host
```

### Distributed PageRank

```cpp
// Distributed PageRank with MPI
void distributed_pagerank(
    const LocalGraph& local_graph,
    const std::vector<int>& ghost_vertices,
    const std::vector<int>& ghost_owners,
    double damping, int max_iter,
    double* local_ranks, MPI_Comm comm) {

    int n_local = local_graph.n;
    std::vector<double> ghost_ranks(ghost_vertices.size());

    for (int iter = 0; iter < max_iter; iter++) {
        // Compute local contribution
        for (int i = 0; i < n_local; i++) {
            double sum = 0.0;
            for (int j : local_graph.in_neighbors(i))
                sum += (j < n_local)
                    ? local_ranks[j] / local_graph.out_degree(j)
                    : ghost_ranks[ghost_index(j)];
            new_ranks[i] = sum;
        }

        // Exchange ghost ranks via MPI_Neighbor_alltoallv
        MPI_Neighbor_alltoallv(
            new_ranks.data(), ..., MPI_DOUBLE,
            ghost_ranks.data(), ..., MPI_DOUBLE,
            comm);
    }
}
```

### Scaling characteristics

With 1 million vertices and 10 million edges across 4 MPI ranks, speedup reaches 3.5x versus single-node with memory per rank of a few gigabytes. With 10 million vertices and 100 million edges across 16 ranks, speedup reaches 14x with memory per rank of a few gigabytes. With 100 million vertices and 1 billion edges across 64 ranks, speedup reaches 55x with memory per rank of a handful of gigabytes. With 1 billion vertices and 10 billion edges across 256 ranks, speedup reaches 200x with memory per rank of several gigabytes.

Scaling is near-linear for PageRank and BFS up to 256 ranks. Beyond that, communication overhead dominates.


## MPI vs multi-GPU

MPI on CPU is best for very large graphs with more than 1 billion edges, using MPI_Allgatherv over CPU memory, distributed RAM aggregate memory, near-linear scaling to over 1000 ranks, and setup via `cmake -DENABLE_MPI=ON`. Multi-GPU with CUDA-aware MPI is best for GPU-accelerated analysis, using NCCL over GPU memory, distributed HBM memory, near-linear scaling to 64 GPUs, and requires an additional CUDA-aware MPI library.

## FAQ

**Q: When should I use MPI versus multi-GPU for distributed graphs?**
A: Use MPI when your graph exceeds GPU memory capacity (more than 1 billion edges) or when GPUs are not available. Use multi-GPU with CUDA-aware MPI when your graph fits in aggregate GPU memory and you need the throughput of GPU-accelerated kernels.

**Q: What is the communication overhead of MPI_Allgatherv in BFS?**
A: The all-gather distributes frontier vertices (up to O(n)) across ranks each iteration. For graphs with small frontiers relative to the total vertex count, communication is negligible. For high-degree graphs with large frontiers, it can become the dominant cost beyond 256 ranks.

**Q: Do I need CUDA-aware MPI to use GPUs with MPI?**
A: No, but without it the runtime must copy data between GPU and host before each MPI call, adding significant overhead. CUDA-aware MPI via NCCL or NVSHMEM passes GPU pointers directly, avoiding these copies.


### Cross-references

- `pynerve.graphs`: Graphs module overview
- `pynerve.graphs.gpu`: GPU graph algorithms
- `pynerve.metrics.gpu`: MPI distributed metrics
- `pynerve.cuda.mpi`: CUDA-aware MPI configuration

## Graph algorithms detail (CPU)

Parallel implementations using OpenMP and SIMD:

```python
from pynerve.graphs import ParallelGraphFiltration

filtration = ParallelGraphFiltration.compute_filtration(
    adjacency_list,
    num_threads=8,
)
```

### BFS

Breadth-first search computes shortest-path distances from a source vertex.

```python
from pynerve.graphs import GraphEngine

engine = GraphEngine()
engine.set_graph(edges)
distances = engine.bfs(source=0)
```

### PageRank

PageRank computes vertex importance via iterative sparse matrix-vector
multiplication.

```python
from pynerve.graphs import GraphEngine

engine = GraphEngine()
engine.set_weighted_graph(edges, weights)
pv = engine.compute_pagerank(damping=0.85, max_iter=100)
```

### Graph filtration

Edges are added in order of increasing weight. H0 persistence tracks
connected components via union-find in near-linear time.

```python
from pynerve.graphs import GraphTopology

filtration = GraphTopology.compute_graph_filtration(graph)
betti = GraphTopology.compute_graph_betti(graph)
diagram = GraphTopology.compute_graph_persistence(graph)
```

### Complexity

Graph filtration costs O(m log m) for sorting edges by weight. H0 persistence costs O(m * alpha(m)) using union-find, which is near-linear. BFS on CPU costs O(n + m) using an adjacency list. PageRank on CPU costs O(m * iter) with one sparse matrix-vector product per iteration.


## Parallel BFS implementation

```cpp
// OpenMP parallel BFS with level synchronization
void parallel_bfs(int source, const CSRGraph& graph,
                  int* distances) {
    std::fill(distances, distances + graph.n, INF);
    distances[source] = 0;

    std::vector<int> frontier = {source};
    int level = 0;

    while (!frontier.empty()) {
        std::vector<int> next;
        #pragma omp parallel
        {
            std::vector<int> local_next;
            #pragma omp for nowait
            for (size_t i = 0; i < frontier.size(); i++) {
                int u = frontier[i];
                for (int v : graph.neighbors(u)) {
                    if (distances[v] == INF) {
                        distances[v] = level + 1;
                        local_next.push_back(v);
                    }
                }
            }
            #pragma omp critical
            next.insert(next.end(),
                local_next.begin(), local_next.end());
        }
        frontier = std::move(next);
        level++;
    }
}
```

## Parallel PageRank

```cpp
void parallel_pagerank(const CSRGraph& graph,
                       double damping, int max_iter,
                       double* ranks) {
    int n = graph.n;
    std::fill(ranks, ranks + n, 1.0 / n);
    std::vector<double> new_ranks(n);
    double dangling_sum = (1.0 - damping) / n;

    for (int iter = 0; iter < max_iter; iter++) {
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            double sum = 0.0;
            for (int j : graph.in_neighbors(i)) {
                sum += ranks[j] / graph.out_degree(j);
            }
            new_ranks[i] = damping * sum + dangling_sum;
        }
        ranks.swap(new_ranks);
    }
}
```

## Graph filtration details

The graph filtration sorts edges by weight and processes them incrementally:

```python
# Manual graph filtration
edges = [(0, 1, 0.3), (1, 2, 0.5), (0, 2, 0.8), (2, 3, 1.0)]
sorted_edges = sorted(edges, key=lambda e: e[2])

# Union-find for H0 tracking
parent = list(range(num_vertices))
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

for u, v, w in sorted_edges:
    ru, rv = find(u), find(v)
    if ru != rv:
        parent[ru] = rv  # merge components
        # birth = w (death of component that merges)
```


## Complexity analysis

BFS has a sequential cost of O(n + m) and a parallel cost of O((n + m) / p + n) with O(n) memory. PageRank has a sequential cost of O(m * iter) and a parallel cost of O(m * iter / p) with O(n) memory. Graph filtration has a sequential cost of O(m log m) and a parallel cost of O(m log m / p) with O(m) memory. H0 persistence has a sequential cost of O(m * alpha(m)) and a parallel cost of O(m * alpha(m) / p) with O(n) memory.

The parallel overhead is dominated by frontier expansion (BFS) and sparse mat-vec (PageRank). Both scale well up to 64 threads.


## Algorithm selection guide

For shortest paths on unweighted graphs, use BFS. For vertex importance on directed or undirected graphs of any size, use PageRank. For component hierarchies on weighted graphs with H0 persistence, use graph filtration. For community detection on large graphs with k clusters, use spectral clustering. For topology analysis on any graph at multiple scales, use persistent homology.

## FAQ

**Q: Why does graph filtration use union-find instead of a more complex data structure?**
A: Union-find with path compression achieves near-linear O(m * alpha(m)) time, which is optimal for tracking connected components during filtration. More complex structures like dynamic connectivity trees are unnecessary since edges are only added, never removed.

**Q: How many threads should I use for parallel BFS or PageRank?**
A: Both scale well up to 64 threads on most systems. Beyond that, memory bandwidth becomes the bottleneck. Start with the number of physical cores and increase if your graph has high-degree vertices that expose more parallelism.

**Q: When does PageRank fail to converge?**
A: PageRank always converges for a damping factor below 1. Convergence is slower when the damping factor is close to 1 or when the graph has many dangling nodes with zero out-degree. Use `max_iter=200` and check the residual for safety.


### Cross-references

- `pynerve.graphs`: Graphs module overview
- `pynerve.graphs.gpu`: GPU graph algorithms
- `pynerve.graphs.mpi`: Distributed graph algorithms
- `pynerve.algorithms`: Algorithm module
- `pynerve.spectral.laplacian`: Spectral clustering on graphs

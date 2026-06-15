# Graph Algorithms

Graph construction from point clouds, distance matrices, and adjacency lists;
persistent homology on graphs (graph filtration, H0 tracking); graph neural
network layers (topology-aware convolution); and GPU-accelerated BFS,
PageRank, and connected components.

```python
import pynerve
import numpy as np

points = np.random.randn(500, 3)
graph = pynerve.graphs.Graph.from_knn(points, k=10)
print(f"Vertices: {graph.num_vertices()}, Edges: {graph.num_edges()}")
```


## Topics

- **[algorithms.md](algorithms.md)** -- BFS, PageRank, graph filtration, parallel CPU algorithms
- **[gnn.md](gnn.md)** -- Graph neural network convolution, attention, GPU benchmarks
- **[gpu.md](gpu.md)** -- CUDA graph algorithms, multi-GPU BFS, connected components
- **[mpi.md](mpi.md)** -- Distributed graph engine, MPI all-gather, CUDA-aware MPI


## API

### `pynerve.graphs.Graph`

Undirected weighted graph with adjacency matrix, Laplacian, and connectivity
utilities.

```python
from pynerve.graphs import Graph

g = Graph(num_vertices=100)

g.add_vertex()
g.add_vertices(50)
g.add_edge(u=0, v=1, weight=0.5)
g.remove_edge(0, 1)
g.set_edge_weight(0, 1, 1.0)

n = g.num_vertices()
m = g.num_edges()
verts = g.get_vertices()
edges = g.get_edges()
neighbors = g.get_neighbors(vertex=5)
adj = g.get_adjacent_vertices(5)
w = g.get_edge_weight(0, 1)
deg = g.get_degree_sequence()

adj_matrix = g.get_adjacency_matrix()
lap_matrix = g.get_laplacian_matrix()

connected = g.is_connected()
components = g.get_connected_components()
comp_id = g.get_component_id(vertex=7)
```

**Factory constructors:**

```python
g = Graph.from_knn(points, k=10, metric="euclidean")
g = Graph.from_radius(points, radius=0.5)
g = Graph.from_distance_matrix(D, threshold=0.5)
g = Graph.from_adjacency_list([(0, 1, 0.5), (1, 2, 0.3)])
g = Graph.from_simplicial_complex(complex)
```

### `pynerve.graphs.WeightedGraph`

Extends `Graph` with weighted Laplacian, spectral clustering, and eigenvalue
computation.

```python
from pynerve.graphs import WeightedGraph

wg = WeightedGraph(num_vertices=100)
wg.add_edge(0, 1, weight=0.5)
wg.set_all_weights(1.0)
wg.normalize_weights()
wg.apply_distance_weights(points)

L = wg.get_weighted_laplacian()

evals = wg.compute_eigenvalues(k=10)
evecs = wg.compute_eigenvectors(k=10)

labels = wg.spectral_cluster(k=5)
labels_n = wg.normalized_spectral_cluster(k=5)
```

### `pynerve.graphs.SimplicialGraph`

A graph that tracks a simplicial complex built on its vertices.

```python
from pynerve.graphs import SimplicialGraph
from pynerve.algebra import Simplex

sg = SimplicialGraph()
sg.add_simplex(Simplex([0, 1]))
sg.add_simplex(Simplex([0, 1, 2]))
sg.add_simplices([Simplex([1,2]), Simplex([2,3])])

skeleton = sg.get_1_skeleton()
weighted = sg.get_weighted_1_skeleton()
clique = sg.get_clique_graph()

n = sg.num_simplices()
d = sg.max_dimension()
edges = sg.get_simplices_of_dimension(dim=1)
neighbors = sg.get_simplex_neighbors(Simplex([0, 1]))
star = sg.get_simplex_star(Simplex([0]))
link = sg.get_simplex_link(Simplex([0]))
```

### `pynerve.graphs.persistent_homology`

Persistent homology on graphs -- graph filtration where edges are added in
order of weight, tracking connected components (H0). CPU and GPU backends.

```python
from pynerve.graphs import PersistentGraph

pg = PersistentGraph(num_vertices=100)

pg.add_vertex_persistent(0)
pg.add_vertex_persistent(1)
pg.add_edge_persistent(0, 1, weight=0.3)
pg.advance_time(0.3)

pg.add_edge_persistent(1, 2, weight=0.5)
pg.advance_time(0.5)

events = pg.get_persistence_events()
diagram = pg.get_persistence_diagram()

dist = pg.compute_persistence_distance(other_graph)
pg.reset_persistence()
```

**Static graph persistence:**

```python
from pynerve.graphs import GraphTopology

diagram = GraphTopology.compute_graph_persistence(graph)
filtration = GraphTopology.compute_graph_filtration(graph)
betti = GraphTopology.compute_graph_betti(graph)

d_bn = GraphTopology.compute_bottleneck_distance(g1, g2)
d_ws = GraphTopology.compute_wasserstein_distance(g1, g2)
d_gh = GraphTopology.compute_gromov_hausdorff_distance(g1, g2)

inv = GraphTopology.compute_graph_invariants(graph)
spec = GraphTopology.compute_spectral_invariants(graph)
```


## Use cases

Common use cases include point cloud analysis with k-NN or radius graphs to compute H0 persistence as a cluster hierarchy; social networks with weighted graphs for spectral clustering as community detection; sensor networks with geometric graphs for graph persistence to detect coverage holes; 3D meshes with simplicial graphs for persistent homology as shape signatures; time-varying data with dynamic graphs for zigzag persistence to capture transient loops; and graph ML with GNN layers for node classification and link prediction.


## Complexity

k-NN graph construction costs O(n log n * d) using a k-d tree or HNSW. Radius graph construction costs O(n^2 * d) by brute force, or O(n^2 / cores) on GPU. Graph filtration costs O(m log m) to sort edges by weight. H0 persistence costs O(m * alpha(m)) using union-find, which is near-linear. GPU-accelerated BFS costs O(n + m) with CSR adjacency. GPU-accelerated PageRank costs O(m * iter) with one sparse matrix-vector product per iteration. GNN convolution on CPU costs O(m * d_in * d_out) where d is the feature dimension, and on GPU costs O(m / cores * d) with Tensor Core acceleration.

## FAQ

**Q: What is the difference between a k-NN graph and a radius graph?**
A: A k-NN graph connects each vertex to its k nearest neighbors, ensuring uniform vertex degree. A radius graph connects vertices within a fixed distance threshold, which adapts to local density but may produce disconnected components in sparse regions.

**Q: When should I use graph persistence versus standard clustering?**
A: Graph persistence (H0) reveals the full hierarchy of clusters at all scales, whereas standard clustering like k-means gives a single partition. Use persistence when you do not know the number of clusters ahead of time or want to study multi-scale structure.

**Q: Can I compute persistence on directed graphs?**
A: The current implementation assumes undirected graphs. For directed graphs, you can symmetrize the edge weights first, or use the specialized directed flag persistence in `pynerve.algebra`.


### Cross-references

- `pynerve.algebra`: Simplicial complex construction
- `pynerve.spectral`: Laplacian eigenanalysis
- `pynerve.cuda`: GPU graph algorithms
- `pynerve.ml`: ML pipeline with graph features

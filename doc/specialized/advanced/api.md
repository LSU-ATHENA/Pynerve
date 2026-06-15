# API

### `pynerve.specialized.cup_product`

Compute the cup product on cohomology classes of a simplicial complex.

```python
from pynerve.specialized import cup_product

# From point cloud (builds VR complex internally)
result = cup_product(
    points,                      # point cloud, shape (n, d)
    max_dim=2,                   # maximum cohomology dimension
    max_radius=2.0,              # VR filtration cutoff
    backend="cuda",              # "cuda" | "cpu"
)

# From explicit simplices and coboundary matrices
result = cup_product(
    simplices=simplices,
    coboundary_matrices=coboundaries,
    max_dim=2,
    backend="cuda",
)
```

**Returns:** ring structure, cohomology basis, cup product table.

**Cup product formula:**
For cochains alpha in C^p and beta in C^q:
(alpha U beta)(sigma) = alpha(sigma|[v0..vp]) * beta(sigma|[vp..vp+q])

**GPU kernel:** `cupProductKernel` -- block size = 256 threads.

### `pynerve.specialized.reeb_graph`

Topological skeleton from a scalar field on a graph or point cloud.

```python
from pynerve.specialized import reeb_graph

result = reeb_graph(
    adjacency=adjacency_list,
    function_values=values,
    backend="cuda",
)
```

**Returns:** nodes, arcs, merge tree, simplified graph.

**Simplification:**
```python
simplified = reeb_graph(
    adjacency=adj_list,
    function_values=values,
    persistence_threshold=0.1,    # remove pairs below this threshold
)
```

**GPU kernels:** `classifyVerticesKernel`, `connectedComponentsKernel`,
`constructArcsKernel`. Block size = 256, max adjacency = 10 per vertex.

### `pynerve.specialized.zigzag`

Persistence for time-varying data -- zigzag diagrams track homology through
insertions AND deletions.

```python
from pynerve.specialized import zigzag_persistence

time_slices = [points_t0, points_t1, points_t2, points_t3]
result = zigzag_persistence(
    time_slices,
    max_dim=2,
    max_radius=1.0,
    backend="cuda",
)
```

**Returns:** birth-death pairs, persistence intervals, interval matching,
computation time.


[Back to index](index.md)

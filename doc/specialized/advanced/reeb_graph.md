# Advanced Reeb graph

### Merge tree and contour tree

```python
from pynerve.specialized import reeb_graph, merge_tree, contour_tree

# Merge tree: tracks connected components of sublevel sets
mt = merge_tree(adjacency, values)

# Contour tree: combines join and split trees
ct = contour_tree(adjacency, values)

# Interactive simplification
for threshold in [0.01, 0.05, 0.1, 0.5]:
    simplified = ct.simplify(persistence_threshold=threshold)
    print(f"Threshold {threshold}: {simplified.num_nodes()} nodes")
```

### Branch decomposition

```python
from pynerve.specialized import BranchDecomposition

bd = BranchDecomposition(contour_tree)
branches = bd.compute_branches(min_size=5)
longest = bd.get_longest_branch()  # main topological feature
for b in branches:
    print(f"Branch: {b.length:.3f}, prominence: {b.prominence:.3f}")
```


[Back to index](index.md)

# Algorithm details

[Back to index](index.md)

### Sparse graph construction

The sparse distance $d_L$ on landmarks is constructed by connecting
each landmark to its nearest neighbors within a scale-dependent radius:

```
For each landmark l_i:
    r_i = covering radius at which l_i was added
    For each landmark l_j with j > i:
        if d(l_i, l_j) <= (1 + epsilon) * max(r_i, r_j) / epsilon:
            add edge (l_i, l_j) with weight d(l_i, l_j)
```

This produces a graph where each vertex has degree $O(\log n)$ in
expectation, compared to $O(n)$ for the full VR proximity graph.

### Edge pruning via bounding boxes

For high-dimensional data, distance computations are pruned using
axis-aligned bounding boxes:

```
For each landmark l_i:
    box = [l_i - r, l_i + r]  where r = (1+eps) * r_i / eps
    candidates = spatial_index.query(box)
    for l_j in candidates:
        compute d(l_i, l_j)
```

This reduces distance computation from $O(m^2)$ to $O(m \log m)$ when
using a kd-tree or R-tree index.

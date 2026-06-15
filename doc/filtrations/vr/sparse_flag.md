# Sparse flag

The `sparse` option enables an approximate VR complex that uses
epsilon-net sampling to reduce the number of points before building
the VR complex:

```python
# Exact VR (default)
result = pynerve.compute_persistence(points, max_dim=2, sparse=False)

# Sparse VR approximation
result = pynerve.compute_persistence(points, max_dim=2, sparse=True, sparse_parameter=0.3)
```

When `sparse=True`, the engine:

1. Builds an $\varepsilon$-net of landmarks via farthest-point sampling.
2. Constructs the VR complex only on the landmark set.
3. Uses scaled distance values that $(1+\varepsilon)$-approximate the
   original distances.
4. Produces a diagram that is $\frac{1}{1-\varepsilon}$-interleaved with
   the exact VR diagram.

See [sparse VR](../sparse_vr.md) for the full theory and API.


<- [Vietoris-Rips Overview](index.md)

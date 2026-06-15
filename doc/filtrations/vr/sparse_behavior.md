# Sparse flag: how it changes behavior

Setting `sparse=True` in `compute_persistence` changes the algorithm
in three substantive ways:

1. **Point reduction.** The full $n$-point cloud is reduced to an
   $m$-point epsilon-net via farthest-point sampling. The number of
   landmarks $m$ depends on `sparse_parameter`:
   - $\varepsilon = 0.1 \Rightarrow m \approx 0.15n$
   - $\varepsilon = 0.3 \Rightarrow m \approx 0.03n$
   - $\varepsilon = 0.5 \Rightarrow m \approx 0.01n$

2. **Distance approximation.** All-pairs distances among landmarks are
   computed, then scaled by $(1+\varepsilon)$ to ensure the interleaving
   guarantee. This is an $O(m^2)$ operation vs $O(n^2)$ for exact VR.

3. **VR construction on landmarks.** The standard VR complex is built
   on the $m$ landmark points. Since $m \ll n$, the simplex count drops
   from $O(n^{k+1})$ to $O(m^{k+1})$, typically a 100-1000x reduction.

The output diagram is guaranteed to be $\frac{1}{1-\varepsilon}$-interleaved
with the exact VR diagram (in bottleneck distance). See [sparse VR](../sparse_vr.md)
for full theoretical details.

```python
# Comparison of exact vs sparse on the same data
import numpy as np
import pynerve

points = np.random.randn(50000, 3)

# Exact VR (may be slow or fail at this size)
# result_exact = pynerve.compute_persistence(points, max_dim=2)

# Sparse VR (fast, approximate)
result_sparse = pynerve.compute_persistence(
    points,
    max_dim=2,
    max_radius=2.0,
    sparse=True,
    sparse_parameter=0.3,
)
```


<- [Vietoris-Rips Overview](index.md)

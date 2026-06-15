# Approximation guarantees

For a witness complex built from points sampled from a compact manifold:

### Bottleneck distance bound

The persistence diagram of the $k$-witness complex is $O(\delta)$-close
to the VR persistence diagram in bottleneck distance, where $\delta$ is
the **maximum distance from any point to its nearest landmark** (the
covering radius of the landmark set):

$$
d_B(\operatorname{Dgm}_{\text{VR}}, \operatorname{Dgm}_{\text{witness}})
\leq 3\delta
$$

With farthest-point sampling, $\delta$ decreases as:

$$
\delta \leq O(m^{-1/d_{\text{intrinsic}}})
$$

where $d_{\text{intrinsic}}$ is the intrinsic dimension of the data
manifold (not the ambient dimension).

### Practical implications

The following values illustrate how intrinsic dimension affects the approximation. For intrinsic dimension 2 with 100 landmarks, $\delta$ is approximately 0.10 and the bottleneck bound is 0.30; with 500 landmarks, $\delta$ is approximately 0.045 and the bound is 0.14. For intrinsic dimension 3 with 100 landmarks, $\delta$ is approximately 0.22 and the bound is 0.66; with 500 landmarks, $\delta$ is approximately 0.13 and the bound is 0.39. For intrinsic dimension 5 with 500 landmarks, $\delta$ is approximately 0.33 and the bound is 0.99. For intrinsic dimension 10 with 1000 landmarks, $\delta$ is approximately 0.50 and the bound is 1.50.

Higher intrinsic dimension requires exponentially more landmarks to
maintain the same approximation quality.

### Weak vs strong witness guarantees

The weak and strong witness constructions differ in several properties. The weak witness has a proven approximation ratio of 3, is more inclusive (more simplices), produces a larger filtration, has higher computational complexity, and is the default in Pynerve. The strong witness has a proven approximation ratio of 2, produces fewer simplices and a smaller filtration, has lower complexity, and is available via a flag.

The strong witness complex provides a better approximation ratio (2 vs 3)
but can miss topological features that the weak witness captures. In
practice, weak witness is preferred for its coverage.

<- [Witness Complex Overview](index.md)

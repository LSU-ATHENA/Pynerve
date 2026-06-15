# Theoretical background

### Witness complex as a relaxation of the Delaunay triangulation

The witness complex generalizes the Delaunay triangulation to arbitrary
metric spaces. In Euclidean space, the $0$-witness complex (using the
$(k+1)$-th nearest landmark condition) recovers the Delaunay triangulation
when the landmarks are chosen as all points.

For a general metric space, the witness complex provides a **Delaunay-like**
triangulation that respects the geometry of the data. This is particularly
valuable when:
- The data lies on a non-Euclidean manifold.
- Only pairwise distances are available (no coordinates).
- The ambient dimension is too high for Delaunay computation.

### Stability under perturbations

The witness complex is **stable** under perturbations of the data:

$$
d_B(\operatorname{Dgm}(X), \operatorname{Dgm}(X')) \leq
2 \cdot d_{\text{GH}}(X, X') + O(\delta)
$$

where $d_{\text{GH}}$ is the Gromov-Hausdorff distance and $\delta$
is the landmark covering radius. This means small perturbations in the
input lead to small changes in the output diagram.

### Relationship to the Cech complex

Let $X$ be sampled from a compact Riemannian manifold $M$ with
condition number $\tau$. If the landmark set $L$ is a $\delta$-net
with $\delta < \tau/4$, then the witness complex on $L$ is
homotopy-equivalent to $M$ up to a factor determined by $\delta$.

This is analogous to the topological guarantee of the Cech complex
but achieved with far fewer simplices.

<- [Witness Complex Overview](index.md)

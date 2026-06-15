# When Alpha beats VR

[Back to index](index.md)

Regarding filtration size, Alpha ranges from $O(n)$ to $O(n^{\lceil d/2 \rceil})$
while VR is $O(n^{k+1})$. For homotopy type, Alpha gives an exact equivalence
to the $\epsilon$-neighborhood, whereas VR provides a factor-2 interleaving
with the Cech complex. Computation time for Alpha is fast (sublinear per
point), while VR slows down at $n > 10^4$. Alpha has a low memory footprint
compared to VR's high memory usage. Differentiability is limited for Alpha
due to CGAL dependency, while VR offers full support via PyTorch autograd.
Alpha is restricted to Euclidean metrics, while VR works with any metric.

For $d \leq 3$, Alpha is strictly preferred over VR when:
- You need exact topology of the $\epsilon$-neighborhood.
- The point set is large ($n > 10^4$) but low-dimensional.
- Memory or compute time is constrained.
- You do not need gradients (Alpha is not differentiable through the
  Delaunay triangulation).

### Comparison: Alpha vs. Vietoris-Rips vs. Witness

Alpha is limited to $d \leq 3$ due to Delaunay constraints, while VR and
Witness support any dimension. Alpha produces a small complex
($O(n)$-$O(n^2)$), VR produces a large one ($O(n^{k})$), and Witness
produces a small complex ($O(m \cdot n)$). Alpha has low memory usage,
VR is high, and Witness is low to moderate. Alpha provides exact homotopy
type, VR provides exact flag complex results, and Witness gives an
approximate $k$-witness result. Alpha is not differentiable, VR is, and
Witness has limited differentiability. Alpha is best for geometric data
requiring exact results, VR for general metric spaces, and Witness for
large $n$ with approximate topology.

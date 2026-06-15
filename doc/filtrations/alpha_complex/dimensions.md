# Dimension constraints

[Back to index](index.md)

The Alpha complex is **limited to $d \leq 3$** (points in $\mathbb{R}^2$ or
$\mathbb{R}^3$). This is a fundamental constraint of the Delaunay
triangulation:

In dimension 2, the Delaunay complexity is $O(n \log n)$, which is fast and
widely used. In dimension 3, the worst-case complexity is $O(n^2)$ but typical
performance is $O(n \log n)$, making it practical up to $10^5$ points. In
dimension 4, the theoretical best is $O(n^2)$, which becomes impractical
beyond $10^4$ points. In dimension 5 and above, the complexity is
$O(n^{\lceil d/2 \rceil})$, which is prohibitively expensive.

For data in $\mathbb{R}^d$ with $d > 3$, use Vietoris-Rips or a
landmark-based approximation.

### Why 3D?

The Delaunay triangulation in $\mathbb{R}^d$ has size $O(n^{\lceil d/2
\rceil})$ in the worst case. For $d = 2$, this is $O(n)$. For $d = 3$,
it is $O(n^2)$ worst-case but $O(n \log n)$ for typical data. For
$d \geq 4$, even the output size grows superlinearly, and numerical
robustness issues with degenerate configurations become severe.

Pynerve uses **CGAL** for Delaunay computation in $\mathbb{R}^2$ and
$\mathbb{R}^3$, with symbolic perturbation to resolve degeneracies.

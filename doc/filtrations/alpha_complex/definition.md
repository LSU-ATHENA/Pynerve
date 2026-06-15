# Definition

[Back to index](index.md)

Let $X \subset \mathbb{R}^d$ be a finite set of points in Euclidean space.
The **Alpha complex** $\operatorname{Alpha}_\epsilon(X)$ at scale
$\epsilon \geq 0$ is defined as:

$$
\operatorname{Alpha}_\epsilon(X) = \{\sigma \in \operatorname{Del}(X) :
\operatorname{circ}(\sigma) \leq \epsilon\}
$$

where $\operatorname{Del}(X)$ is the **Delaunay triangulation** of $X$ and
$\operatorname{circ}(\sigma)$ is the **circumradius** of simplex $\sigma$ --
the radius of the smallest ball passing through all vertices of $\sigma$.

### Delaunay triangulation

The Delaunay triangulation of $X$ is a simplicial complex (in $\mathbb{R}^d$)
such that for every $d$-simplex, the circumscribing ball (the **Delaunay
ball**) contains no other points of $X$ in its interior. In $\mathbb{R}^2$,
this is the familiar empty-circumcircle condition:

$$
\operatorname{Del}(X) = \{\sigma \subset X : \exists B(c, r) \text{ with }
\sigma \subset \partial B(c, r), \; B(c, r) \cap X = \emptyset\}
$$

The Alpha complex is sometimes called the **Delaunay-alpha complex** or
**alpha shapes** in the computational geometry literature.

### Homotopy equivalence

The Alpha complex is **homotopy-equivalent** to the union of
$\epsilon$-balls centered at $X$:

$$
\operatorname{Alpha}_\epsilon(X) \simeq \bigcup_{x \in X} B(x, \epsilon)
$$

This is a stronger guarantee than Vietoris-Rips, which is only
interleaved with the Cech complex (a factor-2 approximation). For
geometric data in $\mathbb{R}^3$, Alpha gives the exact topology of the
$\epsilon$-neighborhood.

### Filtration structure

As $\epsilon$ increases from $0$ to $\infty$, simplices enter in order of
non-decreasing circumradius. The filtration is **thin** compared to VR:
the number of simplices is bounded by the size of the Delaunay triangulation,
which is $O(n^{\lceil d/2 \rceil})$ and typically $O(n)$ for $d \leq 3$.

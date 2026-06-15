# FAQ

[Back to index](index.md)

### Why is Alpha limited to 3 dimensions?

The Alpha complex relies on the Delaunay triangulation, whose size grows as
$O(n^{\lceil d/2 \rceil})$ in the worst case. For $d = 2$ and $d = 3$ this is
manageable ($O(n)$ and $O(n^2)$ worst-case respectively), but for $d \geq 4$
the combinatorial explosion makes construction impractical for all but tiny
point sets. Additionally, numerical robustness in Delaunay algorithms
degrades sharply above 3 dimensions. In practice, CGAL only supports
Delaunay triangulations in $\mathbb{R}^2$ and $\mathbb{R}^3$, which is the
underlying engine Pynerve uses.

### How does Alpha compare to VR in practice?

Alpha typically produces 10-100x fewer simplices than Vietoris-Rips for the
same point cloud, translating to proportionally lower memory usage and faster
matrix reduction. Alpha also provides exact homotopy equivalence to the
$\epsilon$-neighborhood, whereas VR is only a factor-2 interleaving with the
Cech complex. However, Alpha is restricted to Euclidean data in at most 3
dimensions, while VR works in any metric space and any dimension. For
low-dimensional geometric data, Alpha is almost always preferred.

### When should I use weighted Alpha?

Use weighted Alpha when your points carry associated radii or weights, such
as atoms with van der Waals radii in computational chemistry, or when points
have varying influence regions. The weighted variant uses the power diagram
(regular triangulation) instead of the standard Delaunay triangulation,
measuring power-circumradius rather than Euclidean circumradius. This is
essential for computing solvent-accessible surfaces, molecular volumes, and
other applications where point sizes matter.

### Why is Alpha not differentiable?

Alpha persistence requires constructing a Delaunay triangulation, which
involves combinatorial decisions (which points connect to form simplices)
that are discrete and non-smooth with respect to point positions. A small
perturbation to the input can change the triangulation topology, breaking
the gradient path. Vietoris-Rips avoids this by using only pairwise distances
and a sorting-based relaxation that admits subgradients. For differentiable
topological optimization, use VR instead.

### How do I handle cocircular or cospherical points?

Cocircular (2D) and cospherical (3D) configurations produce a non-unique
Delaunay triangulation, which can lead to non-deterministic or
perturbation-sensitive filtrations. Pynerve handles this via symbolic
perturbation to ensure deterministic output. As a preprocessing step,
remove near-duplicate points with `np.unique(np.round(points, decimals=12),
axis=0)`, add tiny jitter to break exact symmetries, and normalize
coordinates to avoid extreme values that amplify numerical issues.

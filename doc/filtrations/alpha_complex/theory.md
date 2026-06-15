# Theoretical background

[Back to index](index.md)

### Weighted Alpha complex

For points with varying radii (e.g., atoms with van der Waals radii),
the **weighted Alpha complex** (also called the **regular** or
**Laguerre** Alpha complex) extends the construction using the
power diagram (weighted Delaunay triangulation):

$$
\operatorname{AlphaWeighted}_\epsilon(X) = \{\sigma \in \operatorname{Reg}(X) :
\operatorname{powerCirc}(\sigma) \leq \epsilon\}
$$

where $\operatorname{Reg}(X)$ is the **regular triangulation** and
$\operatorname{powerCirc}(\sigma)$ is the power circumradius.

Weighted Alpha is used in computational chemistry for computing solvent
accessible surfaces and molecular volumes. Pynerve's C++ backend supports
weighted Delaunay via CGAL, exposed as:

```cpp
// C++ API (via nerve_internal)
WeightedAlpha wa;
wa.setPoints(positions, radii);
wa.buildFiltration(epsilon);
auto result = wa.computePersistence(max_dim);
```

### Relationship to Cech complex

The Alpha complex is the **Cech complex restricted to Delaunay simplices**:

$$
\operatorname{Alpha}_\epsilon(X) = \operatorname{Cech}_\epsilon(X) \cap
\operatorname{Del}(X)
$$

This means Alpha preserves the homotopy type of the Cech complex while
using only Delaunay simplices, which are far fewer than the full Cech
simplices. The Delaunay restriction does not change the homotopy type
because the nerve theorem guarantees homotopy equivalence for good
covers.

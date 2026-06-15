# Alpha complex

The Alpha complex is a subcomplex of the Delaunay triangulation filtered by
circumradius. It produces smaller filtrations than Vietoris-Rips for the same
point cloud and is the preferred choice for low-dimensional geometric data.

## Sections

- [Definition](definition.md)  --  Delaunay triangulation, homotopy equivalence, filtration structure
- [Dimension constraints](dimensions.md)  --  Limited to $d \leq 3$, why 3D matters
- [Complexity analysis](complexity.md)  --  Construction cost, simplex counts, memory usage
- [When Alpha beats VR](vs_vr.md)  --  Filtration size, homotopy type, speed comparison
- [Limitations](limitations.md)  --  Dimensionality, non-geometric data, numerical degeneracy
- [Code examples](code_examples.md)  --  Basic usage, comparison with VR, batched mode
- [API](api.md)  --  PyTorch interface, batched mode, memory management
- [Debugging common issues](debugging.md)  --  Delaunay failures, non-determinism, large values
- [Theoretical background](theory.md)  --  Weighted Alpha, relationship to Cech complex
- [Numerical considerations](numerics.md)  --  Degenerate configurations, floating-point precision
- [Practical guidance](guidance.md)  --  When to use Alpha, tips
- [FAQ](faq.md)  --  Frequently asked questions

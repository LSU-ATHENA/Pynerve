# SimplicialComplex

```cpp
class SimplicialComplex {
public:
    SimplicialComplex();

    void addSimplex(const Simplex& s);
    void addSimplexWithFiltration(const Simplex& s, double filtration);
    void removeSimplex(const Simplex& s);
    void clear();

    Size size() const noexcept;
    Size numSimplices() const noexcept;
    Dimension maxDimension() const noexcept;
    Vector<Simplex> simplicesOfDimension(Dimension dim) const;
    Vector<Simplex> getSimplices() const;
    double getFiltration(const Simplex& s) const;
    void setFiltration(const Simplex& s, double filtration);
    Vector<std::pair<Simplex, double>> getFilteredSimplices() const;
};
```

Container for a filtered simplicial complex. Each simplex has an associated
filtration value.

### Important: face handling

Adding a simplex does NOT automatically add its faces. You must add them
explicitly or use a complex-building helper:

```python
# INCORRECT: triangle without edges
sc = SimplicialComplex()
sc.add_simplex(Simplex([0, 1, 2]))
# sc contains only the triangle, not the edges!

# CORRECT: add all faces
sc.add_simplex(Simplex([0]))
sc.add_simplex(Simplex([1]))
sc.add_simplex(Simplex([2]))
sc.add_simplex(Simplex([0, 1]))
sc.add_simplex(Simplex([1, 2]))
sc.add_simplex(Simplex([0, 2]))
sc.add_simplex(Simplex([0, 1, 2]))

# Or use build_vr_complex which handles this automatically
```

### Filtration values

```python
sc = SimplicialComplex()
sc.addSimplexWithFiltration(Simplex([0, 1]), 0.5)
sc.addSimplexWithFiltration(Simplex([1, 2]), 1.0)

# Query
f = sc.getFiltration(Simplex([0, 1]))  # 0.5

# Modify
sc.setFiltration(Simplex([0, 1]), 0.8)

# Get all simplices with values
filtered = sc.getFilteredSimplices()
for simplex, val in filtered:
    print(f"{simplex} at {val}")
```


### Factory functions

The `pynerve.algebra` module provides helper functions for building common
complexes:

```python
from pynerve.algebra import build_vr_complex, build_cech_complex, build_alpha_complex

# Vietoris-Rips
vr = build_vr_complex(points, max_radius=1.0, max_dim=2)

# Cech
cech = build_cech_complex(points, max_radius=0.5, max_dim=2)

# Alpha (2D/3D Delaunay-based)
alpha = build_alpha_complex(points, max_radius=1.0)
```


### Complexity

Simplex insertion (checked) is O(log m) with hashing. Batch simplex insertion is O(m). Filtration value lookup is O(1) amortized. `getSimplices` costs O(m). `simplicesOfDimension` costs O(m). `getFilteredSimplices` costs O(m log m). `removeSimplex` costs O(m).


### Common pitfalls

1. **Missing faces**: The most common bug. Always add all faces before
   adding a higher-dimensional simplex.

2. **Duplicate vertices**: `Simplex([0, 0, 1])` is invalid. Vertices must
   be unique (duplicates are silently removed).

3. **Filtration monotonicity**: Face filtration values must be <= coface
   values for correct persistence.

4. **Integer overflow**: `Index` is typically `uint32_t`. For complexes
   with >4 billion simplices, use the 64-bit build.


### Cross-references

- `pynerve.algebra.SimplicialComplex` (top-level): module overview
- `pynerve.algebra.BoundaryMatrix`: builds from SimplicialComplex
- `pynerve.algebra.ChainComplex`: builds all boundary matrices
- `pynerve.persistence`: persistence algorithms on simplicial complexes
- `pynerve.torch.SimplexTree`: GPU-accelerated simplex tree equivalent

[Back to index](index.md)

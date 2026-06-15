# Simplex

```cpp
// C++ header: src/include/nerve/algebra/simplex.hpp
namespace nerve::algebra {

class Simplex {
public:
    Simplex();
    explicit Simplex(const std::vector<Index>& vertices);
    Simplex(std::initializer_list<Index> vertices);

    Size dimension() const noexcept;          // numVertices - 1
    Size numVertices() const noexcept;
    const std::vector<Index>& vertices() const noexcept;
    bool contains(Index vertex) const;
    Index operator[](Size i) const;
    Index at(Size i) const;

    Simplex boundary(Size k = 1) const;       // k-th boundary chain
    std::vector<Simplex> faces() const;       // all codim-1 faces
    std::vector<Simplex> cofaces(const std::vector<Simplex>& complex) const;
    std::vector<Simplex> kFaces(Size k) const;
    Simplex faceWithoutVertex(Index vertex) const;

    bool isFaceOf(const Simplex& other) const;
    Simplex join(const Simplex& other) const;
    Simplex meet(const Simplex& other) const;
    Simplex star(const std::vector<Simplex>& complex) const;
    Simplex link(const std::vector<Simplex>& complex) const;

    // Geometry
    double volume(const PointView& coords) const;
    double barycentricCoordinate(const PointView& point, const PointView& coords) const;
    std::vector<double> circumcenter(const PointView& coords) const;
    double circumradius(const PointView& coords) const;
    bool containsPoint(const PointView& point, const PointView& coords) const;

    std::string toString() const;
    void sortVertices();

    struct Hash { Size operator()(const Simplex& s) const noexcept; };
};

}
```


### Constructor details

Build a Simplex from a vector of vertex indices or an initializer list.
Vertices are stored in sorted order (canonical form).

```cpp
Simplex t({0, 1, 2});               // triangle
Simplex e({0, 1});                   // edge
Simplex v({0});                      // vertex
Simplex empty;                       // empty simplex (invalid)
```

The `sortVertices()` method is called automatically in the constructor.
Two simplices with the same vertices in different order are equal.


### Boundary computation

The `boundary(k)` method returns the k-th boundary operator applied to the
simplex. For k = 1, this is the set of (dim-1)-faces (the standard boundary).

```cpp
auto bdy = Simplex({0, 1, 2}).boundary();
// result: Simplex({0,1}), Simplex({1,2}), Simplex({0,2})
// Note: {0,2} not {2,0} because vertices are sorted
```

For k >= 2, `boundary(k)` returns the k-th iterated boundary:
```cpp
auto bdy2 = Simplex({0, 1, 2}).boundary(2);  // boundary of boundary
// Result: empty (because boundary^2 = 0)
```

The boundary is computed by iterating over all vertex subsets of size k:
```text
boundary_k(sigma) = sum_{tau subset sigma, |tau| = |sigma| - k} tau
```


### Faces

`faces()` returns all codimension-1 faces as a vector:

```cpp
auto f = Simplex({0, 1, 2}).faces();
// returns [{1, 2}, {0, 2}, {0, 1}]
// (each vertex removed in order)
```

`kFaces(k)` returns all faces of exactly dimension k:
```cpp
auto edges = Simplex({0, 1, 2, 3}).kFaces(1);
// returns [{1,2}, {0,2}, {0,1}, {0,3}, {1,3}, {2,3}]
// all 6 edges of a tetrahedron
```


### Cofaces, star, link

`cofaces(complex)` returns all simplices in `complex` that contain this
simplex.

```cpp
Simplex edge({0, 1});
auto cof = edge.cofaces(complex);  // all {0, 1, ...} in complex
```

The `star` and `link` methods follow the standard topological definitions:

- `star(s)` = set of cofaces of s (including s)
- `link(s)` = star(s) \ {tau in star(s) : s is not a face of tau}

More concretely:
```text
star(sigma) = {tau in K : sigma subset tau}
link(sigma) = {tau in star(sigma) : sigma intersect tau = empty}
```


### Join and meet

- `join(s1, s2)` -- simplex whose vertices are the union of s1 and s2
- `meet(s1, s2)` -- simplex whose vertices are the intersection of s1 and s2

Two simplices are compatible if their join is valid (no duplicate vertices).

```cpp
Simplex a({0, 1});
Simplex b({2, 3});
auto joined = a.join(b);  // {0, 1, 2, 3}
auto met = a.meet(b);     // {} (empty, disjoint)
```


### Geometry methods

All geometry methods require a `PointView` -- a non-owning reference to a
coordinate table:

```cpp
struct PointView {
    const double* data;       // flat array of coordinates
    size_t num_points;        // number of points in table
    size_t dimension;         // dimension of each point
    // Access: point p, coordinate c -> data[p * dimension + c]
};
```

**Volume** uses the Cayley-Menger determinant. For a k-simplex in R^d:

```cpp
double v = simplex.volume(coords);
// V^2 = (-1)^{k+1} / (2^k * (k!)^2) * det(CM_matrix)
```

**Circumcenter and circumradius** solve for the center of the unique sphere
passing through all vertices:

```cpp
auto center = simplex.circumcenter(coords);
double radius = simplex.circumradius(coords);
```

**Barycentric coordinates** express a point as a convex combination of vertices:

```cpp
double w = simplex.barycentricCoordinate(point, coords);
// returns the weight of the first vertex (others implied)
// sum of all weights = 1, weights >= 0 if inside
```

**Point containment** checks whether a point lies inside the simplex:

```cpp
bool inside = simplex.containsPoint(point, coords);
// Uses barycentric coordinates: all weights >= 0 within tolerance
```


[Back to index](index.md)

# Algebra

Algebraic data structures for simplicial complexes, boundary matrices, and chain complexes. These are the building blocks of persistent homology computation.

```python
from pynerve.algebra import Simplex, SimplicialComplex, BoundaryMatrix, ChainComplex

complex = SimplicialComplex()
complex.add_simplex(Simplex([0, 1, 2]))
complex.add_simplex_with_filtration(Simplex([0, 1]), 0.5)

bm = BoundaryMatrix(complex, dimension=1)
pairs = bm.compute_persistence_pairs()
```

## What's included

| Page | Contents |
|------|----------|
| [Simplicial operations](simplicial_ops.md) | Simplex, SimplexSet, SimplicialComplex; faces, cofaces, star, link, geometry |
| [Boundary matrices](boundary_matrices.md) | BoundaryMatrix construction, access, reduction, persistence pairs |
| [Chain complexes](chain_complexes.md) | ChainComplex, cellular complexes, CW complexes |
| [Distance computation](distance.md) | SIMD-accelerated distance with SSE4.1/AVX2/AVX-512 dispatch |

## Namespace

All types are in `nerve::algebra` (C++) and `pynerve.algebra` (Python).

## Geometry utilities

Free functions for geometric operations on simplices:

```cpp
bool solveLinearSystem(const std::vector<std::vector<double>>& matrix,
                       const std::vector<double>& rhs,
                       std::vector<double>* solution);

std::vector<double> centroid(const std::vector<Index>& simplex_vertices,
                              const std::vector<std::vector<double>>& coord_table);

bool computeBarycentricWeights(const std::vector<Index>& simplex_vertices,
                               const std::vector<std::vector<double>>& coord_table,
                               const PointView& point,
                               std::vector<double>* weights);

double squaredDistance(const std::vector<double>& lhs,
                       const std::vector<double>& rhs);
```

### Volume computation

Volume uses the Cayley-Menger determinant. Complexity is O(d!) in dimension; practical for d <= 10.

## GPU operations

CUDA kernels for algebra are in `src/algebra/gpu/`:

| Kernel | Purpose |
|--------|---------|
| `cech_complex_cuda.cu` | Cech complex construction on GPU |
| `vr_complex_cuda.cu` | Vietoris-Rips complex construction on GPU |
| `distance_kernels.cu` | Pairwise distance on GPU |

Boundary matrix reduction is GPU-accelerated via CUDA kernels in `src/persistence/gpu/`. Enable with `PersistenceBackend.CUDA_HYBRID`.

## Complexity

| Operation | Cost |
|-----------|------|
| Simplex boundary | O(k) for k-face enumeration |
| Simplex cofaces | O(m) in complex size |
| Boundary matrix build | O(m * f) where f = average face count |
| Column reduction (typical) | O(r * cols) |
| Column reduction (worst) | O(cols^2) |
| Pairwise distance (scalar) | O(n^2 * d) |
| Pairwise distance (SIMD) | O(n^2 * d / SIMD_width) |
| Pairwise distance (GPU) | O(n^2 * d / cores) |
| Cayley-Menger volume | O(d!) |
| Centroid computation | O(k * d) |

## Python quick reference

```python
from pynerve.algebra import Simplex, SimplicialComplex, BoundaryMatrix, ChainComplex

# Simplex construction
s = Simplex([0, 1, 2])         # 2-simplex (triangle)
s.dimension()                   # 2
s.vertices()                    # [0, 1, 2]
s.contains(1)                   # True

# Face operations
faces = s.faces()               # all codim-1 faces
bdy = s.boundary()              # boundary as simplices
kfaces = s.kFaces(1)            # 1-dimensional faces (edges)

# Simplicial complex
sc = SimplicialComplex()
sc.add_simplex(Simplex([0, 1, 2]))
sc.add_simplex_with_filtration(Simplex([0, 1]), 0.5)

# Boundary matrix from complex
bm = BoundaryMatrix(sc, dimension=1)
pairs = bm.computePersistencePairs()
for row, col in pairs:
    birth = bm.getFiltrationValue(row)
    death = bm.getFiltrationValue(col)
    print(f"Feature: birth={birth:.3f}, death={death:.3f}")

# Chain complex
cc = ChainComplex(sc)
for d in range(cc.maxDimension() + 1):
    b = cc.bettiNumber(d)
    print(f"Betti_{d} = {b}")
```

## When to use each data structure

| Structure | Use case |
|-----------|----------|
| `Simplex` | Building block for individual simplices. Manipulate faces, cofaces, or geometric properties of a single simplex. |
| `SimplicialComplex` | Container of simplices with filtration values. Use for the main persistence pipeline. |
| `SimplexSet` | Efficient set operations (union, intersection, difference) on simplices. Fast membership testing. |
| `BoundaryMatrix` | Sparse matrix representation of the boundary operator. Linear algebra on chains and cochains. |
| `ChainComplex` | All boundary matrices for dimensions 0..max_dim. Use when you need the full algebraic structure. |

## Performance tips

1. Pre-sort vertices with `sortVertices()` to canonicalize simplices
2. Build complexes in batches rather than one simplex at a time
3. Use `computeCompressedMatrix` instead of full distance matrices when you only need the upper triangle
4. For large complexes (>100k simplices), prefer the GPU path
5. Set `max_radius` on VR complexes to limit the number of edges

## Common pitfalls

- Adding a simplex does NOT automatically add its faces. You must add faces explicitly.
- Filtered simplices require explicit `add_simplex_with_filtration`.
- Boundary matrix indices are row/column positions, not simplex IDs.
- Column reduction mod 2 assumes Z/2 coefficients by default.

## Cross-references

- [Persistence](../persistence/standard_reduction.md): uses BoundaryMatrix
- [Spectral](../spectral/spectral.md): uses BoundaryMatrix for Hodge Laplacian
- [DMT](../dmt/dmt.md): uses Simplex/SimplicialComplex for Morse reduction
- [Torch](../torch/torch.md): provides SimplexTree for GPU
- [IO](../io/io.md): handles serialization of diagrams

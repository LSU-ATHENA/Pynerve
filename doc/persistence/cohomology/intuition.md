# Intuition

[Back to Index](index.md)

Standard (homology) reduction builds the boundary matrix and reduces columns left-to-right. Each column operation can add many rows, making columns *denser* as reduction progresses.

Cohomology reduction instead reduces the *coboundary* matrix -- the transpose of the boundary matrix -- and processes simplices in *decreasing* filtration order. The key benefits:

1. **Sparser initial matrix**: The coboundary of a *d*-simplex is the set of (*d*+1)-simplices that contain it. In sparse filtrations, high-dimensional simplices have few cofaces, so coboundary columns are naturally sparse.

2. **Early columns are sparse**: Processing in reverse filtration order means the first columns processed are the highest-dimensional simplices, which have the fewest coboundary entries. Standard reduction starts with low-dimensional simplices, which have dense boundaries.

3. **Emergent pair detection**: A simplex that enters the filtration late and has a unique oldest coface can be paired immediately without full reduction (de Silva-Morozov 2011).

### Why Coboundary is Sparser

For a d-simplex sigma, its coboundary is:

    coboundary(sigma) = { tau : dim(tau) = d+1, sigma is a face of tau }

In sparse filtrations (e.g., Vietoris-Rips with moderate radius), each d-simplex is a face of only a few (d+1)-simplices. Compare:

For a vertex (dimension 0), the boundary size is 0 while the coboundary size is O(N) because each vertex belongs to many edges. For an edge (dimension 1), the boundary has 2 entries and the coboundary has O(1-10) entries since few triangles contain a given edge. For a triangle (dimension 2), the boundary has 3 entries and the coboundary has O(1-5) entries because few tetrahedra contain a given triangle. For a high-dimensional simplex (dimension d), the boundary has d+1 entries while the coboundary has O(1) entries since such simplices are rarely found in higher-dimensional simplices.

The coboundary matrix has:
- **Vertices** (dim 0): dense coboundary columns (many edges). These are processed *last* in reverse order, when the matrix is already partially reduced.
- **High-dim simplices** (dim >= 2): sparse coboundary columns. Processed *first*, when the matrix is empty, so they are quick to reduce.

Standard reduction does the opposite: it processes vertices first (dense boundaries are not the problem -- boundaries of vertices are empty), then edges (boundaries have 2 entries), then triangles (boundaries have 3 entries), and so on. The problem is that as reduction progresses, columns accumulate entries from pivot elimination, becoming progressively denser. Coboundary reduction avoids this because the pivot elimination process in the coboundary setting tends to *reduce* column density.

### Dual Perspective

Cohomology computes the exact same persistence pairs as homology, but from the dual perspective. The relationship is:

- H_k (homology) = k-dimensional holes
- H^k (cohomology) = dual vector space to H_k

Over a field (including Z2), homology and cohomology contain equivalent information: the persistence barcodes are identical. The choice is purely a matter of computational convenience.

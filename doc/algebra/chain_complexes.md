## ChainComplex

```cpp
class ChainComplex {
public:
    explicit ChainComplex(const SimplicialComplex& complex);

    const BoundaryMatrix& boundary(Size k) const;
    BoundaryMatrix& boundary(Size k);
    Size rank(Size k) const;
    Size bettiNumber(Size k) const;
    Size maxDimension() const noexcept;

    std::vector<double> applyBoundary(Size k, const BufferView<const double>& chain) const;
    std::vector<double> applyCoboundary(Size k, const BufferView<const double>& cochain) const;

    std::vector<std::pair<Index, Index>> compute();
    std::vector<Pair> computePersistenceDiagram();
    std::vector<Size> computeBettiNumbers();
};
```

Constructs all boundary matrices for dimensions 0..max_dim from a
SimplicialComplex. Each `boundary(k)` returns the boundary operator from
k-chains to (k-1)-chains.


### Internal structure

The chain complex stores an array of BoundaryMatrix objects, one per
dimension. Dimension k's boundary matrix B_k maps k-simplices to
(k-1)-simplices as rows.

```
ChainComplex:
  B_0: (empty -- no (-1)-simplices)
  B_1: edges (rows) -> vertices (cols)    [dim=1 boundary]
  B_2: triangles (rows) -> edges (cols)   [dim=2 boundary]
  ...
  B_{max}: (empty if no (max+1)-simplices)
```

Each B_k satisfies the fundamental property: B_k @ B_{k+1} = 0 (the boundary
of a boundary is empty).


### Betti numbers

The k-th Betti number is the rank of the k-th homology group:
beta_k = dim(ker B_k) - dim(im B_{k+1})

```python
cc = ChainComplex(sc)
for d in range(cc.maxDimension() + 1):
    beta = cc.bettiNumber(d)
    print(f"beta_{d} = {beta}")
    rank = cc.rank(d)  # rank of boundary matrix B_d
```

For a 2-sphere (hollow tetrahedron), this produces:
```
beta_0 = 1   (connected)
beta_1 = 0   (no loops -- sphere is simply connected)
beta_2 = 1   (one void)
```


### Computational pipeline

The `compute()` method runs the full persistence pipeline:
1. For each dimension k (starting from 0), reduce B_k
2. Extract persistence pairs from the reduction
3. Return pairs for all dimensions

```python
cc = ChainComplex(sc)
pairs = cc.compute()  # all pairs across all dimensions
diagram = cc.computePersistenceDiagram()  # as (dim, birth, death) tuples
betti = cc.computeBettiNumbers()  # [beta_0, beta_1, ...]
```


### Chain and cochain operations

`applyBoundary(k, chain)` computes B_k @ chain, the boundary of a k-chain. `applyCoboundary(k, cochain)` computes B_{k+1}^T @ cochain, the coboundary of a k-cochain.

```python
# Verify B_1 @ B_2 = 0 property
b2 = cc.applyBoundary(2, chain_on_triangles)
b1 = cc.applyBoundary(1, b2)
assert all(abs(x) < 1e-10 for x in b1)  # boundary of boundary is zero
```


## Cellular and CW complexes

### Cell

```cpp
class Cell {
public:
    explicit Cell(int dimension);
    Cell(int dimension, const std::vector<Index>& boundary);
    int dimension() const noexcept;
    const std::vector<Index>& boundary() const noexcept;
    void addBoundaryCell(Index cell_index);
    void removeBoundaryCell(Index cell_index);
    void clearBoundary();
};
```

A `Cell` represents a single cell in a cell complex. Unlike a simplex (which
is defined by its vertices), a cell is defined by its dimension and its
boundary -- a list of references to lower-dimensional cells.

```python
cell = Cell(dimension=2)
cell.addBoundaryCell(edge_index_0)
cell.addBoundaryCell(edge_index_1)
cell.addBoundaryCell(edge_index_2)
```

### CellularComplex

```cpp
class CellularComplex {
public:
    Index addCell(const Cell& cell);
    void removeCell(Index cell_index);
    void clear();
    Size numCells() const noexcept;
    int maxDimension() const;
    std::vector<Index> cellsOfDimension(int dimension) const;
    const Cell& getCell(Index index) const;
    std::vector<Index> getBoundary(const Cell& cell) const;
    std::vector<Index> getCoboundary(const Cell& cell) const;
    ErrorResult<std::vector<int>> computeEulerCharacteristic() const;
    ErrorResult<std::vector<int>> computeBettiNumbers() const;
};
```

A general cell complex where cells reference their boundary explicitly. This
is more flexible than simplicial complexes because:
- Cells can have arbitrary shapes (not just simplices)
- Attaching maps can be any continuous function
- Fewer cells needed for the same space

**Construction example (2-sphere as cell complex):**

```python
sphere = CellularComplex()
v0 = sphere.addCell(Cell(0))   # vertex
v1 = sphere.addCell(Cell(0))   # vertex
e0 = sphere.addCell(Cell(1, [v0, v1]))  # edge between v0, v1
```

**Euler characteristic:**
```python
chi = sphere.computeEulerCharacteristic()
# chi = V - E + F - ... (alternating sum of cell counts per dimension)
```

### CellularChainComplex

```cpp
class CellularChainComplex {
public:
    explicit CellularChainComplex(const CellularComplex& complex);
    std::vector<std::vector<double>> getBoundaryMatrix() const;
    std::vector<std::vector<double>> getCoboundaryMatrix() const;
    ErrorResult<std::vector<std::vector<int>>> computeHomology() const;
    ErrorResult<std::vector<int>> computeBettiNumbers() const;
};
```

Builds the chain complex from a `CellularComplex` (dense storage). The
boundary matrix is generated by chasing cell boundary references.

```python
ccc = CellularChainComplex(sphere)
betti = ccc.computeBettiNumbers()
homology = ccc.computeHomology()  # generators for each dimension
```

### CWComplex

```cpp
class CWComplex {
public:
    void addSimplices(const std::vector<Simplex>& simplices);
    void addSimplex(const Simplex& simplex);
    Size numSimplices() const;
    int maxDimension() const;
    const Simplex& getSimplex(Index index) const;
    std::vector<Index> getStar(const Simplex& simplex) const;
    std::vector<Index> getLink(const Simplex& simplex) const;
    std::vector<int> computeHomology() const;
    std::vector<int> computeBettiNumbers() const;
};
```

`CWComplex` wraps a simplicial complex and exposes CW-like queries (star,
link, homology). It is a lightweight adapter for simplicial complexes that
provides CW cell structure.


### Practical guidance

**When to use which complex type:** `ChainComplex` is for persistence computation on simplicial complexes. `CellularComplex` is for non-simplicial spaces and attaching maps. `CellularChainComplex` is for homology of cell complexes (with dense matrices). `CWComplex` is for CW-structure queries on simplicial complexes.

**Common pitfalls:**
- `CellularChainComplex` uses dense matrix storage -- limit to small complexes
- Cells must be added in order of dimension (0-cells first, then 1-cells, etc.)
- Euler characteristic only works for finite cell complexes
- `CWComplex` does not support filtrations directly


### Complexity

Boundary matrix build is O(m * f) amortized for ChainComplex and O(cells * avg_boundary) for CellularChainComplex. A single Betti number costs O(nnz_k + nnz_{k+1}) for ChainComplex and O(n^3) dense for CellularChainComplex. All Betti numbers cost O(sum nnz_k) for ChainComplex and O(n^3) worst case for CellularChainComplex. Homology generators require reduction for ChainComplex and Smith normal form for CellularChainComplex.


### Cross-references

- `pynerve.algebra.BoundaryMatrix`: individual boundary operators
- `pynerve.algebra.SimplicialComplex`: source for building chain complexes
- `pynerve.persistence`: reduction algorithms for chain complexes
- `pynerve.dmt`: Morse-theoretic reduction of chain complexes
- `pynerve.spectral.HodgeTheory`: Hodge decomposition on chain complexes

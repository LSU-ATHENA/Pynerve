## BoundaryMatrix

```cpp
class BoundaryMatrix {
public:
    BoundaryMatrix();
    explicit BoundaryMatrix(const SimplicialComplex& complex);
    BoundaryMatrix(const SimplicialComplex& complex, Size dimension);

    void buildFromComplex(const SimplicialComplex& complex);
    void buildKDimensional(const SimplicialComplex& complex, Size k);

    Size rows() const noexcept;
    Size cols() const noexcept;
    Size dimension() const noexcept;
    bool isEmpty() const noexcept;
    Size numNonzeros() const noexcept;
    double sparsityRatio() const noexcept;

    // Linear algebra
    std::vector<double> multiply(const BufferView<const double>& v) const;
    std::vector<double> transposeMultiply(const BufferView<const double>& v) const;
    std::vector<double> applyBoundary(const BufferView<const double>& chain) const;
    std::vector<double> applyCoboundary(const BufferView<const double>& cochain) const;
    BoundaryMatrix transpose() const;

    // Reduction
    void reduceToReducedRowEchelon();
    std::vector<Index> findPivotColumns();
    std::vector<Index> findPivotRows();
    ErrorResult<std::vector<std::pair<Index, Index>>> computePersistencePairs();
    ErrorResult<std::vector<Index>> findEssentialCycles();

    // Access
    double getCoefficient(Size row, Size col) const;
    double getFiltrationValue(Size col) const;
    void setFiltrationValue(Size col, double value);
    int getRowSimplexDimension(Size row) const;
    int getColSimplexDimension(Size col) const;
    std::vector<Simplex> simplicesInRow(Size row) const;
    std::vector<Simplex> simplicesInCol(Size col) const;

    std::string matrixString() const;
    std::string nonzeroPatternString() const;
};
```

Sparse boundary matrix stored in associative map format. `computePersistencePairs`
runs the standard column-reduction algorithm (mod 2 coefficients by default).


### Internal storage

The matrix is stored as a vector of columns, each column being a sorted vector
of row indices where the coefficient is non-zero. This column-major format is
optimized for the reduction algorithm, which primarily traverses columns.

```text
Storage layout:
  columns_[0] = [row_a, row_b, ...]   // non-zero rows in column 0
  columns_[1] = [row_c, row_d, ...]   // non-zero rows in column 1
  ...
  filtration_values_[col] = birth_time_of_col_simplex
```


### Reduction algorithm

```
for col = 0..n_cols-1:
    while pivot(col) != -1 and pivot(col) is paired:
        add column(pivot(col)) to column(col)
    if pivot(col) != -1:
        pair(pivot(col), col)
```

The "add column" operation is XOR (mod 2 addition) of row index sets. This is
implemented as a merge of two sorted vectors, producing a new sorted vector
with symmetric difference elements.

**Pseudocode for column addition (mod 2):**
```text
function addColumn(col_target, col_source):
    i = j = 0
    result = []
    while i < len(col_target) and j < len(col_source):
        if col_target[i] < col_source[j]:
            result.push(col_target[i]); i++
        elif col_target[i] > col_source[j]:
            result.push(col_source[j]); j++
        else:
            // Equal entries cancel in mod 2
            i++; j++
    append remaining elements
    col_target = result
```


### Pivot computation

The pivot of a column is its largest row index (the lowest row in the matrix
with a non-zero entry). For mod 2 coefficients, this corresponds to the
last non-zero element in the sorted row-index vector.

```cpp
Index findPivot(const std::vector<Index>& column) const {
    if (column.empty()) return -1;
    return column.back();
}
```


### Persistence pair interpretation

Pairs returned as `(row_idx, col_idx)` in the matrix. Convert to (birth, death)
by looking up filtration values at those indices:

```python
def pairs_to_diagram(bm, pairs):
    diagram = []
    for row, col in pairs:
        birth = bm.getFiltrationValue(row)
        death = bm.getFiltrationValue(col)
        dim = bm.getRowSimplexDimension(row)  # dimension of the feature
        diagram.append((dim, birth, death))
    return diagram
```

The row index corresponds to the simplex that creates the homology class (birth),
and the column index corresponds to the simplex that destroys it (death).


### Essential cycles

Columns whose pivot rows are not paired correspond to essential cycles --
homology classes that persist to infinity. These are found by
`findEssentialCycles()` which returns the column indices of unpaired columns
after reduction.

```python
essentials = bm.findEssentialCycles()
for col in essentials:
    birth = bm.getFiltrationValue(col)
    print(f"Essential class: birth={birth}, death=inf")
```


### Filtration values vs matrix structure

The boundary matrix stores the filtration value for each column (simplex).
These values determine the order of reduction, which proceeds in increasing
order of filtration. The matrix rows and columns are sorted by filtration
value internally.

```cpp
void sortColumnsByFiltration() {
    // Reorder columns so filtration values are non-decreasing
    // This is essential for correct persistence computation
}
```


### Linear algebra operations

`multiply(v)` computes B * v (boundary applied to a chain) in O(nnz). `transposeMultiply(v)` computes B^T * v (coboundary applied to a cochain) in O(nnz). `applyBoundary(chain)` is an alias for multiply. `applyCoboundary(cochain)` is B^T * cochain in O(nnz). `transpose()` builds B^T as a new BoundaryMatrix in O(nnz).


### Practical usage

```python
from pynerve.algebra import SimplicialComplex, Simplex, BoundaryMatrix

# Build a 2-sphere (hollow tetrahedron)
sphere = SimplicialComplex()
for v in range(4):
    sphere.add_simplex(Simplex([v]))

# Add 6 edges
edges = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
for e in edges:
    sphere.add_simplex(Simplex(e))

# Add 4 triangles
triangles = [(0,1,2), (0,1,3), (0,2,3), (1,2,3)]
for t in triangles:
    sphere.add_simplex(Simplex(t))

# Boundary matrix for dimension 2 (triangles -> edges)
bm2 = BoundaryMatrix(sphere, dimension=2)
print(f"Rows (edges): {bm2.rows()}, Cols (triangles): {bm2.cols()}")

# Apply boundary to a chain
chain = [1.0, 0.0, 0.0, 0.0]  # only first triangle
bdy = bm2.applyBoundary(chain)
print(f"Boundary of triangle 0 has {sum(1 for x in bdy if abs(x) > 0)} edges")
```


### Common pitfalls

1. **Dimension mismatch**: The matrix for dimension k has (k-1)-simplices as
   rows and k-simplices as columns. Attempting to build a matrix for
   dimension 0 returns an empty matrix (there is no B_0).

2. **Filtration monotonicity**: If filtration values are not monotone
   (faces have higher values than cofaces), the persistence pairs will
   be incorrect. Use `sortColumnsByFiltration()` to enforce ordering.

3. **Mod 2 coefficients**: The default implementation uses mod 2 arithmetic.
   For integer or field coefficients, use `setCoefficientType()` or the
   templated variant `BoundaryMatrix<Field>`.

4. **Memory for dense matrices**: Very dense boundary matrices (>10%
   non-zero) may be more efficiently stored as dense arrays. Use
   `sparsityRatio()` to check.


### Cross-references

- `pynerve.algebra.ChainComplex`: constructs all boundary matrices at once
- `pynerve.persistence`: reduction algorithms operating on boundary matrices
- `pynerve.spectral.HodgeTheory`: uses boundary matrices for Hodge decomposition
- `pynerve.dmt`: uses boundary structure for Morse gradient computation
- `pynerve.torch.BoundaryMatrix`: GPU-accelerated TorchScript variant

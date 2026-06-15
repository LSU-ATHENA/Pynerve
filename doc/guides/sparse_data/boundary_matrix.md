# Sparse boundary matrix

Sparse complexes use a **CSR-format boundary matrix**. In terms of storage, full VR has O(n^{d+1}) boundary entries while sparse VR has O(k^{d+1}) for k << n. Memory consumption ranges from terabytes for full VR at n=10^5, d=3 down to gigabytes for sparse VR on the same input. Construction time is O(n^2) distance computations for full VR versus O(k * n) witness-distance computations for sparse VR.

### CSR boundary matrix format specification

```cpp
// The sparse boundary matrix is stored in CSR (Compressed Sparse Row) format.
// Each column of the boundary matrix is a row in CSR.

struct SparseBoundaryMatrix {
    // col_ptr: size (num_columns + 1)
    //   col_ptr[i] is the start index of column i's non-zero entries in indices
    //   col_ptr[i+1] - col_ptr[i] is the number of entries in column i
    std::vector<size_t> col_ptr;

    // indices: all non-zero row indices for each column, concatenated
    //   indices[col_ptr[i] : col_ptr[i+1]] are the vertices of simplex i
    //   (for Z2 coefficients, presence implies coefficient = 1)
    std::vector<Index> indices;

    // Optionally: values for non-Z2 fields
    // std::vector<Field> values;

    size_t num_columns() const { return col_ptr.size() - 1; }
    size_t nnz() const { return col_ptr.back(); }
};
```

### Column operations during reduction

Column operations during reduction iterate only over stored entries:

```
Algorithm: reduce_column(col_idx, pivot_column)

For sparse columns a and b, the XOR (Z2 addition) is:
  result = symmetric_difference(a.indices, b.indices)
  // Merge the two sorted index arrays:
  i = j = 0
  while i < a.size() and j < b.size():
      if a[i] < b[j]: result.push_back(a[i]); i++
      elif b[j] < a[i]: result.push_back(b[j]); j++
      else: i++; j++  // matched entries cancel (1+1 = 0 mod 2)
  // Append remaining entries
  while i < a.size(): result.push_back(a[i++])
  while j < b.size(): result.push_back(b[j++])
```

This is O(nnz(a) + nnz(b)) instead of O(dimension) for dense columns.

### Memory complexity at each scale

At a landmark scale of 0.1% (100 landmarks, ~10^3 boundary entries), memory usage is a few kilobytes. At 1% (1,000 landmarks, ~10^6 boundary entries), memory usage is a few megabytes. At 5% (5,000 landmarks, ~10^8 boundary entries), memory usage is hundreds of megabytes to nearly a gigabyte. At 10% (10,000 landmarks, ~10^9 boundary entries), memory usage is a few gigabytes.

```python
from pynerve.fast_ops import boundary_matrix_sparse, column_reduction_sparse

# Build CSR boundary matrix
boundary = boundary_matrix_sparse(simplices, max_dim=2, max_radius=1.0)

# Direct sparse reduction (bypasses VR)
pairs = column_reduction_sparse(boundary)
```

Back to [Sparse Workflows Overview](index.md)

# Internal Implementation Details

[Back to Index](index.md)

### Sparse Coboundary Construction

The coboundary matrix is not explicitly constructed. Instead, it is built on-the-fly from the boundary matrix:

```cpp
// Internal: compute coboundary on demand
Column getCoboundary(size_t simplex_idx) {
    // For each (d+1)-simplex that contains this simplex,
    // add its index to the coboundary column
    Column result;
    const auto& simplex = simplices[simplex_idx];
    int d = simplex.dimension();

    // Iterate over all (d+1)-simplices that have this simplex as a face
    // This uses an index that maps each simplex to its cofaces
    for (auto coface : coface_index[simplex_idx]) {
        result.push_back(coface);
    }
    return result;
}
```

The coface index is built during filtration construction:

```cpp
// Build coface index for fast coboundary access
void buildCofaceIndex() {
    coface_index.resize(num_simplices);
    for (size_t j = 0; j < num_simplices; ++j) {
        int d = simplices[j].dimension();
        if (d == 0) continue;  // vertices have no boundary
        // For each face of simplex j, add j to that face's coface list
        for (auto face : faces_of(j)) {
            coface_index[face].push_back(j);
        }
    }
}
```

This construction is O(n * avg_faces) and has memory cost O(n * avg_cofaces).

### Bit-Packed Column Storage

```cpp
struct BitPackedColumn {
    // For columns <= 64 rows: single uint64_t
    // For larger columns: pointer to aligned array of uint64_t words
    union {
        uint64_t inline_bits;
        uint64_t* words;  // heap-allocated, cache-line-aligned
    };
    uint32_t size;     // number of rows this column represents
    uint32_t nwords;   // number of 64-bit words
    bool is_inline;    // true if using inline_bits
    int32_t pivot;     // cached pivot, or -1 if empty

    // Bit operations
    bool getbit(uint32_t row) const;
    void setbit(uint32_t row);
    void xorWith(const BitPackedColumn& other);
    int32_t findPivot() const;
};
```

The inline representation (<= 64 rows) avoids heap allocation and is the common case for cohomology columns of high-dimensional simplices. The external representation (heap-allocated, cache-line-aligned words) is used for larger columns (typically vertices and low-dimensional simplices processed late in the algorithm).

### Clearing in Cohomology

Cohomology clearing is the dual of homology clearing:

- **Standard clearing**: After pairing (p, j) where p is the pivot of column j, clear column p (the birth column in homology terms).
- **Cohomology clearing**: After pairing (j, p) where p is the pivot of column j (in coboundary terms), clear column p (the death column in cohomology terms, which corresponds to the birth in homology).

The effect is the same: the column is set to zero and skipped in future reduction. The dimension-cascading variant applies recursively:

```cpp
void clearColumn(size_t col_idx) {
    columns[col_idx].clear();
    cleared_count++;

    // Dimension-cascading: if col_idx was itself a death column,
    // also clear the birth column it was paired with
    if (pair_of[col_idx] != -1) {
        clearColumn(pair_of[col_idx]);
    }
}
```

### Coherence Between Warps (GPU)

When multiple warps process columns on the GPU, the main synchronization challenge is the pivot table. Each warp reads and writes the pivot table entries for its column's current pivot. Two warps might try to claim the same pivot simultaneously.

The solution: the pivot table is partitioned by dimension, and each dimension's pivot entries are processed by a fixed set of warps. Since the cohomology algorithm checks pivots only in dimension d+1 (where d is the current column's dimension), and columns are processed in decreasing dimension order, conflicts are naturally minimized:

- When processing dim-2 columns, only dim-2 pivot entries are accessed.
- When processing dim-1 columns (later), only dim-1 pivot entries are accessed.
- No two warps process columns of the same dimension simultaneously (the columns of a given dimension are too few for this to be a bottleneck in practice).

If needed, atomic compare-and-swap on the pivot table entry provides a fallback:

```cuda
int expected = -1;
int desired = column_idx;
if (atomicCAS(&pivotTable[pivot], expected, desired) == expected) {
    // This warp successfully claimed the pivot
} else {
    // Need to eliminate conflict: add the claiming column
    addColumn(column_idx, pivotTable[pivot]);
}
```

But in practice, the deterministic lock-free path (using `__shfl_sync` for pivot arbitration within a warp) handles all cases without atomics.

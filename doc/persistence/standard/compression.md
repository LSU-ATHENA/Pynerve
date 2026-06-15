# Compression

Compression reduces work by skipping columns that are already fully reduced. Two forms are used:

### Skipped Columns (Column Filtering)

Columns that are zero (after clearing or initial emptiness) are skipped entirely:

```python
# Pseudocode: compression check
for j in range(n):
    if is_column_empty(R, j):
        continue  # skip: already reduced or cleared
    # ... normal reduction ...
```

In sparse filtrations, 30-50% of columns are initially zero (e.g., all vertices have empty boundary columns). With clearing, many more become zero during reduction.

### Trailing Zero Stripping

Partially reduced columns are stored without trailing zeros. After each XOR operation, the column's stored length may shrink if the highest 1 moves lower:

```
Column before XOR:  [1, 0, 1, 0, 1]  pivot = 5 (1-indexed)
Column after XOR:   [1, 0, 0, 0, 0]  pivot = 1
Trailing zeros:     [0, 0, 0, 0] can be stripped
Stored as:          [1]  with length_hint = 1
```

This optimization is especially beneficial for pivot finding, which scans from the end of the column. Shorter columns mean faster pivot lookups.

### Sparse Column Representation

Both forms of compression work naturally with sparse column storage:

```cpp
// C++ internal representation
struct Column {
    std::vector<size_t> nonzero_rows;  // sorted ascending
    size_t pivot;                       // cached, or nonzero_rows.back()
    bool cleared;                       // compression flag
};
```

When `cleared = true`, the column is skipped in the main reduction loop. The `nonzero_rows` vector stores only the indices where ones appear. Trailing zero stripping is automatic: after an XOR, if the last few indices are removed, the vector shrinks.

<- [Standard Reduction Overview](index.md)

# Memory layout: cache-aware boundary matrix

The boundary matrix uses a **compressed sparse row (CSR)** format designed for
cache-efficient reduction:

```
CSR layout:
  row_ptr:    [0, 2, 5, 9, ...]           // offset into col_ind for each row
  col_ind:    [3, 7, 0, 4, 9, 1, 5, ...]  // column indices (simplex IDs)
  col_ptr:    [0, 3, 7, ...]               // alternative: column offsets
  values:     [1, 1, 1, 1, ...]            // coefficient (always 1 for Z/2)
```

Key properties:

- **Cache-line aligned.** The row and column arrays are 64-byte aligned,
  ensuring that adjacent rows fit in the same cache line.
- **NUMA-aware allocation.** For multi-socket systems, memory is allocated
  on the local NUMA node of the reducing thread.
- **Prefetch streaming.** The reduction kernel issues software prefetch
  instructions 3-4 cache lines ahead of the current column position.
- **Sparse elimination.** Zero columns (columns whose lowest-one is above
  the diagonal) are marked as "cleared" and skipped during reduction,
  implementing the clearing optimization.

### Memory modes

```python
from pynerve.nn import PersistentHomology

# Standard: all in RAM (default)
ph = PersistentHomology(max_dim=2, memory_mode="standard")

# Memory-mapped: spill boundary matrix to disk
ph = PersistentHomology(max_dim=3, memory_mode="memory_mapped")

# Streaming: construct and reduce one column at a time
ph = PersistentHomology(max_dim=2, memory_mode="streaming")

# Extreme: bounded-memory with disk-backed swap
ph = PersistentHomology(max_dim=3, memory_mode="extreme", max_memory_gigabytes=2.0)
```

The streaming mode avoids storing the full boundary matrix by computing
each column's boundary on the fly. For extreme memory constraints, set
`max_memory_gigabytes` to bound resident memory; excess columns are swapped to
a temporary file.


<- [Vietoris-Rips Overview](index.md)

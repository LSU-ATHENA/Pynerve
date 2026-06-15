# Column distribution for parallel reduction

The boundary matrix columns are distributed across ranks via `ShardedBoundaryMatrix`:

```cpp
// src/distributed/sharded_boundary_matrix.cpp
class ShardedBoundaryMatrix {
    void distribute_columns(const std::vector<Column>& columns);
    Column get_boundary(Index column);
    DistributedResult distributed_reduce();
    void checkpoint(const std::string& path);
    void restore(const std::string& path);
};
```

### Distribution strategy

Three strategies are available for column distribution. Round-robin assigns columns by `index % world_size` and is best for balanced load with uniform columns. Chunked assigns contiguous blocks per rank and is best for locality-sensitive reduction. Work-stealing provides dynamic load balancing and is best for skewed column sizes.

### Column reduction flow

```
Rank 0 columns     Rank 1 columns     Rank 2 columns     Rank 3 columns
    |                   |                   |                   |
    v                   v                   v                   v
  Local reduce        Local reduce        Local reduce        Local reduce
    |                   |                   |                   |
    `-------------------.-------------------.-------------------'
                              v
                     Pivot exchange (Allgather)
                              v
    .-------------------.-------------------.-------------------.
    v                   v                   v                   v
  Cross-rank pivot     Cross-rank pivot    Cross-rank pivot    Cross-rank pivot
  resolve             resolve              resolve             resolve
```

<- [Distributed Computing Overview](index.md)

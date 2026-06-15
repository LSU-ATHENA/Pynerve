# CPU microarchitecture tuning

### Cache hierarchy considerations

The L1d cache is typically tens of kilobytes per core with 4-5 cycle latency, used for distance temp arrays and reduction worksets. The L2 cache ranges from hundreds of kilobytes to about a megabyte per core with 12-15 cycle latency, used for column reduction buffers and filtration sort. The L3 cache is tens of megabytes shared with 30-50 cycle latency, used for distance matrix chunks and the simplex pool. RAM ranges from tens to hundreds of gigabytes with 100-200 ns latency, used for the distance matrix and boundary matrix.

Pynerve's data structures are sized to fit in L1/L2 for n < 1,000 and in L3 for n < 10,000.

### Prefetching

Pynerve issues software prefetch instructions (`_mm_prefetch`) for:

1. Distance matrix rows during column reduction (prefetch next column's data)
2. Simplex vertex data during boundary matrix construction
3. Sparse CSR index arrays during reduction iteration

### Branch prediction hints

Hot loops use `__builtin_expect` (or `[[likely]]`/`[[unlikely]]`) to guide branch prediction:

```cpp
// Example from matrix reduction inner loop
if (__builtin_expect(pivot < lowest_pivot, 0)) {
    // Rare: pivot reset
    reset_column();
}
```

### Alignment

All dynamically allocated arrays are 64-byte aligned (cache line boundary):

```cpp
// Allocation with alignment
void* ptr = std::aligned_alloc(64, size);
// Or using Pynerve's aligned allocator
auto buf = pool.allocate(size, /* alignment = */ 64);
```

[Back to index](index.md)

# Memory scaling

### Dense vs sparse break-even

Dense VR for n of 1,000 uses a few megabytes, while sparse VR with 1% landmarks uses under a megabyte, making dense the winner with no overhead. At n of 10,000, dense uses hundreds of megabytes versus less than a megabyte for sparse, so sparse wins on memory. At n of 100,000, dense uses tens of gigabytes while sparse uses a few megabytes, making sparse the only option. At n of 1,000,000, dense would require terabytes (not feasible) versus tens of megabytes for sparse. The break-even point is that dense VR is faster for n under 5,000, while sparse VR wins for n over 10,000 in memory and n over 50,000 in time due to cache behavior.

In terms of memory and time complexity by operation, distance computation is O(n^2) memory dense versus O(k*n) sparse, with time O(n^2*d) versus O(k*n*d), breaking even around n of 5,000. Filtration is O(n^{d+1}) memory dense versus O(k^{d+1}) sparse, breaking even around n of 3,000. Reduction is O(n^{2d+2}) memory dense versus O(k^{2d+2}) sparse, breaking even around n of 10,000. For d=2, total complexity is O(n^3) dense versus O(k^3) sparse, with break-even around n of 5,000.

Pynerve uses a hierarchy of allocators. The global allocator is mimalloc, which is general-purpose and used for all host allocations. The hugepage pool operates at the NUMA domain level, using large pages via mmap and MAP_HUGETLB for large arrays. The RawArrayPool is per-thread, using a lock-free bump-allocated arena for temporary reduction buffers.

```python
# Hugepage allocation is automatic for arrays larger than a megabyte.
# No user configuration needed -- Pynerve selects based on size.
```

### Allocator behavior

**mimalloc (global)**
- Default allocator for all C++ allocations
- Fast path: thread-local free lists, no locking
- Slow path: per-page free lists with CAS
- Patch level: mimalloc v2.1+

**Hugepages**
- Enabled automatically for allocations larger than a megabyte
- Uses `/sys/kernel/mm/hugepages/` -- requires kernel configuration
- Falls back to small pages if hugepages unavailable
- Medium pages for most workloads; large pages for very large arrays

**RawArrayPool**
- Per-thread bump allocator for temporary reduction arrays
- Arena released after each `compute_persistence` call
- No bookkeeping -- just a pointer bump
- Size: tens of kilobytes per thread, grown on demand

[Back to index](index.md)

# Memory Management

## Quick start

```python
import pynerve.memory as mem

pool = mem.RawArrayPool(initial_bytes=64 * 1024 * 1024)
buf = pool.allocate(4096)
pool.deallocate(buf, 4096)

slab = mem.SlabAllocator(256)
obj = slab.allocate()
slab.deallocate(obj)
```

Pynerve provides a multi-layered memory subsystem: per-thread slab allocators
for small objects, bump allocators for contiguous arrays, NUMA-aware
allocators, and hugepage-backed pools. All are designed for the high-allocation
patterns of persistence computation.


## Topics

- **[slab_allocator.md](slab_allocator.md)** -- Lock-free slab allocator, thread-local pools, RAII pooled pointers
- **[raw_array_pool.md](raw_array_pool.md)** -- Bump allocator for contiguous arrays, hugepage support, size classes
- **[numa.md](numa.md)** -- NUMA-aware allocation, per-node pools, topology manager
- **[tracking.md](tracking.md)** -- Global allocation counting, peak tracking, core memory pool


## Architecture

```
                      Global / Thread-local allocation tracking
                                         │
         ┌───────────────────────── ─────┼──────────────────────────────┐
         │                               │                              │
    GlobalPagePool               SlabAllocator<T>              NumaAwareAllocator
    (hugepage recycling)     (lock-free CAS free-list)            (libnuma)
         │                               │                              │
         │                         ThreadLocalPool<T>           ┌───────┴───────┐
         │                        (per-thread singleton)   NumaAware    NumaPool
         │                               │                  MemoryPool   Manager
         │                          PooledPtr<T>
         │                        (RAII via ThreadPool)      PoolBackedVector<T>
         │                                                    (backed by RawArrayPool)
    RawArrayPool ───── SizeClassAllocator
    (bump allocator)      (16 size classes)
```


## Which allocator to use

- **`SlabAllocator<T>`** is designed for fixed-size objects (simplices, cells) with many alloc/free cycles of the same type.
- **`RawArrayPool`** is designed for variable-size byte arrays that are allocated once and freed at end.
- **`SizeClassAllocator`** is a general-purpose malloc replacement for mixed sizes smaller than hundreds of kilobytes.
- **`NumaAwareAllocator`** provides NUMA-local allocation for multi-socket systems.
- **`ThreadLocalPool<T>`** provides per-thread allocations with no synchronization overhead.
- **`GlobalPagePool`** handles large allocations and hugepages, serving as backing memory for other pools.


### Choosing a pool

```python
import pynerve.memory as mem

# Small, fixed-size objects (e.g., graph nodes)
slab = mem.SlabAllocator(size=64)  # 64-byte objects

# Large contiguous arrays (e.g., distance matrices)
pool = mem.RawArrayPool(initial_bytes=256 * 1024 * 1024)

# Mixed sizes (general purpose)
salloc = mem.SizeClassAllocator()

# NUMA-local allocation
numa = mem.NumaAwareAllocator(preferred_node=0)

# Per-thread allocation
tl = mem.ThreadLocalPool()
```


## Complexity notes

All allocation and deallocation operations are O(1). `SlabAllocator::allocate` and `deallocate` use CAS on the free-list head. `RawArrayPool::allocate` uses an atomic bump, and `deallocate` is a no-op except for hugepage return. `SizeClassAllocator::allocate` indexes directly into the size class array. `ThreadLocalPool::allocate` needs no synchronization. `NumaAwareAllocator::allocate` calls `numa_alloc_onnode`. `GlobalPagePool::allocatePage` uses a lock-free list with mmap fallback. `PoolBackedVector::push_back` is O(1) amortized via pool allocation.

**Memory overhead per pool:**
- RawArrayPool uses the full `initial_bytes` immediately (virtual, not
  physical until touched).
- SlabAllocator allocates a slab (T * SlabCapacity) on first allocation.
- SizeClassAllocator creates each RawArrayPool lazily.


### Common pitfalls

1. **RawArrayPool deallocation is a no-op**. Memory is only reclaimed on
   `reset()` or destruction. Do not use for long-lived mixed-size allocations.

2. **SlabAllocator requires fixed-size types**. Objects must all be the
   same size. Use SizeClassAllocator for variable sizes.

3. **NUMA allocation falls back to malloc** when NUMA is unavailable.
   Always check for NUMA availability first.

4. **Hugepages require OS configuration**. Use `echo 100 > /proc/sys/vm/nr_hugepages`
   on Linux to enable.

5. **ThreadLocalPool objects must be freed by the allocating thread**.
   Cross-thread deallocation is undefined.


## FAQ

**Why is RawArrayPool deallocation a no-op?**
The bump allocator design avoids per-allocation metadata overhead by not tracking individual allocations. Memory is reclaimed all at once via `reset()` or destruction, which is ideal for phased algorithms where all allocations share the same lifetime.

**How do I choose between SlabAllocator and SizeClassAllocator?**
Use `SlabAllocator` when all objects are the same fixed size (e.g., graph nodes, cells) and you need the fastest possible alloc/free. Use `SizeClassAllocator` when sizes vary but are under hundreds of kilobytes -- it provides 16 size classes with O(1) allocation.

**Does the memory subsystem work without NUMA or hugepages?**
Yes. NUMA allocation falls back to `malloc` when libnuma is unavailable, and `GlobalPagePool` falls back to 4 KB pages when hugepages are exhausted. All allocators degrade gracefully.


### Cross-references

- `pynerve.core`: CPU topology detection for NUMA
- `pynerve.algebra`: Uses slab allocators for simplex construction
- `pynerve.persistence`: Uses RawArrayPool for boundary matrices
- `pynerve.dmt`: Uses SlabAllocator for gradient field storage

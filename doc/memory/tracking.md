## Global allocation tracking

```cpp
Size getGlobalAllocationCount();
Size getGlobalDeallocationCount();
Size getGlobalCurrentBytes();
Size getGlobalPeakBytes();

void trackAllocEvent(Size bytes);
void trackFreeEvent(Size bytes);
void resetGlobalMemoryStats();
Size getSlabAllocatorDiagnosticCount();
double estimateMemoryOverhead();
```

Tracks total bytes allocated, peak usage, and allocation/deallocation counts.
Use for diagnostics and leak detection. Not thread-safe for the global
counters (use only in single-threaded sections or with external locking).


### When to use tracking

For memory leak detection, compare `getGlobalAllocationCount()` against `getGlobalDeallocationCount()`. For peak memory profiling, use `getGlobalPeakBytes()`. For overhead estimation, call `estimateMemoryOverhead()`. For slab fragmentation, use `getSlabAllocatorDiagnosticCount()`.

```python
from pynerve.memory import (
    get_global_allocation_count,
    get_global_current_bytes,
    get_global_peak_bytes,
)

# Before computation
start_bytes = get_global_current_bytes()

# Run computation
result = compute_persistence(points)

# After computation
end_bytes = get_global_current_bytes()
peak = get_global_peak_bytes()
print(f"Allocated: {end_bytes - start_bytes} bytes, peak: {peak} bytes")

# Check for leaks
assert get_global_allocation_count() == get_global_deallocation_count()
```


## Core memory pool

```cpp
namespace nerve::core {

struct MemoryBlock {
    void* ptr;
    std::size_t size;
    std::size_t alignment;
    bool in_use;
};

class MemoryPool {
public:
    explicit MemoryPool(std::size_t pool_size = kDefaultMemoryPoolSize);

    void* allocate(std::size_t size, std::size_t alignment = alignof(std::max_align_t));
    void deallocate(void* ptr, std::size_t size) noexcept;

    std::size_t allocated() const noexcept;
    std::size_t capacity() const noexcept;
    std::size_t numBlocks() const noexcept;
    double fragmentationRatio() const noexcept;

    void defragment();
    void reset();

    bool contains(void* ptr) const noexcept;
    std::vector<MemoryBlock> getBlocks() const;

    void setDeterminismContract(const DeterminismContract& contract);
    bool isDeterministic() const noexcept;
};

}
```

Block-based pool with free-list coalescing. Supports determinism contracts
for reproducible memory layouts. `defragment()` coalesces adjacent free
blocks.


### Block list example

```python
pool = MemoryPool(1024 * 1024)  # a megabyte

a = pool.allocate(128)
b = pool.allocate(256)
pool.deallocate(a, 128)
pool.deallocate(b, 256)

blocks = pool.get_blocks()
# Two free blocks adjacent -- defragment will merge them
pool.defragment()
blocks = pool.get_blocks()
# One combined free block (size 384 + overhead)
```


### Determinism contracts

```python
from pynerve.validation import DeterminismContract

pool = MemoryPool(1024 * 1024)
pool.set_determinism_contract(DeterminismContract(
    level=DeterminismLevel.STRICT,
    seed=42,
))
```

When deterministic, the pool produces the same sequence of block addresses
for the same allocation pattern, enabling bitwise-reproducible computation.


## FAQ

**Are the global tracking counters thread-safe?**
No. The global counters are not thread-safe by design. Use them only in single-threaded sections or with external locking. For per-thread tracking, rely on the thread-local allocator statistics.

**How can I detect memory leaks with tracking?**
Compare `getGlobalAllocationCount()` with `getGlobalDeallocationCount()` after computation completes. If they differ, some allocations were never freed. You can also monitor `getGlobalPeakBytes()` to identify unexpectedly high memory usage.

**What does `estimateMemoryOverhead()` measure?**
It estimates the overhead of internal metadata (free-list nodes, slab headers, pool bookkeeping) relative to user-allocated bytes. Useful for tuning pool sizes and understanding the true memory footprint.


### Cross-references

- `pynerve.memory.memory`: Memory management overview
- `pynerve.core`: Topology detection for allocation strategies
- `pynerve.validation.determinism`: Determinism contracts

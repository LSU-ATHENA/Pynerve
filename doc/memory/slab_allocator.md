## Slab allocator

```cpp
namespace nerve::memory {

template <typename T, Size SlabCapacity = 256>
class SlabAllocator {
public:
    SlabAllocator();
    ~SlabAllocator();
    SlabAllocator(const SlabAllocator&) = delete;
    SlabAllocator& operator=(const SlabAllocator&) = delete;

    T* allocate();
    void deallocate(T* ptr);

    void prefetchSlab();
    void reset();
    Size capacity() const;
    Size available() const;
};

}
```

Lock-free slab allocator using CAS (compare-and-swap) on a free-list of
indices. Each slab holds `SlabCapacity` objects of type T. Allocation and
deallocation are O(1) and do not block.


### Internal structure

```
slab = [T(0), T(1), T(2), ..., T(SlabCapacity - 1)]
free_list: stack of available indices, e.g. [2, 5, 1, ...]
                                        ^ head (CAS target)
```

Allocation:
1. Atomically pop index from free_list head
2. Return &slab[index]

Deallocation:
1. Atomically push index onto free_list

When the free_list is empty, a new slab is allocated (atomics only, no mutex).


### CAS implementation

```cpp
T* allocate() {
    Index expected = free_list_head_.load(std::memory_order_relaxed);
    while (true) {
        if (expected == SENTINEL) {
            // Free list empty -- allocate new slab
            allocateNewSlab();
            expected = free_list_head_.load(std::memory_order_relaxed);
            continue;
        }
        Index next = free_list_[expected].next;
        if (free_list_head_.compare_exchange_weak(expected, next,
                std::memory_order_acquire,
                std::memory_order_relaxed)) {
            break;
        }
    }
    return reinterpret_cast<T*>(slab_base_ + expected * sizeof(T));
}
```


### Thread-local pool

```cpp
namespace nerve::memory {

template <typename T, Size SlabCapacity = 256>
class ThreadLocalPool {
public:
    static ThreadLocalPool& instance();

    T* allocate();
    void deallocate(T* ptr);
    void prefetch();

    SlabAllocator<T, SlabCapacity>& getLocalAllocator();
};

template <typename T>
class PooledPtr {
public:
    PooledPtr(T* ptr);
    ~PooledPtr();
    PooledPtr(PooledPtr&& other) noexcept;
    PooledPtr(const PooledPtr&) = delete;

    T& operator*();
    T* operator->();
    T* get();
    T* release();
    explicit operator bool() const;
};

template <typename T, typename... Args>
PooledPtr<T> makePooled(Args&&... args);

}
```

`ThreadLocalPool<T>` wraps `SlabAllocator<T>` in a `thread_local` singleton.
Each thread has its own slab, so no synchronization is needed for allocations
on the fast path.

`PooledPtr<T>` is a RAII unique-pointer that returns the object to the
thread's local pool on destruction. Use `makePooled<T>(args...)` to construct
in-place.

```cpp
struct Cell { int x, y, z; };
auto ptr = makePooled<Cell>(1, 2, 3);
// ptr is returned to pool when out of scope
```

### Python

```python
import pynerve.memory as mem

slab = mem.SlabAllocator(256)
obj = slab.allocate()
slab.deallocate(obj)
```


### When to use

For scenarios with many allocations of the same type, `SlabAllocator` is 10-100x faster than `malloc/free`. For mixed sizes it is not suitable -- use `SizeClassAllocator` instead. For multithreaded per-type pooling, `ThreadLocalPool` avoids mutex contention. For short-lived temporaries, `PooledPtr` with RAII eliminates manual `free` calls.


## FAQ

**What happens when the free list is empty?**
A new slab is allocated automatically using atomics only -- no mutex involved. The `SlabAllocator` grows dynamically as needed by allocating additional slabs of `SlabCapacity` objects each.

**Can I free a PooledPtr on a different thread?**
No. `PooledPtr` returns the object to the thread's local pool on destruction. Cross-thread deallocation is undefined behavior. Use `ThreadLocalPool` semantics: the allocating thread must also deallocate.

**How does SlabAllocator compare to malloc for small objects?**
For fixed-size objects, `SlabAllocator` is 10-100x faster because allocation is a single CAS operation on a free-list index, with no metadata lookup, no page walk, and no system call overhead.


### Cross-references

- `pynerve.memory.memory`: Memory overview
- `pynerve.memory.raw_array_pool`: Complement for variable-size allocations
- `pynerve.dmt.parallel`: Uses slab allocator for Morse cells

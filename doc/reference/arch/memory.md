# Memory Model

Pynerve uses a multi-tier memory hierarchy designed for zero-fragmentation, NUMA locality, and GPU residency:

The memory hierarchy is structured as a layered tree. At the root, **GlobalPagePool** allocates hugepage-backed slabs (megabytes each) from the OS. Below it, **SlabAllocator\<T\>** provides lock-free fixed-size allocation via CAS on a free list. Each thread accesses its own **ThreadLocalPool\<T\>** singleton, which caches slabs from SlabAllocator. User code receives **PooledPtr\<T\>** RAII handles that automatically return to the thread-local pool on destruction. Alongside this chain, **RawArrayPool** provides a linear bump allocator for temporary arrays, **NumaAwareAllocator** allocates on specific NUMA nodes via libnuma, **SizeClassAllocator** offers 16 size classes for general-purpose mixed-size allocation, and **DeviceMemoryPool** wraps cudaMalloc/cudaFree for GPU memory.

### GlobalPagePool

Singleton that manages hugepage allocations of a few megabytes from the OS. Used as the backing store for slab allocators. Pages are allocated on demand and never returned to the OS during the session.

```cpp
// Single global instance
auto& pool = nerve::memory::GlobalPagePool::instance();
void* page = pool.allocatePage();       // megabytes
pool.deallocatePage(page);
size_t pages = pool.pagesAllocated();   // diagnostic
```

### SlabAllocator\<T\>

Lock-free slab allocator using a CAS free list. Each slab holds `SlabCapacity` objects of type `T`. Allocations are O(1) with no metadata overhead per object.

```cpp
template <typename T, Size SlabCapacity = 256>
class SlabAllocator {
    T* allocate();      // lock-free CAS
    void deallocate(T*); // lock-free CAS
};
```

### ThreadLocalPool\<T\>

Per-thread pool backed by `SlabAllocator`. The singleton pattern (`ThreadLocalPool::instance()`) creates one pool per thread on first access. Provides `UniquePtr` (via custom deleter) and `makeUnique()` for RAII allocation.

```cpp
auto& pool = nerve::memory::ThreadLocalPool<Simplex>::instance();
auto ptr = pool.allocate();          // raw pointer
auto unique = pool.makeUnique();     // PooledPtr<T>, RAII
```

### RawArrayPool

Linear bump allocator for temporary arrays during a single computation. O(1) allocate, O(1) reset. No per-element deallocation -- the entire pool resets at once.

```cpp
nerve::memory::RawArrayPool pool;
void* buf = pool.allocate(1024 * sizeof(double));
// ... use buf ...
pool.reset();  // reclaims all memory
```

### NumaAwareAllocator

Allocates memory on a specific NUMA node via `megabytesind` / `libnuma`. Used by `NumaAwareMemoryPool` for thread-local allocations that stay local to the memory controller.

```cpp
nerve::memory::NumaAwareAllocator alloc(/* numa_node = */ 0);
void* buf = alloc.allocate(4096, /* alignment = */ 64);
alloc.deallocate(buf, 4096);
```

### DeviceMemoryPool (GPU)

Thin wrapper around `cudaMalloc`/`cudaFree` with pool semantics. Configurable initial and max sizes. No fragmentation -- allocations go directly to the CUDA driver.

```cpp
auto pool = nerve::gpu::DeviceMemoryPool::create();
void* gpu_buf = pool->allocate(1024 * sizeof(double));
auto typed = pool->allocateTyped<Pair>(256);
pool->deallocate(gpu_buf, 1024 * sizeof(double));
```

### Object lifecycle

A simplex moves through 7 stages: (1) `compute_persistence()` is called. (2) A simplex is constructed during VR filtration. (3) `ThreadLocalPool\<Simplex\>::instance().allocate()` calls `SlabAllocator\<Simplex\>::allocate()` which CAS-pops from the free list or, if empty, requests a new slab from GlobalPagePool. (4) The simplex is used in the boundary matrix. (5) After reduction completes, the simplex is destroyed. (6) `ThreadLocalPool\<Simplex\>::deallocate()` calls `SlabAllocator\<Simplex\>::deallocate()` which CAS-returns the object to the free list, keeping the slab in the pool for reuse. (7) `Pool::reset()` at the end of `compute_persistence()` frees all slabs back to GlobalPagePool.


[Back to Architecture Index](index.md)

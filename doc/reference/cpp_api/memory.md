# `nerve::memory` -- Memory Pools

### GlobalPagePool

```cpp
#include <nerve/memory/safe_memory_pool.hpp>

namespace nerve::memory;

class GlobalPagePool {
public:
    static GlobalPagePool& instance();
    void* allocatePage();                // megabytes hugepage
    void deallocatePage(void* page);
    size_t pagesAllocated() const;
    size_t hugetlbPagesAllocated() const;
};
```

**Cost:** O(1) allocation, O(1) deallocation. Pages are a few megabytes.

### SlabAllocator<T>

Lock-free slab allocator. O(1) allocate/deallocate, no per-object metadata.

```cpp
template <typename T, Size SlabCapacity = 256>
class SlabAllocator {
public:
    T* allocate();        // lock-free CAS
    void deallocate(T*);  // lock-free CAS
    void prefetchSlab();
    void reset();
    Size capacity() const;
    Size available() const;
};
```

**Cost:** O(1) allocate (single CAS), O(1) deallocate (single CAS). Zero pointer overhead per object.

### ThreadLocalPool<T>

Per-thread pool using `SlabAllocator`. Singleton per thread.

```cpp
template <typename T, Size SlabCapacity = 256>
class ThreadLocalPool {
public:
    static ThreadLocalPool& instance();

    ErrorResult<T*> allocate();
    ErrorResult<void> deallocate(T*);

    struct Deleter;
    using UniquePtr = std::unique_ptr<T, Deleter>;
    static UniquePtr makeUnique();

    Size capacity() const;
    Size available() const;
    Size allocated() const;
};
```

**Example:**

```cpp
using Pool = nerve::memory::ThreadLocalPool<Simplex>;
auto& pool = Pool::instance();
auto ptr = pool.makeUnique();
// ptr is std::unique_ptr<Simplex, Pool::Deleter>
```

### RawArrayPool

Bump allocator. O(1) allocate, O(1) reset.

```cpp
class RawArrayPool {
public:
    void* allocate(size_t bytes);       // O(1) bump
    void deallocate(void* ptr, size_t bytes);  // no-op (reset to reclaim)
    size_t totalAllocated() const;
    double peakUtilization() const;
    void reset();                        // reclaims all memory
};
```

### NumaAwareAllocator

```cpp
class NumaAwareAllocator {
public:
    explicit NumaAwareAllocator(int preferred_node = -1);
    ~NumaAwareAllocator();
    void* allocate(Size bytes, Size alignment);
    void deallocate(void* ptr, Size bytes);
    int getPreferredNode() const noexcept;
    void setPreferredNode(int node);
};
```

**Cost:** O(1) allocation via mbind. Allocation is on the specified NUMA node.

### DeviceMemoryPool (GPU)

```cpp
#include <nerve/gpu/gpu_memory_pool.hpp>

namespace nerve::gpu;

class DeviceMemoryPool {
public:
    static CudaResult<DeviceMemoryPool> create(size_t initial_size = 1ULL << 30);
    static CudaResult<DeviceMemoryPool> create(const Config& config);

    CudaResult<void*> allocate(size_t bytes, size_t alignment = 256);
    template <typename T> CudaResult<T*> allocateTyped(size_t count);
    CudaResult<void> deallocate(void* ptr, size_t bytes);
    template <typename T> CudaResult<void> deallocateTyped(T* ptr, size_t count);

    void reset();
    size_t bytesUsed() const;
    size_t bytesTotal() const;
    double utilization() const;
};
```

**Cost:** O(1) allocation via cudaMalloc wrapper. Initial size a few gigabytes, grows on demand.

<- [C++ API Overview](index.md)

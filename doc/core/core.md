# Core Infrastructure

Core utilities for parallelism, random number generation, and SIMD operations. These components support the persistence pipeline but are also available for direct use.

```python
import pynerve
from pynerve.core import ThreadPool, cpu_features

feats = cpu_features()
print(feats.avx512, feats.avx2, feats.sse4_1, feats.fma)

pool = ThreadPool(num_threads=8)
pool.map(my_function, data)
```

## What's included

| Module | Purpose |
|--------|---------|
| [Thread pool](thread_pool.md) | Per-core pinned workers, NUMA-aware scheduling, lock-free work-stealing |
| [RNG](rng.md) | Deterministic seeded PRNG (mt19937_64), thread-local instances |
| [SIMD ops](simd_ops.md) | memcpy, memset, reduce-sum with AVX-512/AVX2/SSE4.1 dispatch |

## API

### Python

```python
import pynerve.core as core

class ThreadPool:
    def __init__(self, num_threads=None, pin_threads=True): ...
    def map(self, func, iterable): ...
    def submit(self, func, *args): ...
    def shutdown(self): ...

class NumaThreadPool:
    def __init__(self, threads_per_node=None): ...
    def map_node(self, node_id, func, iterable): ...
    def get_node_count(self) -> int: ...

def cpu_features() -> CPUFeatures: ...
def pin_to_core(core_id: int) -> None: ...
def pin_to_numa_node(node_id: int) -> None: ...
```

### C++

```cpp
#include <pynerve/core/thread_pool.hpp>
#include <pynerve/core/affinity.hpp>
#include <pynerve/core/simd_ops.hpp>
#include <pynerve/core/features.hpp>

namespace nerve::core {

class ThreadPool {
    explicit ThreadPool(size_t num_threads = 0, bool pin = true);
    template <typename F, typename... Args>
    auto submit(F&& func, Args&&... args) -> std::future<decltype(func(args...))>;
    template <typename F, typename Iter>
    void map(F&& func, Iter begin, Iter end);
    void shutdown();
    size_t thread_count() const;
};

class NumaThreadPool {
    explicit NumaThreadPool(int threads_per_node = -1);
    size_t node_count() const;
    template <typename F, typename Iter>
    void map_node(int node_id, F&& func, Iter begin, Iter end);
};

class SeededRNG {
    using result_type = uint64_t;
    explicit SeededRNG(uint64_t seed);
    result_type operator()();
    void seed(uint64_t s);
    static constexpr result_type min();
    static constexpr result_type max();
};

struct CPUFeatures {
    bool hasAVX512F() const noexcept;
    bool hasAVX512DQ() const noexcept;
    bool hasAVX2() const noexcept;
    bool hasFMA() const noexcept;
    bool hasSSE4_1() const noexcept;
    bool hasNEON() const noexcept;
    bool hasSVE() const noexcept;
};

struct Topology {
    struct Core { int id; int package_id; int numa_node; };
    static Topology detect();
    const std::vector<Core>& cores() const;
    int core_count() const;
};

void simd_memcpy(void* dst, const void* src, size_t n);
void simd_memset(void* dst, int value, size_t n);
double simd_reduce_sum(const double* data, size_t n);

}
```

## CPU feature detection

Pynerve detects CPU capabilities at runtime using the `cpuid` instruction. Detection happens once at module load time and is cached.

```cpp
CPUFeatures features = CPUFeatures::detect();

if (features.hasAVX512F()) {
    // AVX-512 path
} else if (features.hasAVX2()) {
    // AVX2 path
}
```

Python access:

```python
feats = pynerve.core.cpu_features()
```

Features checked: AVX-512F, AVX-512DQ, AVX2, FMA, SSE4.1, SSE4.2, SSE3, SSSE3, NEON (ARM), SVE (ARM).

## Thread affinity

Pynerve pins worker threads to physical cores for consistent performance. The pinning strategy:

1. Detect topology (cores, packages, NUMA nodes)
2. Assign threads to physical cores first (skip hyperthread siblings)
3. Fill packages round-robin
4. If more threads than physical cores, assign to hyperthread siblings

```cpp
pin_thread_to_core(int core_id);
pin_thread_to_numa_node(int node_id);
```

Uses `pthread_setaffinity_np` on Linux, `SetThreadAffinityMask` on Windows.

## Topology detection

`Topology::detect()` enumerates physical core IDs, package (socket) IDs, NUMA node assignments, and hyperthreading siblings. The thread pool uses this to distribute workers across physical cores first, avoiding hyperthread contention.

## How it works: SIMD dispatch

SIMD operations (memcpy, memset, reduce-sum) are runtime-dispatched:

```cpp
void init_simd_dispatch() {
    auto feats = CPUFeatures::detect();
    if (feats.hasAVX512F()) {
        g_memcpy_impl = simd_memcpy_avx512;
        g_memset_impl = simd_memset_avx512;
    } else if (feats.hasAVX2()) {
        g_memcpy_impl = simd_memcpy_avx2;
    } else if (feats.hasSSE4_1()) {
        g_memcpy_impl = simd_memcpy_sse41;
    } else {
        g_memcpy_impl = memcpy_impl;
    }
}
```

## Complexity

| Operation | Cost |
|-----------|------|
| `ThreadPool::submit` | O(1) via lock-free push |
| `ThreadPool::map` | O(n) work distribution |
| CPU feature detection | O(1) microcode call |
| `Topology::detect` | O(core_count) syscalls |
| `pin_thread_to_core` | O(1) syscall |
| `simd_memcpy` | O(n / SIMD_width) |
| `simd_reduce_sum` | O(n / SIMD_width) |

## Cross-references

- [Memory](../memory/memory.md): NUMA-aware memory pools
- [CUDA](../cuda/cuda.md): GPU backend
- [Algebra distance](../algebra/distance.md): SIMD distance computation
- [Thread pool](thread_pool.md): detailed documentation
- [RNG](rng.md): detailed documentation
- [SIMD ops](simd_ops.md): detailed documentation

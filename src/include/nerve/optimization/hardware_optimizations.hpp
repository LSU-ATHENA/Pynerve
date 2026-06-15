
#pragma once

#include "nerve/config.hpp"
#include "nerve/cpu/x86_intrinsics.hpp"

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>

#if defined(__linux__) && HAS_NUMA && __has_include(<numa.h>)
#include <numa.h>
#include <sched.h>
#define NERVE_OPTIMIZATION_HAS_NUMA 1
#else
#define NERVE_OPTIMIZATION_HAS_NUMA 0
#endif

namespace nerve::optimization
{

// Prefetching

enum class PrefetchLevel
{
    L1 = 3, // Prefetch to L1 cache (immediate use)
    L2 = 2, // Prefetch to L2 cache (moderate temporal locality)
    L3 = 1, // Prefetch to L3 cache (low temporal locality)
    RAM = 0 // Prefetch to RAM only
};

template <PrefetchLevel LEVEL>
inline void prefetch(const void *ptr)
{
#ifdef __GNUC__
    __builtin_prefetch(ptr, 0, static_cast<int>(LEVEL));
#elif defined(_MSC_VER) && NERVE_HAS_X86_INTRINSICS
    _mm_prefetch(reinterpret_cast<const char *>(ptr), static_cast<int>(LEVEL) == 3   ? _MM_HINT_T0
                                                      : static_cast<int>(LEVEL) == 2 ? _MM_HINT_T1
                                                                                     : _MM_HINT_T2);
#else
    (void)ptr;
#endif
}

// Stream prefetch for write-only data
template <PrefetchLevel LEVEL>
inline void prefetchWrite(const void *ptr)
{
#ifdef __GNUC__
    __builtin_prefetch(ptr, 1, static_cast<int>(LEVEL));
#else
    (void)ptr;
#endif
}

// Alignment

constexpr size_t CACHE_LINE_SIZE = 64;
constexpr size_t SIMD_ALIGNMENT = 64; // AVX-512

// Align to cache line boundary
#define ALIGN_CACHE_LINE alignas(64)

// Align to SIMD register width
#define ALIGN_SIMD alignas(64)

// Check if pointer is aligned
inline bool isAligned(const void *ptr, size_t alignment)
{
    return (reinterpret_cast<uintptr_t>(ptr) & (alignment - 1)) == 0;
}

// Pad size to alignment boundary
inline size_t padToAlignment(size_t size, size_t alignment)
{
    return (size + alignment - 1) & ~(alignment - 1);
}

// FMA Operations

// Fused multiply-add: result = a * b + c
// Single rounding, better precision, faster on modern CPUs
inline double fma(double a, double b, double c)
{
    return std::fma(a, b, c);
}

inline float fma(float a, float b, float c)
{
    return std::fma(a, b, c);
}

// Branchless Operations

// Branchless max: returns max(a, b) without branches
inline double branchlessMax(double a, double b)
{
    // Use bit manipulation for branchless max
    // This is often faster than std::max due to branch prediction misses
    return a > b ? a : b;
}

inline int branchlessMax(int a, int b)
{
    int diff = a - b;
    int mask = diff >> (sizeof(int) * 8 - 1); // Sign bit
    return a - (diff & mask);
}

// Branchless min
inline double branchlessMin(double a, double b)
{
    return a < b ? a : b;
}

// Branchless abs
inline double branchlessAbs(double x)
{
    // Clear sign bit
    union
    {
        double d;
        uint64_t u;
    } caster;
    caster.d = x;
    caster.u &= 0x7FFFFFFFFFFFFFFFULL;
    return caster.d;
}

// Likely/Unlikely Hints

#ifdef __GNUC__
#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#endif

// Cache Control

// Non-temporal store (bypasses cache)
inline void streamStore(double *dst, double value)
{
#if defined(__AVX__) && NERVE_HAS_X86_INTRINSICS
    _mm_stream_pd(dst, _mm_set1_pd(value));
#else
    *dst = value;
#endif
}

// Flush cache line
inline void cacheFlush(const void *ptr)
{
#if NERVE_HAS_X86_INTRINSICS && (defined(__GNUC__) || defined(__clang__))
    __builtin_ia32_clflush(ptr);
#elif NERVE_HAS_X86_INTRINSICS
    _mm_clflush(ptr);
#else
    (void)ptr;
#endif
}

// Memory fence
inline void memoryFence()
{
#ifdef __GNUC__
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
#else
    std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
}

// CPU Feature Detection

struct CpuFeatures
{
    bool has_sse2 = false;
    bool has_avx = false;
    bool has_avx2 = false;
    bool has_avx512f = false;
    bool has_fma = false;
    bool has_bmi2 = false;
    bool has_popcnt = false;
    bool has_lzcnt = false;

    static CpuFeatures detect();
};

// NUMA Utilities

#if NERVE_OPTIMIZATION_HAS_NUMA
inline void *numaAllocOnNode(size_t size, int node)
{
    return numa_alloc_onnode(size, node);
}

inline void *numaAllocInterleaved(size_t size)
{
    return numa_alloc_interleaved(size);
}

inline void numaFree(void *ptr, size_t size)
{
    numa_free(ptr, size);
}

inline int getCurrentNumaNode()
{
    return numa_node_of_cpu(sched_getcpu());
}

void runOnNumaNode(int node, std::function<void()> func);

#else
inline void *numaAllocOnNode(size_t size, int)
{
    return std::malloc(size);
}
inline void *numaAllocInterleaved(size_t size)
{
    return std::malloc(size);
}
inline void numaFree(void *ptr, size_t)
{
    std::free(ptr);
}
inline int getCurrentNumaNode()
{
    return 0;
}
#endif

// Hardware Prefetch Distance

// Calculate optimal prefetch distance based on cache size
inline size_t getOptimalPrefetchDistance()
{
    // Conservative default: 8 cache lines ahead
    return 8 * CACHE_LINE_SIZE;
}

// Prefetch distance for random access patterns
inline size_t getRandomAccessPrefetchDistance()
{
    return 4 * CACHE_LINE_SIZE; // Smaller for random access
}

// Prefetch distance for sequential access
inline size_t getSequentialPrefetchDistance()
{
    return 16 * CACHE_LINE_SIZE; // Larger for sequential
}

} // namespace nerve::optimization

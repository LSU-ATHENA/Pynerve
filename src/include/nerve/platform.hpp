#pragma once

// Central platform abstraction layer for Nerve.
// Provides portable wrappers around compiler builtins and platform APIs.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#if NERVE_COMPILER_MSVC
#include <intrin.h>
#include <immintrin.h>
#endif

#if NERVE_HAS_X86_INTRINSICS && (NERVE_COMPILER_GNU_LIKE || defined(__AVX__))
#include <immintrin.h>
#endif

// Compiler detection

#if defined(__GNUC__) || defined(__clang__)
    #define NERVE_COMPILER_GNU_LIKE 1
#else
    #define NERVE_COMPILER_GNU_LIKE 0
#endif

#if defined(_MSC_VER)
    #define NERVE_COMPILER_MSVC 1
#else
    #define NERVE_COMPILER_MSVC 0
#endif

// Platform detection

#if defined(_WIN32) || defined(_WIN64)
    #define NERVE_PLATFORM_WINDOWS 1
#else
    #define NERVE_PLATFORM_WINDOWS 0
#endif

#if defined(__linux__)
    #define NERVE_PLATFORM_LINUX 1
#else
    #define NERVE_PLATFORM_LINUX 0
#endif

#if defined(__APPLE__)
    #define NERVE_PLATFORM_APPLE 1
#else
    #define NERVE_PLATFORM_APPLE 0
#endif

// Likely / Unlikely hints

#if NERVE_COMPILER_GNU_LIKE
    #define NERVE_LIKELY(x)     __builtin_expect(!!(x), 1)
    #define NERVE_UNLIKELY(x)   __builtin_expect(!!(x), 0)
#else
    #define NERVE_LIKELY(x)     (x)
    #define NERVE_UNLIKELY(x)   (x)
#endif

// Prefetch

// Prefetch levels matching hardware cache hierarchy.
enum class NervePrefetchLevel
{
    L1 = 3,
    L2 = 2,
    L3 = 1,
    RAM = 0
};

// Prefetch for read.
inline void nerve_prefetch_read(const void *ptr, NervePrefetchLevel level = NervePrefetchLevel::L1)
{
#if NERVE_COMPILER_GNU_LIKE
    switch (level)
    {
        case NervePrefetchLevel::L1:  __builtin_prefetch(ptr, 0, 3); break;
        case NervePrefetchLevel::L2:  __builtin_prefetch(ptr, 0, 2); break;
        case NervePrefetchLevel::L3:  __builtin_prefetch(ptr, 0, 1); break;
        case NervePrefetchLevel::RAM: __builtin_prefetch(ptr, 0, 0); break;
    }
#elif defined(_MSC_VER) && defined(__AVX__)
    // MSVC on x86 with AVX
    switch (level)
    {
        case NervePrefetchLevel::L1:
            _mm_prefetch(reinterpret_cast<const char *>(ptr), _MM_HINT_T0);
            break;
        case NervePrefetchLevel::L2:
            _mm_prefetch(reinterpret_cast<const char *>(ptr), _MM_HINT_T1);
            break;
        case NervePrefetchLevel::L3:
        case NervePrefetchLevel::RAM:
        default:
            _mm_prefetch(reinterpret_cast<const char *>(ptr), _MM_HINT_T2);
            break;
    }
#else
    (void)ptr;
    (void)level;
#endif
}

// Prefetch for write (read-for-ownership).
// Prefetch for write uses __builtin_prefetch with write hint (rw=1).
// MSVC does not expose an equivalent intrinsic, so this is a no-op on MSVC.
inline void nerve_prefetch_write(const void *ptr, NervePrefetchLevel level = NervePrefetchLevel::L1)
{
#if NERVE_COMPILER_GNU_LIKE
    switch (level)
    {
        case NervePrefetchLevel::L1:  __builtin_prefetch(ptr, 1, 3); break;
        case NervePrefetchLevel::L2:  __builtin_prefetch(ptr, 1, 2); break;
        case NervePrefetchLevel::L3:  __builtin_prefetch(ptr, 1, 1); break;
        case NervePrefetchLevel::RAM: __builtin_prefetch(ptr, 1, 0); break;
    }
#else
    (void)ptr;
    (void)level;
#endif
}

// Bit manipulation builtins

namespace nerve::bits
{

// Count leading zeros (64-bit).
// Returns the number of leading zero bits in x. Undefined for x == 0.
inline int clz64(std::uint64_t x)
{
#if NERVE_COMPILER_GNU_LIKE
    return static_cast<int>(__builtin_clzll(x));
#elif NERVE_COMPILER_MSVC
    unsigned long index = 0;
    _BitScanReverse64(&index, x);
    return static_cast<int>(63 - index);
#else
    // Portable fallback (rarely used)
    int n = 0;
    for (int i = 63; i >= 0; --i)
        if ((x >> static_cast<unsigned>(i)) & 1U) return 63 - i;
    return 64;
#endif
}

// Count trailing zeros (64-bit).
// Returns the number of trailing zero bits in x. Undefined for x == 0.
inline int ctz64(std::uint64_t x)
{
#if NERVE_COMPILER_GNU_LIKE
    return static_cast<int>(__builtin_ctzll(x));
#elif NERVE_COMPILER_MSVC
    unsigned long index = 0;
    _BitScanForward64(&index, x);
    return static_cast<int>(index);
#else
    // Portable fallback
    if (x == 0) return 64;
    int n = 0;
    while ((x & 1U) == 0) { ++n; x >>= 1; }
    return n;
#endif
}

// Count trailing zeros (32-bit).
inline int ctz32(std::uint32_t x)
{
#if NERVE_COMPILER_GNU_LIKE
    return static_cast<int>(__builtin_ctz(x));
#elif NERVE_COMPILER_MSVC
    unsigned long index = 0;
    _BitScanForward(&index, x);
    return static_cast<int>(index);
#else
    if (x == 0) return 32;
    int n = 0;
    while ((x & 1U) == 0) { ++n; x >>= 1; }
    return n;
#endif
}

// Count leading zeros (32-bit).
inline int clz32(std::uint32_t x)
{
#if NERVE_COMPILER_GNU_LIKE
    return static_cast<int>(__builtin_clz(x));
#elif NERVE_COMPILER_MSVC
    unsigned long index = 0;
    _BitScanReverse(&index, x);
    return static_cast<int>(31 - index);
#else
    int n = 0;
    for (int i = 31; i >= 0; --i)
        if ((x >> static_cast<unsigned>(i)) & 1U) return 31 - i;
    return 32;
#endif
}

// Population count (64-bit).
// Returns the number of 1 bits in x.
inline int popcount64(std::uint64_t x)
{
#if NERVE_COMPILER_GNU_LIKE
    return static_cast<int>(__builtin_popcountll(x));
#elif NERVE_COMPILER_MSVC
    return static_cast<int>(__popcnt64(x));
#else
    int count = 0;
    while (x) { count += static_cast<int>(x & 1U); x >>= 1; }
    return count;
#endif
}

// Population count (32-bit).
inline int popcount32(std::uint32_t x)
{
#if NERVE_COMPILER_GNU_LIKE
    return static_cast<int>(__builtin_popcount(x));
#elif NERVE_COMPILER_MSVC
    return static_cast<int>(__popcnt(x));
#else
    int count = 0;
    while (x) { count += static_cast<int>(x & 1U); x >>= 1; }
    return count;
#endif
}

// Find last set bit (1-indexed). Returns 0 for x == 0.
inline int fls64(std::uint64_t x)
{
    if (x == 0) return 0;
    return 64 - clz64(x);
}

// Find first set bit (1-indexed). Returns 0 for x == 0.
inline int ffs64(std::uint64_t x)
{
    if (x == 0) return 0;
    return ctz64(x) + 1;
}

} // namespace nerve::bits

// CPU feature detection (x86/x86_64)

namespace nerve::cpu
{

// CPUID feature flags for x86/x86_64.
// Returns a struct populated via the CPUID instruction.
struct CpuFeatureFlags
{
    // Leaf 1: ECX
    bool has_sse3      : 1 = false;
    bool has_ssse3     : 1 = false;
    bool has_sse41     : 1 = false;
    bool has_sse42     : 1 = false;
    bool has_aes       : 1 = false;
    bool has_avx       : 1 = false;
    bool has_fma       : 1 = false;
    // Leaf 7 (sub-leaf 0): EBX
    bool has_bmi1      : 1 = false;
    bool has_bmi2      : 1 = false;
    bool has_avx2      : 1 = false;
    bool has_avx512f   : 1 = false;
    bool has_avx512dq  : 1 = false;
    bool has_avx512bw  : 1 = false;
    bool has_avx512vl  : 1 = false;

    static CpuFeatureFlags detect() noexcept;
};

// CPUID wrapper. Uses __cpuid/__cpuidex on MSVC, or inline asm on GCC.
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)

inline void cpuid(int info[4], int function_id, int subfunction_id = 0) noexcept
{
#if NERVE_COMPILER_GNU_LIKE && (defined(__x86_64__) || defined(__i386__))
    // GCC/Clang: inline assembly avoids header dependency issues
    __asm__(
        "cpuid\n\t"
        : "=a"(info[0]), "=b"(info[1]), "=c"(info[2]), "=d"(info[3])
        : "a"(function_id), "c"(subfunction_id)
    );
#elif NERVE_COMPILER_MSVC && (defined(_M_X64) || defined(_M_IX86))
    __cpuidex(info, function_id, subfunction_id);
#else
    info[0] = info[1] = info[2] = info[3] = 0;
#endif
}

#endif

inline CpuFeatureFlags CpuFeatureFlags::detect() noexcept
{
    CpuFeatureFlags flags;
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
    int info[4] = {0};

    // Leaf 1: basic feature flags
    cpuid(info, 1);
    flags.has_sse3   = (info[2] >> 0)  & 1;  // ECX bit 0
    flags.has_ssse3  = (info[2] >> 9)  & 1;  // ECX bit 9
    flags.has_sse41  = (info[2] >> 19) & 1;  // ECX bit 19
    flags.has_sse42  = (info[2] >> 20) & 1;  // ECX bit 20
    flags.has_aes    = (info[2] >> 25) & 1;  // ECX bit 25
    flags.has_avx    = (info[2] >> 28) & 1;  // ECX bit 28
    flags.has_fma    = (info[2] >> 12) & 1;  // ECX bit 12

    // Leaf 7 (sub-leaf 0): extended feature flags
    cpuid(info, 7, 0);
    flags.has_bmi1     = (info[1] >> 3)  & 1;  // EBX bit 3
    flags.has_bmi2     = (info[1] >> 8)  & 1;  // EBX bit 8
    flags.has_avx2     = (info[1] >> 5)  & 1;  // EBX bit 5
    flags.has_avx512f  = (info[1] >> 16) & 1;  // EBX bit 16
    flags.has_avx512dq = (info[1] >> 17) & 1;  // EBX bit 17
    flags.has_avx512bw = (info[1] >> 30) & 1;  // EBX bit 30
    flags.has_avx512vl = (info[1] >> 31) & 1;  // EBX bit 31
#else
    (void)flags;
#endif
    return flags;
}

} // namespace nerve::cpu

// Cache line flush

inline void nerve_cache_flush(const void *ptr)
{
#if NERVE_HAS_X86_INTRINSICS
    _mm_clflush(ptr);
#else
    (void)ptr;
#endif
}

// Memory fence

inline void nerve_memory_fence()
{
    std::atomic_thread_fence(std::memory_order_seq_cst);
}

// Attribute wrappers

#if NERVE_COMPILER_GNU_LIKE
    #define NERVE_NOINLINE      __attribute__((noinline))
#elif NERVE_COMPILER_MSVC
    #define NERVE_NOINLINE      __declspec(noinline)
#else
    #define NERVE_NOINLINE
#endif

// Platform-specific includes for system call wrappers

#if NERVE_PLATFORM_WINDOWS
// Keep windows.h lean
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <io.h>       // _get_osfhandle
#elif NERVE_PLATFORM_LINUX
#include <sys/mman.h>  // mmap, munmap, madvise, mprotect
#include <unistd.h>    // sysconf, _SC_PAGESIZE
#include <pthread.h>   // pthread_t, pthread_self, pthread_setaffinity_np
#include <sched.h>     // CPU_ZERO, CPU_SET, cpu_set_t, sched_getcpu
#ifdef NERVE_HAS_NUMA
#include <numa.h>      // numa_node_of_cpu
#endif
#else
// macOS / other POSIX
#include <sys/mman.h>
#include <unistd.h>
#include <pthread.h>
#include <sched.h>
#endif

// POSIX-to-Windows system call wrappers
//
// These provide portable abstractions for mmap/pthread/thread-affinity
// operations that are used throughout the codebase.  All call sites
// use nerve::sys::* uniformly; the wrappers expand to native syscalls.

namespace nerve::sys
{

// Page size

/// Portable page size query.  Returns the system page size (typically 4 KiB).
inline size_t page_size() noexcept
{
#if NERVE_PLATFORM_WINDOWS
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return static_cast<size_t>(si.dwPageSize);
#else
    return static_cast<size_t>(sysconf(_SC_PAGESIZE));
#endif
}

/// Portable allocation granularity query.
///
/// On Windows this returns the allocation granularity (typically 64 KiB)
/// which is the minimum alignment required for offsets passed to
/// MapViewOfFile (via nerve::sys::map()).  On POSIX this is the same
/// as page_size() because mmap() accepts page-aligned offsets.
inline size_t allocation_granularity() noexcept
{
#if NERVE_PLATFORM_WINDOWS
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return static_cast<size_t>(si.dwAllocationGranularity);
#else
    return page_size();
#endif
}

// Memory-mapped I/O

// Protection flags (match POSIX PROT_* values).
// Plain enum (not enum class) so callers can use bitwise OR naturally.
enum MapProt : int
{
    MAP_PROT_NONE  = 0,
    MAP_PROT_READ  = 1,       // = PROT_READ
    MAP_PROT_WRITE = 2,       // = PROT_WRITE
    MAP_PROT_RW    = MAP_PROT_READ | MAP_PROT_WRITE,
};

// Mapping flags (combinations of POSIX MAP_*).
enum MapFlags : int
{
    MAP_FLAG_NONE      = 0,
    MAP_FLAG_SHARED    = 1 << 0,  // MAP_SHARED
    MAP_FLAG_PRIVATE   = 1 << 1,  // MAP_PRIVATE
    MAP_FLAG_ANONYMOUS = 1 << 2,  // MAP_ANONYMOUS
    MAP_FLAG_HUGETLB   = 1 << 3,  // MAP_HUGETLB (best-effort on Windows)
};

// Advice values (match POSIX MADV_*).
enum MapAdvice : int
{
    MAP_ADV_NORMAL    = 0,  // MADV_NORMAL
    MAP_ADV_RANDOM    = 1,  // MADV_RANDOM
    MAP_ADV_SEQUENTIAL = 2, // MADV_SEQUENTIAL
    MAP_ADV_WILLNEED  = 3,  // MADV_WILLNEED
    MAP_ADV_HUGEPAGE  = 14, // MADV_HUGEPAGE (Linux 2.6.38+)
};

/// Sentinel returned by map() on failure (equivalent to MAP_FAILED).
inline void *kMapFailed = reinterpret_cast<void *>(static_cast<uintptr_t>(-1));

/// Portable mmap().  Returns kMapFailed on error.
///
/// For file-backed mappings on Windows, the file descriptor is converted
/// via _get_osfhandle().  For anonymous mappings, VirtualAlloc is used.
/// The offset must be aligned to the allocation granularity (typically
/// 64 KiB) on Windows; call site alignment to page_size() is NOT sufficient.
inline void *map(void *addr, size_t length, int prot, int flags, int fd = -1,
                 size_t offset = 0) noexcept
{
#if NERVE_PLATFORM_WINDOWS
    (void)addr;

    // Translate protection flags to Windows page protection
    DWORD flProtect = PAGE_NOACCESS;
    if (prot & (MAP_PROT_READ | MAP_PROT_WRITE))
        flProtect = (prot & MAP_PROT_WRITE) ? PAGE_READWRITE : PAGE_READONLY;

    if (flags & MAP_FLAG_ANONYMOUS)
    {
        // Anonymous mapping via VirtualAlloc
        DWORD allocType = MEM_COMMIT | MEM_RESERVE;
        if (flags & MAP_FLAG_HUGETLB)
            allocType |= MEM_LARGE_PAGES;
        void *p = VirtualAlloc(nullptr, length, allocType, flProtect);
        return p ? p : kMapFailed;
    }

    // File-backed mapping: convert fd to Windows HANDLE
    HANDLE hFile = INVALID_HANDLE_VALUE;
    if (fd >= 0)
    {
        intptr_t osHandle = _get_osfhandle(fd);
        if (osHandle == -1L)
            return kMapFailed;
        hFile = reinterpret_cast<HANDLE>(osHandle);
    }

    DWORD dwDesiredAccess = (prot & MAP_PROT_WRITE) ? FILE_MAP_WRITE : FILE_MAP_READ;
    DWORD flProtectMap = flProtect;

    HANDLE hMap = CreateFileMappingW(hFile, nullptr, flProtectMap, 0, 0, nullptr);
    if (!hMap)
        return kMapFailed;

    LARGE_INTEGER liOff;
    liOff.QuadPart = static_cast<LONGLONG>(offset);
    void *p = MapViewOfFile(hMap, dwDesiredAccess, liOff.HighPart, liOff.LowPart, length);
    CloseHandle(hMap);
    return p ? p : kMapFailed;
#else
    // POSIX: translate enums to native constants
    int posix_flags = 0;
    if (flags & MAP_FLAG_SHARED)    posix_flags |= MAP_SHARED;
    if (flags & MAP_FLAG_PRIVATE)   posix_flags |= MAP_PRIVATE;
    if (flags & MAP_FLAG_ANONYMOUS) posix_flags |= MAP_ANONYMOUS;
#if defined(MAP_HUGETLB)
    if (flags & MAP_FLAG_HUGETLB)   posix_flags |= MAP_HUGETLB;
#endif

    int posix_prot = 0;
    if (prot & MAP_PROT_READ)  posix_prot |= PROT_READ;
    if (prot & MAP_PROT_WRITE) posix_prot |= PROT_WRITE;

    void *result = ::mmap(addr, length, posix_prot, posix_flags, fd,
                          static_cast<off_t>(offset));
    return (result == MAP_FAILED) ? kMapFailed : result;
#endif
}

/// Portable munmap().
inline int unmap(void *addr, size_t length) noexcept
{
    if (!addr || addr == kMapFailed)
        return -1;
#if NERVE_PLATFORM_WINDOWS
    (void)length;
    return UnmapViewOfFile(addr) ? 0 : -1;
#else
    return ::munmap(addr, length);
#endif
}

/// Portable msync().
inline int sync_map(void *addr, size_t length, bool async = false) noexcept
{
    if (!addr || addr == kMapFailed)
        return -1;
#if NERVE_PLATFORM_WINDOWS
    (void)async;
    return FlushViewOfFile(addr, length) ? 0 : -1;
#else
    return ::msync(addr, length, async ? MS_ASYNC : MS_SYNC);
#endif
}

/// Portable madvise().  Best-effort on Windows — uses PrefetchVirtualMemory
/// for WILLNEED; otherwise a no-op that returns 0.
inline int advise(void *addr, size_t length, int advice) noexcept
{
    if (!addr || addr == kMapFailed)
        return -1;
#if NERVE_PLATFORM_WINDOWS
    if (advice == MAP_ADV_WILLNEED)
    {
        WIN32_MEMORY_RANGE_ENTRY entry;
        entry.VirtualAddress = addr;
        entry.NumberOfBytes = length;
        return PrefetchVirtualMemory(GetCurrentProcess(), 1, &entry, 0) ? 0 : -1;
    }
    // MADV_SEQUENTIAL, MADV_RANDOM, MADV_HUGEPAGE have no direct Windows equivalent
    return 0;
#else
    int posix_advice;
    switch (advice)
    {
    case MAP_ADV_NORMAL:     posix_advice = MADV_NORMAL;     break;
    case MAP_ADV_RANDOM:     posix_advice = MADV_RANDOM;     break;
    case MAP_ADV_SEQUENTIAL: posix_advice = MADV_SEQUENTIAL; break;
    case MAP_ADV_WILLNEED:   posix_advice = MADV_WILLNEED;   break;
#if defined(MADV_HUGEPAGE)
    case MAP_ADV_HUGEPAGE:   posix_advice = MADV_HUGEPAGE;   break;
#endif
    default:                 return 0;
    }
    return ::madvise(addr, length, posix_advice);
#endif
}

// Thread affinity

/// Portable CPU set, large enough for at least 1024 logical processors.
/// On POSIX this wraps cpu_set_t semantics; on Windows it wraps a
/// bitmask used with SetThreadAffinityMask.
struct CpuSet
{
    static constexpr int kMaxCpus = 1024;
    unsigned long bits[(kMaxCpus + (sizeof(unsigned long) * 8 - 1)) /
                       (sizeof(unsigned long) * 8)]{};

    void clear() noexcept
    {
        for (auto &b : bits) b = 0UL;
    }

    void set(int cpu) noexcept
    {
        if (cpu >= 0 && cpu < kMaxCpus)
        {
            auto idx = static_cast<size_t>(cpu) / (sizeof(unsigned long) * 8);
            auto bit = static_cast<size_t>(cpu) % (sizeof(unsigned long) * 8);
            bits[idx] |= (1UL << bit);
        }
    }

    void clr(int cpu) noexcept
    {
        if (cpu >= 0 && cpu < kMaxCpus)
        {
            auto idx = static_cast<size_t>(cpu) / (sizeof(unsigned long) * 8);
            auto bit = static_cast<size_t>(cpu) % (sizeof(unsigned long) * 8);
            bits[idx] &= ~(1UL << bit);
        }
    }

    bool isset(int cpu) const noexcept
    {
        if (cpu >= 0 && cpu < kMaxCpus)
        {
            auto idx = static_cast<size_t>(cpu) / (sizeof(unsigned long) * 8);
            auto bit = static_cast<size_t>(cpu) % (sizeof(unsigned long) * 8);
            return (bits[idx] & (1UL << bit)) != 0;
        }
        return false;
    }
};

/// Portable thread handle type.
#if NERVE_PLATFORM_WINDOWS
using ThreadHandle = void *;  // HANDLE (opaque)
#else
using ThreadHandle = pthread_t;
#endif

/// Portable pthread_self().
inline ThreadHandle thread_self() noexcept
{
#if NERVE_PLATFORM_WINDOWS
    return GetCurrentThread();  // pseudo-handle (no need to close)
#else
    return ::pthread_self();
#endif
}

/// Portable pthread_setaffinity_np().  Returns 0 on success.
inline int thread_set_affinity(ThreadHandle thread, const CpuSet *set) noexcept
{
#if NERVE_PLATFORM_WINDOWS
    // Windows uses a 64-bit KAFFINITY per processor group.
    // This wrapper pins to the primary group only.
    DWORD_PTR mask = 0;
    for (int i = 0; i < 64 && i < CpuSet::kMaxCpus; ++i)
        if (set->isset(i))
            mask |= (static_cast<DWORD_PTR>(1) << i);

    if (mask == 0)
        return -1;

    DWORD_PTR prev = SetThreadAffinityMask(thread, mask);
    return (prev != 0) ? 0 : -1;
#else
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (int i = 0; i < CPU_SETSIZE && i < CpuSet::kMaxCpus; ++i)
        if (set->isset(i))
            CPU_SET(i, &cpuset);
    return ::pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
#endif
}

/// Portable sched_getcpu().  Returns the current CPU number, or 0 if
/// the platform does not support this query.
inline int sched_getcpu() noexcept
{
#if defined(__linux__)
    return ::sched_getcpu();
#elif NERVE_PLATFORM_WINDOWS
    return static_cast<int>(GetCurrentProcessorNumber());
#else
    return 0;
#endif
}

/// Portable numa_node_of_cpu().  Returns the NUMA node for a given CPU,
/// or 0 if NUMA is not available.
///
/// On Windows this calls GetNumaProcessorNode() which takes a processor
/// number and returns its NUMA node.  (GetNumaNodeProcessorNode is the
/// inverse — it takes a node and returns a processor in that node.)
inline int numa_node_of_cpu(int cpu) noexcept
{
#if defined(__linux__) && defined(NERVE_HAS_NUMA)
    return ::numa_node_of_cpu(cpu);
#elif NERVE_PLATFORM_WINDOWS
    USHORT node = 0;
    if (GetNumaProcessorNode(static_cast<UCHAR>(cpu), &node))
        return static_cast<int>(node);
    return 0;
#else
    (void)cpu;
    return 0;
#endif
}

} // namespace nerve::sys

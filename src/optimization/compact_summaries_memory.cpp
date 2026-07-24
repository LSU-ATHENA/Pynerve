// Platform memory probes for AcceleratedCompactSummaries.

#include "nerve/optimization/component_optimizations.hpp"

#include <cstddef>
#include <cstdio>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#elif defined(__linux__)
#include <malloc.h>
#include <unistd.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#endif

namespace nerve::optimization
{

size_t AcceleratedCompactSummaries::getCurrentMemoryUsage() const
{
    size_t usage = 0;
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS memCounters;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &memCounters, sizeof(memCounters)))
    {
        usage = memCounters.WorkingSetSize;
    }
#elif defined(__linux__)
#ifdef __GLIBC__
#if defined(__GLIBC__) && defined(__GLIBC_MINOR__) &&                                              \
    ((__GLIBC__ > 2) || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 33))
    const struct mallinfo2 mi = mallinfo2();
    usage = static_cast<size_t>(mi.hblkhd) + static_cast<size_t>(mi.uordblks);
#else
    const struct mallinfo mi = mallinfo();
    usage = static_cast<size_t>(mi.hblkhd) + static_cast<size_t>(mi.uordblks);
#endif
#else
    FILE *fp = std::fopen("/proc/self/statm", "r");
    if (fp != nullptr)
    {
        long long rss_pages = 0;
        const bool parsed = std::fscanf(fp, "%*s %lld", &rss_pages) == 1;
        std::fclose(fp);
        if (parsed)
        {
            usage = static_cast<size_t>(rss_pages * sysconf(_SC_PAGESIZE));
        }
    }
#endif
#elif defined(__APPLE__)
    struct task_basic_info info;
    mach_msg_type_number_t size = sizeof(info);
    if (task_info(mach_task_self(), TASK_BASIC_INFO, reinterpret_cast<task_info_t>(&info), &size) ==
        KERN_SUCCESS)
    {
        usage = info.resident_size;
    }
#else
    usage = 0;
#endif

    size_t peak = peak_memory_usage_bytes_.load(std::memory_order_relaxed);
    while (peak < usage && !peak_memory_usage_bytes_.compare_exchange_weak(
                               peak, usage, std::memory_order_relaxed, std::memory_order_relaxed))
    {
    }
    return usage;
}

} // namespace nerve::optimization

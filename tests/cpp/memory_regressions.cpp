#include "nerve/common/accelerated_types.hpp"
#include "nerve/core/memory/memory_pool.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core_types.hpp"
#include "memory/safe_memory_pool.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <tuple>
#include <vector>

namespace
{

using nerve::Dimension;
using nerve::Field;
using nerve::Size;
using nerve::core::BufferView;

constexpr double kTol = 1e-10;

bool check_raw_array_pool_alloc_dealloc()
{
    nerve::memory::RawArrayPool pool(4096, false);

    void *ptr = pool.allocate(128);
    if (!ptr)
    {
        std::cerr << "allocation returned null\n";
        return false;
    }

    Size total_before = pool.totalAllocated();
    if (total_before < 128)
    {
        std::cerr << "totalAllocated too small: " << total_before << "\n";
        return false;
    }

    pool.deallocate(ptr, 128);

    Size peak = pool.peakUtilization();
    if (peak < 128)
    {
        std::cerr << "peakUtilization too small: " << peak << "\n";
        return false;
    }

    return true;
}

bool check_raw_array_pool_multiple_allocations()
{
    nerve::memory::RawArrayPool pool(8192, false);

    void *ptrs[4];
    for (int i = 0; i < 4; ++i)
    {
        ptrs[i] = pool.allocate(256);
        if (!ptrs[i])
        {
            std::cerr << "allocation " << i << " returned null\n";
            return false;
        }
    }

    for (int i = 0; i < 4; ++i)
    {
        pool.deallocate(ptrs[i], 256);
    }

    pool.reset();
    if (pool.peakUtilization() < 1024)
    {
        std::cerr << "peak should reflect allocations before reset\n";
        return false;
    }

    return true;
}

bool check_slab_allocator_basic()
{
    nerve::memory::SlabAllocator<int, 64> slab;

    if (!slab.empty())
    {
        std::cerr << "fresh slab should be empty\n";
        return false;
    }

    int *a = slab.allocate();
    if (!a)
    {
        std::cerr << "slab allocate returned null\n";
        return false;
    }
    *a = 42;

    int *b = slab.allocate();
    if (!b)
    {
        std::cerr << "second slab allocate returned null\n";
        return false;
    }
    *b = 99;

    if (slab.empty())
    {
        std::cerr << "slab should not be empty after allocations\n";
        return false;
    }

    slab.deallocate(a);
    slab.deallocate(b);

    return true;
}

bool check_slab_allocator_reset()
{
    nerve::memory::SlabAllocator<double, 32> slab;

    double *p = slab.allocate();
    *p = 3.14;

    if (slab.empty())
    {
        std::cerr << "slab should have elements after allocate\n";
        return false;
    }

    slab.reset();
    if (!slab.empty())
    {
        std::cerr << "slab should be empty after reset\n";
        return false;
    }

    return true;
}

bool check_pool_alignment()
{
    nerve::memory::RawArrayPool pool(4096, false);

    void *ptr = pool.allocate(64);
    if (!ptr)
    {
        std::cerr << "allocation failed\n";
        return false;
    }

    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    if (addr % alignof(std::max_align_t) != 0)
    {
        std::cerr << "misaligned allocation: " << addr << "\n";
        return false;
    }

    pool.deallocate(ptr, 64);
    return true;
}

bool check_memory_pool_allocate_deallocate()
{
    nerve::core::MemoryPool pool(4096);
    void *ptr = pool.allocate(128);
    if (!ptr)
    {
        std::cerr << "MemoryPool allocate returned null\n";
        return false;
    }

    if (pool.allocated() < 128)
    {
        std::cerr << "MemoryPool allocated < 128\n";
        return false;
    }

    pool.deallocate(ptr, 128);

    return true;
}

bool check_memory_pool_fragmentation()
{
    nerve::core::MemoryPool pool(4096);
    void *a = pool.allocate(128);
    void *b = pool.allocate(128);
    static_cast<void>(b);

    pool.deallocate(a, 128);
    double frag = pool.fragmentationRatio();
    if (frag < 0.0 || frag > 1.0)
    {
        std::cerr << "fragmentation ratio out of range: " << frag << "\n";
        return false;
    }

    return true;
}

} // namespace

int main()
{
    if (!check_raw_array_pool_alloc_dealloc())
    {
        std::cerr << "FAIL: raw array pool alloc dealloc\n";
        return 1;
    }
    if (!check_raw_array_pool_multiple_allocations())
    {
        std::cerr << "FAIL: raw array pool multiple allocations\n";
        return 1;
    }
    if (!check_slab_allocator_basic())
    {
        std::cerr << "FAIL: slab allocator basic\n";
        return 1;
    }
    if (!check_slab_allocator_reset())
    {
        std::cerr << "FAIL: slab allocator reset\n";
        return 1;
    }
    if (!check_pool_alignment())
    {
        std::cerr << "FAIL: pool alignment\n";
        return 1;
    }
    if (!check_memory_pool_allocate_deallocate())
    {
        std::cerr << "FAIL: memory pool allocate deallocate\n";
        return 1;
    }
    if (!check_memory_pool_fragmentation())
    {
        std::cerr << "FAIL: memory pool fragmentation\n";
        return 1;
    }
    return 0;
}

#include "nerve/core/memory/numa_aware_pool.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

namespace
{

using nerve::core::NumaAwareMemoryPool;
using nerve::core::NumaPolicy;
using nerve::core::NumaPoolConfig;

bool check_numa_pool_construction()
{
    NumaPoolConfig config;
    config.policy = NumaPolicy::AUTO;
    config.enableNumaBinding = false;
    NumaAwareMemoryPool pool(config);
    return true;
}

bool check_alloc_dealloc()
{
    NumaPoolConfig config;
    config.enableNumaBinding = false;
    NumaAwareMemoryPool pool(config);
    void *ptr = pool.allocate(256);
    if (ptr == nullptr)
    {
        std::cerr << "allocation returned null\n";
        return false;
    }
    std::memset(ptr, 0xAB, 256);
    pool.deallocate(ptr, 256);
    return true;
}

bool check_pool_stats_non_negative()
{
    NumaPoolConfig config;
    config.enableNumaBinding = false;
    NumaAwareMemoryPool pool(config);
    std::size_t allocated = pool.totalAllocated();
    std::size_t capacity = pool.totalCapacity();
    if (allocated > capacity)
    {
        std::cerr << "allocated should not exceed capacity\n";
        return false;
    }
    return true;
}

bool check_multiple_alloc_free_cycles()
{
    NumaPoolConfig config;
    config.enableNumaBinding = false;
    NumaAwareMemoryPool pool(config);
    std::vector<void *> ptrs;
    ptrs.reserve(10);
    for (int i = 0; i < 10; ++i)
    {
        void *p = pool.allocate(64);
        if (p == nullptr)
        {
            std::cerr << "cycle allocation failed at " << i << "\n";
            for (auto *pp : ptrs)
                pool.deallocate(pp, 64);
            return false;
        }
        ptrs.push_back(p);
    }
    for (auto *p : ptrs)
        pool.deallocate(p, 64);
    return true;
}

bool check_allocate_on_node()
{
    NumaPoolConfig config;
    config.enableNumaBinding = false;
    NumaAwareMemoryPool pool(config);
    void *ptr = pool.allocateOnNode(128, 0);
    if (ptr == nullptr)
    {
        std::cerr << "node allocation returned null\n";
        return false;
    }
    pool.deallocate(ptr, 128);
    return true;
}

} // namespace

int main()
{
    if (!check_numa_pool_construction())
    {
        std::cerr << "FAIL: numa_pool_construction\n";
        return 1;
    }
    if (!check_alloc_dealloc())
    {
        std::cerr << "FAIL: alloc_dealloc\n";
        return 1;
    }
    if (!check_pool_stats_non_negative())
    {
        std::cerr << "FAIL: pool_stats_non_negative\n";
        return 1;
    }
    if (!check_multiple_alloc_free_cycles())
    {
        std::cerr << "FAIL: multiple_alloc_free_cycles\n";
        return 1;
    }
    if (!check_allocate_on_node())
    {
        std::cerr << "FAIL: allocate_on_node\n";
        return 1;
    }
    return 0;
}

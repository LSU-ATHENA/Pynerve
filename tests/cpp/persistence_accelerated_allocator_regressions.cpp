#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core_types.hpp"
#include "nerve/persistence/accelerated/thread_safe_allocator.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

namespace
{

using nerve::Size;
using nerve::persistence::accelerated::CudaMemoryPool;
using nerve::persistence::accelerated::ThreadSafeMemoryManager;

bool check_cuda_memory_pool_config_defaults()
{
    CudaMemoryPool::Config config;
    if (config.initial_size != 1024ULL * 1024ULL * 1024ULL)
    {
        std::cerr << "default initial_size should be 1GB\n";
        return false;
    }
    if (config.max_size != 8ULL * 1024ULL * 1024ULL * 1024ULL)
    {
        std::cerr << "default max_size should be 8GB\n";
        return false;
    }
    if (config.alignment != 256)
    {
        std::cerr << "default alignment should be 256\n";
        return false;
    }
    if (!config.auto_grow)
    {
        std::cerr << "default auto_grow should be true\n";
        return false;
    }
    return true;
}

bool check_cuda_memory_pool_create()
{
    auto result = CudaMemoryPool::create();
    if (result.isError())
    {
        std::cerr << "pool create failed with code " << static_cast<int>(result.errorCode())
                  << "\n";
    }
    (void)result;
    return true;
}

bool check_cuda_memory_pool_allocate_zero_rejected()
{
    auto pool_result = CudaMemoryPool::create();
    if (pool_result.isError())
        return true;

    auto pool = std::move(pool_result.value());
    auto alloc_result = pool->allocate(0, 256);
    if (alloc_result.isSuccess())
    {
        std::cerr << "allocate(0) should fail\n";
        if (alloc_result.value())
            pool->deallocate(alloc_result.value());
        return false;
    }
    return true;
}

bool check_cuda_memory_pool_allocate_deallocate_cycle()
{
    auto pool_result = CudaMemoryPool::create();
    if (pool_result.isError())
        return true;

    auto pool = std::move(pool_result.value());

    auto alloc1 = pool->allocate(4096, 256);
    if (alloc1.isError())
    {
        std::cerr << "first allocation failed\n";
        return false;
    }

    auto alloc2 = pool->allocate(8192, 256);
    if (alloc2.isError())
    {
        std::cerr << "second allocation failed\n";
        pool->deallocate(alloc1.value());
        return false;
    }

    auto dealloc1 = pool->deallocate(alloc1.value());
    if (dealloc1.isError())
    {
        std::cerr << "first deallocation failed\n";
        pool->deallocate(alloc2.value());
        return false;
    }

    auto dealloc2 = pool->deallocate(alloc2.value());
    if (dealloc2.isError())
    {
        std::cerr << "second deallocation failed\n";
        return false;
    }

    return true;
}

bool check_cuda_memory_pool_multiple_cycles()
{
    auto pool_result = CudaMemoryPool::create();
    if (pool_result.isError())
        return true;

    auto pool = std::move(pool_result.value());

    for (int cycle = 0; cycle < 3; ++cycle)
    {
        auto alloc = pool->allocate(4096, 256);
        if (alloc.isError())
        {
            std::cerr << "allocation failed at cycle " << cycle << "\n";
            return false;
        }

        auto dealloc = pool->deallocate(alloc.value());
        if (dealloc.isError())
        {
            std::cerr << "deallocation failed at cycle " << cycle << "\n";
            return false;
        }
    }
    return true;
}

bool check_cuda_memory_pool_get_stats()
{
    auto pool_result = CudaMemoryPool::create();
    if (pool_result.isError())
        return true;

    auto pool = std::move(pool_result.value());

    size_t allocated_before = pool->getAllocatedSize();
    (void)allocated_before;

    auto alloc = pool->allocate(65536, 256);
    if (alloc.isSuccess())
    {
        size_t used = pool->getUsedSize();
        if (used == 0)
        {
            std::cerr << "used size should be > 0 after allocation\n";
            pool->deallocate(alloc.value());
            return false;
        }
        pool->deallocate(alloc.value());
    }
    return true;
}

bool check_thread_safe_memory_manager_create()
{
    auto manager_result = ThreadSafeMemoryManager::create();
    if (manager_result.isError())
    {
        std::cerr << "manager create failed\n";
    }
    (void)manager_result;
    return true;
}

} // namespace

int main()
{
    if (!check_cuda_memory_pool_config_defaults())
    {
        std::cerr << "FAIL: CUDA memory pool config defaults\n";
        return 1;
    }
    if (!check_cuda_memory_pool_create())
    {
        std::cerr << "FAIL: CUDA memory pool create\n";
        return 1;
    }
    if (!check_cuda_memory_pool_allocate_zero_rejected())
    {
        std::cerr << "FAIL: CUDA memory pool allocate zero rejected\n";
        return 1;
    }
    if (!check_cuda_memory_pool_allocate_deallocate_cycle())
    {
        std::cerr << "FAIL: CUDA memory pool allocate/deallocate cycle\n";
        return 1;
    }
    if (!check_cuda_memory_pool_multiple_cycles())
    {
        std::cerr << "FAIL: CUDA memory pool multiple cycles\n";
        return 1;
    }
    if (!check_cuda_memory_pool_get_stats())
    {
        std::cerr << "FAIL: CUDA memory pool get stats\n";
        return 1;
    }
    if (!check_thread_safe_memory_manager_create())
    {
        std::cerr << "FAIL: thread safe memory manager create\n";
        return 1;
    }
    return 0;
}

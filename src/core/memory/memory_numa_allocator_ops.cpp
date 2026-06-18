#include "nerve/core/memory/numa_aware_pool.hpp"

#include <limits>
#include <mutex>
#include <unordered_map>

namespace nerve::core
{

[[nodiscard]] NumaAwareMemoryPool &NumaPoolManager::getGlobalPool()
{
    std::lock_guard<std::mutex> lock(manager_mutex_);
    if (!global_pool_)
    {
        global_pool_ = std::make_unique<NumaAwareMemoryPool>(global_config_);
    }
    return *global_pool_;
}

[[nodiscard]] NumaAwareMemoryPool &NumaPoolManager::getPoolForThread(std::thread::id thread_id)
{
    std::lock_guard<std::mutex> lock(manager_mutex_);
    auto it = thread_pools_.find(thread_id);
    if (it != thread_pools_.end())
    {
        return *it->second;
    }
    auto pool = std::make_unique<NumaAwareMemoryPool>(global_config_);
    auto *pool_ptr = pool.get();
    thread_pools_[thread_id] = std::move(pool);
    return *pool_ptr;
}

void NumaPoolManager::configureGlobalPool(const NumaPoolConfig &config)
{
    std::lock_guard<std::mutex> lock(manager_mutex_);
    global_config_ = config;
    global_pool_ = std::make_unique<NumaAwareMemoryPool>(global_config_);
}

void NumaPoolManager::configureThreadPool(std::thread::id thread_id, const NumaPoolConfig &config)
{
    std::lock_guard<std::mutex> lock(manager_mutex_);
    thread_pools_[thread_id] = std::make_unique<NumaAwareMemoryPool>(config);
}

[[nodiscard]] bool NumaPoolManager::bindCurrentThreadToNode(int node_id)
{
    return getPoolForThread().bindToNode(node_id);
}

[[nodiscard]] bool NumaPoolManager::bindThreadToNode(std::thread::id thread_id, int node_id)
{
    return getPoolForThread(thread_id).bindThreadToNode(thread_id, node_id);
}

int NumaPoolManager::getThreadNumaNode(std::thread::id thread_id)
{
    return getPoolForThread(thread_id).getCurrentNumaNode();
}

bool NumaPoolManager::validateAllThreadsNoHiddenAllocations()
{
    std::lock_guard<std::mutex> lock(manager_mutex_);
    for (const auto &[thread_id, pool] : thread_pools_)
    {
        (void)thread_id;
        if (!pool->validateNoHiddenAllocations())
        {
            return false;
        }
    }
    if (global_pool_ && !global_pool_->validateNoHiddenAllocations())
    {
        return false;
    }
    return true;
}

std::unordered_map<std::thread::id, std::size_t> NumaPoolManager::getThreadAllocationCounts()
{
    std::lock_guard<std::mutex> lock(manager_mutex_);
    std::unordered_map<std::thread::id, std::size_t> counts;
    for (const auto &[thread_id, pool] : thread_pools_)
    {
        counts[thread_id] = pool->countHotPathAllocations();
    }
    if (global_pool_)
    {
        counts[std::thread::id{}] = global_pool_->countHotPathAllocations();
    }
    return counts;
}

void NumaPoolManager::resetAllTracking()
{
    std::lock_guard<std::mutex> lock(manager_mutex_);
    for (const auto &[thread_id, pool] : thread_pools_)
    {
        (void)thread_id;
        pool->resetHotPathTracking();
    }
    if (global_pool_)
    {
        global_pool_->resetHotPathTracking();
    }
}

std::size_t NumaPoolManager::getTotalAllocated() const
{
    std::lock_guard<std::mutex> lock(manager_mutex_);
    std::size_t total = 0;
    for (const auto &[thread_id, pool] : thread_pools_)
    {
        (void)thread_id;
        const std::size_t allocated = pool->totalAllocated();
        total = allocated > std::numeric_limits<std::size_t>::max() - total
                    ? std::numeric_limits<std::size_t>::max()
                    : total + allocated;
    }
    if (global_pool_)
    {
        const std::size_t allocated = global_pool_->totalAllocated();
        total = allocated > std::numeric_limits<std::size_t>::max() - total
                    ? std::numeric_limits<std::size_t>::max()
                    : total + allocated;
    }
    return total;
}

std::size_t NumaPoolManager::getTotalCapacity() const
{
    std::lock_guard<std::mutex> lock(manager_mutex_);
    std::size_t total = 0;
    for (const auto &[thread_id, pool] : thread_pools_)
    {
        (void)thread_id;
        const std::size_t capacity = pool->totalCapacity();
        total = capacity > std::numeric_limits<std::size_t>::max() - total
                    ? std::numeric_limits<std::size_t>::max()
                    : total + capacity;
    }
    if (global_pool_)
    {
        const std::size_t capacity = global_pool_->totalCapacity();
        total = capacity > std::numeric_limits<std::size_t>::max() - total
                    ? std::numeric_limits<std::size_t>::max()
                    : total + capacity;
    }
    return total;
}

std::vector<NumaNodeInfo> NumaPoolManager::getNumaTopology() const
{
    return const_cast<NumaPoolManager *>(this)->getGlobalPool().getNumaNodes();
}

} // namespace nerve::core

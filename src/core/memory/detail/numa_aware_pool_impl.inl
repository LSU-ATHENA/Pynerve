
#include "nerve/core/memory/numa_aware_pool.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#if defined(__linux__) && HAS_NUMA
#include <numa.h>
#include <sched.h>
#endif

namespace nerve::core
{

namespace
{

constexpr int kMaxNumaNodes = 256;

void subtractSaturating(std::atomic<std::size_t> &value, std::size_t amount) noexcept
{
    std::size_t current = value.load(std::memory_order_relaxed);
    while (!value.compare_exchange_weak(current, current >= amount ? current - amount : 0,
                                        std::memory_order_relaxed))
    {
    }
}

void addSaturating(std::atomic<std::size_t> &value, std::size_t amount) noexcept
{
    std::size_t current = value.load(std::memory_order_relaxed);
    while (!value.compare_exchange_weak(current,
                                        amount > std::numeric_limits<std::size_t>::max() - current
                                            ? std::numeric_limits<std::size_t>::max()
                                            : current + amount,
                                        std::memory_order_relaxed))
    {
    }
}

bool isKnownNode(const std::vector<NumaNodeInfo> &nodes, int nodeId)
{
    return std::any_of(nodes.begin(), nodes.end(), [nodeId](const NumaNodeInfo &node) {
        return node.isAvailable && node.nodeId == nodeId;
    });
}

int defaultNode(const std::vector<NumaNodeInfo> &nodes)
{
    auto it = std::find_if(nodes.begin(), nodes.end(),
                           [](const NumaNodeInfo &node) { return node.isAvailable; });
    return it == nodes.end() ? 0 : it->nodeId;
}

int normalizeNode(const std::vector<NumaNodeInfo> &nodes, int nodeId)
{
    return isKnownNode(nodes, nodeId) ? nodeId : defaultNode(nodes);
}

void ensureDefaultNode(std::vector<NumaNodeInfo> &nodes)
{
    if (!nodes.empty())
    {
        return;
    }
    NumaNodeInfo node;
    node.nodeId = 0;
    node.isAvailable = true;
    nodes.push_back(std::move(node));
}

} // namespace

NumaAwareMemoryPool::NumaAwareMemoryPool(const NumaPoolConfig &config)
    : config_(config)
    , current_node_(-1)
    , numa_available_(false)
    , hot_path_tracking_enabled_(config.enableHotPathTracking)
    , is_deterministic_(false)
{
    initializeNuma();
    updateStatistics();
}
NumaAwareMemoryPool::NumaAwareMemoryPool(const DeterminismContract &contract,
                                         const NumaPoolConfig &config)
    : config_(config)
    , current_node_(-1)
    , numa_available_(false)
    , hot_path_tracking_enabled_(config.enableHotPathTracking)
    , determinism_contract_(contract)
    , is_deterministic_(true)
{
    initializeNuma();
    updateStatistics();
    if (!DeterminismEnforcer::canSatisfyContract(contract))
    {
        auto violations = DeterminismEnforcer::getContractViolations(contract);
        if (contract.fail_on_non_deterministic)
        {
            const std::string reason = violations.empty() ? "unspecified violation" : violations[0];
            throw std::runtime_error("Cannot satisfy determinism contract: " + reason);
        }
    }
}
NumaAwareMemoryPool::NumaAwareMemoryPool(NumaAwareMemoryPool &&other) noexcept
{
    std::lock_guard<std::mutex> lock(other.pools_mutex_);
    config_ = other.config_;
    node_pools_ = std::move(other.node_pools_);
    numa_nodes_ = std::move(other.numa_nodes_);
    current_node_ = other.current_node_;
    numa_available_ = other.numa_available_;
    hot_path_allocations_ = std::move(other.hot_path_allocations_);
    hot_path_violations_ = other.hot_path_violations_.load(std::memory_order_relaxed);
    hot_path_tracking_enabled_ = other.hot_path_tracking_enabled_;
    is_deterministic_ = other.is_deterministic_;
    determinism_contract_ = other.determinism_contract_;
    total_allocated_ = other.total_allocated_.load(std::memory_order_relaxed);
    total_capacity_ = other.total_capacity_.load(std::memory_order_relaxed);
    other.current_node_ = -1;
    other.numa_available_ = false;
    other.is_deterministic_ = false;
    other.hot_path_violations_ = 0;
    other.total_allocated_ = 0;
    other.total_capacity_ = 0;
}
NumaAwareMemoryPool &NumaAwareMemoryPool::operator=(NumaAwareMemoryPool &&other) noexcept
{
    if (this != &other)
    {
        std::lock(pools_mutex_, other.pools_mutex_);
        std::lock_guard<std::mutex> lock1(pools_mutex_, std::adopt_lock);
        std::lock_guard<std::mutex> lock2(other.pools_mutex_, std::adopt_lock);
        config_ = other.config_;
        node_pools_ = std::move(other.node_pools_);
        numa_nodes_ = std::move(other.numa_nodes_);
        current_node_ = other.current_node_;
        numa_available_ = other.numa_available_;
        hot_path_allocations_ = std::move(other.hot_path_allocations_);
        hot_path_violations_ = other.hot_path_violations_.load(std::memory_order_relaxed);
        hot_path_tracking_enabled_ = other.hot_path_tracking_enabled_;
        is_deterministic_ = other.is_deterministic_;
        determinism_contract_ = other.determinism_contract_;
        total_allocated_ = other.total_allocated_.load(std::memory_order_relaxed);
        total_capacity_ = other.total_capacity_.load(std::memory_order_relaxed);
        other.current_node_ = -1;
        other.numa_available_ = false;
        other.is_deterministic_ = false;
        other.hot_path_violations_ = 0;
        other.total_allocated_ = 0;
        other.total_capacity_ = 0;
    }
    return *this;
}
void *NumaAwareMemoryPool::allocate(std::size_t size, std::size_t alignment)
{
    if (is_deterministic_)
    {
        return allocate(size, alignment, determinism_contract_);
    }
    return allocateWithNumaPolicy(size, alignment);
}
void *NumaAwareMemoryPool::allocate(std::size_t size, std::size_t alignment,
                                    const DeterminismContract &contract)
{
    int nodeId = getPreferredNode();
    auto *pool = getOrCreatePoolForNode(nodeId);
    void *ptr = pool->allocate(size, alignment, contract);
    if (config_.allocationCallback)
    {
        config_.allocationCallback(nodeId, size);
    }
    addSaturating(total_allocated_, size);
    return ptr;
}
void *NumaAwareMemoryPool::allocateOnNode(std::size_t size, int nodeId, std::size_t alignment)
{
    if (nodeId < 0 || !numa_available_)
    {
        return allocate(size, alignment);
    }
    if (!isKnownNode(numa_nodes_, nodeId))
    {
        throw std::invalid_argument("Requested NUMA node is unknown");
    }
    auto *pool = getOrCreatePoolForNode(nodeId);
    void *ptr = pool->allocate(size, alignment);
    if (config_.allocationCallback)
    {
        config_.allocationCallback(nodeId, size);
    }
    addSaturating(total_allocated_, size);
    return ptr;
}
void NumaAwareMemoryPool::deallocate(void *ptr, std::size_t size) noexcept
{
    if (!ptr)
        return;
    untrackHotPathAllocation(ptr);
    std::lock_guard<std::mutex> lock(pools_mutex_);
    for (auto &[nodeId, pool] : node_pools_)
    {
        if (pool && pool->contains(ptr))
        {
            pool->deallocate(ptr, size);
            subtractSaturating(total_allocated_, size);
            return;
        }
    }
}
void *NumaAwareMemoryPool::allocateHotPath(std::size_t size, std::size_t alignment,
                                           const char *function, const char *file, int line)
{
    void *ptr = allocate(size, alignment);
    trackHotPathAllocation(ptr, size, true, function, file, line);
    return ptr;
}
void NumaAwareMemoryPool::deallocateHotPath(void *ptr, std::size_t size) noexcept
{
    deallocate(ptr, size);
}
std::vector<NumaNodeInfo> NumaAwareMemoryPool::getNumaNodes() const
{
    return numa_nodes_;
}
int NumaAwareMemoryPool::getCurrentNumaNode() const
{
    if (!numa_available_)
    {
        return 0;
    }
#if defined(__linux__) && HAS_NUMA
    const int cpu = sched_getcpu();
    if (cpu < 0)
    {
        return defaultNode(numa_nodes_);
    }
    const int node = numa_node_of_cpu(cpu);
    return normalizeNode(numa_nodes_, node);
#else
    return defaultNode(numa_nodes_);
#endif
}
bool NumaAwareMemoryPool::bindToNode(int nodeId)
{
    if (!numa_available_ || !isKnownNode(numa_nodes_, nodeId))
    {
        return false;
    }
#if defined(__linux__) && HAS_NUMA
    if (numa_run_on_node(nodeId) != 0)
    {
        return false;
    }
    numa_set_preferred(nodeId);
#endif
    current_node_ = nodeId;
    return true;
}
bool NumaAwareMemoryPool::bindThreadToNode(std::thread::id threadId, int nodeId)
{
    if (threadId != std::this_thread::get_id())
    {
        return false;
    }
    return bindToNode(nodeId);
}
void NumaAwareMemoryPool::setNumaPolicy(NumaPolicy policy, int preferredNode)
{
    config_.policy = policy;
    config_.preferredNode = preferredNode;
}
std::size_t NumaAwareMemoryPool::totalAllocated() const noexcept
{
    return total_allocated_.load();
}
std::size_t NumaAwareMemoryPool::totalCapacity() const noexcept
{
    return total_capacity_.load();
}
std::vector<HotPathAllocation> NumaAwareMemoryPool::getHotPathAllocations() const
{
    std::lock_guard<std::mutex> lock(hot_path_mutex_);
    return hot_path_allocations_;
}
std::size_t NumaAwareMemoryPool::countHotPathAllocations() const
{
    std::lock_guard<std::mutex> lock(hot_path_mutex_);
    return hot_path_allocations_.size();
}
std::size_t NumaAwareMemoryPool::countHotPathViolations() const
{
    return hot_path_violations_.load();
}
bool NumaAwareMemoryPool::validateNoHiddenAllocations() const
{
    std::lock_guard<std::mutex> lock(hot_path_mutex_);
    for (const auto &alloc : hot_path_allocations_)
    {
        if (!alloc.wasFromPool)
        {
            return false;
        }
    }
    return true;
}
void NumaAwareMemoryPool::resetHotPathTracking()
{
    std::lock_guard<std::mutex> lock(hot_path_mutex_);
    hot_path_allocations_.clear();
    hot_path_violations_ = 0;
}
void NumaAwareMemoryPool::defragmentAllPools()
{
    std::lock_guard<std::mutex> lock(pools_mutex_);
    for (auto &[nodeId, pool] : node_pools_)
    {
        if (pool)
        {
            pool->defragment();
        }
    }
}
void NumaAwareMemoryPool::resetAllPools()
{
    std::lock_guard<std::mutex> lock(pools_mutex_);
    for (auto &[nodeId, pool] : node_pools_)
    {
        if (pool)
        {
            if (is_deterministic_)
            {
                pool->reset(determinism_contract_);
            }
            else
            {
                pool->reset();
            }
        }
    }
    total_allocated_ = 0;
}
void NumaAwareMemoryPool::preallocate_pools(std::size_t size_per_pool)
{
    std::lock_guard<std::mutex> lock(pools_mutex_);
    for (const auto &node : numa_nodes_)
    {
        if (node.isAvailable)
        {
            auto pool = is_deterministic_
                            ? std::make_unique<MemoryPool>(determinism_contract_, size_per_pool)
                            : std::make_unique<MemoryPool>(size_per_pool);
            node_pools_[node.nodeId] = std::move(pool);
        }
    }
    std::size_t totalCapacity = 0;
    for (const auto &[nodeId, pool] : node_pools_)
    {
        if (pool)
        {
            const std::size_t capacity = pool->capacity();
            totalCapacity = capacity > std::numeric_limits<std::size_t>::max() - totalCapacity
                                ? std::numeric_limits<std::size_t>::max()
                                : totalCapacity + capacity;
        }
    }
    total_capacity_ = totalCapacity;
}
bool NumaAwareMemoryPool::isDeterministic() const noexcept
{
    return is_deterministic_;
}
void NumaAwareMemoryPool::setDeterminismContract(const DeterminismContract &contract)
{
    determinism_contract_ = contract;
    is_deterministic_ = true;
    std::lock_guard<std::mutex> lock(pools_mutex_);
    for (auto &[nodeId, pool] : node_pools_)
    {
        if (pool)
        {
            pool->setDeterminismContract(contract);
        }
    }
}
DeterminismMetadata NumaAwareMemoryPool::getDeterminismMetadata() const
{
    if (!is_deterministic_)
    {
        return DeterminismMetadata{};
    }
    std::lock_guard<std::mutex> lock(pools_mutex_);
    for (const auto &[nodeId, pool] : node_pools_)
    {
        if (pool)
        {
            return pool->getDeterminismMetadata();
        }
    }
    return DeterminismMetadata{};
}
void NumaAwareMemoryPool::enableAllocationTracking(bool enable)
{
    hot_path_tracking_enabled_ = enable;
}
void NumaAwareMemoryPool::setAllocationCallback(std::function<void(int, std::size_t)> callback)
{
    config_.allocationCallback = callback;
}
std::unordered_map<int, std::vector<MemoryBlock>> NumaAwareMemoryPool::getPools() const
{
    std::lock_guard<std::mutex> lock(pools_mutex_);
    std::unordered_map<int, std::vector<MemoryBlock>> pools;
    pools.reserve(node_pools_.size());
    for (const auto &[nodeId, pool] : node_pools_)
    {
        if (pool)
        {
            pools.emplace(nodeId, pool->getBlocks());
        }
    }
    return pools;
}
void NumaAwareMemoryPool::initializeNuma()
{
#if defined(__linux__) && HAS_NUMA
    if (config_.enableNumaBinding && numa_available() != -1)
    {
        numa_available_ = true;
        detectNumaTopology();
    }
    else
    {
        numa_available_ = false;
        ensureDefaultNode(numa_nodes_);
        auto pool = is_deterministic_ ? std::make_unique<MemoryPool>(determinism_contract_,
                                                                     config_.poolSizePerNode)
                                      : std::make_unique<MemoryPool>(config_.poolSizePerNode);
        node_pools_[0] = std::move(pool);
        addSaturating(total_capacity_, config_.poolSizePerNode);
    }
#else
    numa_available_ = false;
    ensureDefaultNode(numa_nodes_);
    auto pool = is_deterministic_
                    ? std::make_unique<MemoryPool>(determinism_contract_, config_.poolSizePerNode)
                    : std::make_unique<MemoryPool>(config_.poolSizePerNode);
    node_pools_[0] = std::move(pool);
    addSaturating(total_capacity_, config_.poolSizePerNode);
#endif
}
void NumaAwareMemoryPool::detectNumaTopology()
{
#if defined(__linux__) && HAS_NUMA
    numa_nodes_.clear();
    const int max_node = std::min(numa_max_node(), kMaxNumaNodes - 1);
    for (int i = 0; i <= max_node; ++i)
    {
        NumaNodeInfo node_info;
        node_info.nodeId = i;
        const long long node_size = numa_node_size64(i, nullptr);
        node_info.isAvailable = node_size > 0;
        if (node_info.isAvailable)
        {
            node_info.memorySize = static_cast<std::size_t>(node_size);
            for (int cpu = 0; cpu < numa_num_configured_cpus(); ++cpu)
            {
                if (numa_node_of_cpu(cpu) == i)
                {
                    node_info.cpu_list.push_back(cpu);
                }
            }
            node_info.cpuCount = static_cast<int>(node_info.cpu_list.size());
            numa_nodes_.push_back(node_info);
        }
    }
    ensureDefaultNode(numa_nodes_);
#else
    ensureDefaultNode(numa_nodes_);
#endif
}
int NumaAwareMemoryPool::getPreferredNode() const
{
    switch (config_.policy)
    {
        case NumaPolicy::PREFERRED:
            return normalizeNode(numa_nodes_, config_.preferredNode);
        case NumaPolicy::LOCAL:
            return getCurrentNumaNode();
        case NumaPolicy::BIND:
            return normalizeNode(numa_nodes_, current_node_);
        case NumaPolicy::INTERLEAVE:
            if (numa_nodes_.empty() || config_.poolSizePerNode == 0)
            {
                return 0;
            }
            return numa_nodes_[static_cast<std::size_t>(total_allocated_.load() /
                                                        config_.poolSizePerNode) %
                               numa_nodes_.size()]
                .nodeId;
        case NumaPolicy::AUTO:
        default:
            return getCurrentNumaNode();
    }
}
MemoryPool *NumaAwareMemoryPool::getOrCreatePoolForNode(int nodeId)
{
    std::lock_guard<std::mutex> lock(pools_mutex_);
    nodeId = normalizeNode(numa_nodes_, nodeId);
    auto it = node_pools_.find(nodeId);
    if (it != node_pools_.end())
    {
        return it->second.get();
    }
    auto pool = is_deterministic_
                    ? std::make_unique<MemoryPool>(determinism_contract_, config_.poolSizePerNode)
                    : std::make_unique<MemoryPool>(config_.poolSizePerNode);
    auto *pool_ptr = pool.get();
    node_pools_[nodeId] = std::move(pool);
    addSaturating(total_capacity_, config_.poolSizePerNode);
    return pool_ptr;
}
void NumaAwareMemoryPool::trackHotPathAllocation(void *ptr, std::size_t size, bool from_pool,
                                                 const char *function, const char *file, int line)
{
    if (!hot_path_tracking_enabled_ || ptr == nullptr)
    {
        return;
    }
    std::lock_guard<std::mutex> lock(hot_path_mutex_);
    HotPathAllocation alloc;
    alloc.ptr = ptr;
    alloc.size = size;
    alloc.threadId = std::this_thread::get_id();
    alloc.timestamp = std::chrono::high_resolution_clock::now();
    alloc.functionName = function;
    alloc.fileName = file;
    alloc.lineNumber = line;
    alloc.wasFromPool = from_pool;
    hot_path_allocations_.push_back(alloc);
    if (!from_pool)
    {
        hot_path_violations_.fetch_add(1, std::memory_order_relaxed);
    }
}
void NumaAwareMemoryPool::untrackHotPathAllocation(void *ptr)
{
    if (!hot_path_tracking_enabled_ || ptr == nullptr)
    {
        return;
    }
    std::lock_guard<std::mutex> lock(hot_path_mutex_);
    auto it = std::find_if(hot_path_allocations_.begin(), hot_path_allocations_.end(),
                           [ptr](const HotPathAllocation &alloc) { return alloc.ptr == ptr; });
    if (it != hot_path_allocations_.end())
    {
        hot_path_allocations_.erase(it);
    }
}
void *NumaAwareMemoryPool::allocateWithNumaPolicy(std::size_t size, std::size_t alignment)
{
    int nodeId = getPreferredNode();
    auto *pool = getOrCreatePoolForNode(nodeId);
    void *ptr = pool->allocate(size, alignment);
    if (config_.allocationCallback)
    {
        config_.allocationCallback(nodeId, size);
    }
    addSaturating(total_allocated_, size);
    return ptr;
}
void NumaAwareMemoryPool::updateStatistics()
{
    std::size_t totalCapacity = 0;
    std::lock_guard<std::mutex> lock(pools_mutex_);
    for (const auto &[nodeId, pool] : node_pools_)
    {
        if (pool)
        {
            const std::size_t capacity = pool->capacity();
            totalCapacity = capacity > std::numeric_limits<std::size_t>::max() - totalCapacity
                                ? std::numeric_limits<std::size_t>::max()
                                : totalCapacity + capacity;
        }
    }
    total_capacity_ = totalCapacity;
}
} // namespace nerve::core

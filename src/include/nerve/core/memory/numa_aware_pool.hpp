
#pragma once
#include "nerve/config.hpp"
#include "nerve/core/memory/memory_pool.hpp"
#include "nerve/core/rng/determinism_contract.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <thread>
#include <unordered_map>
#include <vector>
#if defined(__linux__) && HAS_NUMA
#include <numa.h>
#include <sched.h>
#include <unistd.h>
#endif

namespace nerve::core
{

struct NumaNodeInfo
{
    int nodeId;
    std::size_t memorySize;
    int cpuCount;
    std::vector<int> cpu_list;
    bool isAvailable;
    NumaNodeInfo()
        : nodeId(-1)
        , memorySize(0)
        , cpuCount(0)
        , isAvailable(false)
    {}
};

enum class NumaPolicy
{
    AUTO,
    LOCAL,
    INTERLEAVE,
    PREFERRED,
    BIND
};

struct NumaPoolConfig
{
    NumaPolicy policy;
    int preferredNode;
    std::size_t poolSizePerNode;
    bool enableNumaBinding;
    bool enableHotPathTracking;
    std::function<void(int, std::size_t)> allocationCallback;
    NumaPoolConfig()
        : policy(NumaPolicy::AUTO)
        , preferredNode(-1)
        , poolSizePerNode(nerve::kDefaultMemoryPoolSize)
        , enableNumaBinding(true)
        , enableHotPathTracking(true)
    {}
};

struct HotPathAllocation
{
    void *ptr;
    std::size_t size;
    std::thread::id threadId;
    std::chrono::high_resolution_clock::time_point timestamp;
    const char *functionName;
    const char *fileName;
    int lineNumber;
    bool wasFromPool;
    HotPathAllocation()
        : ptr(nullptr)
        , size(0)
        , threadId()
        , functionName(nullptr)
        , fileName(nullptr)
        , lineNumber(0)
        , wasFromPool(false)
    {}
};
class NumaAwareMemoryPool
{
public:
    explicit NumaAwareMemoryPool(const NumaPoolConfig &config = NumaPoolConfig{});
    explicit NumaAwareMemoryPool(const DeterminismContract &contract,
                                 const NumaPoolConfig &config = NumaPoolConfig{});
    ~NumaAwareMemoryPool();
    NumaAwareMemoryPool(const NumaAwareMemoryPool &) = delete;
    NumaAwareMemoryPool &operator=(const NumaAwareMemoryPool &) = delete;
    NumaAwareMemoryPool(NumaAwareMemoryPool &&other) noexcept;
    NumaAwareMemoryPool &operator=(NumaAwareMemoryPool &&other) noexcept;
    void *allocate(std::size_t size, std::size_t alignment = alignof(std::max_align_t));
    void *allocate(std::size_t size, std::size_t alignment, const DeterminismContract &contract);
    void *allocateOnNode(std::size_t size, int nodeId,
                         std::size_t alignment = alignof(std::max_align_t));
    void deallocate(void *ptr, std::size_t size) noexcept;
    void *allocateHotPath(std::size_t size, std::size_t alignment = alignof(std::max_align_t),
                          const char *function = nullptr, const char *file = nullptr, int line = 0);
    void deallocateHotPath(void *ptr, std::size_t size) noexcept;
    std::vector<NumaNodeInfo> getNumaNodes() const;
    int getCurrentNumaNode() const;
    bool bindToNode(int nodeId);
    bool bindThreadToNode(std::thread::id threadId, int nodeId);
    void setNumaPolicy(NumaPolicy policy, int preferredNode = -1);
    std::size_t totalAllocated() const noexcept;
    std::size_t totalCapacity() const noexcept;
    std::vector<HotPathAllocation> getHotPathAllocations() const;
    std::size_t countHotPathAllocations() const;
    std::size_t countHotPathViolations() const;
    bool validateNoHiddenAllocations() const;
    void resetHotPathTracking();
    void defragmentAllPools();
    void resetAllPools();
    void preallocate_pools(std::size_t size_per_pool);
    bool isDeterministic() const noexcept;
    void setDeterminismContract(const DeterminismContract &contract);
    DeterminismMetadata getDeterminismMetadata() const;
    void enableAllocationTracking(bool enable);
    void setAllocationCallback(std::function<void(int, std::size_t)> callback);
    std::unordered_map<int, std::vector<MemoryBlock>> getPools() const;

private:
    NumaPoolConfig config_;
    std::unordered_map<int, std::unique_ptr<MemoryPool>> node_pools_;
    mutable std::mutex pools_mutex_;
    std::vector<NumaNodeInfo> numa_nodes_;
    int current_node_;
    bool numa_available_;
    std::vector<HotPathAllocation> hot_path_allocations_;
    mutable std::mutex hot_path_mutex_;
    std::atomic<std::size_t> hot_path_violations_{0};
    bool hot_path_tracking_enabled_;
    DeterminismContract determinism_contract_;
    bool is_deterministic_;
    std::atomic<std::size_t> total_allocated_{0};
    std::atomic<std::size_t> total_capacity_{0};
    void initializeNuma();
    void detectNumaTopology();
    int getPreferredNode() const;
    MemoryPool *getOrCreatePoolForNode(int nodeId);
    void trackHotPathAllocation(void *ptr, std::size_t size, bool from_pool, const char *function,
                                const char *file, int line);
    void untrackHotPathAllocation(void *ptr);
    void *allocateWithNumaPolicy(std::size_t size, std::size_t alignment);
    void updateStatistics();
};
class NumaPoolManager
{
public:
    static NumaPoolManager &instance();
    NumaAwareMemoryPool &getGlobalPool();
    NumaAwareMemoryPool &getPoolForThread(std::thread::id threadId = std::this_thread::get_id());
    void configureGlobalPool(const NumaPoolConfig &config);
    void configureThreadPool(std::thread::id threadId, const NumaPoolConfig &config);
    bool bindCurrentThreadToNode(int nodeId);
    bool bindThreadToNode(std::thread::id threadId, int nodeId);
    int getThreadNumaNode(std::thread::id threadId = std::this_thread::get_id());
    bool validateAllThreadsNoHiddenAllocations();
    std::unordered_map<std::thread::id, std::size_t> getThreadAllocationCounts();
    void resetAllTracking();
    std::size_t getTotalAllocated() const;
    std::size_t getTotalCapacity() const;
    std::vector<NumaNodeInfo> getNumaTopology() const;

private:
    NumaPoolManager() = default;
    ~NumaPoolManager() = default;
    std::unique_ptr<NumaAwareMemoryPool> global_pool_;
    std::unordered_map<std::thread::id, std::unique_ptr<NumaAwareMemoryPool>> thread_pools_;
    mutable std::mutex manager_mutex_;
    NumaPoolConfig global_config_;
};
class HotPathAllocationTracker
{
public:
    HotPathAllocationTracker(NumaAwareMemoryPool &pool, const char *function, const char *file,
                             int line)
        : pool_(pool)
        , function_(function)
        , file_(file)
        , line_(line)
    {}
    void *allocate(std::size_t size, std::size_t alignment = alignof(std::max_align_t))
    {
        void *ptr = pool_.allocateHotPath(size, alignment, function_, file_, line_);
        allocations_.push_back({ptr, size});
        return ptr;
    }
    ~HotPathAllocationTracker()
    {
        for (const auto &allocation : allocations_)
        {
            pool_.deallocateHotPath(allocation.first, allocation.second);
        }
    }

private:
    NumaAwareMemoryPool &pool_;
    const char *function_;
    const char *file_;
    int line_;
    std::vector<std::pair<void *, std::size_t>> allocations_;
};
#define NERVE_ALLOCATE_HOT_PATH(pool, size)                                                        \
    (pool).allocateHotPath((size), alignof(std::max_align_t), __FUNCTION__, __FILE__, __LINE__)
#define NERVE_HOT_PATH_TRACKER(pool)                                                               \
    HotPathAllocationTracker tracker((pool), __FUNCTION__, __FILE__, __LINE__)
} // namespace nerve::core

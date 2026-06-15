
#pragma once

#include "nerve/core_types.hpp"
#include "nerve/errors/errors.hpp"
#include "nerve/persistence/adaptive_acceleration/adaptive_acceleration_system_capabilities.hpp"
#include "nerve/persistence/core/core_types.hpp"

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace nerve::persistence::adaptive_acceleration
{

struct LockfreeConfig
{
    std::size_t max_threads = 0;
    bool enable_numa = true;
    bool enable_memory_interleaving = true;
    std::size_t preferred_node = 0;
    bool enable_thread_affinity = true;
};

struct ReductionStats
{
    double computation_time_ms = 0.0;
    std::size_t columns_processed = 0;
    std::size_t threads_used = 0;
    std::size_t atomic_operations = 0;
    double scaling_efficiency = 1.0;
    std::string threading_details;
};

class LockfreeMatrixColumn
{
public:
    explicit LockfreeMatrixColumn(std::size_t initial_capacity = 16);

    bool push_back(int value);
    bool pop_back(int &value);
    bool push_front(int value);
    bool pop_front(int &value);

    [[nodiscard]] std::size_t size() const;
    [[nodiscard]] bool empty() const;
    void reserve(std::size_t new_capacity);
    void shrinkToFit();
    void clear();

    [[nodiscard]] const std::vector<int> &values() const;
    void assignSortedUnique(std::vector<int> values);

private:
    std::vector<int> values_;
};

class LockfreeReducer
{
public:
    static errors::ErrorResult<std::unique_ptr<LockfreeReducer>>
    create(const LockfreeConfig &config);

    ~LockfreeReducer();

    errors::ErrorResult<std::vector<::nerve::persistence::Pair>>
    reduceParallel(const std::vector<LockfreeMatrixColumn> &columns, std::size_t num_threads);

    void addColumnAtomic(std::size_t target_col, const LockfreeMatrixColumn &source_col);
    [[nodiscard]] bool findPivotAtomic(std::size_t col, int &pivot);

    [[nodiscard]] const ReductionStats &getPerformanceStats() const;

private:
    explicit LockfreeReducer(const LockfreeConfig &config);
    class Impl;
    std::unique_ptr<Impl> impl_;
};

class NUMAScheduler
{
public:
    struct NUMAConfig
    {
        bool enable_numa = true;
        std::size_t preferred_node = 0;
        bool memory_interleaving = true;
        bool set_thread_affinity = true;
        bool round_robin_scheduling = false;
    };

    static errors::ErrorResult<void> optimizeThreadAffinity(std::size_t num_threads,
                                                            const NUMAConfig &config);

    static errors::ErrorResult<std::vector<CPUInfo>> getCpuTopology();
    static void bindThreadToCpu(std::thread::id thread_id, std::size_t cpu_id);
    static void setMemoryPolicy(std::size_t node_id);
    static std::vector<std::size_t> getNumaNodes();

private:
    static NUMAConfig current_config_;
};

} // namespace nerve::persistence::adaptive_acceleration

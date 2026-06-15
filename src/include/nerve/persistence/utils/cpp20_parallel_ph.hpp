#pragma once

#include "nerve/core.hpp"
#include "nerve/persistence/core/roaring_bitmap.hpp"
#include "nerve/persistence/utils/bit_parallel_z2.hpp"

#include <compare>
#include <condition_variable>
#include <coroutine>
#include <execution>
#include <functional>
#include <mutex>
#include <optional>
#include <queue>
#include <ranges>
#include <shared_mutex>
#include <thread>

namespace nerve::persistence::cpp20
{

// Import BitColumn from bitparallel namespace
using bitparallel::BitColumn;

/**
 * @brief C++20 helpers for dependency-aware persistence reductions.
 *
 * Column reduction has true data dependencies through the pivot table, so this
 * module keeps the pivot-table commit deterministic and uses C++20 facilities
 * for safe preprocessing, task scheduling, lazy column construction, and
 * benchmark instrumentation.
 */

/**
 * @brief Parallel configuration
 */
struct ParallelConfig
{
    bool use_parallel = true;
    std::execution::parallel_unsequenced_policy
        execution_policy; // Retained for source compatibility.
    size_t chunk_size = 64;
    int num_threads = 0; // 0 = auto
};

/**
 * @brief Task-based configuration
 */
struct TaskConfig
{
    int num_threads = 0; // 0 = auto
    bool work_stealing = true;
    bool dynamic_scheduling = true;
};

/**
 * @brief Range-based configuration
 */
struct RangeConfig
{
    size_t max_columns = 0; // 0 = unlimited
    size_t batch_size = 1000;
};

/**
 * @brief Parallel reduction result
 */
struct ParallelReductionResult
{
    double reduction_time_ms = 0.0;
    int num_columns_processed = 0;

    [[nodiscard]] auto operator<=>(const ParallelReductionResult &other) const = default;
    [[nodiscard]] bool operator==(const ParallelReductionResult &other) const = default;
};

/**
 * @brief Task-based reduction result
 */
struct TaskBasedReductionResult
{
    double reduction_time_ms = 0.0;
    int num_threads_used = 0;

    [[nodiscard]] auto operator<=>(const TaskBasedReductionResult &other) const = default;
    [[nodiscard]] bool operator==(const TaskBasedReductionResult &other) const = default;
};

/**
 * @brief Benchmark results
 */
struct ParallelBenchmark
{
    double sequential_time_ms = 0.0;
    double parallel_time_ms = 0.0;
    double speedup = 1.0;
    double theoretical_speedup = 1.0;
    int num_threads = 1;

    [[nodiscard]] auto operator<=>(const ParallelBenchmark &other) const = default;
    [[nodiscard]] bool operator==(const ParallelBenchmark &other) const = default;
};

/**
 * @brief Speedup estimate
 */
struct ParallelSpeedupEstimate
{
    double thread_scaling = 1.0;
    double cache_efficiency = 1.0;
    double total_speedup = 1.0;

    [[nodiscard]] auto operator<=>(const ParallelSpeedupEstimate &other) const = default;
    [[nodiscard]] bool operator==(const ParallelSpeedupEstimate &other) const = default;
};

/**
 * @brief Thread-safe pivot table
 */
class ThreadSafePivotTable
{
public:
    int find(int pivot);
    bool tryRegister(int pivot, int column_idx);

private:
    std::unordered_map<int, int> pivot_to_column_;
    std::shared_mutex mutex_;
};

/**
 * @brief Task queue for work-stealing
 */
class TaskQueue
{
public:
    void push(std::function<void()> task);
    std::optional<std::function<void()>> pop();
    void done();

private:
    std::queue<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool done_ = false;
};

/**
 * @brief Simple coroutine generator
 */
template <typename T>
struct generator
{
    struct promise_type
    {
        T current_value;

        auto get_return_object() { return generator{this}; }
        std::suspend_always initial_suspend() { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }
        void unhandled_exception() { std::terminate(); }
        std::suspend_always yield_value(T value)
        {
            current_value = value;
            return {};
        }
        void return_void() {}
    };

    using handle_type = std::coroutine_handle<promise_type>;
    handle_type handle;

    generator(promise_type *p)
        : handle(handle_type::from_promise(*p))
    {}
    generator(const generator &) = delete;
    generator &operator=(const generator &) = delete;
    generator(generator &&other) noexcept
        : handle(other.handle)
    {
        other.handle = nullptr;
    }
    generator &operator=(generator &&other) noexcept
    {
        if (this != &other)
        {
            if (handle)
                handle.destroy();
            handle = other.handle;
            other.handle = nullptr;
        }
        return *this;
    }
    ~generator()
    {
        if (handle)
            handle.destroy();
    }

    // Iterator support
    struct iterator
    {
        handle_type handle;
        bool done = true;
        bool operator!=(const iterator &other) const { return done != other.done; }
        iterator &operator++()
        {
            handle.resume();
            done = handle.done();
            return *this;
        }
        const T &operator*() const { return handle.promise().current_value; }
    };

    iterator begin()
    {
        if (!handle)
            return {nullptr, true};
        handle.resume();
        return {handle, handle.done()};
    }

    iterator end() { return {nullptr, true}; }
};

/**
 * @brief Parallel column reduction
 *
 * Uses parallel preprocessing when requested, then commits the pivot table in
 * column order so the reduction is deterministic and data-race free.
 *
 * @tparam ColumnType Column type (must satisfy SparseColumn concept)
 * @param columns Columns to reduce
 * @param config Parallel configuration
 * @return Reduction result
 */
template <typename ColumnType>
ParallelReductionResult reduceColumnsParallel(std::vector<ColumnType> &columns,
                                              const ParallelConfig &config);

// Explicit instantiation declaration
extern template ParallelReductionResult
reduceColumnsParallel<roaring::HybridColumn>(std::vector<roaring::HybridColumn> &,
                                             const ParallelConfig &);
extern template ParallelReductionResult reduceColumnsParallel<BitColumn>(std::vector<BitColumn> &,
                                                                         const ParallelConfig &);

/**
 * @brief Task-based parallel reduction
 *
 * Uses a task queue for independent pivot preprocessing, then performs the
 * dependency-ordered reduction commit.
 *
 * @param columns Columns to reduce
 * @param config Task configuration
 * @return Reduction result
 */
TaskBasedReductionResult reduceColumnsTaskBased(std::vector<BitColumn> &columns,
                                                const TaskConfig &config);

/**
 * @brief Ranges-based column processing
 *
 * Uses C++20 ranges for lazy, composable pipelines.
 *
 * @param columns Input columns
 * @param config Range configuration
 * @return Processed columns
 */
std::vector<BitColumn> filterAndProcessColumns(std::vector<BitColumn> &columns,
                                               const RangeConfig &config);

/**
 * @brief Coroutine-based lazy column generator
 *
 * Generates columns on-demand for memory efficiency.
 *
 * @param boundary_matrix Boundary matrix data
 * @param filtration_values Filtration values
 * @return Generator yielding columns
 */
generator<BitColumn> lazyColumnGenerator(const std::vector<std::vector<int>> &boundary_matrix,
                                         const std::vector<double> &filtration_values);

/**
 * @brief Benchmark parallel vs sequential
 *
 * @param columns Test columns
 * @param iterations Number of iterations
 * @return Benchmark results
 */
ParallelBenchmark benchmarkParallel(std::vector<BitColumn> &columns, int iterations = 100);

/**
 * @brief Get optimal parallel configuration
 *
 * @param num_columns Number of columns
 * @param num_rows Number of rows
 * @return Optimal configuration
 */
ParallelConfig getOptimalParallelConfig(size_t num_columns, int num_rows);

/**
 * @brief Get optimal task configuration
 *
 * @param num_columns Number of columns
 * @return Optimal configuration
 */
TaskConfig getOptimalTaskConfig(size_t num_columns);

/**
 * @brief Estimate parallel speedup
 *
 * @param num_columns Number of columns
 * @param num_threads Number of threads
 * @return Speedup estimate
 */
ParallelSpeedupEstimate estimateParallelSpeedup(size_t num_columns, size_t num_threads);

/**
 * @brief Check if parallel execution beneficial
 *
 * @param num_columns Number of columns
 * @return true if parallel recommended
 */
inline bool shouldUseParallel(size_t num_columns)
{
    return num_columns > 500; // Threshold for parallel overhead
}

} // namespace nerve::persistence::cpp20

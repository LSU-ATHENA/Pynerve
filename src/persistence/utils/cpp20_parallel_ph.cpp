// C++20 dependency-aware persistent-homology utilities.
//
// Pivot-table updates are committed in column order. Independent work such as
// initial pivot discovery can run concurrently, but the algebraic reduction
// step remains deterministic.

#include "nerve/persistence/utils/cpp20_parallel_ph.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <thread>
#include <utility>
#include <vector>

namespace nerve::persistence::cpp20
{

// Concept: Sparse column requirements
template <typename T>
concept SparseColumn = requires(T col) {
    { col.computePivot() } -> std::convertible_to<int>;
    { col.isEmpty() } -> std::convertible_to<bool>;
    { col.xorInPlace(col) } -> std::same_as<void>;
    { col.sparsity() } -> std::convertible_to<double>;
};

// Concept: Dense column requirements
template <typename T>
concept DenseColumn = requires(T col) {
    { col.computePivot() } -> std::convertible_to<int>;
    { col.words } -> std::convertible_to<std::span<uint64_t>>;
};

namespace
{

double finiteBenchmarkSpeedup(double baseline_ms, double accelerated_ms)
{
    if (!std::isfinite(baseline_ms) || baseline_ms < 0.0 || !std::isfinite(accelerated_ms) ||
        accelerated_ms <= 0.0)
    {
        return 1.0;
    }
    const double speedup = baseline_ms / accelerated_ms;
    return std::isfinite(speedup) && speedup >= 0.0 ? speedup : 1.0;
}

size_t resolveThreadCount(int requested_threads, size_t work_items)
{
    if (work_items == 0)
        return 0;
    const size_t hardware_threads =
        static_cast<size_t>(std::max(1U, std::thread::hardware_concurrency()));
    const size_t requested =
        requested_threads > 0 ? static_cast<size_t>(requested_threads) : hardware_threads;
    return std::clamp(requested, size_t{1}, work_items);
}

template <typename Func>
void parallelFor(size_t work_items, int requested_threads, Func &&func)
{
    const size_t num_threads = resolveThreadCount(requested_threads, work_items);
    if (num_threads <= 1 || work_items < 2)
    {
        for (size_t i = 0; i < work_items; ++i)
        {
            func(i);
        }
        return;
    }

    std::vector<std::thread> workers;
    workers.reserve(num_threads);
    const size_t chunk = (work_items + num_threads - 1) / num_threads;
    for (size_t worker = 0; worker < num_threads; ++worker)
    {
        const size_t begin = worker * chunk;
        const size_t end = std::min(work_items, begin + chunk);
        if (begin >= end)
            break;
        workers.emplace_back([begin, end, &func]() {
            for (size_t i = begin; i < end; ++i)
            {
                func(i);
            }
        });
    }
    for (auto &worker : workers)
    {
        worker.join();
    }
}

} // namespace

// Parallel column reduction using C++20 std::execution
template <SparseColumn ColumnType>
ParallelReductionResult reduceColumnsParallel(std::vector<ColumnType> &columns,
                                              const ParallelConfig &config)
{
    ParallelReductionResult result{};
    auto start = std::chrono::high_resolution_clock::now();

    std::unordered_map<int, int> pivot_to_column;
    pivot_to_column.reserve(columns.size());

    std::vector<size_t> indices(columns.size());
    std::iota(indices.begin(), indices.end(), 0);
    if (config.use_parallel && columns.size() > 1)
    {
        parallelFor(indices.size(), config.num_threads,
                    [&](size_t index) { columns[indices[index]].computePivot(); });
    }

    for (size_t i = 0; i < columns.size(); ++i)
    {
        auto &col = columns[i];
        if (col.isEmpty())
            continue;

        int pivot = col.computePivot();
        while (pivot >= 0)
        {
            auto it = pivot_to_column.find(pivot);
            if (it == pivot_to_column.end())
            {
                pivot_to_column[pivot] = static_cast<int>(i);
                break;
            }
            col.xorInPlace(columns[static_cast<size_t>(it->second)]);
            pivot = col.computePivot();
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.reduction_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.num_columns_processed = static_cast<int>(columns.size());

    return result;
}

// Explicit instantiation for common types
template ParallelReductionResult
reduceColumnsParallel<roaring::HybridColumn>(std::vector<roaring::HybridColumn> &,
                                             const ParallelConfig &);
template ParallelReductionResult reduceColumnsParallel<BitColumn>(std::vector<BitColumn> &,
                                                                  const ParallelConfig &);

// Task-based parallel reduction with custom scheduler
TaskBasedReductionResult reduceColumnsTaskBased(std::vector<BitColumn> &columns,
                                                const TaskConfig &config)
{
    TaskBasedReductionResult result{};
    auto start = std::chrono::high_resolution_clock::now();

    const size_t num_threads = resolveThreadCount(config.num_threads, columns.size());

    TaskQueue task_queue;
    std::vector<int> initial_pivots(columns.size(), -1);
    for (size_t i = 0; i < columns.size(); ++i)
    {
        task_queue.push([&, i]() { initial_pivots[i] = columns[i].computePivot(); });
    }
    task_queue.done();

    std::vector<std::thread> workers;
    workers.reserve(num_threads);
    for (size_t i = 0; i < num_threads; ++i)
    {
        workers.emplace_back([&task_queue]() {
            while (auto task = task_queue.pop())
            {
                task.value()();
            }
        });
    }
    for (auto &worker : workers)
    {
        worker.join();
    }

    ThreadSafePivotTable pivot_table;

    // Persistent reduction depends on finalized lower-index columns. Keep this
    // commit ordered while using the task queue above for independent work.
    for (size_t i = 0; i < columns.size(); ++i)
    {
        if (initial_pivots[i] < 0 || columns[i].isEmpty())
            continue;
        int pivot = columns[i].computePivot();
        while (pivot >= 0)
        {
            const int other_idx = pivot_table.find(pivot);
            if (other_idx < 0)
            {
                pivot_table.tryRegister(pivot, static_cast<int>(i));
                break;
            }
            columns[i].xorInPlace(columns[static_cast<size_t>(other_idx)]);
            pivot = columns[i].computePivot();
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.reduction_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.num_threads_used = static_cast<int>(num_threads);

    return result;
}

// Ranges-based pipeline for column processing
std::vector<BitColumn> filterAndProcessColumns(std::vector<BitColumn> &columns,
                                               const RangeConfig &config)
{
    std::vector<BitColumn> result;
    for (auto &col : columns)
    {
        const bool under_limit = config.max_columns == 0 || result.size() < config.max_columns;
        if (!col.isEmpty() && under_limit)
        {
            result.push_back(col);
        }
    }

    parallelFor(result.size(), 0, [&](size_t i) { result[i].computePivot(); });

    return result;
}

// Coroutine-based lazy evaluation (C++20)
generator<BitColumn> lazyColumnGenerator(const std::vector<std::vector<int>> &boundary_matrix,
                                         const std::vector<double> &filtration_values)
{
    for (size_t i = 0; i < boundary_matrix.size(); ++i)
    {
        int max_row = static_cast<int>(filtration_values.size());
        if (max_row == 0 && !boundary_matrix[i].empty())
        {
            const auto max_it =
                std::max_element(boundary_matrix[i].begin(), boundary_matrix[i].end());
            max_row = *max_it + 1;
        }
        co_yield BitColumn::fromSparseIndices(boundary_matrix[i], max_row);
    }
}

// Benchmark C++20 parallel vs sequential
ParallelBenchmark benchmarkParallel(std::vector<BitColumn> &columns, int iterations)
{
    ParallelBenchmark bench{};
    const unsigned hardware_threads = std::max(1U, std::thread::hardware_concurrency());
    bench.num_threads = static_cast<int>(hardware_threads);
    bench.theoretical_speedup = static_cast<double>(hardware_threads);
    if (columns.empty() || iterations <= 0)
    {
        return bench;
    }

    auto start_seq = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter)
    {
        for (auto &col : columns)
        {
            col.computePivot();
        }
    }
    auto end_seq = std::chrono::high_resolution_clock::now();
    bench.sequential_time_ms =
        std::chrono::duration<double, std::milli>(end_seq - start_seq).count();

    auto start_par = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter)
    {
        parallelFor(columns.size(), bench.num_threads,
                    [&](size_t i) { columns[i].computePivot(); });
    }
    auto end_par = std::chrono::high_resolution_clock::now();
    bench.parallel_time_ms = std::chrono::duration<double, std::milli>(end_par - start_par).count();

    bench.speedup = finiteBenchmarkSpeedup(bench.sequential_time_ms, bench.parallel_time_ms);

    return bench;
}

// Thread-safe pivot table implementation
int ThreadSafePivotTable::find(int pivot)
{
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = pivot_to_column_.find(pivot);
    if (it != pivot_to_column_.end())
    {
        return it->second;
    }
    return -1;
}

bool ThreadSafePivotTable::tryRegister(int pivot, int column_idx)
{
    std::unique_lock<std::shared_mutex> lock(mutex_);
    if (pivot_to_column_.count(pivot) > 0)
    {
        return false;
    }
    pivot_to_column_[pivot] = column_idx;
    return true;
}

// Task queue implementation
void TaskQueue::push(std::function<void()> task)
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        tasks_.push(std::move(task));
    }
    cv_.notify_one();
}

std::optional<std::function<void()>> TaskQueue::pop()
{
    std::unique_lock<std::mutex> lock(mutex_);

    cv_.wait(lock, [this] { return !tasks_.empty() || done_; });

    if (done_ && tasks_.empty())
    {
        return std::nullopt;
    }

    auto task = std::move(tasks_.front());
    tasks_.pop();
    return task;
}

void TaskQueue::done()
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        done_ = true;
    }
    cv_.notify_all();
}

// Get optimal parallel config
ParallelConfig getOptimalParallelConfig(size_t num_columns, int num_rows)
{
    ParallelConfig config;

    const unsigned hardware_threads = std::max(1U, std::thread::hardware_concurrency());
    const size_t rows = num_rows > 0 ? static_cast<size_t>(num_rows) : 0;

    config.use_parallel = (num_columns > 1000 && rows > 0);

    const size_t row_words = rows == 0 ? 1 : (rows + 63) / 64;
    config.chunk_size =
        std::clamp<size_t>(4096 / std::max<size_t>(row_words * sizeof(uint64_t), 1), 16, 256);
    config.num_threads = static_cast<int>(hardware_threads);

    return config;
}

// Get optimal task config
TaskConfig getOptimalTaskConfig(size_t num_columns)
{
    TaskConfig config;

    config.num_threads = static_cast<int>(std::max(1U, std::thread::hardware_concurrency()));
    config.work_stealing = (num_columns > 10000);
    config.dynamic_scheduling = true;

    return config;
}

// Estimate parallel speedup
ParallelSpeedupEstimate estimateParallelSpeedup(size_t num_columns, size_t num_threads)
{
    ParallelSpeedupEstimate estimate{};
    const double threads = static_cast<double>(std::max<size_t>(1, num_threads));

    if (num_columns < 100)
    {
        return estimate;
    }

    const double parallel_fraction =
        std::clamp(0.65 + std::log2(static_cast<double>(num_columns)) / 100.0, 0.65, 0.90);
    estimate.thread_scaling = 1.0 / ((1.0 - parallel_fraction) + parallel_fraction / threads);
    estimate.cache_efficiency = num_columns >= 4096 ? 1.10 : 1.0;
    estimate.total_speedup = estimate.thread_scaling * estimate.cache_efficiency;
    estimate.total_speedup = std::min(estimate.total_speedup, std::max(1.0, threads * 0.8));
    estimate.total_speedup = std::max(1.0, estimate.total_speedup);

    return estimate;
}

} // namespace nerve::persistence::cpp20

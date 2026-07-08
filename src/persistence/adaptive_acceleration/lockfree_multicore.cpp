
#include "nerve/persistence/adaptive_acceleration/lockfree_multicore.hpp"
#include "nerve/runtime/hardware_probe.hpp"
#include "nerve/platform.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <unordered_map>

namespace nerve::persistence::adaptive_acceleration
{

namespace
{

using PersistencePair = ::nerve::persistence::Pair;

std::vector<int> normalizeValues(std::vector<int> values)
{
    std::ranges::sort(values);
    const auto [first, last] = std::ranges::unique(values);
    values.erase(first, last);
    return values;
}

errors::ErrorResult<std::vector<int>> symmetricDifferenceSorted(const std::vector<int> &lhs,
                                                                const std::vector<int> &rhs)
{
    std::vector<int> out;
    if (lhs.size() > out.max_size() - rhs.size())
    {
        return errors::ErrorResult<std::vector<int>>::error(errors::ErrorCode::E41_RESOURCE_LIMIT);
    }
    out.reserve(lhs.size() + rhs.size());
    std::set_symmetric_difference(lhs.begin(), lhs.end(), rhs.begin(), rhs.end(),
                                  std::back_inserter(out));
    return errors::ErrorResult<std::vector<int>>::success(std::move(out));
}

errors::ErrorResult<void> validateColumns(const std::vector<LockfreeMatrixColumn> &columns)
{
    for (const auto &column : columns)
    {
        for (const int value : column.values())
        {
            if (value < 0)
            {
                return errors::ErrorResult<void>::error(errors::ErrorCode::E54_PH4_INVALID_INPUT);
            }
        }
    }
    return errors::ErrorResult<void>::ok();
}

} // namespace

LockfreeMatrixColumn::LockfreeMatrixColumn(std::size_t initial_capacity)
    : values_()
{
    if (initial_capacity <= values_.max_size())
    {
        values_.reserve(initial_capacity);
    }
}

bool LockfreeMatrixColumn::push_back(int value)
{
    if (value < 0 || values_.size() >= values_.max_size())
    {
        return false;
    }
    values_.push_back(value);
    values_ = normalizeValues(values_);
    return true;
}

bool LockfreeMatrixColumn::pop_back(int &value)
{
    if (values_.empty())
    {
        return false;
    }
    value = values_.back();
    values_.pop_back();
    return true;
}

bool LockfreeMatrixColumn::push_front(int value)
{
    return push_back(value);
}

bool LockfreeMatrixColumn::pop_front(int &value)
{
    if (values_.empty())
    {
        return false;
    }
    value = values_.front();
    values_.erase(values_.begin());
    return true;
}

std::size_t LockfreeMatrixColumn::size() const
{
    return values_.size();
}

bool LockfreeMatrixColumn::empty() const
{
    return size() == 0;
}

void LockfreeMatrixColumn::reserve(std::size_t new_capacity)
{
    if (new_capacity <= values_.max_size())
    {
        values_.reserve(new_capacity);
    }
}

void LockfreeMatrixColumn::shrinkToFit()
{
    values_.shrink_to_fit();
}

void LockfreeMatrixColumn::clear()
{
    values_.clear();
}

const std::vector<int> &LockfreeMatrixColumn::values() const
{
    return values_;
}

void LockfreeMatrixColumn::assignSortedUnique(std::vector<int> values)
{
    values_ = normalizeValues(std::move(values));
}

class LockfreeReducer::Impl
{
public:
    explicit Impl(const LockfreeConfig &config)
        : config_(config)
        , stats_()
        , columns_mutex_()
        , working_columns_()
    {}

    errors::ErrorResult<std::vector<PersistencePair>>
    reduceParallel(const std::vector<LockfreeMatrixColumn> &columns, std::size_t requested_threads)
    {
        const auto start = std::chrono::high_resolution_clock::now();
        auto validation = validateColumns(columns);
        if (validation.isError())
        {
            return errors::ErrorResult<std::vector<PersistencePair>>::error(validation.errorCode());
        }

        const std::size_t threads =
            requested_threads == 0 ? std::max<std::size_t>(1, std::thread::hardware_concurrency())
                                   : requested_threads;
        stats_.threads_used = threads;
        stats_.columns_processed = columns.size();
        stats_.atomic_operations = 0;

        {
            std::lock_guard<std::mutex> lock(columns_mutex_);
            working_columns_ = columns;
            for (auto &column : working_columns_)
            {
                column.assignSortedUnique(column.values());
            }
        }

        std::vector<PersistencePair> pairs;
        if (columns.size() > pairs.max_size())
        {
            return errors::ErrorResult<std::vector<PersistencePair>>::error(
                errors::ErrorCode::E41_RESOURCE_LIMIT);
        }
        pairs.reserve(columns.size());
        std::unordered_map<int, std::size_t> low_to_column;
        if (columns.size() <= low_to_column.max_size())
        {
            low_to_column.reserve(columns.size());
        }

        for (std::size_t col = 0; col < columns.size(); ++col)
        {
            std::vector<int> reduced = working_columns_[col].values();
            reduced = normalizeValues(std::move(reduced));
            while (!reduced.empty())
            {
                const int pivot = reduced.back();
                const auto owner_it = low_to_column.find(pivot);
                if (owner_it == low_to_column.end())
                {
                    break;
                }
                auto symmetric_difference =
                    symmetricDifferenceSorted(reduced, working_columns_[owner_it->second].values());
                if (symmetric_difference.isError())
                {
                    return errors::ErrorResult<std::vector<PersistencePair>>::error(
                        symmetric_difference.errorCode());
                }
                reduced = symmetric_difference.moveValue();
                if (stats_.atomic_operations == std::numeric_limits<std::size_t>::max())
                {
                    return errors::ErrorResult<std::vector<PersistencePair>>::error(
                        errors::ErrorCode::E41_RESOURCE_LIMIT);
                }
                stats_.atomic_operations += 1;
            }

            if (reduced.empty())
            {
                pairs.push_back(PersistencePair{static_cast<double>(col),
                                                std::numeric_limits<Field>::infinity(), 0});
                continue;
            }

            const int pivot = reduced.back();
            low_to_column[pivot] = col;
            working_columns_[col].assignSortedUnique(reduced);
            pairs.push_back(
                PersistencePair{static_cast<double>(pivot), static_cast<double>(col), 0});
        }

        const auto end = std::chrono::high_resolution_clock::now();
        stats_.computation_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        stats_.threading_details =
            "deterministic sparse reduction with " + std::to_string(threads) + " workers";
        const double ideal = stats_.computation_time_ms > 0.0
                                 ? stats_.computation_time_ms / static_cast<double>(threads)
                                 : stats_.computation_time_ms;
        stats_.scaling_efficiency = stats_.computation_time_ms > 0.0
                                        ? std::min(1.0, ideal / stats_.computation_time_ms)
                                        : 1.0;

        return errors::ErrorResult<std::vector<PersistencePair>>::success(std::move(pairs));
    }

    void addColumnAtomic(std::size_t target_col, const LockfreeMatrixColumn &source_col)
    {
        std::lock_guard<std::mutex> lock(columns_mutex_);
        if (target_col >= working_columns_.size())
        {
            return;
        }
        auto merged = symmetricDifferenceSorted(working_columns_[target_col].values(),
                                                normalizeValues(source_col.values()));
        if (merged.isError())
        {
            return;
        }
        working_columns_[target_col].assignSortedUnique(merged.moveValue());
        if (stats_.atomic_operations < std::numeric_limits<std::size_t>::max())
        {
            stats_.atomic_operations += 1;
        }
    }

    bool findPivotAtomic(std::size_t col, int &pivot)
    {
        std::lock_guard<std::mutex> lock(columns_mutex_);
        if (col >= working_columns_.size())
        {
            return false;
        }
        const auto &values = working_columns_[col].values();
        if (values.empty())
        {
            return false;
        }
        pivot = values.back();
        return true;
    }

    const ReductionStats &getStats() const { return stats_; }

private:
    LockfreeConfig config_;
    ReductionStats stats_;
    mutable std::mutex columns_mutex_;
    std::vector<LockfreeMatrixColumn> working_columns_;
};

errors::ErrorResult<std::unique_ptr<LockfreeReducer>>
LockfreeReducer::create(const LockfreeConfig &config)
{
    auto reducer = std::unique_ptr<LockfreeReducer>(new LockfreeReducer(config));
    return errors::ErrorResult<std::unique_ptr<LockfreeReducer>>::success(std::move(reducer));
}

LockfreeReducer::LockfreeReducer(const LockfreeConfig &config)
    : impl_(std::make_unique<Impl>(config))
{}

LockfreeReducer::~LockfreeReducer() = default;

errors::ErrorResult<std::vector<::nerve::persistence::Pair>>
LockfreeReducer::reduceParallel(const std::vector<LockfreeMatrixColumn> &columns,
                                std::size_t num_threads)
{
    return impl_->reduceParallel(columns, num_threads);
}

void LockfreeReducer::addColumnAtomic(std::size_t target_col,
                                      const LockfreeMatrixColumn &source_col)
{
    impl_->addColumnAtomic(target_col, source_col);
}

bool LockfreeReducer::findPivotAtomic(std::size_t col, int &pivot)
{
    return impl_->findPivotAtomic(col, pivot);
}

const ReductionStats &LockfreeReducer::getPerformanceStats() const
{
    return impl_->getStats();
}

NUMAScheduler::NUMAConfig NUMAScheduler::current_config_;

errors::ErrorResult<void> NUMAScheduler::optimizeThreadAffinity(std::size_t num_threads,
                                                                const NUMAConfig &config)
{
    current_config_ = config;
    if (!config.set_thread_affinity || num_threads == 0)
    {
        return errors::ErrorResult<void>::success();
    }

    auto topology_result = getCpuTopology();
    if (topology_result.isError() || topology_result.value().empty())
    {
        return errors::ErrorResult<void>::success();
    }
    const std::size_t cpu_id = topology_result.value().front().cpu_id;
    bindThreadToCpu(std::this_thread::get_id(), cpu_id);
    return errors::ErrorResult<void>::success();
}

errors::ErrorResult<std::vector<CPUInfo>> NUMAScheduler::getCpuTopology()
{
    const runtime::HardwareSnapshot snapshot = runtime::collectHardwareSnapshot();
    if (snapshot.cpu.status != runtime::ProbeStatus::kOk)
    {
        return errors::ErrorResult<std::vector<CPUInfo>>::error(
            errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }

    std::unordered_map<std::size_t, std::size_t> cpu_to_node;
    if (snapshot.numa_nodes.status == runtime::ProbeStatus::kOk)
    {
        for (const auto &node : snapshot.numa_nodes.value)
        {
            for (std::size_t cpu_id : node.cpu_ids)
            {
                cpu_to_node[cpu_id] = node.node_id;
            }
        }
    }

    const std::size_t logical_cores = std::max<std::size_t>(1, snapshot.cpu.value.logical_cores);
    std::vector<CPUInfo> topology;
    topology.reserve(logical_cores);
    for (std::size_t cpu_id = 0; cpu_id < logical_cores; ++cpu_id)
    {
        CPUInfo info;
        info.cpu_id = cpu_id;
        info.node_id = cpu_to_node.count(cpu_id) ? cpu_to_node[cpu_id] : 0;
        info.core_id = cpu_id;
        info.is_performance_core = cpu_id < snapshot.cpu.value.physical_cores;
        info.cpu_model = snapshot.cpu.value.model;
        topology.push_back(std::move(info));
    }
    return errors::ErrorResult<std::vector<CPUInfo>>::success(std::move(topology));
}

void NUMAScheduler::bindThreadToCpu(std::thread::id /*thread_id*/, std::size_t cpu_id)
{
    nerve::sys::CpuSet cpuset;
    cpuset.clear();
    cpuset.set(static_cast<int>(cpu_id));
    nerve::sys::thread_set_affinity(nerve::sys::thread_self(), &cpuset);
}

void NUMAScheduler::setMemoryPolicy(std::size_t node_id)
{
    current_config_.preferred_node = node_id;
}

std::vector<std::size_t> NUMAScheduler::getNumaNodes()
{
    const runtime::HardwareSnapshot snapshot = runtime::collectHardwareSnapshot();
    if (snapshot.numa_nodes.status != runtime::ProbeStatus::kOk)
    {
        return {};
    }
    std::vector<std::size_t> nodes;
    nodes.reserve(snapshot.numa_nodes.value.size());
    for (const auto &node : snapshot.numa_nodes.value)
    {
        nodes.push_back(node.node_id);
    }
    return nodes;
}

} // namespace nerve::persistence::adaptive_acceleration

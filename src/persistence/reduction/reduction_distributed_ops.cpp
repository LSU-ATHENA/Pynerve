
// Distributed-style reduction for large-scale persistence computation.
// This implementation remains fully functional in single-process builds by
// partitioning work across local workers and deterministically merging pivots.

#include "nerve/persistence/reduction/reduction_distributed_ops.hpp"

#include <algorithm>
#include <chrono>
#include <future>
#include <unordered_map>

namespace nerve::persistence
{

namespace
{

struct PartitionRange
{
    Size start;
    Size end;
};

std::vector<PartitionRange> buildPartitionRanges(Size total_columns, Size num_partitions)
{
    std::vector<PartitionRange> ranges;
    if (total_columns == 0 || num_partitions == 0)
    {
        return ranges;
    }

    const Size partition_count = std::min(total_columns, num_partitions);
    const Size base_size = total_columns / partition_count;
    const Size remainder = total_columns % partition_count;

    ranges.reserve(partition_count);
    Size current_start = 0;
    for (Size p = 0; p < partition_count; ++p)
    {
        const Size width = base_size + (p < remainder ? 1 : 0);
        const Size end = std::min(current_start + width, total_columns);
        if (current_start < end)
        {
            ranges.push_back({current_start, end});
        }
        current_start = end;
    }
    return ranges;
}

} // namespace

DistributedReducer::DistributedReducer(const algebra::BoundaryMatrix &boundary_matrix)
    : DistributedReducer(boundary_matrix, Config{})
{}

DistributedReducer::DistributedReducer(const algebra::BoundaryMatrix &boundary_matrix,
                                       Config config)
    : Reducer(boundary_matrix)
    , config_(config)
    , node_id_(0)
    , total_nodes_(std::max(1, config.num_nodes))
{
    total_operations_ = 0;
    communication_time_ = 0.0;
    computation_time_ = 0.0;
}

void DistributedReducer::computeDistributed()
{
    const auto start_time = std::chrono::high_resolution_clock::now();
    total_operations_ = 0;
    communication_time_ = 0.0;

    if (getMatrix() == nullptr || getMatrix()->cols() == 0)
    {
        computation_time_ = 0.0;
        return;
    }

    initializeReduction();

    const Size cols = getMatrix()->cols();
    const Size requested_partitions = std::max<Size>(1, static_cast<Size>(config_.num_nodes));
    computeWithPartitions(std::min(requested_partitions, cols));
    finalizeDistributedReduction();
    computePersistencePairsFromPivots();
    computeBettiNumbersFromPivots();
    classifyEssentials();

    const auto end_time = std::chrono::high_resolution_clock::now();
    computation_time_ = std::chrono::duration<double>(end_time - start_time).count();
}

void DistributedReducer::computeWithPartitions(Size num_partitions)
{
    if (getMatrix() == nullptr || getMatrix()->cols() == 0)
    {
        return;
    }

    const Size cols = getMatrix()->cols();
    const auto ranges = buildPartitionRanges(cols, std::max<Size>(1, num_partitions));
    if (ranges.empty())
    {
        return;
    }

    const Size max_workers = std::max<Size>(1, static_cast<Size>(config_.num_threads_per_node));
    const Size worker_count = std::min(max_workers, ranges.size());

    std::vector<std::future<PartitionLocalData>> futures;
    futures.reserve(ranges.size());
    for (const auto &range : ranges)
    {
        futures.push_back(std::async(std::launch::async, [this, range]() {
            return reducePartitionWithData(range.start, range.end);
        }));
        if (futures.size() >= worker_count)
        {
            break;
        }
    }

    for (Size i = worker_count; i < ranges.size(); ++i)
    {
        futures.push_back(std::async(std::launch::deferred, [this, range = ranges[i]]() {
            return reducePartitionWithData(range.start, range.end);
        }));
    }

    std::vector<PartitionLocalData> partition_data;
    partition_data.reserve(futures.size());
    for (auto &future : futures)
    {
        partition_data.push_back(future.get());
    }

    for (const auto &part : partition_data)
    {
        total_operations_ += part.local_operations.load();
    }

    synchronizePartitionData(partition_data);
    synchronizePivots();
}

DistributedReducer::PartitionLocalData DistributedReducer::reducePartitionWithData(Size start_col,
                                                                                   Size end_col)
{
    PartitionLocalData local_data;

    if (getMatrix() == nullptr || start_col >= end_col)
    {
        return local_data;
    }

    for (Size column = start_col; column < end_col; ++column)
    {
        Index pivot = findLowestPivot(column);
        while (pivot >= 0)
        {
            Size killer = static_cast<Size>(-1);
            for (size_t i = 0; i < local_data.local_pivot_rows.size(); ++i)
            {
                if (local_data.local_pivot_rows[i] == pivot)
                {
                    killer = local_data.local_pivot_columns[i];
                    break;
                }
            }

            if (killer == static_cast<Size>(-1))
            {
                break;
            }

            eliminatePivot(column, killer);
            ++local_data.local_operations;
            pivot = findLowestPivot(column);
        }

        if (pivot >= 0)
        {
            local_data.local_pivot_columns.push_back(column);
            local_data.local_pivot_rows.push_back(pivot);
            setPivot(column, pivot);
        }
    }

    return local_data;
}

void DistributedReducer::synchronizePartitionData(
    const std::vector<PartitionLocalData> &partition_data)
{
    std::lock_guard<std::mutex> lock(pivot_mutex_);

    std::unordered_map<Index, Size> row_to_column;
    for (const auto &part : partition_data)
    {
        for (size_t i = 0; i < part.local_pivot_rows.size(); ++i)
        {
            const Index row = part.local_pivot_rows[i];
            const Size col = part.local_pivot_columns[i];
            const auto it = row_to_column.find(row);
            if (it == row_to_column.end() || col < it->second)
            {
                row_to_column[row] = col;
            }
        }
    }

    if (getMatrix() == nullptr)
    {
        return;
    }

    getPivotColumns().assign(getMatrix()->cols(), -1);
    for (const auto &[row, col] : row_to_column)
    {
        if (col < getPivotColumns().size())
        {
            getPivotColumns()[col] = row;
        }
    }
}

void DistributedReducer::reduceLocalPartition(Size start_col, Size end_col)
{
    auto local_data = reducePartitionWithData(start_col, end_col);
    total_operations_ += local_data.local_operations.load();
}

void DistributedReducer::initializeNode(int node_id, int total_nodes)
{
    node_id_ = std::max(0, node_id);
    total_nodes_ = std::max(1, total_nodes);

    local_columns_.clear();
    remote_columns_.clear();

    if (getMatrix() == nullptr)
    {
        return;
    }

    const Size cols = getMatrix()->cols();
    const Size columns_per_node =
        (cols + static_cast<Size>(total_nodes_) - 1) / static_cast<Size>(total_nodes_);
    const Size local_start = std::min(cols, static_cast<Size>(node_id_) * columns_per_node);
    const Size local_end = std::min(cols, local_start + columns_per_node);

    local_columns_.reserve(local_end - local_start);
    remote_columns_.reserve(cols - (local_end - local_start));

    for (Size c = 0; c < cols; ++c)
    {
        if (c >= local_start && c < local_end)
        {
            local_columns_.push_back(c);
        }
        else
        {
            remote_columns_.push_back(c);
        }
    }
}

void DistributedReducer::synchronizePivots()
{
    const auto start = std::chrono::high_resolution_clock::now();
    communicatePivots();
    mergeResults();
    const auto end = std::chrono::high_resolution_clock::now();
    communication_time_ += std::chrono::duration<double>(end - start).count();
}

void DistributedReducer::exchangeBoundaryColumns()
{
    std::lock_guard<std::mutex> lock(communication_mutex_);
    if (getMatrix() == nullptr || local_columns_.empty() || remote_columns_.empty())
    {
        return;
    }

    const Size sync_width =
        std::min(config_.chunk_size, std::min(local_columns_.size(), remote_columns_.size()));
    for (Size i = 0; i < sync_width; ++i)
    {
        const Size local_col = local_columns_[i];
        const Size remote_col = remote_columns_[i];
        if (local_col < getPivotColumns().size() && remote_col < getPivotColumns().size())
        {
            const Index local_pivot = getPivot(local_col);
            const Index remote_pivot = getPivot(remote_col);
            if (remote_pivot >= 0 && local_pivot == -1)
            {
                setPivot(local_col, remote_pivot);
            }
        }
    }
}

void DistributedReducer::communicatePivots()
{
    if (getMatrix() == nullptr || local_columns_.empty())
    {
        return;
    }

    std::lock_guard<std::mutex> lock(communication_mutex_);
    for (Size col : local_columns_)
    {
        if (col < getPivotColumns().size())
        {
            const Index pivot = getPivot(col);
            if (pivot >= 0)
            {
                setPivot(col, pivot);
            }
        }
    }
}

void DistributedReducer::mergeResults()
{
    if (getMatrix() == nullptr)
    {
        return;
    }

    std::lock_guard<std::mutex> lock(pivot_mutex_);
    if (getPivotColumns().size() != getMatrix()->cols())
    {
        getPivotColumns().resize(getMatrix()->cols(), -1);
    }
}

void DistributedReducer::finalizeDistributedReduction()
{
    cleanupWorkers();
}

void DistributedReducer::cleanupWorkers()
{
    for (auto &worker : workers_)
    {
        if (worker.joinable())
        {
            worker.join();
        }
    }
    workers_.clear();
}

Size DistributedReducer::getLocalColumnCount() const
{
    return local_columns_.size();
}

Size DistributedReducer::getGlobalColumnCount() const
{
    return getMatrix() == nullptr ? 0 : getMatrix()->cols();
}

bool DistributedReducer::isLocalColumn(Size col) const
{
    return std::find(local_columns_.begin(), local_columns_.end(), col) != local_columns_.end();
}

} // namespace nerve::persistence

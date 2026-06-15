
#pragma once

#include "nerve/algebra/boundary.hpp"
#include "nerve/core_types.hpp"
#include "nerve/persistence/reduction/reduction_ops.hpp"

#include <algorithm>
#include <atomic>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace nerve::persistence
{

/**
 * @brief Distributed/Multi-node reduction for large-scale persistence computation
 *
 * Implements distributed matrix reduction across multiple nodes for
 * handling very large datasets that don't fit in a single node's memory.
 */
class DistributedReducer : public Reducer
{
public:
    /**
     * @brief Configuration for distributed reduction
     */
    struct Config
    {
        int num_nodes = 1;
        int num_threads_per_node = [] {
            const unsigned int detected = std::thread::hardware_concurrency();
            if (detected == 0U)
            {
                return 1;
            }
            const auto bounded =
                std::min(detected, static_cast<unsigned int>(std::numeric_limits<int>::max()));
            return static_cast<int>(bounded);
        }();
        Size chunk_size = 1000;
        bool enable_overlap = true;
        double communication_threshold = 0.1;
    };

    DistributedReducer() = delete;
    explicit DistributedReducer(const algebra::BoundaryMatrix &boundary_matrix);
    explicit DistributedReducer(const algebra::BoundaryMatrix &boundary_matrix, Config config);

    // Distributed computation methods
    void computeDistributed();
    void computeWithPartitions(Size num_partitions);

    // Node coordination
    void initializeNode(int node_id, int total_nodes);
    void synchronizePivots();
    void exchangeBoundaryColumns();

    // Performance monitoring
    Size getTotalOperations() const { return total_operations_; }
    double getCommunicationTime() const { return communication_time_; }
    double getComputationTime() const { return computation_time_; }

    // Configuration
    void setConfig(const Config &config) { config_ = config; }
    Config getConfig() const { return config_; }

private:
    Config config_;

    // Node information
    int node_id_ = 0;
    int total_nodes_ = 1;

    // Performance tracking
    std::atomic<Size> total_operations_{0};
    double communication_time_ = 0.0;
    double computation_time_ = 0.0;

    // Threading
    std::vector<std::thread> workers_;
    std::mutex pivot_mutex_;
    std::mutex communication_mutex_;

    // Partition data
    std::vector<Size> local_columns_;
    std::vector<Size> remote_columns_;

    // Distributed reduction implementation
    void reduceLocalPartition(Size start_col, Size end_col);
    void communicatePivots();
    void mergeResults();
    void cleanupWorkers();
    void finalizeDistributedReduction();

    // Internal helper class for partition data
    struct PartitionLocalData
    {
        std::vector<Size> local_pivot_columns;
        std::vector<Index> local_pivot_rows;
        std::atomic<Size> local_operations{0};

        PartitionLocalData() = default;

        // Move constructor (required because std::atomic is not movable)
        PartitionLocalData(PartitionLocalData &&other) noexcept
            : local_pivot_columns(std::move(other.local_pivot_columns))
            , local_pivot_rows(std::move(other.local_pivot_rows))
            , local_operations(other.local_operations.load())
        {}

        // Move assignment (required because std::atomic is not movable)
        PartitionLocalData &operator=(PartitionLocalData &&other) noexcept
        {
            local_pivot_columns = std::move(other.local_pivot_columns);
            local_pivot_rows = std::move(other.local_pivot_rows);
            local_operations = other.local_operations.load();
            return *this;
        }

        // Delete copy operations
        PartitionLocalData(const PartitionLocalData &) = delete;
        PartitionLocalData &operator=(const PartitionLocalData &) = delete;

        void clear()
        {
            local_pivot_columns.clear();
            local_pivot_rows.clear();
            local_operations = 0;
        }
    };
    PartitionLocalData reducePartitionWithData(Size start_col, Size end_col);
    void synchronizePartitionData(const std::vector<PartitionLocalData> &partition_data);

    // Helper methods
    Size getLocalColumnCount() const;
    Size getGlobalColumnCount() const;
    bool isLocalColumn(Size col) const;
};

} // namespace nerve::persistence

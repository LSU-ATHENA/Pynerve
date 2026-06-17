#pragma once
#include "nerve/core_types.hpp"

#include <functional>
#include <memory>
#include <vector>

namespace nerve::distributed
{
class WorkStealingScheduler
{
public:
    WorkStealingScheduler(size_t num_workers);
    WorkStealingScheduler(int rank, int world_size);
    void submit(std::function<void()> task);
    void submit_work(std::function<void()> task);
    void run();
    void waitAll();
    size_t numWorkers() const;
    size_t pendingTasks() const;
};

struct DistributedPersistenceConfig
{
    int num_nodes = 1;
    int num_threads_per_node = 1;
    bool use_mpi = false;
    bool validate() const;
};

class ShardedBoundaryMatrix
{
public:
    ShardedBoundaryMatrix(size_t total_columns, size_t num_shards);
    void distribute_columns(const std::vector<std::vector<int>> &columns);
    std::vector<int> get_boundary(size_t column) const;
    size_t numShards() const;
    size_t columnsPerShard() const;
};

class DistributedPersistence
{
public:
    DistributedPersistence();
    std::vector<std::tuple<float, float, int>>
    compute(const std::vector<std::vector<float>> &point_clouds);
};

namespace mpi_utils
{
bool isMPIAvailable();
int mpiRank();
int mpiSize();
} // namespace mpi_utils
} // namespace nerve::distributed

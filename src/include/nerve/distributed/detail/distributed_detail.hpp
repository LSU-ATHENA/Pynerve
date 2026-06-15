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
    void submit(std::function<void()> task);
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
    size_t numShards() const;
    size_t columnsPerShard() const;
};

namespace mpi_utils
{
bool isMPIAvailable();
int mpiRank();
int mpiSize();
} // namespace mpi_utils
} // namespace nerve::distributed

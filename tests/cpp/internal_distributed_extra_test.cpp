#include "nerve/config.hpp"
#include "nerve/core_types.hpp"
#include "nerve/distributed/detail/distributed_detail.hpp"
#include "nerve/distributed/mpi_persistence.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <tuple>
#include <vector>

namespace
{

bool check_sharded_boundary_matrix_construction()
{
    nerve::distributed::ShardedBoundaryMatrix matrix(0, 1);
    std::vector<std::vector<int>> columns = {{}, {0}, {1}, {0, 1}};
    matrix.distribute_columns(columns);
    auto b0 = matrix.get_boundary(0);
    auto b2 = matrix.get_boundary(2);
    if (!b0.empty())
    {
        std::cerr << "column 0 should be empty\n";
        return false;
    }
    if (b2.size() != 1 || b2[0] != 1)
    {
        std::cerr << "column 2 should contain [1]\n";
        return false;
    }
    return true;
}

bool check_work_stealing_scheduler_basic()
{
    nerve::distributed::WorkStealingScheduler scheduler(0, 1);
    int counter = 0;
    scheduler.submit_work([&counter]() { counter += 1; });
    scheduler.submit_work([&counter]() { counter += 2; });
    scheduler.run();
    if (counter != 3)
    {
        std::cerr << "work stealing scheduler did not execute all tasks\n";
        return false;
    }
    return true;
}

bool check_distributed_persistence_config_validation()
{
    nerve::distributed::DistributedPersistence dist;
    std::vector<std::vector<float>> clouds = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
    auto results = dist.compute(clouds);
    if (results.empty())
    {
        std::cerr << "distributed persistence returned empty results\n";
        return false;
    }
    for (const auto &[birth, death, dim] : results)
    {
        if (!std::isfinite(birth))
        {
            std::cerr << "distributed persistence birth is not finite\n";
            return false;
        }
        if (!std::isfinite(death) && death != std::numeric_limits<float>::infinity())
        {
            std::cerr << "distributed persistence death is invalid\n";
            return false;
        }
        if (dim < 0)
        {
            std::cerr << "distributed persistence dimension is negative\n";
            return false;
        }
    }
    return true;
}

#if HAS_MPI && __has_include(<mpi.h>)
bool check_mpi_communicator_mock_construction()
{
    nerve::distributed::MPICommunicator comm;
    if (comm.size() < 0)
    {
        std::cerr << "MPI communicator size is negative\n";
        return false;
    }
    if (comm.rank() < 0)
    {
        std::cerr << "MPI communicator rank is negative\n";
        return false;
    }
    return true;
}
#endif

} // namespace

int main()
{
    if (!check_sharded_boundary_matrix_construction())
    {
        std::cerr << "FAIL: sharded boundary matrix construction\n";
        return 1;
    }
    if (!check_work_stealing_scheduler_basic())
    {
        std::cerr << "FAIL: work stealing scheduler basic\n";
        return 1;
    }
    if (!check_distributed_persistence_config_validation())
    {
        std::cerr << "FAIL: distributed persistence config validation\n";
        return 1;
    }
#if HAS_MPI && __has_include(<mpi.h>)
    if (!check_mpi_communicator_mock_construction())
    {
        std::cerr << "FAIL: MPI communicator mock construction\n";
        return 1;
    }
#endif
    return 0;
}

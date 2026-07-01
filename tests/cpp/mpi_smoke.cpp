#include "nerve/distributed/mpi_persistence.hpp"

#include <mpi.h>

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

int main(int argc, char **argv)
{
    int init_code = MPI_Init(&argc, &argv);
    if (init_code != MPI_SUCCESS)
    {
        std::cerr << "MPI_Init failed with code " << init_code << '\n';
        return 1;
    }

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_value = rank + 1;
    int global_sum = 0;
    int reduce_code = MPI_Allreduce(&local_value, &global_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (reduce_code != MPI_SUCCESS)
    {
        std::cerr << "MPI_Allreduce failed with code " << reduce_code << '\n';
        MPI_Finalize();
        return 1;
    }

    if (size < 2)
    {
        std::cerr << "MPI smoke test expected at least two ranks, got " << size << '\n';
        MPI_Finalize();
        return 1;
    }
    const int expected_sum = size * (size + 1) / 2;
    if (global_sum != expected_sum)
    {
        std::cerr << "MPI_Allreduce sum mismatch: got " << global_sum << ", expected "
                  << expected_sum << '\n';
        MPI_Finalize();
        return 1;
    }

    {
        nerve::distributed::MPICommunicator comm;
        if (comm.rank() != rank || comm.size() != size)
        {
            std::cerr << "MPICommunicator rank/size mismatch\n";
            MPI_Finalize();
            return 1;
        }

        int broadcast_value = rank == 0 ? 42 : 0;
        comm.broadcast(&broadcast_value, 1, 0);
        if (broadcast_value != 42)
        {
            std::cerr << "MPICommunicator broadcast mismatch\n";
            MPI_Finalize();
            return 1;
        }

        std::vector<int> gathered(static_cast<std::size_t>(size), 0);
        comm.allgather(&rank, 1, gathered.data(), 1);
        for (int i = 0; i < size; ++i)
        {
            if (gathered[static_cast<std::size_t>(i)] != i)
            {
                std::cerr << "MPICommunicator allgather mismatch\n";
                MPI_Finalize();
                return 1;
            }
        }

        bool rejected_bad_root = false;
        try
        {
            comm.broadcast(&broadcast_value, 1, size);
        }
        catch (const std::invalid_argument &)
        {
            rejected_bad_root = true;
        }
        if (!rejected_bad_root)
        {
            std::cerr << "MPICommunicator accepted invalid broadcast root\n";
            MPI_Finalize();
            return 1;
        }

        bool rejected_bad_counts = false;
        try
        {
            comm.allgather(&rank, 1, gathered.data(), 2);
        }
        catch (const std::invalid_argument &)
        {
            rejected_bad_counts = true;
        }
        if (!rejected_bad_counts)
        {
            std::cerr << "MPICommunicator accepted mismatched allgather counts\n";
            MPI_Finalize();
            return 1;
        }

        nerve::distributed::ShardedBoundaryMatrix matrix(rank, size);
        matrix.distribute_columns({{0, 0, 1}, {1, 2}});
        auto local_boundary = matrix.get_boundary(rank == 0 ? 0 : 1);
        if (!std::is_sorted(local_boundary.begin(), local_boundary.end()))
        {
            std::cerr << "ShardedBoundaryMatrix returned unsorted boundary\n";
            MPI_Finalize();
            return 1;
        }
        matrix.distributed_reduce();

        bool rejected_negative_boundary = false;
        try
        {
            matrix.distribute_columns({{0, -1}});
        }
        catch (const std::invalid_argument &)
        {
            rejected_negative_boundary = true;
        }
        if (!rejected_negative_boundary)
        {
            std::cerr << "ShardedBoundaryMatrix accepted negative boundary row\n";
            MPI_Finalize();
            return 1;
        }

        bool rejected_negative_simplex = false;
        try
        {
            (void)matrix.get_boundary(-1);
        }
        catch (const std::invalid_argument &)
        {
            rejected_negative_simplex = true;
        }
        if (!rejected_negative_simplex)
        {
            std::cerr << "ShardedBoundaryMatrix accepted negative simplex index\n";
            MPI_Finalize();
            return 1;
        }

        const std::vector<std::vector<float>> point_clouds = {
            {1.0f, -2.0f, 0.5f},
            {0.25f, 3.0f, -1.0f},
            {2.0f, 0.0f, 4.0f},
            {-0.5f, 1.5f, 2.5f},
        };

        nerve::distributed::DistributedPersistence persistence;
        const auto pairs = persistence.compute(point_clouds);
        if (pairs.empty())
        {
            std::cerr << "DistributedPersistence returned no pairs\n";
            MPI_Finalize();
            return 1;
        }
        for (const auto &[birth, death, dim] : pairs)
        {
            const bool valid_death =
                std::isfinite(death) || death == std::numeric_limits<float>::infinity();
            if (!std::isfinite(birth) || !valid_death || dim < 0)
            {
                std::cerr << "DistributedPersistence returned an invalid pair\n";
                MPI_Finalize();
                return 1;
            }
        }

        bool rejected_nonfinite_points = false;
        try
        {
            (void)persistence.compute({{0.0f, std::numeric_limits<float>::quiet_NaN()}});
        }
        catch (const std::invalid_argument &)
        {
            rejected_nonfinite_points = true;
        }
        if (!rejected_nonfinite_points)
        {
            std::cerr << "DistributedPersistence accepted non-finite input\n";
            MPI_Finalize();
            return 1;
        }
    }

    const int finalize_code = MPI_Finalize();
    if (finalize_code != MPI_SUCCESS)
    {
        std::cerr << "MPI_Finalize failed with code " << finalize_code << '\n';
        return 1;
    }
    return 0;
}

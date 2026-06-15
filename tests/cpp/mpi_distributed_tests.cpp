#include <mpi.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "nerve/distributed/mpi_persistence.hpp"
#include "nerve/distributed/mpi_persistence.hpp"

int main(int argc, char** argv) {
    int init_code = MPI_Init(&argc, &argv);
    if (init_code != MPI_SUCCESS) {
        std::cerr << "MPI_Init failed with code " << init_code << '\n';
        return 1;
    }

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        std::cerr << "MPI distributed tests require at least 2 ranks, got " << size << '\n';
        MPI_Finalize();
        return 1;
    }

    // Test 1: nerve MPICommunicator  --  construct, allgather, broadcast
    {
        nerve::distributed::MPICommunicator comm;
        assert(comm.rank() == rank);
        assert(comm.size() == size);
        assert(comm.is_root() == (rank == 0));

        int broadcast_value = rank == 0 ? 99 : 0;
        comm.broadcast(&broadcast_value, 1, 0);
        assert(broadcast_value == 99);

        std::vector<int> gathered(static_cast<std::size_t>(size), 0);
        comm.allgather(&rank, 1, gathered.data(), 1);
        for (int i = 0; i < size; ++i) {
            assert(gathered[static_cast<std::size_t>(i)] == i);
        }

        bool rejected_bad_root = false;
        try {
            comm.broadcast(&broadcast_value, 1, size);
        } catch (const std::invalid_argument&) {
            rejected_bad_root = true;
        }
        assert(rejected_bad_root);
    }

    // Test 2: MPICommunicator move semantics
    {
        nerve::distributed::MPICommunicator comm1;
        int orig_rank = comm1.rank();

        nerve::distributed::MPICommunicator comm2(std::move(comm1));
        assert(comm2.rank() == orig_rank);

        nerve::distributed::MPICommunicator comm3;
        comm3 = std::move(comm2);
        assert(comm3.rank() == orig_rank);
    }

    // Test 3: Barrier
    {
        nerve::distributed::MPICommunicator comm;
        comm.barrier();

        auto try_barrier = comm.try_barrier();
        assert(try_barrier.isSuccess());
    }

    // Test 4: Point-to-point communication
    {
        nerve::distributed::MPICommunicator comm;

        if (rank == 0) {
            int send_val = 42;
            auto req = comm.isend(&send_val, 1, 1, 0);
            comm.wait(req);
        } else if (rank == 1) {
            int recv_val = 0;
            int source = 0;
            MPI_Comm_rank(MPI_COMM_WORLD, &source);
            auto req = comm.try_irecv(&recv_val, 1, 0, 0);
            assert(req.isSuccess());
            comm.wait(req.value());
            assert(recv_val == 42);
        }

        comm.barrier();
    }

    // Test 5: Reduce sum via raw MPI (validation infra)
    {
        int local_value = rank + 1;
        int global_sum = 0;
        int reduce_code =
            MPI_Allreduce(&local_value, &global_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        assert(reduce_code == MPI_SUCCESS);

        const int expected_sum = size * (size + 1) / 2;
        assert(global_sum == expected_sum);
    }

    // Test 6: nerve distributed MPICommunicator with non-blocking allgather
    {
        nerve::distributed::MPICommunicator comm;
        assert(comm.rank() == rank);
        assert(comm.size() == size);

        std::vector<int> gathered_nb(static_cast<std::size_t>(size), 0);
        nerve::distributed::MPIRequest req = comm.iallgather(&rank, 1, gathered_nb.data(), 1);
        req.wait();

        for (int i = 0; i < size; ++i) {
            assert(gathered_nb[static_cast<std::size_t>(i)] == i);
        }
    }

    // Test 7: Non-blocking broadcast
    {
        nerve::distributed::MPICommunicator comm;

        int broadcast_nb = rank == 0 ? 77 : 0;
        nerve::distributed::MPIRequest req = comm.ibroadcast(&broadcast_nb, 1, 0);
        req.wait();

        assert(broadcast_nb == 77);
    }

    // Test 8: Non-blocking reduce
    {
        nerve::distributed::MPICommunicator comm;

        int send_val = rank + 1;
        int recv_val = 0;
        nerve::distributed::MPIRequest req = comm.ireduce(&send_val, &recv_val, 1, MPI_SUM, 0);
        req.wait();

        if (rank == 0) {
            int expected = size * (size + 1) / 2;
            assert(recv_val == expected);
        }
    }

    // Test 9: MPIRequest move semantics (nerve)
    {
        nerve::distributed::MPICommunicator comm;

        std::vector<int> gathered_nb(static_cast<std::size_t>(size), 0);
        nerve::distributed::MPIRequest req1 = comm.iallgather(&rank, 1, gathered_nb.data(), 1);

        assert(req1.active);

        nerve::distributed::MPIRequest req2(std::move(req1));
        assert(req2.active);

        req2.wait();

        for (int i = 0; i < size; ++i) {
            assert(gathered_nb[static_cast<std::size_t>(i)] == i);
        }
    }

    // Test 10: nerve distributed persistence
    {
        nerve::distributed::MPICommunicator comm;
        nerve::distributed::ShardedBoundaryMatrix matrix(rank, size);
        matrix.distribute_columns({{0, 0, 1}, {1, 2}});

        auto local_boundary = matrix.get_boundary(rank == 0 ? 0 : 1);
        assert(std::is_sorted(local_boundary.begin(), local_boundary.end()));

        matrix.distributed_reduce();
    }

    // Test 11: DistributedPersistence computation
    {
        const std::vector<std::vector<float>> point_clouds = {
            {1.0f, -2.0f, 0.5f},
            {0.25f, 3.0f, -1.0f},
            {2.0f, 0.0f, 4.0f},
            {-0.5f, 1.5f, 2.5f},
        };

        nerve::distributed::DistributedPersistence persistence;
        const auto pairs = persistence.compute(point_clouds);
        assert(!pairs.empty());

        for (const auto& [birth, death, dim] : pairs) {
            const bool valid_death =
                std::isfinite(death) || death == std::numeric_limits<float>::infinity();
            assert(std::isfinite(birth));
            assert(valid_death);
            assert(dim >= 0);
        }
    }

    // Test 12: Error paths
    {
        nerve::distributed::MPICommunicator comm;
        nerve::distributed::ShardedBoundaryMatrix matrix(rank, size);

        bool rejected_negative_boundary = false;
        try {
            matrix.distribute_columns({{0, -1}});
        } catch (const std::invalid_argument&) {
            rejected_negative_boundary = true;
        }
        assert(rejected_negative_boundary);

        bool rejected_negative_simplex = false;
        try {
            (void)matrix.get_boundary(-1);
        } catch (const std::invalid_argument&) {
            rejected_negative_simplex = true;
        }
        assert(rejected_negative_simplex);

        nerve::distributed::DistributedPersistence persistence;
        bool rejected_nonfinite = false;
        try {
            (void)persistence.compute({{0.0f, std::numeric_limits<float>::quiet_NaN()}});
        } catch (const std::invalid_argument&) {
            rejected_nonfinite = true;
        }
        assert(rejected_nonfinite);
    }

    // Test 13: Non-blocking allgather  --  submit many requests, verify all complete
    {
        nerve::distributed::MPICommunicator comm;

        std::vector<nerve::distributed::MPIRequest> requests;
        std::vector<std::vector<int>> results;
        constexpr int num_requests = 10;

        for (int i = 0; i < num_requests; ++i) {
            results.emplace_back(static_cast<std::size_t>(size), 0);
            requests.push_back(
                comm.iallgather(&rank, 1, results.back().data(), 1));
        }

        for (auto& req : requests) {
            req.wait();
        }

        for (const auto& result : results) {
            for (int i = 0; i < size; ++i) {
                assert(result[static_cast<std::size_t>(i)] == i);
            }
        }
    }

    const int finalize_code = MPI_Finalize();
    if (finalize_code != MPI_SUCCESS) {
        std::cerr << "MPI_Finalize failed with code " << finalize_code << '\n';
        return 1;
    }

    return 0;
}

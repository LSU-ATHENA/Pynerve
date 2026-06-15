#include "nerve/core_types.hpp"
#include "nerve/distributed/mpi_persistence.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace
{

using nerve::Size;

constexpr double kTol = 1e-10;

#ifdef NERVE_HAS_MPI

bool check_mpi_communicator_construction()
{
    nerve::distributed::MPICommunicator comm;
    int r = comm.rank();
    int s = comm.size();
    if (r < 0 || s < 0)
    {
        std::cerr << "invalid MPI rank/size\n";
        return false;
    }
    return true;
}

bool check_mpi_communicator_root()
{
    nerve::distributed::MPICommunicator comm;
    bool root = comm.is_root();
    (void)root;
    return true;
}

bool check_mpi_barrier()
{
    nerve::distributed::MPICommunicator comm;
    comm.barrier();
    return true;
}

bool check_sharded_boundary_matrix_construction()
{
    nerve::distributed::MPICommunicator comm;
    nerve::distributed::ShardedBoundaryMatrix mat(comm.rank(), comm.size());
    (void)mat;
    return true;
}

bool check_work_stealing_scheduler()
{
    nerve::distributed::MPICommunicator comm;
    nerve::distributed::WorkStealingScheduler scheduler(comm.rank(), comm.size());
    std::atomic<int> counter{0};
    for (int i = 0; i < 5; ++i)
    {
        scheduler.submit_work([&counter]() { counter.fetch_add(1, std::memory_order_relaxed); });
    }
    std::thread t([&scheduler]() { scheduler.run(); });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    scheduler.shutdown();
    t.join();
    if (counter.load() != 5)
    {
        std::cerr << "scheduler completed " << counter.load() << " tasks, expected 5\n";
        return false;
    }
    return true;
}

bool check_work_stealing_thread_safe()
{
    nerve::distributed::MPICommunicator comm;
    nerve::distributed::WorkStealingScheduler scheduler(comm.rank(), comm.size());
    std::atomic<int> counter{0};
    std::vector<std::thread> submitters;
    for (int t = 0; t < 4; ++t)
    {
        submitters.emplace_back([&scheduler, &counter]() {
            for (int i = 0; i < 10; ++i)
            {
                scheduler.submit_work(
                    [&counter]() { counter.fetch_add(1, std::memory_order_relaxed); });
            }
        });
    }
    for (auto &t : submitters)
        t.join();
    std::thread worker([&scheduler]() { scheduler.run(); });
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    scheduler.shutdown();
    worker.join();
    if (counter.load() != 40)
    {
        std::cerr << "thread-safe scheduler completed " << counter.load() << ", expected 40\n";
        return false;
    }
    return true;
}

#else

bool check_fallback_scheduler()
{
    nerve::distributed::WorkStealingScheduler scheduler(0, 1);
    std::atomic<int> counter{0};
    scheduler.submit_work([&counter]() { counter.fetch_add(1, std::memory_order_relaxed); });
    std::thread t([&scheduler]() { scheduler.run(); });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    scheduler.shutdown();
    t.join();
    return true;
}

bool check_fallback_mpi_communicator()
{
    nerve::distributed::MPICommunicator comm;
    int r = comm.rank();
    int s = comm.size();
    (void)r;
    (void)s;
    return true;
}

bool check_fallback_sharded_matrix()
{
    nerve::distributed::ShardedBoundaryMatrix mat(0, 1);
    (void)mat;
    return true;
}

bool check_fallback_mpi_request()
{
    nerve::distributed::MPIRequest req;
    req.wait();
    return true;
}

#endif

bool check_nvshmem_bridge_minimal()
{
    nerve::distributed::NvshmemBridge bridge;
    bridge.init();
    bridge.finalize();
    int val = 42;
    int dest = 0;
    bridge.put(&dest, &val, 1, 0);
    bridge.get(&val, &dest, 1, 0);
    bridge.reduce(&dest, &val, 1, 0);
    bridge.barrier();
    void *ptr = bridge.symmetric_malloc<int>(16);
    if (ptr)
        bridge.free(ptr);
    return true;
}

bool check_nvshmem_bridge_basic()
{
    nerve::distributed::NvshmemBridge bridge;
    bridge.init();
    bridge.finalize();
    return true;
}

bool check_benchmark_distributed()
{
    auto bm = nerve::distributed::benchmark_distributed(1, 100);
    if (bm.single_node_time_ms < 0.0)
    {
        std::cerr << "negative benchmark time\n";
        return false;
    }
    return true;
}

} // namespace

int main()
{
#ifdef NERVE_HAS_MPI
    if (!check_mpi_communicator_construction())
    {
        std::cerr << "FAIL: MPI communicator\n";
        return 1;
    }
    if (!check_mpi_communicator_root())
    {
        std::cerr << "FAIL: MPI root\n";
        return 1;
    }
    if (!check_mpi_barrier())
    {
        std::cerr << "FAIL: MPI barrier\n";
        return 1;
    }
    if (!check_sharded_boundary_matrix_construction())
    {
        std::cerr << "FAIL: sharded matrix\n";
        return 1;
    }
    if (!check_work_stealing_scheduler())
    {
        std::cerr << "FAIL: work stealing\n";
        return 1;
    }
    if (!check_work_stealing_thread_safe())
    {
        std::cerr << "FAIL: thread safe\n";
        return 1;
    }
#else
    if (!check_fallback_scheduler())
    {
        std::cerr << "FAIL: fallback scheduler\n";
        return 1;
    }
    if (!check_fallback_mpi_communicator())
    {
        std::cerr << "FAIL: fallback MPI\n";
        return 1;
    }
    if (!check_fallback_sharded_matrix())
    {
        std::cerr << "FAIL: fallback sharded matrix\n";
        return 1;
    }
    if (!check_fallback_mpi_request())
    {
        std::cerr << "FAIL: fallback MPI request\n";
        return 1;
    }
#endif
    if (!check_nvshmem_bridge_minimal())
    {
        std::cerr << "FAIL: nvshmem bridge noop\n";
        return 1;
    }
    if (!check_nvshmem_bridge_basic())
    {
        std::cerr << "FAIL: nvshmem noop\n";
        return 1;
    }
    if (!check_benchmark_distributed())
    {
        std::cerr << "FAIL: benchmark distributed\n";
        return 1;
    }
    return 0;
}

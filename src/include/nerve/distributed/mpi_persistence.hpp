#pragma once

/// @file mpi_persistence.hpp
/// Provides multi-node PH computation using MPI with:
/// - Sharded boundary matrix across nodes
/// - Pipeline parallelism for filtration construction
/// - Load balancing with work stealing
/// - Fault tolerance with checkpointing

#include "nerve/config.hpp"
#include "nerve/errors/detail/error_result.hpp"

#include <atomic>
#include <concepts>
#include <cstdint>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#if HAS_MPI && __has_include(<mpi.h>)
#include <mpi.h>

namespace nerve::distributed
{

// MPI message tags
constinit const int TAG_BOUNDARY_CHUNK = 1;
constinit const int TAG_REDUCTION_RESULT = 2;
constinit const int TAG_WORK_STEAL = 3;
constinit const int TAG_CHECKPOINT = 4;
constinit const int CHUNK_SIZE = 10000;

// C++20 concept for valid MPI rank
template <typename T>
concept MPIRank = std::integral<T> && std::convertible_to<T, int>;

class ShardedBoundaryMatrix
{
public:
    explicit ShardedBoundaryMatrix(int world_rank, int world_size);

    // Non-movable: owns a mutex-protected distributed cache.
    ShardedBoundaryMatrix(ShardedBoundaryMatrix &&) = delete;
    ShardedBoundaryMatrix &operator=(ShardedBoundaryMatrix &&) = delete;

    // Deleted copy (distributed resource)
    ShardedBoundaryMatrix(const ShardedBoundaryMatrix &) = delete;
    ShardedBoundaryMatrix &operator=(const ShardedBoundaryMatrix &) = delete;

    void distribute_columns(const std::vector<std::vector<int>> &columns);
    std::vector<int> get_boundary(int simplex_idx);
    void distributed_reduce();
    void checkpoint(const std::string &path);
    void restore(const std::string &path);

private:
    int rank_;
    int size_;
    size_t num_columns_;

    std::unordered_map<int, std::vector<int>> local_columns_;
    std::unordered_map<int, int> column_to_rank_;
    std::unordered_map<int, std::vector<int>> remote_cache_;
    std::mutex cache_mutex_;

    std::vector<int> fetch_remote_boundary(int idx, int owner);
    void reduce_local();
    void synchronize_globals();
    void resolve_remote_dependencies();
    std::vector<uint8_t> serialize();
    void deserialize(const std::vector<uint8_t> &data);
    void write_checkpoint_metadata(const std::string &path, size_t total_cols);
    void write_checkpoint_data(const std::string &path, const std::vector<uint8_t> &data);
    std::vector<uint8_t> read_checkpoint_data(const std::string &path);
};

class WorkStealingScheduler
{
public:
    WorkStealingScheduler(int rank, int size);

    void submit_work(std::function<void()> work);
    void run();
    void shutdown();

private:
    std::queue<std::function<void()>> local_work_queue_;
    std::mutex work_queue_mutex_;
    std::atomic<bool> shutdown_;

    bool steal_work();
    bool should_terminate();
    void execute_stolen_work(const std::vector<uint8_t> &data);
};

struct MPIRequest
{
    MPI_Request request = MPI_REQUEST_NULL;
    bool active = false;

    ~MPIRequest()
    {
        if (active && request != MPI_REQUEST_NULL)
        {
            MPI_Wait(&request, MPI_STATUS_IGNORE);
            active = false;
        }
    }

    MPIRequest() = default;
    MPIRequest(const MPIRequest &) = delete;
    MPIRequest &operator=(const MPIRequest &) = delete;
    MPIRequest(MPIRequest &&other) noexcept
        : request(other.request)
        , active(other.active)
    {
        other.request = MPI_REQUEST_NULL;
        other.active = false;
    }
    MPIRequest &operator=(MPIRequest &&other) noexcept
    {
        if (this != &other)
        {
            if (active && request != MPI_REQUEST_NULL)
            {
                MPI_Wait(&request, MPI_STATUS_IGNORE);
            }
            request = other.request;
            active = other.active;
            other.request = MPI_REQUEST_NULL;
            other.active = false;
        }
        return *this;
    }

    void wait()
    {
        if (active && request != MPI_REQUEST_NULL)
        {
            MPI_Wait(&request, MPI_STATUS_IGNORE);
            active = false;
        }
    }
};

class MPICommunicator
{
public:
    MPICommunicator();
    ~MPICommunicator();

    // Move semantics  --  moved-from object becomes empty (rank/size == -1)
    MPICommunicator(MPICommunicator &&other) noexcept;
    MPICommunicator &operator=(MPICommunicator &&other) noexcept;

    // Deleted copy (owns MPI lifecycle contribution)
    MPICommunicator(const MPICommunicator &) = delete;
    MPICommunicator &operator=(const MPICommunicator &) = delete;

    int rank() const;
    int size() const;
    bool is_root() const;

    /// Collective operations  --  throw on MPI error
    template <typename T>
    void broadcast(T *data, int count, int root = 0);

    template <typename T>
    void allgather(const T *send_data, int send_count, T *recv_data, int recv_count);

    template <typename T>
    void alltoall(const T *sendbuf, int sendcount, T *recvbuf, int recvcount);

    template <typename T>
    MPIRequest iallgather(const T *sendbuf, int sendcount, T *recvbuf, int recvcount);

    template <typename T>
    MPIRequest ibroadcast(T *buffer, int count, int root = 0);

    void barrier();

    template <typename T>
    void reduce(const T *sendbuf, T *recvbuf, int count, MPI_Op op, int root = 0);

    template <typename T>
    MPIRequest ireduce(const T *sendbuf, T *recvbuf, int count, MPI_Op op, int root = 0);

    /// Non-blocking point-to-point  --  returns a handle that must be waited on
    template <typename T>
    MPIRequest isend(const T *data, int count, int dest, int tag = 0);

    template <typename T>
    MPIRequest irecv(T *data, int count, int source, int tag = 0);

    static void wait(MPIRequest &req);
    static void waitall(std::vector<MPIRequest> &reqs);

    /// ErrorResult-based overloads  --  no exception thrown on MPI error
    template <typename T>
    ::nerve::errors::ErrorResult<void> try_broadcast(T *data, int count, int root = 0);

    template <typename T>
    ::nerve::errors::ErrorResult<void> try_allgather(const T *send_data, int send_count,
                                                     T *recv_data, int recv_count);

    ::nerve::errors::ErrorResult<void> try_barrier();

    template <typename T>
    ::nerve::errors::ErrorResult<MPIRequest> try_isend(const T *data, int count, int dest,
                                                       int tag = 0);

    template <typename T>
    ::nerve::errors::ErrorResult<MPIRequest> try_irecv(T *data, int count, int source, int tag = 0);

private:
    int world_rank_;
    int world_size_;
};

class DistributedPersistence
{
public:
    DistributedPersistence();

    std::vector<std::tuple<float, float, int>>
    compute(const std::vector<std::vector<float>> &point_clouds);

private:
    MPICommunicator comm_;
    ShardedBoundaryMatrix matrix_;
    WorkStealingScheduler scheduler_;
    size_t num_columns_;

    void distribute_filtration(const std::vector<std::vector<float>> &point_clouds);
    std::vector<std::tuple<float, float, int>> gather_results();
};

struct DistributedBenchmark
{
    double single_node_time_ms = 0.0;
    double distributed_time_ms = 0.0;
    double speedup = 1.0;
    int num_nodes = 0;
    int num_gpus_per_node = 0;
    size_t total_data_size = 0;
};

DistributedBenchmark benchmark_distributed(int num_nodes, int data_size_per_node);

} // namespace nerve::distributed
#else

namespace nerve::distributed
{

using MPI_Op = int;

struct MPIRequest
{
    bool active = false;

    ~MPIRequest() { active = false; }

    MPIRequest() = default;
    MPIRequest(const MPIRequest &) = delete;
    MPIRequest &operator=(const MPIRequest &) = delete;
    MPIRequest(MPIRequest &&other) noexcept
        : active(other.active)
    {
        other.active = false;
    }
    MPIRequest &operator=(MPIRequest &&other) noexcept
    {
        if (this != &other)
        {
            active = other.active;
            other.active = false;
        }
        return *this;
    }

    void wait() { active = false; }
};

class MPICommunicator
{
public:
    MPICommunicator();
    ~MPICommunicator();

    // Move semantics
    MPICommunicator(MPICommunicator &&other) noexcept;
    MPICommunicator &operator=(MPICommunicator &&other) noexcept;

    MPICommunicator(const MPICommunicator &) = delete;
    MPICommunicator &operator=(const MPICommunicator &) = delete;

    int rank() const;
    int size() const;
    bool is_root() const;

    template <typename T>
    void broadcast(T *data, int count, int root = 0);

    template <typename T>
    void allgather(const T *send_data, int send_count, T *recv_data, int recv_count);

    template <typename T>
    void alltoall(const T *sendbuf, int sendcount, T *recvbuf, int recvcount);

    template <typename T>
    MPIRequest iallgather(const T *sendbuf, int sendcount, T *recvbuf, int recvcount);

    template <typename T>
    MPIRequest ibroadcast(T *buffer, int count, int root = 0);

    void barrier();

    template <typename T>
    void reduce(const T *sendbuf, T *recvbuf, int count, MPI_Op op, int root = 0);

    template <typename T>
    MPIRequest ireduce(const T *sendbuf, T *recvbuf, int count, MPI_Op op, int root = 0);

    // Non-blocking no-ops  --  execute synchronously in single-process mode
    template <typename T>
    MPIRequest isend(const T *data, int count, int dest, int tag = 0);

    template <typename T>
    MPIRequest irecv(T *data, int count, int source, int tag = 0);

    static void wait(MPIRequest &req) { req.active = false; }
    static void waitall(std::vector<MPIRequest> &reqs)
    {
        for (auto &r : reqs)
            r.active = false;
    }

    // ErrorResult overloads
    template <typename T>
    ::nerve::errors::ErrorResult<void> try_broadcast(T *data, int count, int root = 0);

    template <typename T>
    ::nerve::errors::ErrorResult<void> try_allgather(const T *send_data, int send_count,
                                                     T *recv_data, int recv_count);

    ::nerve::errors::ErrorResult<void> try_barrier();

    template <typename T>
    ::nerve::errors::ErrorResult<MPIRequest> try_isend(const T *data, int count, int dest,
                                                       int tag = 0);

    template <typename T>
    ::nerve::errors::ErrorResult<MPIRequest> try_irecv(T *data, int count, int source, int tag = 0);

private:
    int world_rank_;
    int world_size_;
};

} // namespace nerve::distributed
#endif

#pragma once

#if NERVE_HAS_MPI && __has_include(<mpi.h>)

#include <mpi.h>

#ifdef NERVE_HAS_CUDA
#if __has_include(<nccl.h>)
#include <cuda_runtime.h>
#include <nccl.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nerve::distributed
{

struct NcclMpiConfig
{
    int gpus_per_node = 0;
    int local_world_rank = -1;
    int local_world_size = 0;
    int node_id = -1;
    ncclUniqueId nccl_id;
};

struct NcclMpiWorld
{
    ncclComm_t nccl_comm_local = nullptr;
    ncclComm_t nccl_comm_global = nullptr;
    std::vector<int> gpu_ranks_local;
    std::vector<int> gpu_ranks_global;
    int local_rank = -1;
    int local_size = 0;
    int world_rank = -1;
    int world_size = 0;
    int node_id = -1;
};

class NcclMpiBridge
{
public:
    NcclMpiBridge() = default;

    explicit NcclMpiBridge(MPI_Comm mpi_comm, int gpus_per_node,
                           cudaStream_t default_stream = nullptr);

    ~NcclMpiBridge();

    NcclMpiBridge(NcclMpiBridge &&) = delete;
    NcclMpiBridge &operator=(NcclMpiBridge &&) = delete;
    NcclMpiBridge(const NcclMpiBridge &) = delete;
    NcclMpiBridge &operator=(const NcclMpiBridge &) = delete;

    bool is_initialized() const { return initialized_; }
    int local_rank() const { return world_.local_rank; }
    int local_size() const { return world_.local_size; }
    int node_id() const { return world_.node_id; }

    ncclComm_t local_comm() const { return world_.nccl_comm_local; }
    ncclComm_t global_comm() const { return world_.nccl_comm_global; }
    cudaStream_t stream() const { return stream_; }

    void allreduce(const double *send, double *recv, int count, ncclRedOp_t op = ncclSum)
    {
        if (!initialized_)
            return;
        ncclAllReduce(send, recv, count, ncclFloat64, op, world_.nccl_comm_local, stream_);
    }

    void allreduce(const float *send, float *recv, int count, ncclRedOp_t op = ncclSum)
    {
        if (!initialized_)
            return;
        ncclAllReduce(send, recv, count, ncclFloat32, op, world_.nccl_comm_local, stream_);
    }

    void broadcast(double *data, int count, int root)
    {
        if (!initialized_)
            return;
        ncclBroadcast(data, data, count, ncclFloat64, root, world_.nccl_comm_local, stream_);
    }

    void broadcast(float *data, int count, int root)
    {
        if (!initialized_)
            return;
        ncclBroadcast(data, data, count, ncclFloat32, root, world_.nccl_comm_local, stream_);
    }

    void allgather(const double *send, double *recv, int count)
    {
        if (!initialized_)
            return;
        ncclAllGather(send, recv, count, ncclFloat64, world_.nccl_comm_local, stream_);
    }

    void allgather(const float *send, float *recv, int count)
    {
        if (!initialized_)
            return;
        ncclAllGather(send, recv, count, ncclFloat32, world_.nccl_comm_local, stream_);
    }

    void barrier();

    void sync_stream() { cudaStreamSynchronize(stream_); }

    void cross_node_reduce(const double *local_data, double *global_result, int count,
                           MPI_Comm mpi_comm = MPI_COMM_WORLD, ncclRedOp_t op = ncclSum)
    {
        if (!initialized_)
            return;
        std::vector<double> host_local(count);
        std::vector<double> host_global(count);
        cudaMemcpy(host_local.data(), local_data, count * sizeof(double), cudaMemcpyDeviceToHost);
        MPI_Allreduce(host_local.data(), host_global.data(), count, MPI_DOUBLE,
                      op == ncclSum ? MPI_SUM : MPI_MAX, mpi_comm);
        cudaMemcpyAsync(global_result, host_global.data(), count * sizeof(double),
                        cudaMemcpyHostToDevice, stream_);
    }

    void cross_node_reduce(const float *local_data, float *global_result, int count,
                           MPI_Comm mpi_comm = MPI_COMM_WORLD, ncclRedOp_t op = ncclSum)
    {
        if (!initialized_)
            return;
        std::vector<float> host_local(count);
        std::vector<float> host_global(count);
        cudaMemcpy(host_local.data(), local_data, count * sizeof(float), cudaMemcpyDeviceToHost);
        MPI_Allreduce(host_local.data(), host_global.data(), count, MPI_FLOAT,
                      op == ncclSum ? MPI_SUM : MPI_MAX, mpi_comm);
        cudaMemcpyAsync(global_result, host_global.data(), count * sizeof(float),
                        cudaMemcpyHostToDevice, stream_);
    }

private:
    bool initialized_ = false;
    NcclMpiWorld world_;
    cudaStream_t stream_ = nullptr;
    bool owns_stream_ = false;

    void init_local_comm(MPI_Comm mpi_comm, int gpus_per_node);
    void init_global_comm(MPI_Comm mpi_comm);

    inline static std::atomic<int> active_bridge_count_{0};
    inline static std::mutex init_mutex_;
};

inline NcclMpiBridge::NcclMpiBridge(MPI_Comm mpi_comm, int gpus_per_node,
                                    cudaStream_t default_stream)
    : stream_(default_stream)
{
    if (gpus_per_node <= 0)
        return;

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count < 1)
        return;

    std::lock_guard<std::mutex> lock(init_mutex_);

    MPI_Comm local_comm;
    MPI_Comm_split_type(mpi_comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);
    int local_rank_val, local_size_val, world_rank_val, world_size_val;
    MPI_Comm_rank(local_comm, &local_rank_val);
    MPI_Comm_size(local_comm, &local_size_val);
    MPI_Comm_rank(mpi_comm, &world_rank_val);
    MPI_Comm_size(mpi_comm, &world_size_val);

    int gpu_id = local_rank_val % gpus_per_node;
    if (gpu_id >= device_count)
    {
        MPI_Comm_free(&local_comm);
        return;
    }
    cudaSetDevice(gpu_id);

    if (!stream_)
    {
        cudaStreamCreate(&stream_);
        owns_stream_ = true;
    }

    world_.local_rank = local_rank_val;
    world_.local_size = local_size_val;
    world_.world_rank = world_rank_val;
    world_.world_size = world_size_val;
    world_.node_id = world_rank_val / gpus_per_node;

    init_local_comm(mpi_comm, gpus_per_node);
    init_global_comm(mpi_comm);

    initialized_ = true;
    active_bridge_count_++;
}

inline NcclMpiBridge::~NcclMpiBridge()
{
    if (!initialized_)
        return;
    if (world_.nccl_comm_local)
    {
        ncclCommDestroy(world_.nccl_comm_local);
        world_.nccl_comm_local = nullptr;
    }
    if (world_.nccl_comm_global)
    {
        ncclCommDestroy(world_.nccl_comm_global);
        world_.nccl_comm_global = nullptr;
    }
    if (owns_stream_ && stream_)
    {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
    active_bridge_count_--;
}

inline void NcclMpiBridge::init_local_comm(MPI_Comm mpi_comm, int gpus_per_node)
{
    MPI_Comm local_comm;
    MPI_Comm_split_type(mpi_comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);
    std::lock_guard<std::mutex> lock(init_mutex_);

    int node_rank;
    MPI_Comm_rank(local_comm, &node_rank);
    int gpu_id = node_rank % gpus_per_node;

    ncclUniqueId id;
    if (node_rank == 0)
    {
        ncclGetUniqueId(&id);
    }
    MPI_Bcast(&id, sizeof(ncclUniqueId), MPI_BYTE, 0, local_comm);

    int local_gpus;
    MPI_Comm_size(local_comm, &local_gpus);
    if (local_gpus > gpus_per_node)
        local_gpus = gpus_per_node;

    ncclCommInitRank(&world_.nccl_comm_local, local_gpus, id, gpu_id);

    MPI_Comm_free(&local_comm);
}

inline void NcclMpiBridge::init_global_comm(MPI_Comm mpi_comm)
{
    int node_count = world_.world_size / world_.local_size;
    if (node_count <= 1)
        return;
    if (world_.local_rank != 0)
        return;

    ncclUniqueId id;
    ncclGetUniqueId(&id);
    MPI_Bcast(&id, sizeof(ncclUniqueId), MPI_BYTE, 0, mpi_comm);

    ncclCommInitRank(&world_.nccl_comm_global, node_count, id, world_.node_id);
}

inline void NcclMpiBridge::barrier()
{
    if (!initialized_)
        return;
    cudaError_t err = cudaStreamSynchronize(stream_);
    (void)err;
    MPI_Barrier(MPI_COMM_WORLD);
}

} // namespace nerve::distributed

#endif // __has_include(<nccl.h>)
#else
namespace nerve::distributed
{
class NcclMpiBridge
{
public:
    NcclMpiBridge() = default;
    explicit NcclMpiBridge(MPI_Comm, int) {}
    bool is_initialized() const { return false; }
};
} // namespace nerve::distributed
#endif // NERVE_HAS_CUDA
#endif // NERVE_HAS_MPI

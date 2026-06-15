#pragma once

#include "nerve/config.hpp"

#ifdef NERVE_HAS_CUDA

#include <cuda_runtime.h>

#if NERVE_HAS_MPI && __has_include(<mpi.h>)
#include <mpi.h>
#endif

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace nerve::distributed
{

struct IpcHandleEntry
{
    cudaIpcMemHandle_t handle;
    std::size_t size;
};

class CudaIpcSharedMemory
{
public:
#ifdef NERVE_HAS_MPI
    explicit CudaIpcSharedMemory(MPI_Comm comm = MPI_COMM_WORLD)
        : comm_(comm)
    {
        int rank = 0;
        int size = 0;
        MPI_Comm_rank(comm_, &rank);
        MPI_Comm_size(comm_, &size);
        rank_ = rank;
        world_size_ = size;
    }
#else
    CudaIpcSharedMemory()
        : rank_(0)
        , world_size_(1)
    {}
#endif

    int rank() const { return rank_; }
    int worldSize() const { return world_size_; }

    void publish(int source_rank, void *d_ptr, std::size_t size)
    {
        cudaIpcMemHandle_t handle;
        cudaError_t err = cudaIpcGetMemHandle(&handle, d_ptr);
        if (err != cudaSuccess)
        {
            throw std::runtime_error("cudaIpcGetMemHandle failed: " +
                                     std::string(cudaGetErrorString(err)));
        }

        broadcastHandle(source_rank, handle, size);
        cacheHandle(source_rank, handle, size);
    }

    void *open(int source_rank, std::size_t size)
    {
        auto it = handle_cache_.find(source_rank);
        if (it == handle_cache_.end())
        {
            throw std::runtime_error("CudaIpcSharedMemory::open: no handle for rank " +
                                     std::to_string(source_rank));
        }

        if (it->second.size < size)
        {
            throw std::runtime_error("CudaIpcSharedMemory::open: requested size " +
                                     std::to_string(size) + " exceeds published size " +
                                     std::to_string(it->second.size));
        }

        void *ptr = nullptr;
        cudaError_t err =
            cudaIpcOpenMemHandle(&ptr, it->second.handle, cudaIpcMemLazyEnablePeerAccess);
        if (err != cudaSuccess)
        {
            throw std::runtime_error("cudaIpcOpenMemHandle failed: " +
                                     std::string(cudaGetErrorString(err)));
        }

        opened_ptrs_.push_back({source_rank, ptr});
        return ptr;
    }

    void close(void *d_ptr)
    {
        for (auto it = opened_ptrs_.begin(); it != opened_ptrs_.end(); ++it)
        {
            if (it->second == d_ptr)
            {
                cudaError_t err = cudaIpcCloseMemHandle(d_ptr);
                if (err != cudaSuccess)
                {
                    throw std::runtime_error("cudaIpcCloseMemHandle failed: " +
                                             std::string(cudaGetErrorString(err)));
                }
                opened_ptrs_.erase(it);
                return;
            }
        }
        throw std::runtime_error("CudaIpcSharedMemory::close: handle not found in opened set");
    }

private:
#ifdef NERVE_HAS_MPI
    void broadcastHandle(int source_rank, cudaIpcMemHandle_t handle, std::size_t size)
    {
        MPI_Bcast(&handle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, source_rank, comm_);
        std::uint64_t sz = static_cast<std::uint64_t>(size);
        MPI_Bcast(&sz, 1, MPI_UINT64_T, source_rank, comm_);
    }
#else
    void broadcastHandle(int, cudaIpcMemHandle_t, std::size_t) {}
#endif

    void cacheHandle(int source_rank, cudaIpcMemHandle_t handle, std::size_t size)
    {
        IpcHandleEntry entry;
        entry.handle = handle;
        entry.size = size;
        handle_cache_[source_rank] = entry;
    }

    int rank_;
    int world_size_;

#ifdef NERVE_HAS_MPI
    MPI_Comm comm_;
#endif

    std::unordered_map<int, IpcHandleEntry> handle_cache_;
    std::vector<std::pair<int, void *>> opened_ptrs_;
};

} // namespace nerve::distributed

#endif // NERVE_HAS_CUDA

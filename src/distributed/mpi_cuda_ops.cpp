#include "nerve/config.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <vector>

#if defined(NERVE_HAS_MPI)
#include <mpi.h>
#endif

#include "nerve/distributed/cuda_aware_mpi.hpp"

namespace nerve::distributed
{

#if defined(NERVE_HAS_CUDA) && defined(NERVE_HAS_MPI)

static int mpiCudaRank()
{
    int rank = 0;
    int mpi_err = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (mpi_err != MPI_SUCCESS)
    {
        std::cerr << "MPI_Comm_rank failed in mpiCudaRank" << std::endl;
    }
    return rank;
}

static int mpiCudaSize()
{
    int size = 1;
    int mpi_err = MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (mpi_err != MPI_SUCCESS)
    {
        std::cerr << "MPI_Comm_size failed in mpiCudaSize" << std::endl;
    }
    return size;
}

// AllGather
cudaError_t mpiCudaAllGather(const void *sendbuf, int sendcount, void *recvbuf, int recvcount,
                             MPI_Datatype datatype)
{
    if (mpiCudaSize() <= 1)
    {
        std::memcpy(recvbuf, sendbuf, static_cast<size_t>(sendcount) * sizeof(double));
        return cudaSuccess;
    }

    void *host_send = nullptr;
    bool is_device = false;
    cudaPointerAttributes attr;
    cudaPointerGetAttributes(&attr, sendbuf);

    if (attr.type == cudaMemoryTypeDevice)
    {
        is_device = true;
        cudaMallocHost(&host_send, sendcount * sizeof(double));
        cudaMemcpy(host_send, sendbuf, sendcount * sizeof(double), cudaMemcpyDeviceToHost);
    }
    else
    {
        host_send = const_cast<void *>(sendbuf);
    }

    void *host_recv = nullptr;
    bool recv_is_device = false;
    cudaPointerAttributes recv_attr;
    cudaPointerGetAttributes(&recv_attr, recvbuf);
    if (recv_attr.type == cudaMemoryTypeDevice)
    {
        recv_is_device = true;
        cudaMallocHost(&host_recv, recvcount * mpiCudaSize() * sizeof(double));
    }
    else
    {
        host_recv = recvbuf;
    }

    int mpi_err = MPI_Allgather(host_send, sendcount, datatype, host_recv, recvcount, datatype,
                                MPI_COMM_WORLD);
    if (mpi_err != MPI_SUCCESS)
    {
        std::cerr << "MPI_Allgather failed in mpiCudaAllGather" << std::endl;
        if (recv_is_device)
            cudaFreeHost(host_recv);
        if (is_device)
            cudaFreeHost(host_send);
        return cudaErrorUnknown;
    }

    if (recv_is_device)
    {
        cudaMemcpy(recvbuf, host_recv, recvcount * mpiCudaSize() * sizeof(double),
                   cudaMemcpyHostToDevice);
        cudaFreeHost(host_recv);
    }

    if (is_device)
        cudaFreeHost(host_send);

    return cudaSuccess;
}

// Reduce
cudaError_t mpiCudaReduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
                          MPI_Op op, int root)
{
    void *host_send = nullptr;
    bool send_is_device = false;
    cudaPointerAttributes attr;
    cudaPointerGetAttributes(&attr, sendbuf);
    if (attr.type == cudaMemoryTypeDevice)
    {
        send_is_device = true;
        cudaMallocHost(&host_send, count * sizeof(double));
        cudaMemcpy(host_send, sendbuf, count * sizeof(double), cudaMemcpyDeviceToHost);
    }
    else
    {
        host_send = const_cast<void *>(sendbuf);
    }

    void *host_recv = nullptr;
    bool recv_is_device = false;
    cudaPointerAttributes recv_attr;
    cudaPointerGetAttributes(&recv_attr, recvbuf);
    if (recv_attr.type == cudaMemoryTypeDevice)
    {
        recv_is_device = true;
        cudaMallocHost(&host_recv, count * sizeof(double));
    }
    else
    {
        host_recv = recvbuf;
    }

    int mpi_err = MPI_Reduce(host_send, host_recv, count, datatype, op, root, MPI_COMM_WORLD);
    if (mpi_err != MPI_SUCCESS)
    {
        std::cerr << "MPI_Reduce failed in mpiCudaReduce" << std::endl;
        if (recv_is_device)
            cudaFreeHost(host_recv);
        if (send_is_device)
            cudaFreeHost(host_send);
        return cudaErrorUnknown;
    }

    if (recv_is_device)
    {
        int rank = mpiCudaRank();
        if (rank == root)
        {
            cudaMemcpy(recvbuf, host_recv, count * sizeof(double), cudaMemcpyHostToDevice);
        }
        cudaFreeHost(host_recv);
    }

    if (send_is_device)
        cudaFreeHost(host_send);

    return cudaSuccess;
}

// Broadcast
cudaError_t mpiCudaBroadcast(void *buf, int count, MPI_Datatype datatype, int root)
{
    void *host_buf = nullptr;
    bool is_device = false;
    cudaPointerAttributes attr;
    cudaPointerGetAttributes(&attr, buf);
    if (attr.type == cudaMemoryTypeDevice)
    {
        is_device = true;
        cudaMallocHost(&host_buf, count * sizeof(double));
        int rank = mpiCudaRank();
        if (rank == root)
        {
            cudaMemcpy(host_buf, buf, count * sizeof(double), cudaMemcpyDeviceToHost);
        }
    }
    else
    {
        host_buf = buf;
    }

    int mpi_err = MPI_Bcast(host_buf, count, datatype, root, MPI_COMM_WORLD);
    if (mpi_err != MPI_SUCCESS)
    {
        std::cerr << "MPI_Bcast failed in mpiCudaBroadcast" << std::endl;
        if (is_device)
            cudaFreeHost(host_buf);
        return cudaErrorUnknown;
    }

    if (is_device)
    {
        cudaMemcpy(buf, host_buf, count * sizeof(double), cudaMemcpyHostToDevice);
        cudaFreeHost(host_buf);
    }

    return cudaSuccess;
}

#else

cudaError_t mpiCudaAllGather(const void *, int, void *, int, MPI_Datatype)
{
    return cudaErrorNotSupported;
}

cudaError_t mpiCudaReduce(const void *, void *, int, MPI_Datatype, MPI_Op, int)
{
    return cudaErrorNotSupported;
}

cudaError_t mpiCudaBroadcast(void *, int, MPI_Datatype, int)
{
    return cudaErrorNotSupported;
}

#endif

} // namespace nerve::distributed

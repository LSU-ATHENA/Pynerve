#pragma once

#if NERVE_HAS_MPI && __has_include(<mpi.h>)

#include <mpi.h>

#ifdef NERVE_HAS_CUDA
#include <cuda_runtime.h>
#endif

#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace nerve::distributed
{

#ifdef NERVE_HAS_CUDA

inline bool is_cuda_aware_mpi()
{
    int flag = 0;
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
    if (MPIX_Query_cuda_support() == 1)
        flag = 1;
#endif
    return flag != 0;
}

template <typename T>
void mpi_send_device(const T *d_data, int count, int dest, int tag, cudaStream_t stream,
                     MPI_Comm comm = MPI_COMM_WORLD)
{
    cudaStreamSynchronize(stream);
    if (is_cuda_aware_mpi())
    {
        MPI_Send(d_data, count * sizeof(T), MPI_BYTE, dest, tag, comm);
    }
    else
    {
        std::vector<T> host(count);
        cudaMemcpy(host.data(), d_data, count * sizeof(T), cudaMemcpyDeviceToHost);
        MPI_Send(host.data(), count * sizeof(T), MPI_BYTE, dest, tag, comm);
    }
}

template <typename T>
void mpi_recv_device(T *d_data, int count, int source, int tag, cudaStream_t stream,
                     MPI_Comm comm = MPI_COMM_WORLD)
{
    if (is_cuda_aware_mpi())
    {
        MPI_Recv(d_data, count * sizeof(T), MPI_BYTE, source, tag, comm, MPI_STATUS_IGNORE);
        cudaStreamSynchronize(stream);
    }
    else
    {
        std::vector<T> host(count);
        MPI_Recv(host.data(), count * sizeof(T), MPI_BYTE, source, tag, comm, MPI_STATUS_IGNORE);
        cudaMemcpyAsync(d_data, host.data(), count * sizeof(T), cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);
    }
}

template <typename T>
void mpi_irecv_device(T *d_data, int count, int source, int tag, MPI_Request *request,
                      cudaStream_t stream, MPI_Comm comm = MPI_COMM_WORLD)
{
    if (is_cuda_aware_mpi())
    {
        MPI_Irecv(d_data, count * sizeof(T), MPI_BYTE, source, tag, comm, request);
    }
    else
    {
        T *host = new T[count];
        MPI_Irecv(host, count * sizeof(T), MPI_BYTE, source, tag, comm, request);
        MPI_Wait(request, MPI_STATUS_IGNORE);
        cudaMemcpyAsync(d_data, host, count * sizeof(T), cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);
        delete[] host;
    }
}

template <typename T>
void mpi_isend_device(const T *d_data, int count, int dest, int tag, MPI_Request *request,
                      cudaStream_t stream, MPI_Comm comm = MPI_COMM_WORLD)
{
    cudaStreamSynchronize(stream);
    if (is_cuda_aware_mpi())
    {
        MPI_Isend(d_data, count * sizeof(T), MPI_BYTE, dest, tag, comm, request);
    }
    else
    {
        T *host = new T[count];
        cudaMemcpy(host, d_data, count * sizeof(T), cudaMemcpyDeviceToHost);
        MPI_Isend(host, count * sizeof(T), MPI_BYTE, dest, tag, comm, request);
        MPI_Wait(request, MPI_STATUS_IGNORE);
        delete[] host;
    }
}

template <typename T>
void mpi_bcast_device(T *d_data, int count, int root, cudaStream_t stream,
                      MPI_Comm comm = MPI_COMM_WORLD)
{
    cudaStreamSynchronize(stream);
    if (is_cuda_aware_mpi())
    {
        MPI_Bcast(d_data, count * sizeof(T), MPI_BYTE, root, comm);
    }
    else
    {
        std::vector<T> host(count);
        int rank;
        MPI_Comm_rank(comm, &rank);
        if (rank == root)
        {
            cudaMemcpy(host.data(), d_data, count * sizeof(T), cudaMemcpyDeviceToHost);
        }
        MPI_Bcast(host.data(), count * sizeof(T), MPI_BYTE, root, comm);
        if (rank != root)
        {
            cudaMemcpyAsync(d_data, host.data(), count * sizeof(T), cudaMemcpyHostToDevice, stream);
        }
        cudaStreamSynchronize(stream);
    }
}

template <typename T>
void mpi_allreduce_device(const T *d_send, T *d_recv, int count, MPI_Op op, cudaStream_t stream,
                          MPI_Comm comm = MPI_COMM_WORLD)
{
    cudaStreamSynchronize(stream);
    if (is_cuda_aware_mpi())
    {
        MPI_Allreduce(d_send, d_recv, count * sizeof(T), MPI_BYTE, op, comm);
    }
    else
    {
        std::vector<T> send_host(count), recv_host(count);
        cudaMemcpy(send_host.data(), d_send, count * sizeof(T), cudaMemcpyDeviceToHost);
        MPI_Allreduce(send_host.data(), recv_host.data(), count * sizeof(T), MPI_BYTE, op, comm);
        cudaMemcpyAsync(d_recv, recv_host.data(), count * sizeof(T), cudaMemcpyHostToDevice,
                        stream);
        cudaStreamSynchronize(stream);
    }
}

template <typename T>
void mpi_allgather_device(const T *d_send, int send_count, T *d_recv, int recv_count,
                          cudaStream_t stream, MPI_Comm comm = MPI_COMM_WORLD)
{
    cudaStreamSynchronize(stream);
    if (is_cuda_aware_mpi())
    {
        MPI_Allgather(d_send, send_count * sizeof(T), MPI_BYTE, d_recv, recv_count * sizeof(T),
                      MPI_BYTE, comm);
    }
    else
    {
        std::vector<T> send_host(send_count), recv_host(recv_count);
        cudaMemcpy(send_host.data(), d_send, send_count * sizeof(T), cudaMemcpyDeviceToHost);
        MPI_Allgather(send_host.data(), send_count * sizeof(T), MPI_BYTE, recv_host.data(),
                      recv_count * sizeof(T), MPI_BYTE, comm);
        cudaMemcpyAsync(d_recv, recv_host.data(), recv_count * sizeof(T), cudaMemcpyHostToDevice,
                        stream);
        cudaStreamSynchronize(stream);
    }
}

#endif

} // namespace nerve::distributed

#endif // NERVE_HAS_MPI

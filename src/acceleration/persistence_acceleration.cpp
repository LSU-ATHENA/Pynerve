#include <cstdint>
#include <cstring>
#include <vector>

#ifdef NERVE_HAS_CUDA
#include <cuda_runtime.h>
#endif
#ifdef NERVE_HAS_MPI
#include <mpi.h>
#endif

#include "nerve/acceleration_fwd.hpp"
#include "nerve/types.hpp"

namespace nerve::persistence::mpi_cuda
{
using nerve::algorithms::mpi::algoSize;

cudaError_t distributeColumnsGPU(const int *d_columns, Size ncols, int ngpus, int **d_partitions,
                                 Size *counts)
{
    if (!d_columns || !d_partitions || !counts || ncols == 0 || ngpus <= 0)
        return cudaErrorInvalidValue;
#ifdef NERVE_HAS_CUDA
    Size per = ncols / static_cast<Size>(ngpus);
    Size rem = ncols % static_cast<Size>(ngpus);
    Size offset = 0;
    for (int g = 0; g < ngpus; ++g)
    {
        cudaError_t err = cudaSetDevice(g);
        if (err != cudaSuccess)
            return err;
        Size chunk = per + (static_cast<Size>(g) < rem ? 1 : 0);
        counts[g] = chunk;
        err = cudaMalloc(&d_partitions[g], chunk * sizeof(int));
        if (err != cudaSuccess)
            return err;
        err = cudaMemcpy(d_partitions[g], d_columns + offset, chunk * sizeof(int),
                         cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess)
            return err;
        offset += chunk;
    }
    return cudaSuccess;
#else
    return cudaErrorNotSupported;
#endif
}
cudaError_t exchangePivotsGPU(const int *d_local, Size nlocal, int *d_all, MPI_Comm comm,
                              cudaStream_t s)
{
    if (!d_local || !d_all || nlocal == 0)
        return cudaErrorInvalidValue;
#ifdef NERVE_HAS_CUDA
    std::vector<int> host(nlocal);
    cudaError_t err =
        cudaMemcpyAsync(host.data(), d_local, nlocal * sizeof(int), cudaMemcpyDeviceToHost, s);
    if (err != cudaSuccess)
        return err;
    err = cudaStreamSynchronize(s);
    if (err != cudaSuccess)
        return err;
#ifdef NERVE_HAS_MPI
    int sz = 0;
    MPI_Comm_size(comm, &sz);
    int cnt = static_cast<int>(nlocal);
    std::vector<int> host_all(nlocal * static_cast<Size>(sz));
    MPI_Allgather(host.data(), cnt, MPI_INT, host_all.data(), cnt, MPI_INT, comm);
    err = cudaMemcpyAsync(d_all, host_all.data(), nlocal * static_cast<Size>(sz) * sizeof(int),
                          cudaMemcpyHostToDevice, s);
    return (err != cudaSuccess) ? err : cudaStreamSynchronize(s);
#else
    (void)comm;
    (void)d_all;
    return cudaErrorNotSupported;
#endif
#else
    (void)comm;
    (void)s;
    return cudaErrorNotSupported;
#endif
}
cudaError_t allGatherPairsGPU(const Pair *d_local, Size nlocal, Pair *d_all, MPI_Comm comm)
{
    if (!d_local || !d_all || nlocal == 0)
        return cudaErrorInvalidValue;
#ifdef NERVE_HAS_CUDA
    Size bytes = nlocal * sizeof(Pair);
    std::vector<Pair> host(nlocal);
    cudaError_t err = cudaMemcpy(host.data(), d_local, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
        return err;
    int sz = algoSize();
#ifdef NERVE_HAS_MPI
    std::vector<Pair> host_all(nlocal * static_cast<Size>(sz));
    MPI_Allgather(host.data(), static_cast<int>(bytes), MPI_BYTE, host_all.data(),
                  static_cast<int>(bytes), MPI_BYTE, comm);
#else
    std::vector<Pair> host_all = host;
    (void)comm;
#endif
    err = cudaMemcpy(d_all, host_all.data(), nlocal * static_cast<Size>(sz) * sizeof(Pair),
                     cudaMemcpyHostToDevice);
    return err;
#else
    (void)comm;
    return cudaErrorNotSupported;
#endif
}
} // namespace nerve::persistence::mpi_cuda

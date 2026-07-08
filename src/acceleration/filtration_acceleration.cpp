#include <cstdint>
#include <cstring>
#include <vector>

#ifdef NERVE_HAS_CUDA
#include <cuda_runtime.h>
#endif
#ifdef NERVE_HAS_MPI
#include <mpi.h>
#endif

#include "nerve/types.hpp"
#include "nerve/acceleration_fwd.hpp"

namespace nerve::filtration::multi_gpu
{
cudaError_t distributeFiltration(const int *d_edges, Size nedges, int ngpus, int **d_partitions)
{
    if (!d_edges || !d_partitions || nedges == 0 || ngpus <= 0)
        return cudaErrorInvalidValue;
#ifdef NERVE_HAS_CUDA
    Size per = nedges / static_cast<Size>(ngpus);
    Size rem = nedges % static_cast<Size>(ngpus);
    Size offset = 0;
    for (int g = 0; g < ngpus; ++g)
    {
        cudaError_t err = cudaSetDevice(g);
        if (err != cudaSuccess)
            return err;
        Size chunk = per + (static_cast<Size>(g) < rem ? 1 : 0);
        err = cudaMalloc(&d_partitions[g], chunk * sizeof(int));
        if (err != cudaSuccess)
            return err;
        err = cudaMemcpy(d_partitions[g], d_edges + offset, chunk * sizeof(int),
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
cudaError_t gatherPairs(const Pair *const *d_partials, const int *counts, int ngpus, Pair *d_result)
{
    if (!d_partials || !counts || !d_result || ngpus <= 0)
        return cudaErrorInvalidValue;
#ifdef NERVE_HAS_CUDA
    Size offset = 0;
    for (int g = 0; g < ngpus; ++g)
    {
        if (counts[g] <= 0 || !d_partials[g])
            continue;
        cudaError_t err = cudaSetDevice(g);
        if (err != cudaSuccess)
            return err;
        err = cudaMemcpy(d_result + offset, d_partials[g],
                         static_cast<Size>(counts[g]) * sizeof(Pair), cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess)
            return err;
        offset += static_cast<Size>(counts[g]);
    }
    return cudaSuccess;
#else
    return cudaErrorNotSupported;
#endif
}
} // namespace nerve::filtration::multi_gpu
namespace nerve::filtration::mpi_cuda
{
cudaError_t allGatherFiltrationGPU(const double *d_local, Size nlocal, double *d_all,
                                   cudaStream_t s)
{
    if (!d_local || !d_all || nlocal == 0)
        return cudaErrorInvalidValue;
#ifdef NERVE_HAS_CUDA
    std::vector<double> host(nlocal);
    cudaError_t err =
        cudaMemcpyAsync(host.data(), d_local, nlocal * sizeof(double), cudaMemcpyDeviceToHost, s);
    if (err != cudaSuccess)
        return err;
    err = cudaStreamSynchronize(s);
    if (err != cudaSuccess)
        return err;
#ifdef NERVE_HAS_MPI
    int sz = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &sz);
    int cnt = static_cast<int>(nlocal);
    std::vector<double> host_all(nlocal * static_cast<Size>(sz));
    MPI_Allgather(host.data(), cnt, MPI_DOUBLE, host_all.data(), cnt, MPI_DOUBLE, MPI_COMM_WORLD);
    err = cudaMemcpyAsync(d_all, host_all.data(), nlocal * static_cast<Size>(sz) * sizeof(double),
                          cudaMemcpyHostToDevice, s);
    return (err != cudaSuccess) ? err : cudaStreamSynchronize(s);
#else
    (void)d_all;
    return cudaErrorNotSupported;
#endif
#else
    (void)s;
    return cudaErrorNotSupported;
#endif
}
} // namespace nerve::filtration::mpi_cuda

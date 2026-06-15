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

namespace nerve::sheaf::multi_gpu
{
cudaError_t distributeStalks(const double *d_stalks, Size dim, int ngpus, double **d_partitions)
{
    if (!d_stalks || !d_partitions || dim == 0 || ngpus <= 0)
        return cudaErrorInvalidValue;
#ifdef NERVE_HAS_CUDA
    Size per = dim / static_cast<Size>(ngpus);
    Size rem = dim % static_cast<Size>(ngpus);
    Size offset = 0;
    for (int g = 0; g < ngpus; ++g)
    {
        cudaError_t err = cudaSetDevice(g);
        if (err != cudaSuccess)
            return err;
        Size chunk = per + (static_cast<Size>(g) < rem ? 1 : 0);
        err = cudaMalloc(&d_partitions[g], chunk * sizeof(double));
        if (err != cudaSuccess)
            return err;
        err = cudaMemcpy(d_partitions[g], d_stalks + offset, chunk * sizeof(double),
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
cudaError_t gatherCocycles(const double *const *d_cocycles, const int *sizes, int ngpus,
                           double *d_result)
{
    if (!d_cocycles || !sizes || !d_result || ngpus <= 0)
        return cudaErrorInvalidValue;
#ifdef NERVE_HAS_CUDA
    Size offset = 0;
    for (int g = 0; g < ngpus; ++g)
    {
        if (sizes[g] <= 0 || !d_cocycles[g])
            continue;
        cudaError_t err = cudaSetDevice(g);
        if (err != cudaSuccess)
            return err;
        err = cudaMemcpy(d_result + offset, d_cocycles[g],
                         static_cast<Size>(sizes[g]) * sizeof(double), cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess)
            return err;
        offset += static_cast<Size>(sizes[g]);
    }
    return cudaSuccess;
#else
    return cudaErrorNotSupported;
#endif
}
} // namespace nerve::sheaf::multi_gpu
namespace nerve::sheaf::mpi
{
void allGatherCocycles(const double *l, Size nl, double *a)
{
    if (!l || !a || nl == 0)
        return;
#ifdef NERVE_HAS_MPI
    MPI_Allgather(l, static_cast<int>(nl), MPI_DOUBLE, a, static_cast<int>(nl), MPI_DOUBLE,
                  MPI_COMM_WORLD);
#else
    std::memcpy(a, l, nl * sizeof(double));
#endif
}
} // namespace nerve::sheaf::mpi

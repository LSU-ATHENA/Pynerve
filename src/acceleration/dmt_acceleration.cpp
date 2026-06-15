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

namespace nerve::dmt::multi_gpu
{
cudaError_t scatterCells(const int *d_cells, Size n, int ngpus, int **d_partitions)
{
    if (!d_cells || !d_partitions || n == 0 || ngpus <= 0)
        return cudaErrorInvalidValue;
#ifdef NERVE_HAS_CUDA
    Size per = n / static_cast<Size>(ngpus);
    Size rem = n % static_cast<Size>(ngpus);
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
        err = cudaMemcpy(d_partitions[g], d_cells + offset, chunk * sizeof(int),
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
cudaError_t gatherGradients(const int *const *d_grads, const int *sizes, int ngpus, int *d_result)
{
    if (!d_grads || !sizes || !d_result || ngpus <= 0)
        return cudaErrorInvalidValue;
#ifdef NERVE_HAS_CUDA
    Size offset = 0;
    for (int g = 0; g < ngpus; ++g)
    {
        if (sizes[g] <= 0 || !d_grads[g])
            continue;
        cudaError_t err = cudaSetDevice(g);
        if (err != cudaSuccess)
            return err;
        err = cudaMemcpy(d_result + offset, d_grads[g], static_cast<Size>(sizes[g]) * sizeof(int),
                         cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess)
            return err;
        offset += static_cast<Size>(sizes[g]);
    }
    return cudaSuccess;
#else
    return cudaErrorNotSupported;
#endif
}
} // namespace nerve::dmt::multi_gpu
namespace nerve::dmt::mpi
{
void allGatherGradientField(const int *l, Size nl, int *a)
{
    if (!l || !a || nl == 0)
        return;
#ifdef NERVE_HAS_MPI
    MPI_Allgather(l, static_cast<int>(nl), MPI_INT, a, static_cast<int>(nl), MPI_INT,
                  MPI_COMM_WORLD);
#else
    std::memcpy(a, l, nl * sizeof(int));
#endif
}
} // namespace nerve::dmt::mpi

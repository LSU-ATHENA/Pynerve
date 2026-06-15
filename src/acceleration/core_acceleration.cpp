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

namespace nerve::core::multi_gpu
{
cudaError_t scatterInput(const double *d_input, Size n, int num_gpus, double **d_partitions)
{
    if (!d_input || !d_partitions || n == 0 || num_gpus <= 0)
        return cudaErrorInvalidValue;
#ifdef NERVE_HAS_CUDA
    Size per = n / static_cast<Size>(num_gpus);
    Size rem = n % static_cast<Size>(num_gpus);
    Size offset = 0;
    for (int g = 0; g < num_gpus; ++g)
    {
        cudaError_t err = cudaSetDevice(g);
        if (err != cudaSuccess)
            return err;
        Size chunk = per + (static_cast<Size>(g) < rem ? 1 : 0);
        err = cudaMalloc(&d_partitions[g], chunk * sizeof(double));
        if (err != cudaSuccess)
            return err;
        err = cudaMemcpy(d_partitions[g], d_input + offset, chunk * sizeof(double),
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
cudaError_t gatherOutput(double **d_partitions, const int *sizes, int num_gpus, double *d_out)
{
    if (!d_partitions || !sizes || !d_out || num_gpus <= 0)
        return cudaErrorInvalidValue;
#ifdef NERVE_HAS_CUDA
    Size offset = 0;
    for (int g = 0; g < num_gpus; ++g)
    {
        if (sizes[g] <= 0 || !d_partitions[g])
            continue;
        cudaError_t err = cudaSetDevice(g);
        if (err != cudaSuccess)
            return err;
        err = cudaMemcpy(d_out + offset, d_partitions[g],
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
} // namespace nerve::core::multi_gpu
namespace nerve::core::mpi
{
void broadcastConfig(void *b, Size n, int r)
{
    if (!b || n == 0)
        return;
#ifdef NERVE_HAS_MPI
    MPI_Bcast(b, static_cast<int>(n), MPI_BYTE, r, MPI_COMM_WORLD);
#else
    (void)r;
#endif
}
void reduceMetrics(const double *l, double *g, Size n)
{
    if (!l || !g || n == 0)
        return;
#ifdef NERVE_HAS_MPI
    MPI_Reduce(l, g, static_cast<int>(n), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
#else
    std::memcpy(g, l, n * sizeof(double));
#endif
}
} // namespace nerve::core::mpi

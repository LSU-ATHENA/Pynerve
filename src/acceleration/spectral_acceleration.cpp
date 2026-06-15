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

namespace nerve::spectral::multi_gpu
{
cudaError_t distributeMatrices(const double *d_matrix, Size n, int ngpus, double **d_partitions)
{
    if (!d_matrix || !d_partitions || n == 0 || ngpus <= 0)
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
        err = cudaMalloc(&d_partitions[g], chunk * sizeof(double));
        if (err != cudaSuccess)
            return err;
        err = cudaMemcpy(d_partitions[g], d_matrix + offset, chunk * sizeof(double),
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
cudaError_t gatherEigenvectors(const double *const *d_vectors, const int *sizes, int ngpus,
                               double *d_result)
{
    if (!d_vectors || !sizes || !d_result || ngpus <= 0)
        return cudaErrorInvalidValue;
#ifdef NERVE_HAS_CUDA
    Size offset = 0;
    for (int g = 0; g < ngpus; ++g)
    {
        if (sizes[g] <= 0 || !d_vectors[g])
            continue;
        cudaError_t err = cudaSetDevice(g);
        if (err != cudaSuccess)
            return err;
        err = cudaMemcpy(d_result + offset, d_vectors[g],
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
} // namespace nerve::spectral::multi_gpu
namespace nerve::spectral::mpi
{
void allReduceEigenvalues(const double *l, double *g, Size n)
{
    if (!l || !g || n == 0)
        return;
#ifdef NERVE_HAS_MPI
    MPI_Reduce(l, g, static_cast<int>(n), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
#else
    std::memcpy(g, l, n * sizeof(double));
#endif
}
} // namespace nerve::spectral::mpi

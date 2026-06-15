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

namespace nerve::optimization::multi_gpu
{
cudaError_t distributeParams(double *d_params, Size n, int ngpus, double **d_partitions)
{
    if (!d_params || !d_partitions || n == 0 || ngpus <= 0)
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
        err = cudaMemcpy(d_partitions[g], d_params + offset, chunk * sizeof(double),
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
cudaError_t reduceParamGrads(const double *const *d_grads, const int *sizes, int ngpus,
                             double *d_out)
{
    if (!d_grads || !sizes || !d_out || ngpus <= 0)
        return cudaErrorInvalidValue;
#ifdef NERVE_HAS_CUDA
    int total = 0;
    for (int g = 0; g < ngpus; ++g)
        total += sizes[g];
    std::vector<double> host_out(static_cast<Size>(total), 0.0);
    Size offset = 0;
    for (int g = 0; g < ngpus; ++g)
    {
        if (sizes[g] <= 0 || !d_grads[g])
            continue;
        cudaError_t err = cudaSetDevice(g);
        if (err != cudaSuccess)
            return err;
        std::vector<double> host(static_cast<Size>(sizes[g]));
        err = cudaMemcpy(host.data(), d_grads[g], static_cast<Size>(sizes[g]) * sizeof(double),
                         cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
            return err;
        for (int i = 0; i < sizes[g]; ++i)
            host_out[offset + static_cast<Size>(i)] += host[static_cast<Size>(i)];
        offset += static_cast<Size>(sizes[g]);
    }
    cudaSetDevice(0);
    return cudaMemcpy(d_out, host_out.data(), static_cast<Size>(total) * sizeof(double),
                      cudaMemcpyHostToDevice);
#else
    return cudaErrorNotSupported;
#endif
}
} // namespace nerve::optimization::multi_gpu
namespace nerve::optimization::mpi
{
void allReduceOptimizerState(double *s, Size n)
{
    if (!s || n == 0)
        return;
#ifdef NERVE_HAS_MPI
    MPI_Allreduce(MPI_IN_PLACE, s, static_cast<int>(n), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#else
    (void)s;
    (void)n;
#endif
}
} // namespace nerve::optimization::mpi

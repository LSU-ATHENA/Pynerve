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

namespace nerve::autodiff::multi_gpu
{
cudaError_t distributeAutodiffWork(Size n_params, int *d_offsets, int num_gpus)
{
    if (n_params == 0 || num_gpus <= 0)
        return cudaErrorInvalidValue;
#ifdef NERVE_HAS_CUDA
    Size per = n_params / static_cast<Size>(num_gpus);
    Size rem = n_params % static_cast<Size>(num_gpus);
    if (d_offsets)
    {
        d_offsets[0] = 0;
        for (int g = 1; g < num_gpus; ++g)
            d_offsets[g] =
                d_offsets[g - 1] + static_cast<int>(per + (static_cast<Size>(g - 1) < rem ? 1 : 0));
    }
    for (int g = 0; g < num_gpus; ++g)
    {
        cudaError_t err = cudaSetDevice(g);
        if (err != cudaSuccess)
            return err;
    }
    return cudaSuccess;
#else
    (void)d_offsets;
    return cudaErrorNotSupported;
#endif
}
cudaError_t reduceGradients(const double *const *d_grads, const int *sizes, int num_gpus,
                            double *d_result)
{
    if (!d_grads || !sizes || !d_result || num_gpus <= 0)
        return cudaErrorInvalidValue;
#ifdef NERVE_HAS_CUDA
    int total = 0;
    for (int g = 0; g < num_gpus; ++g)
        total += sizes[g];
    std::vector<double> host_result(static_cast<Size>(total), 0.0);
    Size offset = 0;
    for (int g = 0; g < num_gpus; ++g)
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
            host_result[offset + static_cast<Size>(i)] += host[static_cast<Size>(i)];
        offset += static_cast<Size>(sizes[g]);
    }
    cudaSetDevice(0);
    return cudaMemcpy(d_result, host_result.data(), static_cast<Size>(total) * sizeof(double),
                      cudaMemcpyHostToDevice);
#else
    return cudaErrorNotSupported;
#endif
}
} // namespace nerve::autodiff::multi_gpu
namespace nerve::autodiff::mpi
{
void syncGradients(double *g, Size n)
{
    if (!g || n == 0)
        return;
#ifdef NERVE_HAS_MPI
    MPI_Allreduce(MPI_IN_PLACE, g, static_cast<int>(n), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#else
    (void)g;
    (void)n;
#endif
}
} // namespace nerve::autodiff::mpi

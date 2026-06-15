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

namespace nerve::algebra::multi_gpu
{
cudaError_t distributeGramMatrix(Size n, int num_gpus)
{
    if (n == 0 || num_gpus <= 0)
        return cudaErrorInvalidValue;
#ifdef NERVE_HAS_CUDA
    Size rows_per = n / static_cast<Size>(num_gpus);
    for (int g = 0; g < num_gpus; ++g)
    {
        cudaError_t err = cudaSetDevice(g);
        if (err != cudaSuccess)
            return err;
    }
    return cudaSuccess;
#else
    (void)n;
    (void)num_gpus;
    return cudaErrorNotSupported;
#endif
}
} // namespace nerve::algebra::multi_gpu
namespace nerve::algebra::mpi_cuda
{
cudaError_t allGatherGramGPU(const double *d_local, Size nlocal, double *d_all, cudaStream_t s)
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
    std::vector<double> host_all(nlocal * static_cast<Size>(sz));
    MPI_Allgather(host.data(), static_cast<int>(nlocal), MPI_DOUBLE, host_all.data(),
                  static_cast<int>(nlocal), MPI_DOUBLE, MPI_COMM_WORLD);
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
} // namespace nerve::algebra::mpi_cuda

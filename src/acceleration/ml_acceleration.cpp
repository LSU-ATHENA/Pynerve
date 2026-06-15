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

namespace nerve::ml::multi_gpu
{
cudaError_t distributeTrainingData(const double *d_data, Size n, int ngpus, double **d_partitions)
{
    if (!d_data || !d_partitions || n == 0 || ngpus <= 0)
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
        err = cudaMemcpy(d_partitions[g], d_data + offset, chunk * sizeof(double),
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
cudaError_t syncModelParams(double *d_params, Size nparams, int ngpus)
{
    if (!d_params || nparams == 0 || ngpus <= 1)
        return cudaErrorInvalidValue;
#ifdef NERVE_HAS_CUDA
    std::vector<double> host(nparams);
    cudaError_t err =
        cudaMemcpy(host.data(), d_params, nparams * sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
        return err;
    for (int g = 0; g < ngpus; ++g)
    {
        err = cudaSetDevice(g);
        if (err != cudaSuccess)
            return err;
        err = cudaMemcpy(d_params, host.data(), nparams * sizeof(double), cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
            return err;
    }
    return cudaSuccess;
#else
    return cudaErrorNotSupported;
#endif
}
} // namespace nerve::ml::multi_gpu
namespace nerve::ml::mpi
{
void allReduceGradients(double *g, Size n)
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
} // namespace nerve::ml::mpi

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

namespace nerve::metrics::multi_gpu
{
cudaError_t distributeDiagramPairs(const double *d_dgm, Size npairs, int ngpus,
                                   double **d_partitions)
{
    if (!d_dgm || !d_partitions || npairs == 0 || ngpus <= 0)
        return cudaErrorInvalidValue;
#ifdef NERVE_HAS_CUDA
    const Size stride = 2;
    Size per = npairs / static_cast<Size>(ngpus);
    Size rem = npairs % static_cast<Size>(ngpus);
    Size offset = 0;
    for (int g = 0; g < ngpus; ++g)
    {
        cudaError_t err = cudaSetDevice(g);
        if (err != cudaSuccess)
            return err;
        Size chunk = per + (static_cast<Size>(g) < rem ? 1 : 0);
        Size bytes = chunk * stride * sizeof(double);
        err = cudaMalloc(&d_partitions[g], bytes);
        if (err != cudaSuccess)
            return err;
        err = cudaMemcpy(d_partitions[g], d_dgm + offset * stride, bytes, cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess)
            return err;
        offset += chunk;
    }
    return cudaSuccess;
#else
    return cudaErrorNotSupported;
#endif
}
cudaError_t gatherDistances(double **d_partials, const int *sizes, int ngpus, double *d_out)
{
    if (!d_partials || !sizes || !d_out || ngpus <= 0)
        return cudaErrorInvalidValue;
#ifdef NERVE_HAS_CUDA
    for (int g = 0; g < ngpus; ++g)
    {
        if (sizes[g] <= 0 || !d_partials[g])
            continue;
        cudaError_t err = cudaSetDevice(g);
        if (err != cudaSuccess)
            return err;
        err = cudaMemcpy(d_out + g, d_partials[g], static_cast<Size>(sizes[g]) * sizeof(double),
                         cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess)
            return err;
    }
    return cudaSuccess;
#else
    return cudaErrorNotSupported;
#endif
}
} // namespace nerve::metrics::multi_gpu
namespace nerve::metrics::mpi_cuda
{
cudaError_t bottleneckDistMPI(const double *d_dgm, Size nlocal, int root, double *d_result,
                              cudaStream_t s)
{
    if (!d_dgm || !d_result || nlocal == 0 || root < 0)
        return cudaErrorInvalidValue;
#ifdef NERVE_HAS_CUDA
    const Size stride = 2;
    Size bytes = nlocal * stride * sizeof(double);
    std::vector<double> host(bytes / sizeof(double));
    cudaError_t err = cudaMemcpyAsync(host.data(), d_dgm, bytes, cudaMemcpyDeviceToHost, s);
    if (err != cudaSuccess)
        return err;
    err = cudaStreamSynchronize(s);
    if (err != cudaSuccess)
        return err;
#ifdef NERVE_HAS_MPI
    int cnt = static_cast<int>(nlocal * stride);
    if (algoRank() == root)
    {
        int sz = algoSize();
        std::vector<double> host_all(static_cast<Size>(cnt * sz));
        MPI_Gather(host.data(), cnt, MPI_DOUBLE, host_all.data(), cnt, MPI_DOUBLE, root,
                   MPI_COMM_WORLD);
        err =
            cudaMemcpyAsync(d_result, host_all.data(), static_cast<Size>(cnt * sz) * sizeof(double),
                            cudaMemcpyHostToDevice, s);
    }
    else
    {
        MPI_Gather(host.data(), cnt, MPI_DOUBLE, nullptr, cnt, MPI_DOUBLE, root, MPI_COMM_WORLD);
        err = cudaSuccess;
    }
    return (err != cudaSuccess) ? err : cudaStreamSynchronize(s);
#else
    (void)d_result;
    (void)root;
    return cudaErrorNotSupported;
#endif
#else
    (void)s;
    return cudaErrorNotSupported;
#endif
}
} // namespace nerve::metrics::mpi_cuda

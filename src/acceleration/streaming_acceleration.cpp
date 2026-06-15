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

namespace nerve::streaming::mpi_cuda
{
cudaError_t allGatherPointsGPU(const double *d_local, Size nlocal, Size dim, double *d_all,
                               MPI_Comm comm, cudaStream_t s)
{
    if (!d_local || !d_all || nlocal == 0 || dim == 0)
        return cudaErrorInvalidValue;
#ifdef NERVE_HAS_CUDA
    Size cnt = nlocal * dim;
    std::vector<double> host(cnt);
    cudaError_t err =
        cudaMemcpyAsync(host.data(), d_local, cnt * sizeof(double), cudaMemcpyDeviceToHost, s);
    if (err != cudaSuccess)
        return err;
    err = cudaStreamSynchronize(s);
    if (err != cudaSuccess)
        return err;
#ifdef NERVE_HAS_MPI
    int sz = 0;
    MPI_Comm_size(comm, &sz);
    int cnt_int = static_cast<int>(cnt);
    std::vector<double> host_all(cnt * static_cast<Size>(sz));
    MPI_Allgather(host.data(), cnt_int, MPI_DOUBLE, host_all.data(), cnt_int, MPI_DOUBLE, comm);
    err = cudaMemcpyAsync(d_all, host_all.data(), cnt * static_cast<Size>(sz) * sizeof(double),
                          cudaMemcpyHostToDevice, s);
    return (err != cudaSuccess) ? err : cudaStreamSynchronize(s);
#else
    (void)comm;
    (void)d_all;
    return cudaErrorNotSupported;
#endif
#else
    (void)comm;
    (void)s;
    return cudaErrorNotSupported;
#endif
}
cudaError_t reduceStabilityGPU(const double *d_local, Size nlocal, double *d_global, MPI_Comm comm)
{
    if (!d_local || !d_global || nlocal == 0)
        return cudaErrorInvalidValue;
#ifdef NERVE_HAS_CUDA
    std::vector<double> host(nlocal);
    cudaError_t err =
        cudaMemcpy(host.data(), d_local, nlocal * sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
        return err;
#ifdef NERVE_HAS_MPI
    std::vector<double> host_global(nlocal);
    MPI_Reduce(host.data(), host_global.data(), static_cast<int>(nlocal), MPI_DOUBLE, MPI_SUM, 0,
               comm);
    err = cudaMemcpy(d_global, host_global.data(), nlocal * sizeof(double), cudaMemcpyHostToDevice);
#else
    (void)comm;
    err = cudaMemcpy(d_global, host.data(), nlocal * sizeof(double), cudaMemcpyHostToDevice);
#endif
    return err;
#else
    (void)comm;
    return cudaErrorNotSupported;
#endif
}
} // namespace nerve::streaming::mpi_cuda

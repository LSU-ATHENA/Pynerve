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

namespace nerve::graphs::mpi_cuda
{
cudaError_t allGatherGraphGPU(const int *d_local, Size nlocal, int *d_counts, int *d_all,
                              cudaStream_t s)
{
    if (!d_local || !d_counts || !d_all || nlocal == 0)
        return cudaErrorInvalidValue;
#ifdef NERVE_HAS_CUDA
    std::vector<int> host(nlocal);
    cudaError_t err =
        cudaMemcpyAsync(host.data(), d_local, nlocal * sizeof(int), cudaMemcpyDeviceToHost, s);
    if (err != cudaSuccess)
        return err;
    err = cudaStreamSynchronize(s);
    if (err != cudaSuccess)
        return err;
#ifdef NERVE_HAS_MPI
    int sz = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &sz);
    int cnt = static_cast<int>(nlocal);
    std::vector<int> host_counts(static_cast<Size>(sz));
    MPI_Allgather(&cnt, 1, MPI_INT, host_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    int total = 0;
    std::vector<int> displs(static_cast<Size>(sz));
    for (int i = 0; i < sz; ++i)
    {
        displs[static_cast<Size>(i)] = total;
        total += host_counts[static_cast<Size>(i)];
    }
    std::vector<int> host_all(static_cast<Size>(total));
    MPI_Allgatherv(host.data(), cnt, MPI_INT, host_all.data(), host_counts.data(), displs.data(),
                   MPI_INT, MPI_COMM_WORLD);
    err = cudaMemcpyAsync(d_counts, host_counts.data(), static_cast<Size>(sz) * sizeof(int),
                          cudaMemcpyHostToDevice, s);
    if (err != cudaSuccess)
        return err;
    err = cudaMemcpyAsync(d_all, host_all.data(), static_cast<Size>(total) * sizeof(int),
                          cudaMemcpyHostToDevice, s);
    return (err != cudaSuccess) ? err : cudaStreamSynchronize(s);
#else
    (void)d_counts;
    (void)d_all;
    return cudaErrorNotSupported;
#endif
#else
    (void)s;
    return cudaErrorNotSupported;
#endif
}
} // namespace nerve::graphs::mpi_cuda

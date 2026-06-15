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

namespace nerve::algorithms::multi_gpu
{
cudaError_t distributeDistanceWork(Size n, int *d_counts, int ngpus)
{
    if (n == 0 || ngpus <= 0)
        return cudaErrorInvalidValue;
#ifdef NERVE_HAS_CUDA
    Size per = n / static_cast<Size>(ngpus);
    Size rem = n % static_cast<Size>(ngpus);
    for (int g = 0; g < ngpus; ++g)
    {
        cudaError_t err = cudaSetDevice(g);
        if (err != cudaSuccess)
            return err;
    }
    if (d_counts)
        for (int g = 0; g < ngpus; ++g)
            d_counts[g] = static_cast<int>(per + (static_cast<Size>(g) < rem ? 1 : 0));
    return cudaSuccess;
#else
    (void)d_counts;
    return cudaErrorNotSupported;
#endif
}
cudaError_t gatherDistanceResults(const double *const *d_partials, const int *counts, int num_gpus,
                                  double *d_result, cudaStream_t s)
{
    if (!d_partials || !counts || !d_result || num_gpus <= 0)
        return cudaErrorInvalidValue;
#ifdef NERVE_HAS_CUDA
    Size offset = 0;
    for (int g = 0; g < num_gpus; ++g)
    {
        if (counts[g] <= 0 || !d_partials[g])
            continue;
        cudaError_t err = cudaSetDevice(g);
        if (err != cudaSuccess)
            return err;
        int cnt = counts[g];
        std::vector<double> host(static_cast<Size>(cnt));
        err = cudaMemcpyAsync(host.data(), d_partials[g], static_cast<Size>(cnt) * sizeof(double),
                              cudaMemcpyDeviceToHost, s);
        if (err != cudaSuccess)
            return err;
        err = cudaStreamSynchronize(s);
        if (err != cudaSuccess)
            return err;
        cudaSetDevice(0);
        err = cudaMemcpyAsync(d_result + offset, host.data(),
                              static_cast<Size>(cnt) * sizeof(double), cudaMemcpyHostToDevice, s);
        if (err != cudaSuccess)
            return err;
        offset += static_cast<Size>(cnt);
    }
    return cudaStreamSynchronize(s);
#else
    (void)s;
    return cudaErrorNotSupported;
#endif
}
} // namespace nerve::algorithms::multi_gpu
namespace nerve::algorithms::mpi
{
void initAlgorithmsMPI(int *argc, char ***argv)
{
#ifdef NERVE_HAS_MPI
    int provided = 0;
    MPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, &provided);
    (void)provided;
#else
    (void)argc;
    (void)argv;
#endif
}
int algoRank()
{
#ifdef NERVE_HAS_MPI
    int r = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    return r;
#else
    return 0;
#endif
}
int algoSize()
{
#ifdef NERVE_HAS_MPI
    int s = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &s);
    return s;
#else
    return 1;
#endif
}
void allGatherDistances(const double *l, Size nl, Size d, double *a)
{
    if (!l || !a || nl == 0 || d == 0)
        return;
#ifdef NERVE_HAS_MPI
    int cnt = static_cast<int>(nl * d);
    MPI_Allgather(l, cnt, MPI_DOUBLE, a, cnt, MPI_DOUBLE, MPI_COMM_WORLD);
#else
    std::memcpy(a, l, nl * d * sizeof(double));
#endif
}
} // namespace nerve::algorithms::mpi
namespace nerve::algorithms::mpi_cuda
{
cudaError_t allGatherDeviceVectors(const double *d_local, Size nlocal, Size stride, double *d_all,
                                   cudaStream_t s)
{
    if (!d_local || !d_all || nlocal == 0 || stride == 0)
        return cudaErrorInvalidValue;
#ifdef NERVE_HAS_CUDA
    Size bytes = nlocal * stride * sizeof(double);
    std::vector<double> host(nlocal * stride);
    cudaError_t err = cudaMemcpyAsync(host.data(), d_local, bytes, cudaMemcpyDeviceToHost, s);
    if (err != cudaSuccess)
        return err;
    err = cudaStreamSynchronize(s);
    if (err != cudaSuccess)
        return err;
#ifdef NERVE_HAS_MPI
    int sz = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &sz);
    int cnt = static_cast<int>(nlocal * stride);
    std::vector<double> host_all(static_cast<Size>(sz) * nlocal * stride);
    MPI_Allgather(host.data(), cnt, MPI_DOUBLE, host_all.data(), cnt, MPI_DOUBLE, MPI_COMM_WORLD);
    err = cudaMemcpyAsync(d_all, host_all.data(), static_cast<Size>(sz) * bytes,
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
} // namespace nerve::algorithms::mpi_cuda

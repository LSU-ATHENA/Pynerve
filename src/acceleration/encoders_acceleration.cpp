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

namespace nerve::encoders::multi_gpu
{
cudaError_t scatterBatches(const double *d_data, Size total, int ngpus, double **d_batches,
                           Size *batch_sizes)
{
    if (!d_data || !d_batches || !batch_sizes || total == 0 || ngpus <= 0)
        return cudaErrorInvalidValue;
#ifdef NERVE_HAS_CUDA
    Size per = total / static_cast<Size>(ngpus);
    Size rem = total % static_cast<Size>(ngpus);
    Size offset = 0;
    for (int g = 0; g < ngpus; ++g)
    {
        cudaError_t err = cudaSetDevice(g);
        if (err != cudaSuccess)
            return err;
        Size chunk = per + (static_cast<Size>(g) < rem ? 1 : 0);
        batch_sizes[g] = chunk;
        err = cudaMalloc(&d_batches[g], chunk * sizeof(double));
        if (err != cudaSuccess)
            return err;
        err = cudaMemcpy(d_batches[g], d_data + offset, chunk * sizeof(double),
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
cudaError_t gatherEncoded(double **d_encoded, const Size *sizes, int ngpus, double *d_out)
{
    if (!d_encoded || !sizes || !d_out || ngpus <= 0)
        return cudaErrorInvalidValue;
#ifdef NERVE_HAS_CUDA
    Size offset = 0;
    for (int g = 0; g < ngpus; ++g)
    {
        if (sizes[g] == 0 || !d_encoded[g])
            continue;
        cudaError_t err = cudaSetDevice(g);
        if (err != cudaSuccess)
            return err;
        err = cudaMemcpy(d_out + offset, d_encoded[g], sizes[g] * sizeof(double),
                         cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess)
            return err;
        offset += sizes[g];
    }
    return cudaSuccess;
#else
    return cudaErrorNotSupported;
#endif
}
} // namespace nerve::encoders::multi_gpu
namespace nerve::encoders::mpi
{
void gatherEncodedVectors(const double *l, Size nl, Size cd, double *a)
{
    if (!l || !a || nl == 0 || cd == 0)
        return;
#ifdef NERVE_HAS_MPI
    int cnt = static_cast<int>(nl * cd);
    MPI_Gather(l, cnt, MPI_DOUBLE, a, cnt, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#else
    std::memcpy(a, l, nl * cd * sizeof(double));
#endif
}
} // namespace nerve::encoders::mpi

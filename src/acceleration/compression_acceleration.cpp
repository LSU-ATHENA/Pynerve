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

namespace nerve::compression::multi_gpu
{
cudaError_t distributeCompression(Size data_size, int num_gpus)
{
    if (data_size == 0 || num_gpus <= 0)
        return cudaErrorInvalidValue;
#ifdef NERVE_HAS_CUDA
    for (int g = 0; g < num_gpus; ++g)
    {
        cudaError_t err = cudaSetDevice(g);
        if (err != cudaSuccess)
            return err;
    }
    return cudaSuccess;
#else
    return cudaErrorNotSupported;
#endif
}
} // namespace nerve::compression::multi_gpu
namespace nerve::compression::mpi
{
void allGatherCompressed(const uint8_t *l, Size nl, uint8_t *a)
{
    if (!l || !a || nl == 0)
        return;
#ifdef NERVE_HAS_MPI
    MPI_Allgather(l, static_cast<int>(nl), MPI_BYTE, a, static_cast<int>(nl), MPI_BYTE,
                  MPI_COMM_WORLD);
#else
    std::memcpy(a, l, nl);
#endif
}
} // namespace nerve::compression::mpi

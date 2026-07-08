#include "nerve/core_types.hpp"
#include "nerve/gpu/gpu_ptx_ops.cuh"

#include <cuda_runtime.h>

#include <cstdint>

namespace nerve::algorithms::gpu
{
using namespace nerve::gpu::ptx;

__global__ void batchDistanceKernel(const double *__restrict__ a, Size na,
                                    const double *__restrict__ b, Size nb, Size dim,
                                    double *__restrict__ distances);

template <int kBlockSize>
__global__ void pairwiseDistanceKernel(const double *__restrict__ points, Size n, Size dim,
                                       double *__restrict__ matrix)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= static_cast<int>(n))
        return;

    for (int j = 0; j < static_cast<int>(n); ++j)
    {
        double sum = 0.0;
        for (Size d = 0; d < dim; ++d)
        {
            double diff =
                points[static_cast<size_t>(i) * dim + d] - points[static_cast<size_t>(j) * dim + d];
            sum = fma_f64(diff, diff, sum);
        }
        matrix[static_cast<size_t>(i) * n + j] = sqrt(sum);
    }
}

__global__ void batchDistanceKernel(const double *__restrict__ a, Size na,
                                    const double *__restrict__ b, Size nb, Size dim,
                                    double *__restrict__ distances)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= static_cast<int>(na) || j >= static_cast<int>(nb))
        return;
    double sum = 0.0;
    for (Size d = 0; d < dim; ++d)
    {
        double diff = a[static_cast<size_t>(i) * dim + d] - b[static_cast<size_t>(j) * dim + d];
        sum = fma_f64(diff, diff, sum);
    }
    distances[static_cast<size_t>(j) * na + i] = sqrt(sum);
}

} // namespace nerve::algorithms::gpu

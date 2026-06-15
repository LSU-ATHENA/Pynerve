#pragma once
#include <cuda_runtime.h>

namespace nerve::gpu::tedjoin
{

cudaError_t launchFp64TensorDistance(const double *points, int n_points, int dim, double *distances,
                                     int dist_stride, cudaStream_t stream = nullptr);

} // namespace nerve::gpu::tedjoin

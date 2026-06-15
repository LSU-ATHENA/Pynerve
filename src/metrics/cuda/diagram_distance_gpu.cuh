#pragma once
#include "nerve/core_types.hpp"

#include <cuda_runtime.h>

namespace nerve::metrics::cuda
{

cudaError_t launchBottleneckDistance(const double *d_dgm1, Size n1, const double *d_dgm2, Size n2,
                                     double *d_result, cudaStream_t stream = 0);

cudaError_t launchWassersteinDistance(const double *d_dgm1, Size n1, const double *d_dgm2, Size n2,
                                      double p, double *d_result, cudaStream_t stream = 0);

} // namespace nerve::metrics::cuda

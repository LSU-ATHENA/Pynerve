#pragma once
#include "nerve/core_types.hpp"

#include <cuda_runtime.h>

namespace nerve::optimization::gpu
{

cudaError_t launchSgdStep(double *d_params, const double *d_grads, Size n, double lr,
                          double momentum, Size step);

cudaError_t launchAdamStep(double *d_params, double *d_m, double *d_v, const double *d_grads,
                           Size n, double lr, double beta1, double beta2, double eps, Size t);

cudaError_t launchClippedGradient(const double *d_grads, double *d_clipped, Size n,
                                  double max_norm);

} // namespace nerve::optimization::gpu

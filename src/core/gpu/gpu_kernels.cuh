#pragma once
#include "nerve/core_types.hpp"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand_uniform.h>

namespace nerve::core::gpu
{

__global__ void fillUniformKernel(float *buffer, Size n, unsigned long long seed);
__global__ void fillNormalKernel(double *buffer, Size n, double mean, double stddev,
                                 unsigned long long seed);
__global__ void fillPoissonKernel(int *buffer, Size n, double lambda, unsigned long long seed);

cudaError_t launchFillUniform(float *d_buffer, Size n, unsigned long long seed);
cudaError_t launchFillNormal(double *d_buffer, Size n, double mean, double stddev,
                             unsigned long long seed);
cudaError_t launchFillPoisson(int *d_buffer, Size n, double lambda, unsigned long long seed);

cudaError_t launchVectorAdd(const double *a, const double *b, double *c, Size n);
cudaError_t launchVectorScale(double *buffer, double alpha, Size n);
cudaError_t launchVectorDot(const double *a, const double *b, double *result, Size n);
cudaError_t launchMatrixTranspose(const double *input, double *output, Size rows, Size cols);

} // namespace nerve::core::gpu

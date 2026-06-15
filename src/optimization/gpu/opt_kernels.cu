#include "nerve/optimization/gpu/opt_kernels.cuh"

#include <cuda_runtime.h>

#include <cmath>

namespace nerve::optimization::gpu
{

__global__ void sgdStepKernel(double *params, const double *grads, Size n, double lr,
                              double momentum, Size step)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= static_cast<int>(n))
        return;
    params[idx] -= lr * (grads[idx] + momentum / (step + 1) * params[idx]);
}

__global__ void adamStepKernel(double *params, double *m, double *v, const double *grads, Size n,
                               double lr, double beta1, double beta2, double eps, Size t)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= static_cast<int>(n))
        return;
    double g = grads[idx];
    m[idx] = beta1 * m[idx] + (1.0 - beta1) * g;
    v[idx] = beta2 * v[idx] + (1.0 - beta2) * g * g;
    double m_hat = m[idx] / (1.0 - pow(beta1, t));
    double v_hat = v[idx] / (1.0 - pow(beta2, t));
    params[idx] -= lr * m_hat / (sqrt(v_hat) + eps);
}

__global__ void clipGradKernel(const double *grads, double *clipped, Size n, double max_norm)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= static_cast<int>(n))
        return;
    double g = grads[idx];
    double abs_g = fabs(g);
    if (abs_g > max_norm)
        g = copysign(max_norm, g);
    clipped[idx] = g;
}

cudaError_t launchSgdStep(double *d_params, const double *d_grads, Size n, double lr,
                          double momentum, Size step)
{
    dim3 block(256);
    dim3 grid((static_cast<unsigned int>(n) + 255) / 256);
    sgdStepKernel<<<grid, block>>>(d_params, d_grads, n, lr, momentum, step);
    return cudaGetLastError();
}

cudaError_t launchAdamStep(double *d_params, double *d_m, double *d_v, const double *d_grads,
                           Size n, double lr, double beta1, double beta2, double eps, Size t)
{
    dim3 block(256);
    dim3 grid((static_cast<unsigned int>(n) + 255) / 256);
    adamStepKernel<<<grid, block>>>(d_params, d_m, d_v, d_grads, n, lr, beta1, beta2, eps, t);
    return cudaGetLastError();
}

cudaError_t launchClippedGradient(const double *d_grads, double *d_clipped, Size n, double max_norm)
{
    dim3 block(256);
    dim3 grid((static_cast<unsigned int>(n) + 255) / 256);
    clipGradKernel<<<grid, block>>>(d_grads, d_clipped, n, max_norm);
    return cudaGetLastError();
}

} // namespace nerve::optimization::gpu

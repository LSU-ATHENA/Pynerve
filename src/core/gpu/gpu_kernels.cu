#include "gpu_kernels.cuh"

#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace nerve::core::gpu
{

__global__ void fillUniformKernel(float *buffer, Size n, unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= static_cast<int>(n))
        return;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, 0, &state);
    buffer[idx] = curand_uniform(&state);
}

__global__ void fillNormalKernel(double *buffer, Size n, double mean, double stddev,
                                 unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= static_cast<int>(n))
        return;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, 0, &state);
    buffer[idx] = mean + stddev * curand_normal_double(&state);
}

__global__ void fillPoissonKernel(int *buffer, Size n, double lambda, unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= static_cast<int>(n))
        return;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, 0, &state);
    buffer[idx] = curand_poisson(&state, lambda);
}

__global__ void vectorAddKernel(const double *a, const double *b, double *c, Size n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= static_cast<int>(n))
        return;
    c[idx] = a[idx] + b[idx];
}

__global__ void vectorScaleKernel(double *buffer, double alpha, Size n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= static_cast<int>(n))
        return;
    buffer[idx] *= alpha;
}

__global__ void vectorDotKernel(const double *a, const double *b, double *partial, Size n)
{
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    double sum = 0.0;
    while (i < static_cast<int>(n))
    {
        sum += a[i] * b[i];
        i += gridDim.x * blockDim.x;
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0)
        partial[blockIdx.x] = sdata[0];
}

__global__ void transposeKernel(const double *input, double *output, Size rows, Size cols)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < static_cast<int>(cols) && y < static_cast<int>(rows))
    {
        output[static_cast<size_t>(x) * rows + y] = input[static_cast<size_t>(y) * cols + x];
    }
}

cudaError_t launchFillUniform(float *d_buffer, Size n, unsigned long long seed)
{
    dim3 block(256);
    dim3 grid((static_cast<unsigned int>(n) + 255) / 256);
    fillUniformKernel<<<grid, block>>>(d_buffer, n, seed);
    return cudaGetLastError();
}

cudaError_t launchFillNormal(double *d_buffer, Size n, double mean, double stddev,
                             unsigned long long seed)
{
    dim3 block(256);
    dim3 grid((static_cast<unsigned int>(n) + 255) / 256);
    fillNormalKernel<<<grid, block>>>(d_buffer, n, mean, stddev, seed);
    return cudaGetLastError();
}

cudaError_t launchFillPoisson(int *d_buffer, Size n, double lambda, unsigned long long seed)
{
    dim3 block(256);
    dim3 grid((static_cast<unsigned int>(n) + 255) / 256);
    fillPoissonKernel<<<grid, block>>>(d_buffer, n, lambda, seed);
    return cudaGetLastError();
}

cudaError_t launchVectorAdd(const double *a, const double *b, double *c, Size n)
{
    dim3 block(256);
    dim3 grid((static_cast<unsigned int>(n) + 255) / 256);
    vectorAddKernel<<<grid, block>>>(a, b, c, n);
    return cudaGetLastError();
}

cudaError_t launchVectorScale(double *buffer, double alpha, Size n)
{
    dim3 block(256);
    dim3 grid((static_cast<unsigned int>(n) + 255) / 256);
    vectorScaleKernel<<<grid, block>>>(buffer, alpha, n);
    return cudaGetLastError();
}

cudaError_t launchVectorDot(const double *a, const double *b, double *result, Size n)
{
    dim3 block(256);
    dim3 grid(32);
    size_t shared_mem = 256 * sizeof(double);
    vectorDotKernel<<<grid, block, shared_mem>>>(a, b, result, n);
    double sum = 0.0;
    cudaMemcpy(&sum, &result[0], sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(result, &sum, sizeof(double), cudaMemcpyHostToDevice);
    return cudaGetLastError();
}

cudaError_t launchMatrixTranspose(const double *input, double *output, Size rows, Size cols)
{
    dim3 block(16, 16);
    dim3 grid((static_cast<unsigned int>(cols) + 15) / 16,
              (static_cast<unsigned int>(rows) + 15) / 16);
    transposeKernel<<<grid, block>>>(input, output, rows, cols);
    return cudaGetLastError();
}

} // namespace nerve::core::gpu

#include <cuda_runtime.h>

#include <cmath>

namespace nerve::nn::cuda
{

__global__ void reluKernel(double *data, Size n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= static_cast<int>(n))
        return;
    if (data[idx] < 0.0)
        data[idx] = 0.0;
}

__global__ void sigmoidKernel(const double *input, double *output, Size n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= static_cast<int>(n))
        return;
    output[idx] = 1.0 / (1.0 + exp(-input[idx]));
}

__global__ void tanhKernel(const double *input, double *output, Size n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= static_cast<int>(n))
        return;
    output[idx] = tanh(input[idx]);
}

__global__ void batchNormKernel(double *data, Size n, double mean, double std_inv)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= static_cast<int>(n))
        return;
    data[idx] = (data[idx] - mean) * std_inv;
}

__global__ void softmaxKernel(const double *input, double *output, Size n)
{
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    if (tid == 0)
    {
        double max_val = input[0];
        for (int i = 1; i < static_cast<int>(n); ++i)
            if (input[i] > max_val)
                max_val = input[i];
        sdata[0] = max_val;
    }
    __syncthreads();
    double max_val = sdata[0];
    if (tid < static_cast<int>(n))
    {
        output[tid] = exp(input[tid] - max_val);
    }
    __syncthreads();
    if (tid == 0)
    {
        double sum = 0.0;
        for (int i = 0; i < static_cast<int>(n); ++i)
            sum += output[i];
        sdata[0] = sum;
    }
    __syncthreads();
    double inv_sum = 1.0 / sdata[0];
    if (tid < static_cast<int>(n))
        output[tid] *= inv_sum;
}

cudaError_t launchReLU(double *d_data, Size n)
{
    if (n == 0)
        return cudaErrorInvalidValue;
    dim3 block(256);
    dim3 grid((static_cast<unsigned int>(n) + 255) / 256);
    reluKernel<<<grid, block>>>(d_data, n);
    return cudaGetLastError();
}

cudaError_t launchSigmoid(const double *d_input, double *d_output, Size n)
{
    if (n == 0)
        return cudaErrorInvalidValue;
    dim3 block(256);
    dim3 grid((static_cast<unsigned int>(n) + 255) / 256);
    sigmoidKernel<<<grid, block>>>(d_input, d_output, n);
    return cudaGetLastError();
}

cudaError_t launchTanh(const double *d_input, double *d_output, Size n)
{
    if (n == 0)
        return cudaErrorInvalidValue;
    dim3 block(256);
    dim3 grid((static_cast<unsigned int>(n) + 255) / 256);
    tanhKernel<<<grid, block>>>(d_input, d_output, n);
    return cudaGetLastError();
}

cudaError_t launchBatchNorm(double *d_data, Size n, double mean, double std_inv)
{
    if (n == 0)
        return cudaErrorInvalidValue;
    dim3 block(256);
    dim3 grid((static_cast<unsigned int>(n) + 255) / 256);
    batchNormKernel<<<grid, block>>>(d_data, n, mean, std_inv);
    return cudaGetLastError();
}

cudaError_t launchSoftmax(const double *d_input, double *d_output, Size n)
{
    if (n == 0)
        return cudaErrorInvalidValue;
    dim3 block(256);
    size_t shared = sizeof(double);
    softmaxKernel<<<1, block, shared>>>(d_input, d_output, n);
    return cudaGetLastError();
}

} // namespace nerve::nn::cuda

#include <cuda_runtime.h>

#include <cmath>

#include "nerve/gpu/gpu_ptx_ops.cuh"

namespace nerve::nn::cuda
{
using namespace nerve::gpu::ptx;

// Internal helpers

__device__ __forceinline__ double warp_reduce_sum_f64(double val)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
#else
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor(val, offset);
#endif
    return val;
}

// F64 Kernels (PTX-optimised)

__global__ void reluKernel(double *data, Size n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= static_cast<int>(n))
        return;
    data[idx] = hwmax_f64(data[idx], 0.0);
}

__global__ void sigmoidKernel(const double *input, double *output, Size n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= static_cast<int>(n))
        return;
    double neg_x = -input[idx];
    output[idx] = rcp_approx_f64(1.0 + exp(neg_x));
}

__global__ void tanhKernel(const double *input, double *output, Size n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= static_cast<int>(n))
        return;
    double neg_two_x = -2.0 * input[idx];
    double inv_denom = rcp_approx_f64(1.0 + exp(neg_two_x));
    output[idx] = fma_f64(2.0, inv_denom, -1.0);
}

__global__ void batchNormKernel(double *data, Size n, double mean, double std_inv)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= static_cast<int>(n))
        return;
    data[idx] = fma_f64(-mean, std_inv, fma_f64(data[idx], std_inv, 0.0));
}

__global__ void softmaxKernel(const double *input, double *output, Size n)
{
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int lane_id = tid & 31;
    int warp_id = tid >> 5;
    int num_warps = blockDim.x >> 5;

    double local_val = (tid < static_cast<int>(n)) ? input[tid] : -1.0e300;
    double warp_max = warp_reduce_max_f64(local_val);
    if (lane_id == 0)
        sdata[warp_id] = warp_max;
    __syncthreads();

    double cross_val = (tid < num_warps) ? sdata[tid] : -1.0e300;
    double global_max = warp_reduce_max_f64(cross_val);
    if (lane_id == 0 && tid == 0)
        sdata[0] = global_max;
    __syncthreads();
    global_max = sdata[0];

    double exp_val = 0.0;
    if (tid < static_cast<int>(n))
    {
        exp_val = exp(input[tid] - global_max);
        output[tid] = exp_val;
    }
    double warp_sum = warp_reduce_sum_f64(exp_val);
    if (lane_id == 0)
        sdata[warp_id] = warp_sum;
    __syncthreads();

    double cross_sum = (tid < num_warps) ? sdata[tid] : 0.0;
    double global_sum = warp_reduce_sum_f64(cross_sum);
    if (lane_id == 0 && tid == 0)
        sdata[0] = global_sum;
    __syncthreads();
    global_sum = sdata[0];

    double inv_sum = rcp_approx_f64(global_sum);
    if (tid < static_cast<int>(n))
        output[tid] *= inv_sum;
}

// F32 Kernels

__global__ void reluKernel_f32(float *data, Size n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= static_cast<int>(n))
        return;
    data[idx] = hwmax_f32(data[idx], 0.0f);
}

__global__ void sigmoidKernel_f32(const float *input, float *output, Size n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= static_cast<int>(n))
        return;
    output[idx] = fast_sigmoid_f32(input[idx]);
}

__global__ void tanhKernel_f32(const float *input, float *output, Size n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= static_cast<int>(n))
        return;
    constexpr float NEG_TWO_OVER_LN2 = -2.88539008f;
    float exp_neg = ex2_approx_f32(input[idx] * NEG_TWO_OVER_LN2);
    float inv_denom = rcp_approx_f32(1.0f + exp_neg);
    output[idx] = fma_f32(2.0f, inv_denom, -1.0f);
}

__global__ void batchNormKernel_f32(float *data, Size n, float mean, float std_inv)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= static_cast<int>(n))
        return;
    data[idx] = fma_f32(-mean, std_inv, fma_f32(data[idx], std_inv, 0.0f));
}

__global__ void softmaxKernel_f32(const float *input, float *output, Size n)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int lane_id = tid & 31;
    int warp_id = tid >> 5;
    int num_warps = blockDim.x >> 5;

    float local_val = (tid < static_cast<int>(n)) ? input[tid] : -1.0e38f;
    float warp_max = warp_reduce_max_f32(local_val);
    if (lane_id == 0)
        sdata[warp_id] = warp_max;
    __syncthreads();

    float cross_val = (tid < num_warps) ? sdata[tid] : -1.0e38f;
    float global_max = warp_reduce_max_f32(cross_val);
    if (lane_id == 0 && tid == 0)
        sdata[0] = global_max;
    __syncthreads();
    global_max = sdata[0];

    float exp_val = 0.0f;
    if (tid < static_cast<int>(n))
    {
        exp_val = fast_exp_f32(input[tid] - global_max);
        output[tid] = exp_val;
    }
    float warp_sum = warp_reduce_sum_f32(exp_val);
    if (lane_id == 0)
        sdata[warp_id] = warp_sum;
    __syncthreads();

    float cross_sum = (tid < num_warps) ? sdata[tid] : 0.0f;
    float global_sum = warp_reduce_sum_f32(cross_sum);
    if (lane_id == 0 && tid == 0)
        sdata[0] = global_sum;
    __syncthreads();
    global_sum = sdata[0];

    float inv_sum = rcp_approx_f32(global_sum);
    if (tid < static_cast<int>(n))
        output[tid] *= inv_sum;
}

// PTX-Optimised F64 Variants

__global__ void reluPtxKernel(double *data, Size n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= static_cast<int>(n))
        return;
    data[idx] = hwmax_f64(data[idx], 0.0);
}

__global__ void sigmoidPtxKernel(const double *input, double *output, Size n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= static_cast<int>(n))
        return;
    double neg_x = -input[idx];
    output[idx] = rcp_approx_f64(1.0 + exp(neg_x));
}

__global__ void tanhPtxKernel(const double *input, double *output, Size n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= static_cast<int>(n))
        return;
    double neg_two_x = -2.0 * input[idx];
    double inv_denom = rcp_approx_f64(1.0 + exp(neg_two_x));
    output[idx] = fma_f64(2.0, inv_denom, -1.0);
}

__global__ void batchNormPtxKernel(double *data, Size n, double mean, double std_inv)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= static_cast<int>(n))
        return;
    data[idx] = fma_f64(-mean, std_inv, fma_f64(data[idx], std_inv, 0.0));
}

__global__ void softmaxPtxKernel(const double *input, double *output, Size n)
{
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int lane_id = tid & 31;
    int warp_id = tid >> 5;
    int num_warps = blockDim.x >> 5;

    double local_val = (tid < static_cast<int>(n)) ? input[tid] : -1.0e300;
    double warp_max = warp_reduce_max_f64(local_val);
    if (lane_id == 0)
        sdata[warp_id] = warp_max;
    __syncthreads();

    double cross_val = (tid < num_warps) ? sdata[tid] : -1.0e300;
    double global_max = warp_reduce_max_f64(cross_val);
    if (lane_id == 0 && tid == 0)
        sdata[0] = global_max;
    __syncthreads();
    global_max = sdata[0];

    double exp_val = 0.0;
    if (tid < static_cast<int>(n))
    {
        exp_val = exp(input[tid] - global_max);
        output[tid] = exp_val;
    }
    double warp_sum = warp_reduce_sum_f64(exp_val);
    if (lane_id == 0)
        sdata[warp_id] = warp_sum;
    __syncthreads();

    double cross_sum = (tid < num_warps) ? sdata[tid] : 0.0;
    double global_sum = warp_reduce_sum_f64(cross_sum);
    if (lane_id == 0 && tid == 0)
        sdata[0] = global_sum;
    __syncthreads();
    global_sum = sdata[0];

    double inv_sum = rcp_approx_f64(global_sum);
    if (tid < static_cast<int>(n))
        output[tid] *= inv_sum;
}

// PTX-Optimised F32 Variants

__global__ void reluPtxKernel_f32(float *data, Size n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= static_cast<int>(n))
        return;
    data[idx] = hwmax_f32(data[idx], 0.0f);
}

__global__ void sigmoidPtxKernel_f32(const float *input, float *output, Size n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= static_cast<int>(n))
        return;
    output[idx] = fast_sigmoid_f32(input[idx]);
}

__global__ void tanhPtxKernel_f32(const float *input, float *output, Size n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= static_cast<int>(n))
        return;
    constexpr float NEG_TWO_OVER_LN2 = -2.88539008f;
    float exp_neg = ex2_approx_f32(input[idx] * NEG_TWO_OVER_LN2);
    float inv_denom = rcp_approx_f32(1.0f + exp_neg);
    output[idx] = fma_f32(2.0f, inv_denom, -1.0f);
}

__global__ void batchNormPtxKernel_f32(float *data, Size n, float mean, float std_inv)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= static_cast<int>(n))
        return;
    data[idx] = fma_f32(-mean, std_inv, fma_f32(data[idx], std_inv, 0.0f));
}

__global__ void softmaxPtxKernel_f32(const float *input, float *output, Size n)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int lane_id = tid & 31;
    int warp_id = tid >> 5;
    int num_warps = blockDim.x >> 5;

    float local_val = (tid < static_cast<int>(n)) ? input[tid] : -1.0e38f;
    float warp_max = warp_reduce_max_f32(local_val);
    if (lane_id == 0)
        sdata[warp_id] = warp_max;
    __syncthreads();

    float cross_val = (tid < num_warps) ? sdata[tid] : -1.0e38f;
    float global_max = warp_reduce_max_f32(cross_val);
    if (lane_id == 0 && tid == 0)
        sdata[0] = global_max;
    __syncthreads();
    global_max = sdata[0];

    float exp_val = 0.0f;
    if (tid < static_cast<int>(n))
    {
        exp_val = fast_exp_f32(input[tid] - global_max);
        output[tid] = exp_val;
    }
    float warp_sum = warp_reduce_sum_f32(exp_val);
    if (lane_id == 0)
        sdata[warp_id] = warp_sum;
    __syncthreads();

    float cross_sum = (tid < num_warps) ? sdata[tid] : 0.0f;
    float global_sum = warp_reduce_sum_f32(cross_sum);
    if (lane_id == 0 && tid == 0)
        sdata[0] = global_sum;
    __syncthreads();
    global_sum = sdata[0];

    float inv_sum = rcp_approx_f32(global_sum);
    if (tid < static_cast<int>(n))
        output[tid] *= inv_sum;
}

// Architecture detection

__host__ inline int getComputeCapability()
{
    int dev = 0;
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess)
        return 0;
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, dev);
    if (err != cudaSuccess)
        return 0;
    return prop.major * 10 + prop.minor;
}

// F64 Host Launchers (preserved signatures, architecture-aware dispatch)

cudaError_t launchReLU(double *d_data, Size n)
{
    if (n == 0)
        return cudaErrorInvalidValue;
    dim3 block(256);
    dim3 grid((static_cast<unsigned int>(n) + 255) / 256);
    if (getComputeCapability() >= 80)
        reluPtxKernel<<<grid, block>>>(d_data, n);
    else
        reluKernel<<<grid, block>>>(d_data, n);
    return cudaGetLastError();
}

cudaError_t launchSigmoid(const double *d_input, double *d_output, Size n)
{
    if (n == 0)
        return cudaErrorInvalidValue;
    dim3 block(256);
    dim3 grid((static_cast<unsigned int>(n) + 255) / 256);
    if (getComputeCapability() >= 80)
        sigmoidPtxKernel<<<grid, block>>>(d_input, d_output, n);
    else
        sigmoidKernel<<<grid, block>>>(d_input, d_output, n);
    return cudaGetLastError();
}

cudaError_t launchTanh(const double *d_input, double *d_output, Size n)
{
    if (n == 0)
        return cudaErrorInvalidValue;
    dim3 block(256);
    dim3 grid((static_cast<unsigned int>(n) + 255) / 256);
    if (getComputeCapability() >= 80)
        tanhPtxKernel<<<grid, block>>>(d_input, d_output, n);
    else
        tanhKernel<<<grid, block>>>(d_input, d_output, n);
    return cudaGetLastError();
}

cudaError_t launchBatchNorm(double *d_data, Size n, double mean, double std_inv)
{
    if (n == 0)
        return cudaErrorInvalidValue;
    dim3 block(256);
    dim3 grid((static_cast<unsigned int>(n) + 255) / 256);
    if (getComputeCapability() >= 80)
        batchNormPtxKernel<<<grid, block>>>(d_data, n, mean, std_inv);
    else
        batchNormKernel<<<grid, block>>>(d_data, n, mean, std_inv);
    return cudaGetLastError();
}

cudaError_t launchSoftmax(const double *d_input, double *d_output, Size n)
{
    if (n == 0)
        return cudaErrorInvalidValue;
    dim3 block(256);
    int num_warps = (block.x + 31) >> 5;
    size_t shared = static_cast<size_t>(num_warps) * sizeof(double);
    if (getComputeCapability() >= 80)
        softmaxPtxKernel<<<1, block, shared>>>(d_input, d_output, n);
    else
        softmaxKernel<<<1, block, shared>>>(d_input, d_output, n);
    return cudaGetLastError();
}

// F32 Host Launchers

cudaError_t launchReLU_f32(float *d_data, Size n)
{
    if (n == 0)
        return cudaErrorInvalidValue;
    dim3 block(256);
    dim3 grid((static_cast<unsigned int>(n) + 255) / 256);
    if (getComputeCapability() >= 80)
        reluPtxKernel_f32<<<grid, block>>>(d_data, n);
    else
        reluKernel_f32<<<grid, block>>>(d_data, n);
    return cudaGetLastError();
}

cudaError_t launchSigmoid_f32(const float *d_input, float *d_output, Size n)
{
    if (n == 0)
        return cudaErrorInvalidValue;
    dim3 block(256);
    dim3 grid((static_cast<unsigned int>(n) + 255) / 256);
    if (getComputeCapability() >= 80)
        sigmoidPtxKernel_f32<<<grid, block>>>(d_input, d_output, n);
    else
        sigmoidKernel_f32<<<grid, block>>>(d_input, d_output, n);
    return cudaGetLastError();
}

cudaError_t launchTanh_f32(const float *d_input, float *d_output, Size n)
{
    if (n == 0)
        return cudaErrorInvalidValue;
    dim3 block(256);
    dim3 grid((static_cast<unsigned int>(n) + 255) / 256);
    if (getComputeCapability() >= 80)
        tanhPtxKernel_f32<<<grid, block>>>(d_input, d_output, n);
    else
        tanhKernel_f32<<<grid, block>>>(d_input, d_output, n);
    return cudaGetLastError();
}

cudaError_t launchBatchNorm_f32(float *d_data, Size n, float mean, float std_inv)
{
    if (n == 0)
        return cudaErrorInvalidValue;
    dim3 block(256);
    dim3 grid((static_cast<unsigned int>(n) + 255) / 256);
    if (getComputeCapability() >= 80)
        batchNormPtxKernel_f32<<<grid, block>>>(d_data, n, mean, std_inv);
    else
        batchNormKernel_f32<<<grid, block>>>(d_data, n, mean, std_inv);
    return cudaGetLastError();
}

cudaError_t launchSoftmax_f32(const float *d_input, float *d_output, Size n)
{
    if (n == 0)
        return cudaErrorInvalidValue;
    dim3 block(256);
    int num_warps = (block.x + 31) >> 5;
    size_t shared = static_cast<size_t>(num_warps) * sizeof(float);
    if (getComputeCapability() >= 80)
        softmaxPtxKernel_f32<<<1, block, shared>>>(d_input, d_output, n);
    else
        softmaxKernel_f32<<<1, block, shared>>>(d_input, d_output, n);
    return cudaGetLastError();
}

} // namespace nerve::nn::cuda

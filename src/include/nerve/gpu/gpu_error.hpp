#pragma once

#include "cuda/error/cuda_error.hpp"

namespace nerve::gpu
{

template <typename T>
using CudaResult = nerve::error::Result<T>;

inline CudaResult<void> checkCudaCall(cudaError_t status, const char *expression)
{
    return cuda_check(status, expression);
}

inline CudaResult<void> checkCudaKernel(const char *kernel_name)
{
    return cuda_check_kernel(kernel_name);
}

} // namespace nerve::gpu

#ifndef CUDA_CALL
#define CUDA_CALL(expr)                                                                            \
    do                                                                                             \
    {                                                                                              \
        auto _cuda_result = ::nerve::gpu::checkCudaCall((expr), #expr);                            \
        if (!_cuda_result.isOk())                                                                  \
        {                                                                                          \
            return _cuda_result;                                                                   \
        }                                                                                          \
    } while (0)
#endif

#ifndef CUDA_KERNEL_CHECK
#define CUDA_KERNEL_CHECK(name)                                                                    \
    do                                                                                             \
    {                                                                                              \
        auto _cuda_result = ::nerve::gpu::checkCudaKernel(name);                                   \
        if (!_cuda_result.isOk())                                                                  \
        {                                                                                          \
            return _cuda_result;                                                                   \
        }                                                                                          \
    } while (0)
#endif

#ifndef GPU_CHECK
#define GPU_CHECK(expr)                                                                            \
    do                                                                                             \
    {                                                                                              \
        cudaError_t _gpu_status = (expr);                                                          \
        if (_gpu_status != cudaSuccess)                                                            \
        {                                                                                          \
            throw std::runtime_error(std::string(#expr) + ": " + cudaGetErrorString(_gpu_status)); \
        }                                                                                          \
    } while (0)
#endif

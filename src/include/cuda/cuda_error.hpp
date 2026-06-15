#pragma once

#include "cuda/error/cuda_error.hpp"

namespace nerve::cuda
{

using CudaErrorKind = nerve::gpu::CudaErrorKind;
using CudaError = nerve::gpu::CudaError;

template <typename T>
using CudaResult = nerve::gpu::CudaResult<T>;

using nerve::gpu::cuda_check;
using nerve::gpu::cuda_check_kernel;
using nerve::gpu::cuda_check_kernel_async;

template <typename T>
using RecoveryPolicy = nerve::gpu::RecoveryPolicy<T>;

using CudaErrorLogger = nerve::gpu::CudaErrorLogger;

} // namespace nerve::cuda

#define CUDA_CALL(expr)                                                                            \
    do                                                                                             \
    {                                                                                              \
        auto _r = ::nerve::gpu::cuda_check((expr), #expr);                                         \
        if (!_r.isOk())                                                                            \
            return _r;                                                                             \
    } while (0)

#define CUDA_KERNEL_CHECK(name)                                                                    \
    do                                                                                             \
    {                                                                                              \
        auto _r = ::nerve::gpu::cuda_check_kernel(name);                                           \
        if (!_r.isOk())                                                                            \
            return _r;                                                                             \
    } while (0)

#define CUDA_CALL_LOG(expr)                                                                        \
    do                                                                                             \
    {                                                                                              \
        cudaError_t _e = (expr);                                                                   \
        if (_e != cudaSuccess)                                                                     \
        {                                                                                          \
            ::nerve::gpu::CudaErrorLogger::log(::nerve::gpu::CudaError::from(_e, #expr));          \
        }                                                                                          \
    } while (0)

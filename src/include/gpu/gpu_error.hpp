
#pragma once
#include "nerve/error/error_registry.hpp"

#include <cuda_runtime.h>
#include <stdio.h>

#include <string>

namespace nerve::gpu
{

#define CUDA_CHECK(call)                                                                           \
    do                                                                                             \
    {                                                                                              \
        cudaError_t err = (call);                                                                  \
        if (err != cudaSuccess)                                                                    \
        {                                                                                          \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,                          \
                    cudaGetErrorString(err));                                                      \
            return nerve::error::Result<void>::err(nerve::error::TDAErrorCode::AllocationFailed,   \
                                                   std::string("CUDA error: ") +                   \
                                                       cudaGetErrorString(err));                   \
        }                                                                                          \
    } while (0)

#define CUDA_CHECK_RETURN(call, return_type)                                                       \
    do                                                                                             \
    {                                                                                              \
        cudaError_t err = (call);                                                                  \
        if (err != cudaSuccess)                                                                    \
        {                                                                                          \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,                          \
                    cudaGetErrorString(err));                                                      \
            return nerve::error::Result<return_type>::err(                                         \
                nerve::error::TDAErrorCode::AllocationFailed,                                      \
                std::string("CUDA error: ") + cudaGetErrorString(err));                            \
        }                                                                                          \
    } while (0)

#define CUDA_CHECK_LAST()                                                                          \
    do                                                                                             \
    {                                                                                              \
        cudaError_t err = cudaGetLastError();                                                      \
        if (err != cudaSuccess)                                                                    \
        {                                                                                          \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,                          \
                    cudaGetErrorString(err));                                                      \
            return nerve::error::Result<void>::err(nerve::error::TDAErrorCode::AllocationFailed,   \
                                                   std::string("CUDA kernel launch error: ") +     \
                                                       cudaGetErrorString(err));                   \
        }                                                                                          \
    } while (0)

} // namespace nerve::gpu

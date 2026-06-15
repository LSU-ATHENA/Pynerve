
#pragma once
#include "nerve/config.hpp"
#include "nerve/errors/errors.hpp"

#include <source_location>
#include <string>

#if __has_include(<cuda_runtime.h>)
#include <cuda_runtime.h>
#define NERVE_GPU_HAS_CUDA_HEADERS 1
#else
#define NERVE_GPU_HAS_CUDA_HEADERS 0
using cudaError_t = int;
constexpr int cudaSuccess = 0;
#endif

namespace nerve::gpu
{

// CUDA error handling integrated with Nerve error system.
class CudaErrorHandler
{
public:
    // Convert CUDA errors to Nerve ErrorCode
    static errors::ErrorCode classify_cuda_error(cudaError_t err);

    // Checking with context.
    static errors::ErrorResult<void>
    check_cuda_operation(cudaError_t result, const std::string &operation,
                         const std::source_location &loc = std::source_location::current());

    // Kernel launch validation
    static errors::ErrorResult<void>
    validateKernelLaunch(const std::string &kernel_name,
                         const std::source_location &loc = std::source_location::current());

    // Memory pressure detection
    static errors::ErrorResult<bool> checkMemoryPressure();

    // Recovery strategies
    static errors::ErrorResult<void> attemptRecovery(errors::ErrorCode error_code,
                                                     const std::string &operation);
};

// Macros for CUDA error handling.
#define TOPO_CUDA_CHECK(call)                                                                      \
    do                                                                                             \
    {                                                                                              \
        auto result = nerve::gpu::CudaErrorHandler::check_cuda_operation((call), #call);           \
        if (!result.isSuccess())                                                                   \
        {                                                                                          \
            return result;                                                                         \
        }                                                                                          \
    } while (0)

#define TOPO_CUDA_KERNEL_CHECK(name)                                                               \
    do                                                                                             \
    {                                                                                              \
        auto result = nerve::gpu::CudaErrorHandler::validateKernelLaunch(name);                    \
        if (!result.isSuccess())                                                                   \
        {                                                                                          \
            return result;                                                                         \
        }                                                                                          \
    } while (0)

} // namespace nerve::gpu

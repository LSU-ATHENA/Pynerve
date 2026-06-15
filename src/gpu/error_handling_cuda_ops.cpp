
#include "nerve/gpu/cuda_error_check.hpp"

#include <sstream>
#include <utility>

#if NERVE_GPU_HAS_CUDA_HEADERS
#include <cuda_runtime.h>
#endif

namespace nerve::gpu
{

errors::ErrorCode CudaErrorHandler::classify_cuda_error(cudaError_t err)
{
#if NERVE_GPU_HAS_CUDA_HEADERS
    switch (err)
    {
        case cudaSuccess:
            return errors::ErrorCode::SUCCESS;

        case cudaErrorMemoryAllocation:
        case cudaErrorInvalidValue:
            return errors::ErrorCode::E10_GPU_OOM;

        case cudaErrorLaunchFailure:
        case cudaErrorLaunchOutOfResources:
        case cudaErrorLaunchTimeout:
        case cudaErrorLaunchIncompatibleTexturing:
            return errors::ErrorCode::E11_GPU_LAUNCH_FAIL;

        case cudaErrorInvalidDevice:
        case cudaErrorInvalidDevicePointer:
        case cudaErrorInvalidMemcpyDirection:
            return errors::ErrorCode::E88_INVALID_SIMPLICES;

        case cudaErrorUnknown:
        default:
            return errors::ErrorCode::UNKNOWN;
    }
#else
    (void)err;
    return errors::ErrorCode::E11_GPU_LAUNCH_FAIL;
#endif
}

errors::ErrorResult<void> CudaErrorHandler::check_cuda_operation(cudaError_t result,
                                                                 const std::string &operation,
                                                                 const std::source_location &loc)
{
#if NERVE_GPU_HAS_CUDA_HEADERS
    if (result == cudaSuccess)
    {
        return errors::ErrorResult<void>::success();
    }

    const errors::ErrorCode error_code = classify_cuda_error(result);

    errors::ErrorContext context;
    context.operation_name = operation;
    context.component_name = "CUDA";
    context.addMetadata("cuda_error", cudaGetErrorString(result));
    context.addMetadata("cuda_error_code", std::to_string(static_cast<int>(result)));
    context.addMetadata("file", loc.file_name());
    context.addMetadata("line", std::to_string(loc.line()));
    errors::ErrorRegistry::instance().reportError(error_code, context);

    if (error_code == errors::ErrorCode::E10_GPU_OOM)
    {
        auto recovery_result = attemptRecovery(error_code, operation);
        if (recovery_result.isSuccess())
        {
            return errors::ErrorResult<void>::success();
        }
    }

    return errors::ErrorResult<void>::error(error_code);
#else
    (void)result;
    (void)operation;
    (void)loc;
    return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
#endif
}

errors::ErrorResult<void> CudaErrorHandler::validateKernelLaunch(const std::string &kernel_name,
                                                                 const std::source_location &loc)
{
#if NERVE_GPU_HAS_CUDA_HEADERS
    cudaError_t launch_error = cudaGetLastError();
    if (launch_error != cudaSuccess)
    {
        return check_cuda_operation(launch_error, "kernel launch: " + kernel_name, loc);
    }

    cudaError_t sync_error = cudaDeviceSynchronize();
    if (sync_error != cudaSuccess)
    {
        return check_cuda_operation(sync_error, "kernel execute: " + kernel_name, loc);
    }

    return errors::ErrorResult<void>::success();
#else
    (void)kernel_name;
    (void)loc;
    return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
#endif
}

errors::ErrorResult<bool> CudaErrorHandler::checkMemoryPressure()
{
#if NERVE_GPU_HAS_CUDA_HEADERS
    size_t free_mem = 0;
    size_t total_mem = 0;
    cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);

    if (err != cudaSuccess)
    {
        return errors::ErrorResult<bool>::error(classify_cuda_error(err));
    }

    bool pressure =
        total_mem > 0 && (static_cast<double>(free_mem) / static_cast<double>(total_mem)) < 0.2;
    if (pressure)
    {
        errors::ErrorContext context;
        context.operation_name = "memory_pressure_check";
        context.component_name = "CUDA";
        context.addMetadata("freeMemory", std::to_string(free_mem));
        context.addMetadata("total_memory", std::to_string(total_mem));
        context.addMetadata("free_ratio", std::to_string(static_cast<double>(free_mem) /
                                                         static_cast<double>(total_mem)));
        errors::ErrorRegistry::instance().reportError(errors::ErrorCode::E10_GPU_OOM, context);
    }

    return errors::ErrorResult<bool>::success(std::move(pressure));
#else
    return errors::ErrorResult<bool>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
#endif
}

errors::ErrorResult<void> CudaErrorHandler::attemptRecovery(errors::ErrorCode error_code,
                                                            const std::string &operation)
{
#if NERVE_GPU_HAS_CUDA_HEADERS
    errors::ErrorContext context;
    context.operation_name = "recovery_attempt";
    context.component_name = "CUDA";
    context.addMetadata("original_operation", operation);
    context.addMetadata("error_code", std::to_string(static_cast<int>(error_code)));

    switch (error_code)
    {
        case errors::ErrorCode::E10_GPU_OOM:
        {
            cudaError_t err = cudaDeviceReset();
            if (err == cudaSuccess)
            {
                context.addMetadata("recovery_action", "cudaDeviceReset");
                errors::ErrorRegistry::instance().reportError(
                    errors::ErrorCode::E11_GPU_LAUNCH_FAIL, context);
                return errors::ErrorResult<void>::success();
            }
            break;
        }
        case errors::ErrorCode::E11_GPU_LAUNCH_FAIL:
        {
            cudaError_t err = cudaDeviceSynchronize();
            if (err == cudaSuccess)
            {
                context.addMetadata("recovery_action", "cudaDeviceSynchronize");
                errors::ErrorRegistry::instance().reportError(
                    errors::ErrorCode::E11_GPU_LAUNCH_FAIL, context);
                return errors::ErrorResult<void>::success();
            }
            break;
        }
        default:
            break;
    }

    context.addMetadata("recovery_result", "failed");
    errors::ErrorRegistry::instance().reportError(error_code, context);
    return errors::ErrorResult<void>::error(error_code);
#else
    (void)operation;
    return errors::ErrorResult<void>::error(error_code);
#endif
}

} // namespace nerve::gpu

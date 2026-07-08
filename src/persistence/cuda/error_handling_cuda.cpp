
#include "nerve/persistence/cuda/cuda_error_handling.hpp"

#include <chrono>
#include <sstream>
#include <thread>

namespace nerve::persistence::accelerated
{

void ErrCtx::addMetadata(const std::string &key, const std::string &value)
{
    metadata[key] = value;
}

std::string ErrCtx::toString() const
{
    std::ostringstream out;
    out << "operation=" << operation_name << ", component=" << component_name
        << ", kernel=" << kernel_name << ", file=" << file_name << ", line=" << lineNumber
        << ", function_line=" << functionLine;
    if (!metadata.empty())
    {
        out << ", metadata={";
        bool first = true;
        for (const auto &[key, value] : metadata)
        {
            if (!first)
            {
                out << ", ";
            }
            out << key << ":" << value;
            first = false;
        }
        out << "}";
    }
    return out.str();
}

namespace
{

errors::ErrorCode mapErrorCode(cudaError_t error_code)
{
    if (error_code == cudaErrorMemoryAllocation)
    {
        return errors::ErrorCode::E10_GPU_OOM;
    }
    if (error_code == cudaErrorLaunchFailure || error_code == cudaErrorLaunchTimeout ||
        error_code == cudaErrorLaunchOutOfResources || error_code == cudaErrorInvalidConfiguration)
    {
        return errors::ErrorCode::E11_GPU_LAUNCH_FAIL;
    }
    if (error_code == cudaErrorInitializationError)
    {
        return errors::ErrorCode::E11_GPU_LAUNCH_FAIL;
    }
    return errors::ErrorCode::UNKNOWN;
}

} // namespace

namespace cuda_utils
{

} // namespace cuda_utils

namespace cuda_error_handling
{

errors::ErrorResult<void> check_cuda_operation(cudaError_t result, const std::string &operation,
                                               const std::source_location &loc)
{
    if (result == cudaSuccess)
    {
        return errors::ErrorResult<void>::ok();
    }

    ErrCtx context;
    context.operation_name = operation;
    context.component_name = "cuda_error_handling";
    context.file_name = loc.file_name();
    context.lineNumber = static_cast<int>(loc.line());
    context.functionLine = static_cast<int>(loc.line());
    context.addMetadata("cuda_error", cudaGetErrorString(result));
    cuda_utils::log_cuda_error(result, context, operation);
    return errors::ErrorResult<void>::error(mapErrorCode(result));
}

errors::ErrorResult<void> validateKernelLaunch(const std::string &kernel_name,
                                               const std::source_location &loc)
{
    const cudaError_t launch_error = cudaGetLastError();
    if (launch_error != cudaSuccess)
    {
        return check_cuda_operation(launch_error, "kernel_launch:" + kernel_name, loc);
    }
    return errors::ErrorResult<void>::ok();
}

errors::ErrorResult<void> attemptRecovery(cudaError_t error_code, const std::string &context)
{
    switch (getRecoveryStrategy(error_code))
    {
        case CudaRecoveryStrategy::None:
            return errors::ErrorResult<void>::error(errors::ErrorCode::UNKNOWN);
        case CudaRecoveryStrategy::Retry:
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            return errors::ErrorResult<void>::ok();
        case CudaRecoveryStrategy::ResetDevice:
            return check_cuda_operation(cudaDeviceReset(), "cudaDeviceReset:" + context);
        case CudaRecoveryStrategy::Abort:
            return errors::ErrorResult<void>::error(errors::ErrorCode::E50_PH_ABORT);
    }
    return errors::ErrorResult<void>::error(errors::ErrorCode::UNKNOWN);
}

bool isRecoverableError(cudaError_t error_code)
{
    return error_code == cudaErrorMemoryAllocation || error_code == cudaErrorLaunchTimeout ||
           error_code == cudaErrorLaunchOutOfResources;
}

std::string getErrorDescription(cudaError_t error_code)
{
    return std::string(cudaGetErrorString(error_code));
}

errors::ErrorCode map_cuda_error_to_error_code(cudaError_t error_code)
{
    return mapErrorCode(error_code);
}

CudaErrorSeverity classifyErrorSeverity(cudaError_t error_code)
{
    if (error_code == cudaSuccess)
    {
        return CudaErrorSeverity::Warning;
    }
    if (error_code == cudaErrorMemoryAllocation || error_code == cudaErrorLaunchTimeout)
    {
        return CudaErrorSeverity::Recoverable;
    }
    if (error_code == cudaErrorLaunchFailure || error_code == cudaErrorLaunchOutOfResources ||
        error_code == cudaErrorInvalidConfiguration)
    {
        return CudaErrorSeverity::Critical;
    }
    return CudaErrorSeverity::Fatal;
}

CudaRecoveryStrategy getRecoveryStrategy(cudaError_t error_code)
{
    if (error_code == cudaSuccess)
    {
        return CudaRecoveryStrategy::None;
    }
    if (error_code == cudaErrorMemoryAllocation || error_code == cudaErrorLaunchTimeout ||
        error_code == cudaErrorLaunchOutOfResources)
    {
        return CudaRecoveryStrategy::Retry;
    }
    if (error_code == cudaErrorLaunchFailure || error_code == cudaErrorInvalidConfiguration)
    {
        return CudaRecoveryStrategy::ResetDevice;
    }
    return CudaRecoveryStrategy::Abort;
}

} // namespace cuda_error_handling

OpGuard::OpGuard(const std::string &operation)
    : operation_(operation)
    , start_time_(std::chrono::steady_clock::now())
{}

OpGuard::~OpGuard()
{
    // Non-destructive peek  --  does NOT reset the last-error state so
    // subsequent error checks are not silently cleared.
    const cudaError_t pending = cudaPeekAtLastError();
    if (pending != cudaSuccess)
    {
        (void)cuda_error_handling::check_cuda_operation(pending, operation_ + ":destructor",
                                                        std::source_location::current());
    }
}

errors::ErrorResult<void> OpGuard::checkErrors()
{
    const cudaError_t pending = cudaGetLastError();
    if (pending != cudaSuccess)
    {
        return cuda_error_handling::check_cuda_operation(pending, operation_ + ":manual_check",
                                                         std::source_location::current());
    }
    return errors::ErrorResult<void>::ok();
}

double OpGuard::getDurationMs() const
{
    const auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(now - start_time_).count();
}

KernelGuard::KernelGuard(const std::string &kernel_name)
    : OpGuard(kernel_name)
{}

errors::ErrorResult<void> KernelGuard::validateLaunch()
{
    return cuda_error_handling::validateKernelLaunch(getOperation(),
                                                     std::source_location::current());
}

DeviceGuard::DeviceGuard(int device_id)
    : OpGuard("device_guard")
    , saved_device_(-1)
    , owns_device_(true)
{
    int current = 0;
    if (cudaGetDevice(&current) == cudaSuccess)
    {
        saved_device_ = current;
    }
    if (device_id >= 0)
    {
        (void)cudaSetDevice(device_id);
    }
}

DeviceGuard::~DeviceGuard()
{
    if (owns_device_ && saved_device_ >= 0)
    {
        (void)cudaSetDevice(saved_device_);
    }
}

errors::ErrorResult<void> DeviceGuard::resetDevice()
{
    return cuda_error_handling::check_cuda_operation(cudaDeviceReset(), "cudaDeviceReset",
                                                     std::source_location::current());
}

StreamGuard::StreamGuard(const std::string &stream_name, cudaStream_t stream)
    : OpGuard("stream_guard:" + stream_name)
    , stream_(stream)
{}

errors::ErrorResult<void> StreamGuard::synchronize()
{
    if (stream_)
    {
        return cuda_error_handling::check_cuda_operation(cudaStreamSynchronize(stream_),
                                                         "cudaStreamSynchronize",
                                                         std::source_location::current());
    }
    return cuda_error_handling::check_cuda_operation(
        cudaDeviceSynchronize(), "cudaDeviceSynchronize", std::source_location::current());
}

MemGuard::MemGuard(const std::string &operation)
    : OpGuard("memory_guard:" + operation)
{}

errors::ErrorResult<void> MemGuard::checkMemoryAllocation()
{
    return checkErrors();
}

errors::ErrorResult<void> MemGuard::checkMemoryDeallocation()
{
    return checkErrors();
}

SyncGuard::SyncGuard(const std::string &operation)
    : OpGuard("sync_guard:" + operation)
{}

errors::ErrorResult<void> SyncGuard::synchronizeDevice()
{
    return cuda_error_handling::check_cuda_operation(
        cudaDeviceSynchronize(), "cudaDeviceSynchronize", std::source_location::current());
}

errors::ErrorResult<void> SyncGuard::synchronizeStream(cudaStream_t stream)
{
    return cuda_error_handling::check_cuda_operation(
        cudaStreamSynchronize(stream), "cudaStreamSynchronize", std::source_location::current());
}

} // namespace nerve::persistence::accelerated

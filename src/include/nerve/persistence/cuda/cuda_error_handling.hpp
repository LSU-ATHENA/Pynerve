
#pragma once

#include "nerve/core.hpp"
#include "nerve/core_types.hpp"
#include "nerve/errors/errors.hpp"
#include "nerve/persistence/accelerated/accelerated_api.hpp"
#include "nerve/runtime/hardware_probe.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <iostream>
#include <source_location>
#include <string>
#include <unordered_map>
#include <vector>

namespace nerve::persistence::accelerated
{

enum class CudaErrorSeverity
{
    Warning,
    Recoverable,
    Critical,
    Fatal
};

enum class CudaRecoveryStrategy
{
    None,
    Retry,
    ResetDevice,
    Abort
};

struct ErrCtx
{
    std::string operation_name;
    std::string component_name;
    std::string kernel_name;
    std::string file_name;
    int lineNumber;
    int functionLine;
    std::chrono::steady_clock::time_point timestamp;
    std::unordered_map<std::string, std::string> metadata;

    ErrCtx()
        : lineNumber(0)
        , functionLine(0)
        , timestamp(std::chrono::steady_clock::now())
    {}

    void addMetadata(const std::string &key, const std::string &value);

    std::string toString() const;
};

namespace cuda_error_handling
{

errors::ErrorResult<void>
check_cuda_operation(cudaError_t result, const std::string &operation,
                     const std::source_location &loc = std::source_location::current());

errors::ErrorResult<void>
validateKernelLaunch(const std::string &kernel_name,
                     const std::source_location &loc = std::source_location::current());

errors::ErrorResult<void> attemptRecovery(cudaError_t error_code, const std::string &context);

bool isRecoverableError(cudaError_t error_code);

std::string getErrorDescription(cudaError_t error_code);

errors::ErrorCode map_cuda_error_to_error_code(cudaError_t error_code);

CudaErrorSeverity classifyErrorSeverity(cudaError_t error_code);

CudaRecoveryStrategy getRecoveryStrategy(cudaError_t error_code);

} // namespace cuda_error_handling

class OpGuard
{
public:
    explicit OpGuard(const std::string &operation);

    virtual ~OpGuard();

    OpGuard(OpGuard &&) = default;
    OpGuard &operator=(OpGuard &&) = default;
    OpGuard(const OpGuard &) = delete;
    OpGuard &operator=(const OpGuard &) = delete;

    errors::ErrorResult<void> checkErrors();

    const std::string &getOperation() const { return operation_; }

    double getDurationMs() const;

private:
    std::string operation_;
    std::chrono::steady_clock::time_point start_time_;
};

class KernelGuard : public OpGuard
{
public:
    explicit KernelGuard(const std::string &kernel_name);

    errors::ErrorResult<void> validateLaunch();
};

class DeviceGuard : public OpGuard
{
public:
    explicit DeviceGuard(int device_id = -1);

    ~DeviceGuard() override;

    DeviceGuard(DeviceGuard &&other) noexcept
        : OpGuard(std::move(other))
        , saved_device_(other.saved_device_)
        , owns_device_(other.owns_device_)
    {
        other.owns_device_ = false;
    }
    DeviceGuard &operator=(DeviceGuard &&other) noexcept
    {
        if (this != &other)
        {
            OpGuard::operator=(std::move(other));
            saved_device_ = other.saved_device_;
            owns_device_ = other.owns_device_;
            other.owns_device_ = false;
        }
        return *this;
    }
    DeviceGuard(const DeviceGuard &) = delete;
    DeviceGuard &operator=(const DeviceGuard &) = delete;

    errors::ErrorResult<void> resetDevice();

private:
    int saved_device_;
    bool owns_device_;
};

class StreamGuard : public OpGuard
{
public:
    StreamGuard(const std::string &stream_name, cudaStream_t stream = nullptr);

    ~StreamGuard() override = default;

    StreamGuard(StreamGuard &&) = default;
    StreamGuard &operator=(StreamGuard &&) = default;
    StreamGuard(const StreamGuard &) = delete;
    StreamGuard &operator=(const StreamGuard &) = delete;

    errors::ErrorResult<void> synchronize();

private:
    cudaStream_t stream_;
};

class MemGuard : public OpGuard
{
public:
    explicit MemGuard(const std::string &operation);

    errors::ErrorResult<void> checkMemoryAllocation();

    errors::ErrorResult<void> checkMemoryDeallocation();
};

class SyncGuard : public OpGuard
{
public:
    explicit SyncGuard(const std::string &operation);

    errors::ErrorResult<void> synchronizeDevice();

    errors::ErrorResult<void> synchronizeStream(cudaStream_t stream);
};

namespace cuda_utils
{

bool is_cuda_available();

errors::ErrorResult<cudaDeviceProp> getDeviceProperties(int device_id = 0);

errors::ErrorResult<Size> getAvailableGpuMemory();

errors::ErrorResult<Size> getTotalGpuMemory();

double getMemoryPressure();

std::string get_cuda_driver_version();

std::string get_cuda_runtime_version();

errors::ErrorResult<void> check_cuda_support(const std::vector<std::string> &features);

ErrCtx createErrorContext(const std::string &operation);

void log_cuda_error(cudaError_t error_code, const ErrCtx &context, const std::string &operation);

Size getOptimalBlockSize(int device_id = 0, Size kernel_resource_usage = 0);

Size getOptimalGridSize(Size total_elements, Size block_size);

bool shouldUseStreaming(Size problem_size, Size available_gpu_memory);

Size getStreamingChunkSize(Size problem_size, Size available_gpu_memory);

errors::ErrorResult<void> validateLaunchParams(dim3 grid_size, dim3 block_size,
                                               Size shared_memory_size, int device_id = 0);

errors::ErrorResult<AcceleratedPerformanceStats>
getKernelPerformanceMetrics(const std::string &kernel_name);

errors::ErrorResult<double> profileKernelExecutionTime(const std::string &kernel_name,
                                                       dim3 grid_size, dim3 block_size);

} // namespace cuda_utils

namespace cuda_guards
{

inline OpGuard createOperationGuard(const std::string &operation)
{
    return OpGuard(operation);
}

inline KernelGuard createKernelGuard(const std::string &kernel_name)
{
    return KernelGuard(kernel_name);
}

inline DeviceGuard createDeviceGuard(int device_id = -1)
{
    return DeviceGuard(device_id);
}

inline StreamGuard createStreamGuard(const std::string &stream_name, cudaStream_t stream = nullptr)
{
    return StreamGuard(stream_name, stream);
}

inline MemGuard createMemoryGuard(const std::string &operation)
{
    return MemGuard(operation);
}

inline SyncGuard createSyncGuard(const std::string &operation)
{
    return SyncGuard(operation);
}

} // namespace cuda_guards

#define CUDA_CHECK(result, operation)                                                              \
    do                                                                                             \
    {                                                                                              \
        auto cuda_result = (result);                                                               \
        if (cuda_result != cudaSuccess)                                                            \
        {                                                                                          \
            auto error_result = cuda_error_handling::check_cuda_operation(                         \
                cuda_result, (operation), std::source_location::current());                        \
            if (error_result.isError())                                                            \
            {                                                                                      \
                std::cerr << "CUDA Error in " << (operation) << ": "                               \
                          << error_result.error().message << std::endl;                            \
            }                                                                                      \
        }                                                                                          \
    } while (0)

#define CUDA_CHECK_SYNC(result, operation)                                                         \
    do                                                                                             \
    {                                                                                              \
        auto cuda_result = (result);                                                               \
        if (cuda_result != cudaSuccess)                                                            \
        {                                                                                          \
            auto error_result = cuda_error_handling::check_cuda_operation(                         \
                cuda_result, (operation), std::source_location::current());                        \
            if (error_result.isError())                                                            \
            {                                                                                      \
                std::cerr << "CUDA Error in " << (operation) << ": "                               \
                          << error_result.error().message << std::endl;                            \
            }                                                                                      \
        }                                                                                          \
    } while (0)

#define CUDA_LAUNCH_KERNEL(kernel, grid, block, ...)                                               \
    do                                                                                             \
    {                                                                                              \
        kernel<<<grid, block>>>(__VA_ARGS__);                                                      \
        auto launch_result =                                                                       \
            cuda_error_handling::validateKernelLaunch(#kernel, std::source_location::current());   \
        if (launch_result.isError())                                                               \
        {                                                                                          \
            std::cerr << "CUDA Kernel launch failed: " << launch_result.error().message            \
                      << std::endl;                                                                \
        }                                                                                          \
    } while (0)

#define CUDA_LAUNCH_KERNEL_SYNC(kernel, grid, block, ...)                                          \
    do                                                                                             \
    {                                                                                              \
        kernel<<<grid, block>>>(__VA_ARGS__);                                                      \
        auto launch_result =                                                                       \
            cuda_error_handling::validateKernelLaunch(#kernel, std::source_location::current());   \
        if (launch_result.isError())                                                               \
        {                                                                                          \
            std::cerr << "CUDA Kernel launch failed: " << launch_result.error().message            \
                      << std::endl;                                                                \
        }                                                                                          \
        auto sync_result = cuda_error_handling::check_cuda_operation(                              \
            cudaDeviceSynchronize(), #kernel " synchronization", std::source_location::current()); \
        if (sync_result.isError())                                                                 \
        {                                                                                          \
            std::cerr << "CUDA synchronization failed: " << sync_result.error().message            \
                      << std::endl;                                                                \
        }                                                                                          \
    } while (0)

#define CUDA_SYNC_KERNEL(kernel, ...)                                                              \
    do                                                                                             \
    {                                                                                              \
        kernel<<<1, 1>>>(__VA_ARGS__);                                                             \
        auto sync_result = cuda_error_handling::check_cuda_operation(                              \
            cudaDeviceSynchronize(), #kernel " synchronization", std::source_location::current()); \
        if (sync_result.isError())                                                                 \
        {                                                                                          \
            std::cerr << "CUDA synchronization failed: " << sync_result.error().message            \
                      << std::endl;                                                                \
        }                                                                                          \
    } while (0)

#define CUDA_GUARD_OPERATION(operation)                                                            \
    auto cuda_guard = cuda_guards::createOperationGuard(operation)

#define CUDA_GUARD_KERNEL(kernel) auto cuda_guard = cuda_guards::createKernelGuard(kernel)

#define CUDA_GUARD_MEMORY(operation) auto cuda_guard = cuda_guards::createMemoryGuard(operation)

#define CUDA_GUARD_SYNC(operation) auto cuda_guard = cuda_guards::createSyncGuard(operation)

} // namespace nerve::persistence::accelerated

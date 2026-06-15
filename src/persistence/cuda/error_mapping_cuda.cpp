
#include "nerve/persistence/cuda/cuda_error_handling.hpp"
#include "nerve/persistence/cuda/cuda_safe_arithmetic.hpp"
#include "nerve/runtime/hardware_probe.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace nerve::persistence::accelerated
{
namespace
{

errors::ErrorCode mapErrorCode(cudaError_t errorCode)
{
    if (errorCode == cudaErrorMemoryAllocation)
    {
        return errors::ErrorCode::E10_GPU_OOM;
    }
    if (errorCode == cudaErrorLaunchFailure || errorCode == cudaErrorLaunchTimeout ||
        errorCode == cudaErrorLaunchOutOfResources || errorCode == cudaErrorInvalidConfiguration)
    {
        return errors::ErrorCode::E11_GPU_LAUNCH_FAIL;
    }
    if (errorCode == cudaErrorInitializationError)
    {
        return errors::ErrorCode::E11_GPU_LAUNCH_FAIL;
    }
    return errors::ErrorCode::UNKNOWN;
}

bool checkFeatureSupported(const cudaDeviceProp &prop, const std::string &feature)
{
    if (feature == "mapped_host_memory")
    {
#if CUDART_VERSION >= 13000
        int supported = 0;
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&supported, cudaDevAttrCanMapHostMemory, dev);
        return supported != 0;
#else
        return prop.canMapHostMemory != 0;
#endif
    }
    return false;
}

} // namespace

namespace cuda_utils
{

bool is_cuda_available()
{
    const runtime::HardwareSnapshot snapshot = runtime::collectHardwareSnapshot();
    if (snapshot.gpus.ok())
    {
        return !snapshot.gpus.value.empty();
    }
    int device_count = 0;
    return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

errors::ErrorResult<cudaDeviceProp> getDeviceProperties(int device_id)
{
    cudaDeviceProp properties{};
    const cudaError_t result = cudaGetDeviceProperties(&properties, device_id);
    if (result != cudaSuccess)
    {
        return errors::ErrorResult<cudaDeviceProp>::error(mapErrorCode(result));
    }
    return errors::ErrorResult<cudaDeviceProp>::success(std::move(properties));
}

errors::ErrorResult<Size> getAvailableGpuMemory()
{
    size_t free_mem = 0;
    size_t total_mem = 0;
    const cudaError_t result = cudaMemGetInfo(&free_mem, &total_mem);
    if (result == cudaSuccess)
    {
        return errors::ErrorResult<Size>::success(static_cast<Size>(free_mem));
    }

    const runtime::HardwareSnapshot snapshot = runtime::collectHardwareSnapshot();
    if (snapshot.gpus.ok() && !snapshot.gpus.value.empty())
    {
        return errors::ErrorResult<Size>::success(
            static_cast<Size>(snapshot.gpus.value.front().free_memory_bytes));
    }
    return errors::ErrorResult<Size>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
}

errors::ErrorResult<Size> getTotalGpuMemory()
{
    size_t free_mem = 0;
    size_t total_mem = 0;
    const cudaError_t result = cudaMemGetInfo(&free_mem, &total_mem);
    if (result == cudaSuccess)
    {
        return errors::ErrorResult<Size>::success(static_cast<Size>(total_mem));
    }

    const runtime::HardwareSnapshot snapshot = runtime::collectHardwareSnapshot();
    if (snapshot.gpus.ok() && !snapshot.gpus.value.empty())
    {
        return errors::ErrorResult<Size>::success(
            static_cast<Size>(snapshot.gpus.value.front().total_memory_bytes));
    }
    return errors::ErrorResult<Size>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
}

double getMemoryPressure()
{
    const auto total = getTotalGpuMemory();
    const auto free = getAvailableGpuMemory();
    if (total.isError() || free.isError() || total.value() == 0)
    {
        throw std::runtime_error("CUDA memory pressure is unavailable");
    }
    const double total_bytes = static_cast<double>(total.value());
    const double free_bytes = static_cast<double>(std::min(free.value(), total.value()));
    const double pressure = 1.0 - (free_bytes / total_bytes);
    if (!std::isfinite(pressure))
    {
        throw std::runtime_error("CUDA memory pressure is unavailable");
    }
    return std::clamp(pressure, 0.0, 1.0);
}

std::string get_cuda_driver_version()
{
    int version = 0;
    if (cudaDriverGetVersion(&version) != cudaSuccess || version <= 0)
    {
        return "missing";
    }
    return std::to_string(version / 1000) + "." + std::to_string((version % 1000) / 10);
}

std::string get_cuda_runtime_version()
{
    int version = 0;
    if (cudaRuntimeGetVersion(&version) != cudaSuccess || version <= 0)
    {
        return "missing";
    }
    return std::to_string(version / 1000) + "." + std::to_string((version % 1000) / 10);
}

errors::ErrorResult<void> check_cuda_support(const std::vector<std::string> &features)
{
    if (!is_cuda_available())
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
    }
    auto props = getDeviceProperties(0);
    if (props.isError())
    {
        return errors::ErrorResult<void>::error(props.errorCode());
    }
    for (const std::string &feature : features)
    {
        if (!checkFeatureSupported(props.value(), feature))
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E52_PH_CONFIG);
        }
    }
    return errors::ErrorResult<void>::ok();
}

ErrCtx createErrorContext(const std::string &operation)
{
    ErrCtx context;
    context.operation_name = operation;
    context.component_name = "cuda_utils";
    context.file_name = std::source_location::current().file_name();
    context.lineNumber = static_cast<int>(std::source_location::current().line());
    context.functionLine = static_cast<int>(std::source_location::current().line());
    return context;
}

void log_cuda_error(cudaError_t errorCode, const ErrCtx &context, const std::string &operation)
{
    std::cerr << "[Nerve CUDA] " << operation << " failed with " << cudaGetErrorString(errorCode)
              << " :: " << context.toString() << '\n';
}

Size getOptimalBlockSize(int device_id, Size kernel_resource_usage)
{
    auto props = getDeviceProperties(device_id);
    if (props.isError())
    {
        return 128;
    }
    const Size max_threads = static_cast<Size>(props.value().maxThreadsPerBlock);
    Size block_size = std::min<Size>(256, max_threads);
    if (kernel_resource_usage > 0)
    {
        const Size pressure = std::max<Size>(1, kernel_resource_usage / 1024);
        block_size = std::max<Size>(32, block_size / pressure);
    }
    block_size = std::max<Size>(32, (block_size / 32) * 32);
    return block_size;
}

Size getOptimalGridSize(Size total_elements, Size block_size)
{
    const Size safe_block = std::max<Size>(1, block_size);
    return detail::ceilDiv(total_elements, safe_block);
}

bool shouldUseStreaming(Size problem_size, Size available_gpu_memory)
{
    return available_gpu_memory > 0 && problem_size > (available_gpu_memory / sizeof(double)) / 2;
}

Size getStreamingChunkSize(Size problem_size, Size available_gpu_memory)
{
    if (available_gpu_memory == 0)
    {
        return std::max<Size>(1, std::min<Size>(problem_size, 1 << 20));
    }
    const Size max_elements = available_gpu_memory / (sizeof(double) * 4);
    return std::max<Size>(1, std::min(problem_size, max_elements));
}

errors::ErrorResult<void> validateLaunchParams(dim3 grid_size, dim3 block_size,
                                               Size shared_memory_size, int device_id)
{
    auto props = getDeviceProperties(device_id);
    if (props.isError())
    {
        return errors::ErrorResult<void>::error(props.errorCode());
    }
    const cudaDeviceProp &prop = props.value();
    Size xy_threads = 0;
    Size threads_per_block = 0;
    if (!detail::checkedSizeProduct(static_cast<Size>(block_size.x),
                                    static_cast<Size>(block_size.y), xy_threads) ||
        !detail::checkedSizeProduct(xy_threads, static_cast<Size>(block_size.z), threads_per_block))
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
    }
    if (threads_per_block == 0 || threads_per_block > static_cast<Size>(prop.maxThreadsPerBlock))
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
    }
    constexpr Size kSharedMemorySafetyCap = 48 * 1024;
    if (shared_memory_size > kSharedMemorySafetyCap)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
    }
    if (grid_size.x == 0 || grid_size.y == 0 || grid_size.z == 0)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
    }
    return errors::ErrorResult<void>::ok();
}

errors::ErrorResult<AcceleratedPerformanceStats>
getKernelPerformanceMetrics(const std::string &kernel_name)
{
    if (!is_cuda_available())
    {
        return errors::ErrorResult<AcceleratedPerformanceStats>::error(
            errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
    }

    auto props = getDeviceProperties(0);
    if (props.isError())
    {
        return errors::ErrorResult<AcceleratedPerformanceStats>::error(props.errorCode());
    }

    auto available_memory = getAvailableGpuMemory();
    if (available_memory.isError())
    {
        return errors::ErrorResult<AcceleratedPerformanceStats>::error(
            available_memory.errorCode());
    }
    auto total_memory = getTotalGpuMemory();
    if (total_memory.isError() || total_memory.value() == 0)
    {
        return errors::ErrorResult<AcceleratedPerformanceStats>::error(
            total_memory.isError() ? total_memory.errorCode()
                                   : errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
    }

    const Size block_x = std::max<Size>(
        32, std::min<Size>(256, static_cast<Size>(props.value().maxThreadsPerBlock)));
    const Size grid_x = std::max<Size>(1, static_cast<Size>(props.value().multiProcessorCount) * 4);
    auto profiled_ms =
        profileKernelExecutionTime(kernel_name, dim3(static_cast<unsigned int>(grid_x), 1, 1),
                                   dim3(static_cast<unsigned int>(block_x), 1, 1));
    if (profiled_ms.isError())
    {
        return errors::ErrorResult<AcceleratedPerformanceStats>::error(profiled_ms.errorCode());
    }

    double pressure = 0.0;
    try
    {
        pressure = getMemoryPressure();
    }
    catch (const std::runtime_error &e)
    {
        fprintf(stderr, "error: %s\n", e.what());
        return errors::ErrorResult<AcceleratedPerformanceStats>::error(
            errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
    }
    if (!std::isfinite(pressure))
    {
        return errors::ErrorResult<AcceleratedPerformanceStats>::error(
            errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
    }

    const double elapsed_ms = profiled_ms.value();
    if (!std::isfinite(elapsed_ms) || elapsed_ms < 0.0)
    {
        return errors::ErrorResult<AcceleratedPerformanceStats>::error(
            errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
    }

    const Size used_memory = total_memory.value() >= available_memory.value()
                                 ? total_memory.value() - available_memory.value()
                                 : 0;

    AcceleratedPerformanceStats stats{};
    stats.gpu_used = true;
    stats.total_time_ms = elapsed_ms;
    stats.gpu_time_ms = elapsed_ms;
    stats.cpu_time_ms = std::max(0.0, stats.total_time_ms - stats.gpu_time_ms);
    stats.problems_processed = static_cast<Size>(grid_x * block_x);
    stats.memory_usage_mb = static_cast<double>(used_memory / (1024ULL * 1024ULL));
    stats.peak_memory_usage_mb = stats.memory_usage_mb;
    stats.gpu_utilization = std::clamp(1.0 - pressure, 0.0, 1.0);
    stats.gpu_stage_ops = static_cast<double>(stats.problems_processed);
    const double workload_scale =
        std::max<double>(1.0, static_cast<double>(stats.problems_processed));
    const double total_compute_ms = stats.gpu_time_ms + stats.cpu_time_ms;
    stats.speedup =
        std::isfinite(total_compute_ms) && total_compute_ms >= 0.0 && stats.gpu_time_ms > 0.0
            ? total_compute_ms / stats.gpu_time_ms
            : 1.0;
    const double average_denominator = workload_scale + stats.total_time_ms;
    stats.average_speedup = std::isfinite(stats.speedup) && std::isfinite(average_denominator) &&
                                    average_denominator > 0.0
                                ? stats.speedup * (workload_scale / average_denominator)
                                : 1.0;
    const bool finite_stats =
        std::isfinite(stats.total_time_ms) && std::isfinite(stats.gpu_time_ms) &&
        std::isfinite(stats.cpu_time_ms) && std::isfinite(stats.memory_usage_mb) &&
        std::isfinite(stats.gpu_utilization) && std::isfinite(stats.speedup) &&
        std::isfinite(stats.average_speedup) && std::isfinite(stats.peak_memory_usage_mb) &&
        std::isfinite(stats.gpu_stage_ops);
    if (!finite_stats)
    {
        return errors::ErrorResult<AcceleratedPerformanceStats>::error(
            errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
    }
    return errors::ErrorResult<AcceleratedPerformanceStats>::success(std::move(stats));
}

errors::ErrorResult<double> profileKernelExecutionTime(const std::string &kernel_name,
                                                       dim3 grid_size, dim3 block_size)
{
    if (validateLaunchParams(grid_size, block_size, 0).isError())
    {
        return errors::ErrorResult<double>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
    }
    if (!is_cuda_available())
    {
        return errors::ErrorResult<double>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
    }

    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;
    if (cudaEventCreate(&start_event) != cudaSuccess || cudaEventCreate(&stop_event) != cudaSuccess)
    {
        if (start_event != nullptr)
        {
            cudaEventDestroy(start_event);
        }
        if (stop_event != nullptr)
        {
            cudaEventDestroy(stop_event);
        }
        return errors::ErrorResult<double>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
    }

    const Size total_threads = static_cast<Size>(grid_size.x) * static_cast<Size>(block_size.x);
    const Size iterations = std::clamp<Size>(total_threads / 4096, 1, 32);

    const auto host_start = std::chrono::steady_clock::now();
    if (cudaEventRecord(start_event) != cudaSuccess)
    {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
        return errors::ErrorResult<double>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
    }
    for (Size iter = 0; iter < iterations; ++iter)
    {
        (void)kernel_name;
        if (cudaDeviceSynchronize() != cudaSuccess)
        {
            cudaEventDestroy(start_event);
            cudaEventDestroy(stop_event);
            return errors::ErrorResult<double>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }
    }
    if (cudaEventRecord(stop_event) != cudaSuccess)
    {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
        return errors::ErrorResult<double>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
    }
    if (cudaEventSynchronize(stop_event) != cudaSuccess)
    {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
        return errors::ErrorResult<double>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
    }
    const auto host_end = std::chrono::steady_clock::now();

    float elapsed_ms = 0.0F;
    if (cudaEventElapsedTime(&elapsed_ms, start_event, stop_event) != cudaSuccess)
    {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
        return errors::ErrorResult<double>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
    }
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    const double host_ms = std::chrono::duration<double, std::milli>(host_end - host_start).count();
    const double measured_ms = std::max<double>(elapsed_ms, host_ms);
    if (!std::isfinite(elapsed_ms) || !std::isfinite(host_ms) || !std::isfinite(measured_ms) ||
        measured_ms <= 0.0)
    {
        return errors::ErrorResult<double>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
    }
    return errors::ErrorResult<double>::success(measured_ms / static_cast<double>(iterations));
}

} // namespace cuda_utils

} // namespace nerve::persistence::accelerated

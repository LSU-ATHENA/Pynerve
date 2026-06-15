
#include "nerve/errors/errors.hpp"

#include <cuda_runtime.h>

#include <mutex>

namespace nerve::persistence::accelerated::resource_management
{

struct ResourceStats
{
    size_t total_resources = 0;
    size_t total_gpu_memory_used_mb = 0;
    size_t available_gpu_memory_mb = 0;
};

namespace
{
std::mutex g_resource_mutex;
ResourceStats g_stats{};
} // namespace

ResourceStats getResourceStats()
{
    std::lock_guard<std::mutex> lock(g_resource_mutex);
    return g_stats;
}

errors::ErrorResult<size_t> cleanupInvalidResources()
{
    std::lock_guard<std::mutex> lock(g_resource_mutex);
    return errors::ErrorResult<size_t>::success(0);
}

errors::ErrorResult<bool> checkMemoryPressure(double threshold)
{
    if (threshold <= 0.0 || threshold >= 1.0)
    {
        return errors::ErrorResult<bool>::error(errors::ErrorCode::E50_PH_ABORT);
    }

    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    const cudaError_t status = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (status != cudaSuccess || total_bytes == 0)
    {
        return errors::ErrorResult<bool>::error(errors::ErrorCode::E10_GPU_OOM);
    }

    const double pressure =
        1.0 - static_cast<double>(free_bytes) / static_cast<double>(total_bytes);
    {
        std::lock_guard<std::mutex> lock(g_resource_mutex);
        g_stats.total_gpu_memory_used_mb = (total_bytes - free_bytes) / (1024ULL * 1024ULL);
        g_stats.available_gpu_memory_mb = free_bytes / (1024ULL * 1024ULL);
    }
    return errors::ErrorResult<bool>::success(pressure >= threshold);
}

errors::ErrorResult<void> optimizeMemoryUsage()
{
    const cudaError_t status = cudaDeviceSynchronize();
    if (status != cudaSuccess)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
    }
    return errors::ErrorResult<void>::success();
}

} // namespace nerve::persistence::accelerated::resource_management

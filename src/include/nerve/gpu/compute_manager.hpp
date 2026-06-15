// Provides runtime probing and deterministic behavior in host-only builds.

#pragma once

#include "nerve/runtime/hardware_probe.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <vector>

// Host CUDA type definitions when CUDA headers are missing.
#if !defined(__CUDACC__)
#if __has_include(<cuda_runtime.h>)
#include <cuda_runtime.h>
#else
using cudaStream_t = void *;
using cudaEvent_t = void *;
using cudaError_t = int;
constexpr cudaError_t cudaSuccess = 0;
#endif
#endif

namespace nerve::gpu
{

enum class OperationType
{
    MATRIX_REDUCTION,
    DISTANCE_COMPUTATION,
    VR_CONSTRUCTION,
    PERSISTENCE_COMPUTATION
};

struct ComputeConfig
{
    std::size_t gpu_id = 0;
    std::size_t batch_size = 1024;
    bool use_unified_memory = false;
    bool enable_profiling = false;
};

struct Stats
{
    double computation_time_ms = 0.0;
    std::size_t memory_used_bytes = 0;
    std::size_t num_operations = 0;
};

class ComputeManager
{
public:
    static ComputeManager &instance()
    {
        static ComputeManager singleton;
        return singleton;
    }

    [[nodiscard]] bool isGpuAvailable() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return initialized_ && gpu_count_ > 0;
    }

    [[nodiscard]] std::size_t getGpuCount() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return gpu_count_;
    }

    bool initialize(std::size_t gpu_id = 0)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        refreshProbeUnsafe();
        if (gpu_count_ == 0 || gpu_id >= gpu_count_)
        {
            initialized_ = false;
            active_gpu_id_ = kInvalidGpuId;
            last_error_ = static_cast<cudaError_t>(1);
            return false;
        }
        active_gpu_id_ = gpu_id;
        config_.gpu_id = gpu_id;
        initialized_ = true;
        last_error_ = cudaSuccess;
        return true;
    }

    void shutdown()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        initialized_ = false;
        active_gpu_id_ = kInvalidGpuId;
    }

    [[nodiscard]] cudaStream_t getStream() const
    {
        if (!isGpuAvailable())
        {
            return cudaStream_t{};
        }
        return makeSentinelStreamToken();
    }

    template <typename T>
    bool submit(OperationType, const std::vector<T> &data, Stats *stats = nullptr)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const bool ready = initialized_ && gpu_count_ > 0;
        if (stats != nullptr)
        {
            stats->num_operations += ready ? 1U : 0U;
            stats->memory_used_bytes += data.size() * sizeof(T);
        }
        if (ready)
        {
            aggregate_stats_.num_operations += 1U;
            aggregate_stats_.memory_used_bytes += data.size() * sizeof(T);
        }
        return ready;
    }

    bool synchronize()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return initialized_ && gpu_count_ > 0;
    }

    [[nodiscard]] cudaError_t getLastError() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return last_error_;
    }

    void setConfig(const ComputeConfig &config)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        config_ = config;
    }

    [[nodiscard]] ComputeConfig getConfig() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return config_;
    }

    [[nodiscard]] std::size_t getCachedFreeMemoryBytes() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return cached_free_memory_bytes_;
    }

private:
    static constexpr std::size_t kInvalidGpuId = std::numeric_limits<std::size_t>::max();

    ComputeManager() { refreshProbeUnsafe(); }

    void refreshProbeUnsafe()
    {
        const runtime::HardwareSnapshot snapshot = runtime::collectHardwareSnapshot();
        gpu_count_ = (snapshot.gpus.ok() ? snapshot.gpus.value.size() : 0);
        cached_free_memory_bytes_ =
            (snapshot.gpus.ok() && !snapshot.gpus.value.empty())
                ? static_cast<std::size_t>(snapshot.gpus.value.front().free_memory_bytes)
                : 0U;
    }

    static cudaStream_t makeSentinelStreamToken() { return (cudaStream_t)1; }

    mutable std::mutex mutex_;
    ComputeConfig config_{};
    Stats aggregate_stats_{};
    std::size_t gpu_count_ = 0;
    std::size_t active_gpu_id_ = kInvalidGpuId;
    std::size_t cached_free_memory_bytes_ = 0;
    bool initialized_ = false;
    cudaError_t last_error_ = cudaSuccess;
};

[[nodiscard]] inline bool isCudaRuntimeAvailable()
{
    return ComputeManager::instance().getGpuCount() > 0;
}

[[nodiscard]] inline std::size_t getGpuFreeMemory()
{
    return ComputeManager::instance().getCachedFreeMemoryBytes();
}

} // namespace nerve::gpu

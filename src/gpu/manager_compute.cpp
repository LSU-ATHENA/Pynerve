#include "nerve/gpu/kernel_launcher.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstddef>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <vector>

namespace nerve::gpu
{
namespace
{
constexpr std::size_t kBytesPerMB = 1024ULL * 1024ULL;

std::unordered_map<std::string, Config> g_tuned_configs;

std::string architectureName(int compute_capability)
{
    if (compute_capability <= 0)
    {
        return "missing";
    }
    return "sm_" + std::to_string(compute_capability);
}

} // namespace

class ComputeManager::OperationScope
{};
class ComputeManager::GraphManager
{};

ComputeManager &ComputeManager::getInstance()
{
    static ComputeManager instance;
    return instance;
}

ComputeManager::~ComputeManager()
{
    shutdown();
}

void ComputeManager::initialize()
{
    initialize(Config{});
}

void ComputeManager::shutdown()
{
    std::unique_lock<std::shared_mutex> lock(mutex_);
    graphManager_.reset();
    available_ = false;
    initialized_ = false;
}

bool ComputeManager::isAvailable() const
{
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return initialized_ && available_ && config_.enable_gpu;
}

bool ComputeManager::isSupported(OperationType operation) const
{
    std::shared_lock<std::shared_mutex> lock(mutex_);
    if (!initialized_ || !available_ || !config_.enable_gpu)
    {
        return false;
    }
    const auto it = config_.enabled.find(operation);
    return it == config_.enabled.end() || it->second;
}

void ComputeManager::setConfig(const Config &config)
{
    std::unique_lock<std::shared_mutex> lock(mutex_);
    config_ = config;
}

Config ComputeManager::getConfig() const
{
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return config_;
}

Strategy ComputeManager::selectStrategy(OperationType operation, size_t problem_size) const
{
    std::shared_lock<std::shared_mutex> lock(mutex_);
    if (!initialized_ || !available_ || !config_.enable_gpu)
    {
        return Strategy::kCPUOnly;
    }

    const auto enabled_it = config_.enabled.find(operation);
    if (enabled_it != config_.enabled.end() && !enabled_it->second)
    {
        return Strategy::kCPUOnly;
    }

    size_t threshold = config_.threshold;
    const auto threshold_it = config_.thresholds.find(operation);
    if (threshold_it != config_.thresholds.end())
    {
        threshold = threshold_it->second;
    }

    return problem_size < threshold ? Strategy::kCPUOnly : Strategy::kGPUAccelerated;
}

void ComputeManager::recordPerformance(const PerformanceProfile &profile)
{
    std::unique_lock<std::shared_mutex> lock(mutex_);
    PerformanceProfile stored = profile;
    if (!std::isfinite(stored.cpu_time_ms) || stored.cpu_time_ms < 0.0)
    {
        stored.cpu_time_ms = 0.0;
    }
    if (!std::isfinite(stored.gpu_time_ms) || stored.gpu_time_ms < 0.0)
    {
        stored.gpu_time_ms = 0.0;
    }
    if (!std::isfinite(stored.speedup) || stored.speedup < 0.0)
    {
        stored.speedup = 1.0;
    }
    if (stored.timestamp == std::chrono::system_clock::time_point{})
    {
        stored.timestamp = std::chrono::system_clock::now();
    }
    history_.push_back(stored);
    updateStats(stored);
}

std::vector<PerformanceProfile> ComputeManager::getPerformanceHistory() const
{
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return history_;
}

void ComputeManager::clearPerformanceHistory()
{
    std::unique_lock<std::shared_mutex> lock(mutex_);
    history_.clear();
}

ComputeManager::Stats ComputeManager::getStats() const
{
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return stats_;
}

void ComputeManager::clearStats()
{
    std::unique_lock<std::shared_mutex> lock(mutex_);
    stats_ = Stats{};
}

std::string ComputeManager::getLastError() const
{
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return last_error_;
}

size_t ComputeManager::getAvailableMemoryMB() const
{
    {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        if (!initialized_ || !available_ || !config_.enable_gpu)
        {
            return 0;
        }
    }

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess)
    {
        return 0;
    }
    return free_bytes / kBytesPerMB;
}

void ComputeManager::setMemoryLimitMB(size_t limit)
{
    std::unique_lock<std::shared_mutex> lock(mutex_);
    config_.max_memory_mb = limit;
}

void ComputeManager::setUseOptimizedKernels(bool enable)
{
    std::unique_lock<std::shared_mutex> lock(mutex_);
    useOptimizedKernels_ = enable;
}

bool ComputeManager::getUseOptimizedKernels() const
{
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return useOptimizedKernels_;
}

void ComputeManager::runAutoTune()
{
    std::unique_lock<std::shared_mutex> lock(mutex_);
    if (!initialized_ || !available_ || !config_.enable_gpu)
    {
        last_error_ = "Auto-tuning unavailable: GPU not initialized";
        return;
    }

    constexpr size_t test_size = 10000;
    constexpr int iterations = 5;

    void *dev_buf = nullptr;
    cudaError_t alloc_err = cudaMalloc(&dev_buf, test_size * sizeof(double));
    if (alloc_err != cudaSuccess)
    {
        last_error_ = "Auto-tune allocation failed: " + std::string(cudaGetErrorString(alloc_err));
        return;
    }

    double best_time_ms = std::numeric_limits<double>::max();
    size_t best_streams = config_.num_streams;
    size_t best_chunk = config_.chunk_size;

    for (size_t streams : {1, 2, 4, 8})
    {
        for (size_t chunk : {1024, 4096, 8192, 16384})
        {
            const auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < iterations; ++i)
            {
                cudaMemset(dev_buf, 0, test_size * sizeof(double));
            }
            cudaDeviceSynchronize();
            const auto end = std::chrono::high_resolution_clock::now();
            const double elapsed_ms =
                std::chrono::duration<double, std::milli>(end - start).count() / iterations;

            if (elapsed_ms < best_time_ms)
            {
                best_time_ms = elapsed_ms;
                best_streams = streams;
                best_chunk = chunk;
            }
        }
    }

    cudaFree(dev_buf);

    config_.num_streams = best_streams;
    config_.chunk_size = best_chunk;
    last_error_.clear();
}

void ComputeManager::loadTunedConfiguration(const std::string &filename)
{
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto it = g_tuned_configs.find(filename);
    if (it != g_tuned_configs.end())
    {
        config_ = it->second;
        last_error_.clear();
        return;
    }
    last_error_ = "Tuning persistence not implemented: cannot load from '" + filename + "'";
}

void ComputeManager::saveTunedConfiguration(const std::string &filename)
{
    std::unique_lock<std::shared_mutex> lock(mutex_);
    g_tuned_configs[filename] = config_;
    last_error_.clear();
}

void ComputeManager::setUseCudaGraphs(bool enable)
{
    std::unique_lock<std::shared_mutex> lock(mutex_);
    useCudaGraphs_ = enable;
    if (!useCudaGraphs_)
    {
        graphManager_.reset();
    }
}

bool ComputeManager::getUseCudaGraphs() const
{
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return useCudaGraphs_;
}

std::string ComputeManager::getGPUArchitectureName() const
{
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return architectureName(gpuComputeCapability_);
}

int ComputeManager::getComputeCapability() const
{
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return gpuComputeCapability_;
}

void ComputeManager::updateStats(const PerformanceProfile &profile)
{
    const size_t previous_total = stats_.total_operations;
    stats_.total_operations = previous_total + 1;
    if (profile.used_gpu)
    {
        ++stats_.gpu_operations;
    }
    stats_.average_speedup =
        ((stats_.average_speedup * static_cast<double>(previous_total)) + profile.speedup) /
        static_cast<double>(stats_.total_operations);
    if (profile.cpu_time_ms > profile.gpu_time_ms)
    {
        stats_.total_time_saved_ms += profile.cpu_time_ms - profile.gpu_time_ms;
    }
}

void ComputeManager::recordSuccess(const std::string &operation, double speedup)
{
    std::unique_lock<std::shared_mutex> lock(mutex_);
    PerformanceProfile profile;
    profile.operation_name = operation;
    profile.speedup = speedup;
    profile.used_gpu = speedup > 1.0;
    profile.timestamp = std::chrono::system_clock::now();
    history_.push_back(profile);
    updateStats(profile);
    last_error_.clear();
}

void ComputeManager::recordFailure(const std::string &operation, const std::string &reason)
{
    std::unique_lock<std::shared_mutex> lock(mutex_);
    last_error_ = operation + ": " + reason;
}

} // namespace nerve::gpu

// Runtime-support helpers for Fast VR configuration and capability reporting.

#include "nerve/persistence/detail/fast_vr_accelerated_detail.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"

#include <algorithm>
#include <sstream>
#include <thread>

#ifdef NERVE_HAS_CUDA
#include <cuda_runtime_api.h>
#endif

namespace nerve::persistence
{

namespace
{

constexpr std::size_t SMALL_DATASET_THRESHOLD = 500;
#ifdef NERVE_HAS_CUDA
constexpr std::size_t BYTES_PER_MB = 1024ULL * 1024ULL;
#endif
constexpr double MIN_MEMORY_LIMIT_MB = 512.0;
constexpr double MAX_MEMORY_LIMIT_MB = 16384.0;
constexpr double BYTES_TO_MB_DIVISOR = 1024.0 * 1024.0;
constexpr int DEFAULT_MAX_DIMENSION = 2;

} // namespace

bool is_cuda_available()
{
#ifdef NERVE_HAS_CUDA
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    return status == cudaSuccess && device_count > 0;
#else
    return false;
#endif
}

bool isAdaptiveAccelerationAvailable()
{
    return false;
}

errors::ErrorResult<std::string> getAdaptiveAccelerationCapabilities()
{
    std::ostringstream out;
    out << "Adaptive Acceleration Capabilities:\n";
    out << "CUDA Available: " << (is_cuda_available() ? "Yes" : "No") << "\n";
    out << "CPU Cores: " << std::thread::hardware_concurrency() << "\n";
#ifdef NERVE_HAS_CUDA
    if (is_cuda_available())
    {
        cudaDeviceProp props{};
        if (cudaGetDeviceProperties(&props, 0) == cudaSuccess)
        {
            out << "GPU Memory: " << (props.totalGlobalMem / BYTES_PER_MB) << " MB\n";
            out << "Compute Capability: " << props.major << props.minor << "\n";
            out << "GPU Name: " << props.name << "\n";
        }
    }
#endif
    return errors::ErrorResult<std::string>::success(out.str());
}

VRConfig getOptimalFastvrConfig(Size num_points, Size point_dim)
{
    VRConfig config;
    config.max_dim = DEFAULT_MAX_DIMENSION;
    config.max_radius = 1.0;
    config.num_threads = 0;
    config.use_accelerated_runtime = false;
    config.auto_detect_accelerated_runtime = false;
    config.use_adaptive_acceleration = false;
    config.auto_detect_adaptive_acceleration = false;

    if (num_points < SMALL_DATASET_THRESHOLD)
    {
        config.algorithm = VRAlgorithmSelection::FAST_SIMD;
    }
    else
    {
        config.algorithm = VRAlgorithmSelection::EXACT_STANDARD;
    }
    config.memory_limit_mb = std::max(
        MIN_MEMORY_LIMIT_MB, std::min(MAX_MEMORY_LIMIT_MB, static_cast<double>(num_points) *
                                                               static_cast<double>(point_dim) *
                                                               8.0 / BYTES_TO_MB_DIVISOR));
    return config;
}

namespace accelerated::detail
{

void recordGlobalMetric(const PerformanceMetrics &metric, double memory_usage_mb, bool gpu_used,
                        bool hybrid_used)
{
    auto &state = globalPerformanceState();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.stats.problems_processed++;
    state.stats.total_time_ms += metric.total_time_ms;
    state.stats.cpu_time_ms += metric.cpu_time_ms;
    state.stats.gpu_time_ms += metric.gpu_time_ms;
    state.stats.memory_usage_mb = std::max(state.stats.memory_usage_mb, memory_usage_mb);
    state.stats.peak_memory_usage_mb = std::max(state.stats.peak_memory_usage_mb, memory_usage_mb);
    state.stats.gpu_used = state.stats.gpu_used || gpu_used;
    state.stats.hybrid_used = state.stats.hybrid_used || hybrid_used;
    state.stats.detailed_metrics.push_back(metric);
}

} // namespace accelerated::detail

} // namespace nerve::persistence

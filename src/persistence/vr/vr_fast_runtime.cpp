
#include "nerve/persistence/vr/vr_fast_ops.hpp"
#include "nerve/runtime/hardware_probe.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <thread>

namespace nerve::persistence::accelerated
{
namespace
{

size_t saturatedProduct(size_t lhs, size_t rhs)
{
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs)
    {
        return std::numeric_limits<size_t>::max();
    }
    return lhs * rhs;
}

size_t saturatedAdd(size_t lhs, size_t rhs)
{
    if (rhs > std::numeric_limits<size_t>::max() - lhs)
    {
        return std::numeric_limits<size_t>::max();
    }
    return lhs + rhs;
}

double finiteNonnegative(double value)
{
    return std::isfinite(value) && value >= 0.0 ? value : 0.0;
}

PerformanceMetrics sanitizeMetrics(PerformanceMetrics metric)
{
    metric.total_time_ms = finiteNonnegative(metric.total_time_ms);
    metric.gpu_time_ms = finiteNonnegative(metric.gpu_time_ms);
    metric.cpu_time_ms = finiteNonnegative(metric.cpu_time_ms);
    metric.total_time_ms = std::max(metric.total_time_ms, metric.gpu_time_ms + metric.cpu_time_ms);
    metric.max_radius = finiteNonnegative(metric.max_radius);
    metric.gpu_work_ratio = finiteNonnegative(metric.gpu_work_ratio);
    metric.problem_ops = finiteNonnegative(metric.problem_ops);
    metric.gpu_bytes = finiteNonnegative(metric.gpu_bytes);
    metric.gpu_stage_ops = finiteNonnegative(metric.gpu_stage_ops);
    return metric;
}

size_t completeGraphEdgeCount(size_t n_points)
{
    if (n_points < 2)
    {
        return 0;
    }
    if (n_points % 2 == 0)
    {
        return saturatedProduct(n_points / 2, n_points - 1);
    }
    return saturatedProduct(n_points, (n_points - 1) / 2);
}

} // namespace

SystemCapabilities GPUAccelerationManager::detectSystemCapabilities()
{
    return detail::detectSystemCapabilitiesImpl();
}

AccelerationMode
GPUAccelerationManager::recommendAccelerationMode(const ProblemCharacteristics &problem,
                                                  const SystemCapabilities &capabilities)
{
#ifdef NERVE_HAS_CUDA
    if (capabilities.cuda_device_count > 0)
    {
        size_t estimated_mem = problem.n_points * problem.point_dim * sizeof(double) * 3;
        size_t gpu_mem = capabilities.total_gpu_memory_bytes;
        if (estimated_mem < gpu_mem / 2 && problem.n_points >= 1000)
        {
            return AccelerationMode::GPU_ONLY;
        }
        if (estimated_mem < gpu_mem * 3 / 4 && problem.n_points >= 500)
        {
            return AccelerationMode::CPU_GPU_HYBRID;
        }
    }
#endif
    return AccelerationMode::CPU_ONLY;
}

bool GPUAccelerationManager::isGpuRuntimeAvailable()
{
    return detail::detectSystemCapabilitiesImpl().cuda_available;
}

errors::ErrorResult<std::unique_ptr<GPUAccelerationManager>>
GPUAccelerationManager::create(const VRConfig &config)
{
    return detail::makeDefaultGpuManager(config);
}

errors::ErrorResult<std::unique_ptr<PerformanceOptimizer>>
PerformanceOptimizer::create(const VRConfig &config)
{
    return detail::makeDefaultPerformanceOptimizer(config);
}

namespace utils
{

SystemCapabilities detectSystemCapabilities()
{
    return detail::detectSystemCapabilitiesImpl();
}

AccelerationMode recommendAccelerationMode(const ProblemCharacteristics &problem,
                                           const SystemCapabilities &capabilities)
{
    if (!capabilities.cuda_available)
    {
        return AccelerationMode::CPU_ONLY;
    }
    const size_t problem_size = saturatedProduct(problem.estimated_n_points, problem.point_dim);
    constexpr size_t large_problem_threshold = 10000;
    if (problem_size >= large_problem_threshold || problem.estimated_n_points >= 10000)
    {
        return AccelerationMode::HYBRID_GPU_PREFERRED;
    }
    if (problem_size >= large_problem_threshold / 2 || problem.estimated_n_points >= 1000)
    {
        return AccelerationMode::HYBRID_AUTO;
    }
    return AccelerationMode::CPU_ONLY;
}

size_t estimateMemoryRequirements(size_t n_points, size_t point_dim, size_t max_dim,
                                  const VRConfig & /*config*/)
{
    const size_t point_values = saturatedProduct(n_points, point_dim);
    const size_t point_bytes = saturatedProduct(point_values, sizeof(double));
    const size_t edge_count = completeGraphEdgeCount(n_points);
    const size_t distance_bytes = saturatedProduct(edge_count, sizeof(float));
    const size_t pair_count = saturatedProduct(edge_count, std::max<size_t>(1, max_dim));
    const size_t pair_bytes = saturatedProduct(pair_count, sizeof(Pair)) / 8;
    return saturatedAdd(saturatedAdd(point_bytes, distance_bytes), pair_bytes);
}

bool isAccelerationBeneficial(size_t n_points, size_t point_dim, double max_radius)
{
    if (n_points == 0 || point_dim == 0 || !std::isfinite(max_radius) || max_radius <= 0.0)
    {
        return false;
    }
    constexpr size_t threshold = 10000;
    const size_t problem_size = saturatedProduct(n_points, point_dim);
    if (problem_size < threshold)
    {
        return false;
    }
    try
    {
        return ::nerve::persistence::accelerated::detail::detectSystemCapabilitiesImpl()
            .cuda_available;
    }
    catch (...)
    {
        return false;
    }
}

VRConfig createOptimalConfigForProblem(size_t n_points, size_t point_dim, double max_radius,
                                       const VRConfig &base_config)
{
    if (!std::isfinite(max_radius) || max_radius <= 0.0)
    {
        throw std::invalid_argument("max_radius must be finite and greater than 0");
    }
    if (base_config.validate().isError())
    {
        throw std::invalid_argument("invalid base VRConfig");
    }
    VRConfig config = base_config.getEffectiveConfig();
    config.max_radius = max_radius;

    const auto capabilities = detectSystemCapabilities();
    config.acceleration.mode = AccelerationMode::CPU_ONLY;
    config.use_acceleration = false;
    if (config.num_threads == 0)
    {
        config.num_threads = std::max<size_t>(1, std::thread::hardware_concurrency());
    }

    const size_t estimated_bytes =
        estimateMemoryRequirements(n_points, point_dim, config.max_dim, config);
    if (capabilities.available_memory > 0 &&
        estimated_bytes > (capabilities.available_memory * 7) / 10)
    {
        config.acceleration.enable_streaming = true;
    }
    return config;
}

AcceleratedPerformanceStats estimatePerformance(size_t n_points, size_t point_dim,
                                                double max_radius, const VRConfig &config)
{
    if (n_points == 0 || point_dim == 0 || !std::isfinite(max_radius) || max_radius <= 0.0 ||
        config.validate().isError())
    {
        throw std::invalid_argument("invalid accelerated VR performance estimate parameters");
    }
    AcceleratedPerformanceStats stats;
    stats.problems_processed = 1;

    const VRConfig effective = config.getEffectiveConfig();
    const double ops = detail::estimateProblemOps(n_points, point_dim, effective.max_dim);
    (void)max_radius;

    const double cpu_ops_per_ms = 5.0e4;
    stats.cpu_time_ms = ops / cpu_ops_per_ms;
    stats.gpu_time_ms = 0.0;
    stats.total_time_ms = stats.cpu_time_ms;
    stats.speedup = 1.0;
    stats.average_speedup = stats.speedup;
    stats.gpu_used = false;
    stats.hybrid_used = false;
    stats.memory_usage_mb = detail::bytesToMb(
        estimateMemoryRequirements(n_points, point_dim, effective.max_dim, effective));
    stats.peak_memory_usage_mb = stats.memory_usage_mb;
    stats.gpu_utilization = 0.0;
    return stats;
}

errors::ErrorResult<void> validateAccelerationConfig(const VRConfig &config)
{
    return config.validate();
}

} // namespace utils

void PerformanceMonitor::startMonitoring(const std::string &operation)
{
    start_times_[operation] = std::chrono::steady_clock::now();
}

void PerformanceMonitor::endMonitoring(const std::string &operation)
{
    const auto it = start_times_.find(operation);
    if (it == start_times_.end())
    {
        return;
    }
    const auto end = std::chrono::steady_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(end - it->second).count();
    auto &metric = metrics_[operation];
    metric.total_time_ms = elapsed_ms;
    metric.cpu_time_ms = elapsed_ms;
    metric.timestamp = std::chrono::system_clock::now();
    start_times_.erase(it);
}

void PerformanceMonitor::recordMetrics(const std::string &operation,
                                       const PerformanceMetrics &metrics)
{
    metrics_[operation] = metrics;
}

AcceleratedPerformanceStats PerformanceMonitor::getAggregatedStats() const
{
    AcceleratedPerformanceStats stats;
    if (metrics_.empty())
    {
        return stats;
    }

    stats.problems_processed = metrics_.size();
    double speedup_sum = 0.0;
    for (const auto &entry : metrics_)
    {
        const auto metric = sanitizeMetrics(entry.second);
        stats.total_time_ms += metric.total_time_ms;
        stats.gpu_time_ms += metric.gpu_time_ms;
        stats.cpu_time_ms += metric.cpu_time_ms;
        stats.gpu_used = stats.gpu_used || metric.gpu_time_ms > 0.0 ||
                         metric.execution_mode == ExecutionMode::GPU_ONLY;
        stats.hybrid_used =
            stats.hybrid_used || (metric.execution_mode == ExecutionMode::HYBRID_GPU_PREFERRED ||
                                  metric.execution_mode == ExecutionMode::HYBRID_CPU_PREFERRED ||
                                  metric.execution_mode == ExecutionMode::HYBRID_AUTO);
        stats.gpu_stage_ops += std::max(0.0, metric.gpu_stage_ops);
        stats.kernel_diagnostics.dropped_invalid_distances +=
            metric.kernel_diagnostics.dropped_invalid_distances;
        stats.kernel_diagnostics.invalid_distance_inputs +=
            metric.kernel_diagnostics.invalid_distance_inputs;
        stats.kernel_diagnostics.dimension_tiles_processed +=
            metric.kernel_diagnostics.dimension_tiles_processed;
        stats.kernel_diagnostics.pivot_hits += metric.kernel_diagnostics.pivot_hits;
        speedup_sum += metric.getSpeedup();
        stats.detailed_metrics.push_back(metric);
    }
    const double count = static_cast<double>(metrics_.size());
    stats.average_speedup = speedup_sum / count;
    stats.speedup = stats.average_speedup;
    stats.gpu_utilization =
        stats.total_time_ms > 0.0 ? stats.gpu_time_ms / stats.total_time_ms : 0.0;
    return stats;
}

void PerformanceMonitor::reset()
{
    metrics_.clear();
    start_times_.clear();
}

errors::ErrorResult<std::unique_ptr<AcceleratedVREngine>>
AcceleratedVREngine::create(const VRConfig &config)
{
    const auto validation = config.validate();
    if (validation.isError())
    {
        return errors::ErrorResult<std::unique_ptr<AcceleratedVREngine>>::error(
            validation.errorCode());
    }
    return detail::makeDefaultAcceleratedEngine(config);
}

} // namespace nerve::persistence::accelerated

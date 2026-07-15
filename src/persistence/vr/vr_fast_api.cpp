
#include "nerve/persistence/detail/fast_vr_accelerated_detail.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <limits>
#include <utility>

namespace nerve::persistence::accelerated
{

errors::ErrorResult<std::vector<Pair>>
computeVrPersistenceAccelerated(core::BufferView<const double> points, Size point_dim,
                                const VRConfig &config)
{
    auto engine_result = AcceleratedVREngine::create(config);
    if (engine_result.isError())
    {
        return errors::ErrorResult<std::vector<Pair>>::error(engine_result.errorCode());
    }
    auto engine = std::move(engine_result.value());
    return engine->computeVrPersistence(points, point_dim, config);
}

errors::ErrorResult<std::vector<Pair>>
computeVrPersistenceFast(core::BufferView<const double> points, Size point_dim,
                         const ::nerve::persistence::VRConfig &config)
{
    const ::nerve::persistence::VRConfig effective = config.getEffectiveConfig();
    const auto validation = effective.validate();
    if (validation.isError())
    {
        return errors::ErrorResult<std::vector<Pair>>::error(validation.errorCode());
    }
    if (!effective.use_acceleration && effective.acceleration.mode == AccelerationMode::CPU_ONLY)
    {
        if (point_dim == 0 || points.empty() || (points.size() % point_dim) != 0)
        {
            return errors::ErrorResult<std::vector<Pair>>::error(errors::ErrorCode::E50_PH_ABORT);
        }

        const auto start = std::chrono::steady_clock::now();
        auto pairs = ::nerve::persistence::computeVrPersistenceFast(points, point_dim, effective);
        const auto end = std::chrono::steady_clock::now();

        PerformanceMetrics metric;
        metric.total_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        metric.cpu_time_ms = metric.total_time_ms;
        metric.problem_size = points.size() / point_dim;
        metric.point_dim = point_dim;
        metric.max_radius = effective.max_radius;
        metric.max_dim = effective.max_dim;
        metric.execution_mode = ExecutionMode::CPU_ONLY;
        metric.success = true;
        metric.result_size = pairs.size();
        metric.error_code = errors::ErrorCode::SUCCESS;
        metric.timestamp = std::chrono::system_clock::now();
        metric.problem_ops =
            detail::estimateProblemOps(metric.problem_size, metric.point_dim, metric.max_dim);
        metric.gpu_bytes = 0.0;

        const double memory_usage_mb =
            detail::bytesToMb(points.size() * sizeof(double) + pairs.size() * sizeof(Pair));
        detail::recordGlobalMetric(metric, memory_usage_mb, false, false);
        return errors::ErrorResult<std::vector<Pair>>::success(std::move(pairs));
    }

    return computeVrPersistenceAccelerated(points, point_dim, effective);
}

::nerve::persistence::VRConfig
createOptimalConfig(core::BufferView<const double> points, Size point_dim,
                    const ::nerve::persistence::VRConfig &base_config)
{
    if (point_dim == 0)
    {
        ::nerve::persistence::VRConfig config = base_config.getEffectiveConfig();
        config.acceleration.mode = AccelerationMode::CPU_ONLY;
        config.use_acceleration = false;
        return config;
    }
    return utils::createOptimalConfigForProblem(points.size() / point_dim, point_dim,
                                                base_config.max_radius, base_config);
}

namespace performance
{

AcceleratedPerformanceStats getCurrentPerformanceStats()
{
    auto &state = detail::globalPerformanceState();
    std::lock_guard<std::mutex> lock(state.mutex);
    return state.stats;
}

errors::ErrorResult<void> exportPerformanceReport(const std::string &filename,
                                                  const AcceleratedPerformanceStats &stats)
{
    std::ofstream output(filename);
    if (!output.is_open())
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E00_IO_TIMEOUT);
    }

    output << "# Nerve Accelerated Performance Report\n";
    output << "problems_processed=" << stats.problems_processed << "\n";
    output << "total_time_ms=" << stats.total_time_ms << "\n";
    output << "cpu_time_ms=" << stats.cpu_time_ms << "\n";
    output << "gpu_time_ms=" << stats.gpu_time_ms << "\n";
    output << "memory_usage_mb=" << stats.memory_usage_mb << "\n";
    output << "peak_memory_usage_mb=" << stats.peak_memory_usage_mb << "\n";
    output << "speedup=" << stats.speedup << "\n";
    output << "average_speedup=" << stats.average_speedup << "\n";
    output << "gpu_used=" << (stats.gpu_used ? 1 : 0) << "\n";
    output << "hybrid_used=" << (stats.hybrid_used ? 1 : 0) << "\n";
    output << "dropped_invalid_distances=" << stats.kernel_diagnostics.dropped_invalid_distances
           << "\n";
    output << "invalid_distance_inputs=" << stats.kernel_diagnostics.invalid_distance_inputs
           << "\n";
    output << "dimension_tiles_processed=" << stats.kernel_diagnostics.dimension_tiles_processed
           << "\n";
    output << "pivot_hits=" << stats.kernel_diagnostics.pivot_hits << "\n";
    output << "detailed_metrics=" << stats.detailed_metrics.size() << "\n";

    if (!output.good())
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E00_IO_TIMEOUT);
    }
    return errors::ErrorResult<void>::success();
}

} // namespace performance

} // namespace nerve::persistence::accelerated

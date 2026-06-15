
#include "nerve/common/accelerated_types.hpp"
#include "nerve/errors/errors.hpp"
#include "nerve/persistence/accelerated/work_distributor.hpp"

#include <cmath>

namespace nerve::persistence::accelerated::accelerated_error_tools
{

using ::nerve::common::AcceleratedPerformanceStats;

errors::ErrorResult<void> validateDistribution(const WorkDistribution &distribution,
                                               size_t total_columns)
{
    if (distribution.gpuColumns + distribution.cpuColumns != total_columns)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E50_PH_ABORT);
    }
    if (distribution.gpuColumns > total_columns || distribution.cpuColumns > total_columns)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E50_PH_ABORT);
    }
    return errors::ErrorResult<void>::success();
}

errors::ErrorResult<void> validateMetrics(const AcceleratedPerformanceStats &stats)
{
    const bool finite_metrics =
        std::isfinite(stats.total_time_ms) && std::isfinite(stats.gpu_time_ms) &&
        std::isfinite(stats.cpu_time_ms) && std::isfinite(stats.memory_usage_mb) &&
        std::isfinite(stats.gpu_utilization) && std::isfinite(stats.speedup) &&
        std::isfinite(stats.average_speedup) && std::isfinite(stats.peak_memory_usage_mb) &&
        std::isfinite(stats.gpu_stage_ops);
    if (!finite_metrics)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E20_NUM_NAN);
    }
    for (const auto &metric : stats.detailed_metrics)
    {
        const bool finite_detail =
            std::isfinite(metric.total_time_ms) && std::isfinite(metric.gpu_time_ms) &&
            std::isfinite(metric.cpu_time_ms) && std::isfinite(metric.max_radius) &&
            std::isfinite(metric.gpu_work_ratio) && std::isfinite(metric.problem_ops) &&
            std::isfinite(metric.gpu_bytes) && std::isfinite(metric.gpu_stage_ops);
        if (!finite_detail)
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E20_NUM_NAN);
        }
        if (metric.total_time_ms < 0.0 || metric.gpu_time_ms < 0.0 || metric.cpu_time_ms < 0.0 ||
            metric.max_radius < 0.0 || metric.gpu_work_ratio < 0.0 || metric.problem_ops < 0.0 ||
            metric.gpu_bytes < 0.0 || metric.gpu_stage_ops < 0.0)
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E50_PH_ABORT);
        }
    }
    if (stats.total_time_ms < 0.0 || stats.gpu_time_ms < 0.0 || stats.cpu_time_ms < 0.0 ||
        stats.memory_usage_mb < 0.0 || stats.gpu_utilization < 0.0 || stats.speedup < 0.0 ||
        stats.average_speedup < 0.0 || stats.peak_memory_usage_mb < 0.0 ||
        stats.gpu_stage_ops < 0.0)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E50_PH_ABORT);
    }
    if (stats.gpu_time_ms + stats.cpu_time_ms > stats.total_time_ms * 1.05)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E50_PH_ABORT);
    }
    return errors::ErrorResult<void>::success();
}

errors::ErrorResult<void> validatePairs(const std::vector<Pair> &pairs)
{
    for (const auto &pair : pairs)
    {
        const bool finite_death = std::isfinite(pair.death);
        const bool infinite_death = pair.isInfinite();
        if (!std::isfinite(pair.birth) || (!finite_death && !infinite_death))
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E20_NUM_NAN);
        }
        if (pair.dimension < 0 || (finite_death && pair.death < pair.birth))
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E50_PH_ABORT);
        }
    }
    return errors::ErrorResult<void>::success();
}

} // namespace nerve::persistence::accelerated::accelerated_error_tools

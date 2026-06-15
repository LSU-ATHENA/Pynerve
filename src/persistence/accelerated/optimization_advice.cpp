
#include "nerve/persistence/accelerated/accelerated_api.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace nerve::persistence::accelerated::optimization_recommendations
{

std::vector<std::string> suggestActions(const PerformanceMetrics &metrics)
{
    std::vector<std::string> recommendations;
    const bool finite_metrics =
        std::isfinite(metrics.total_time_ms) && std::isfinite(metrics.gpu_time_ms) &&
        std::isfinite(metrics.cpu_time_ms) && std::isfinite(metrics.gpu_bytes);
    if (!finite_metrics || metrics.total_time_ms < 0.0 || metrics.gpu_time_ms < 0.0 ||
        metrics.cpu_time_ms < 0.0 || metrics.gpu_bytes < 0.0)
    {
        recommendations.emplace_back(
            "Collect finite non-negative performance metrics before tuning.");
        return recommendations;
    }
    const bool gpu_used = metrics.gpu_available || metrics.gpu_compute_stage_executed;
    const double gpu_utilization =
        (metrics.total_time_ms > 0.0) ? (metrics.gpu_time_ms / metrics.total_time_ms) : 0.0;
    const double overhead_ratio =
        (metrics.total_time_ms > 0.0)
            ? std::max(0.0, metrics.total_time_ms - metrics.gpu_time_ms - metrics.cpu_time_ms) /
                  metrics.total_time_ms
            : 0.0;
    const double memory_usage_mb = metrics.gpu_bytes / (1024.0 * 1024.0);

    if (gpu_used && gpu_utilization < 0.65)
    {
        recommendations.emplace_back("Increase GPU batch size or reduce kernel launch overhead.");
    }
    if (overhead_ratio > 0.25)
    {
        recommendations.emplace_back("Reduce transfer/synchronization overhead with fused stages.");
    }
    if (memory_usage_mb > 4096.0)
    {
        recommendations.emplace_back("Enable chunked processing to reduce peak memory.");
    }
    if (recommendations.empty())
    {
        recommendations.emplace_back("No immediate tuning changes required.");
    }
    return recommendations;
}

} // namespace nerve::persistence::accelerated::optimization_recommendations

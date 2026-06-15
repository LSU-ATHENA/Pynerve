
#include "nerve/persistence/accelerated/accelerated_api.hpp"

#include <cmath>

namespace nerve::persistence::accelerated::performance_impact
{

double computeRuntimeChange(const PerformanceMetrics &baseline, const PerformanceMetrics &current)
{
    if (!std::isfinite(baseline.total_time_ms) || !std::isfinite(current.total_time_ms) ||
        baseline.total_time_ms <= 0.0 || current.total_time_ms < 0.0)
    {
        return 0.0;
    }
    return (current.total_time_ms - baseline.total_time_ms) / baseline.total_time_ms;
}

double computeMemoryChange(const PerformanceMetrics &baseline, const PerformanceMetrics &current)
{
    if (!std::isfinite(baseline.gpu_bytes) || !std::isfinite(current.gpu_bytes) ||
        baseline.gpu_bytes <= 0.0 || current.gpu_bytes < 0.0)
    {
        return 0.0;
    }
    return (current.gpu_bytes - baseline.gpu_bytes) / baseline.gpu_bytes;
}

double computeOverallImpactScore(const PerformanceMetrics &baseline,
                                 const PerformanceMetrics &current)
{
    const double runtime_component = 1.0 - computeRuntimeChange(baseline, current);
    const double memory_component = 1.0 - computeMemoryChange(baseline, current);
    const double score = 0.7 * runtime_component + 0.3 * memory_component;
    return std::isfinite(score) ? score : 0.0;
}

} // namespace nerve::persistence::accelerated::performance_impact

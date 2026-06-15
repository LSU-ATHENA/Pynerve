
#include "nerve/persistence/accelerated/accelerated_api.hpp"

#include <mutex>

namespace nerve::persistence::accelerated::performance
{

namespace
{
std::mutex g_metrics_mutex;
PerformanceMetrics g_last_metrics{};
} // namespace

void updateLastMetrics(const PerformanceMetrics &metrics)
{
    std::lock_guard<std::mutex> lock(g_metrics_mutex);
    g_last_metrics = metrics;
}

PerformanceMetrics getLastMetrics()
{
    std::lock_guard<std::mutex> lock(g_metrics_mutex);
    return g_last_metrics;
}

double estimateGpuSpeedupRatio()
{
    std::lock_guard<std::mutex> lock(g_metrics_mutex);
    return g_last_metrics.getSpeedup();
}

} // namespace nerve::persistence::accelerated::performance


#include "nerve/common/accelerated_types.hpp"
#include "nerve/error/error_handling.hpp"
#include "nerve/errors/errors.hpp"
#include "nerve/persistence/accelerated/work_distributor.hpp"
#include "nerve/persistence/core/core_types.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <random>
#include <vector>

namespace
{

using nerve::Dimension;
using nerve::Field;
using nerve::common::AcceleratedPerformanceStats;
using nerve::common::PerformanceMetrics;
using nerve::error::CircuitBreaker;
using nerve::error::CircuitBreakerConfig;
using nerve::error::ErrorCode;
using nerve::error::ErrorEvent;
using nerve::error::ErrorHandlingManager;
using nerve::error::ErrorObservability;
using nerve::error::RetryBackoffManager;
using nerve::optimization::CallContract;
using nerve::persistence::Pair;
using nerve::persistence::accelerated::WorkDistribution;
using nerve::persistence::accelerated::accelerated_error_tools::validateDistribution;
using nerve::persistence::accelerated::accelerated_error_tools::validateMetrics;
using nerve::persistence::accelerated::accelerated_error_tools::validatePairs;

bool check_validate_distribution_valid()
{
    WorkDistribution dist(60, 40, true);
    auto result = validateDistribution(dist, 100);
    if (result.isError())
    {
        std::cerr << "valid distribution flagged as error\n";
        return false;
    }
    return true;
}

bool check_validate_distribution_mismatch()
{
    WorkDistribution dist(60, 30, true);
    auto result = validateDistribution(dist, 100);
    if (!result.isError())
        return false;
    return true;
}

bool check_validate_distribution_oob()
{
    WorkDistribution dist(200, 50, true);
    auto result = validateDistribution(dist, 100);
    if (!result.isError())
        return false;
    return true;
}

bool check_validate_metrics_valid()
{
    AcceleratedPerformanceStats stats;
    stats.total_time_ms = 100.0;
    stats.gpu_time_ms = 30.0;
    stats.cpu_time_ms = 70.0;
    stats.memory_usage_mb = 256.0;
    stats.gpu_utilization = 0.75;
    stats.speedup = 2.5;
    stats.average_speedup = 2.3;
    stats.peak_memory_usage_mb = 512.0;
    stats.gpu_stage_ops = 1.0;
    auto result = validateMetrics(stats);
    if (result.isError())
        return false;
    return true;
}

bool check_validate_metrics_nan_rejected()
{
    AcceleratedPerformanceStats stats;
    stats.total_time_ms = std::numeric_limits<double>::quiet_NaN();
    stats.gpu_time_ms = 0.0;
    stats.cpu_time_ms = 0.0;
    stats.memory_usage_mb = 0.0;
    stats.gpu_utilization = 0.0;
    stats.speedup = 0.0;
    stats.average_speedup = 0.0;
    stats.peak_memory_usage_mb = 0.0;
    stats.gpu_stage_ops = 0.0;
    auto result = validateMetrics(stats);
    if (!result.isError())
        return false;
    return true;
}

bool check_validate_metrics_time_mismatch()
{
    AcceleratedPerformanceStats stats;
    stats.total_time_ms = 10.0;
    stats.gpu_time_ms = 100.0;
    stats.cpu_time_ms = 100.0;
    stats.memory_usage_mb = 0.0;
    stats.gpu_utilization = 0.0;
    stats.speedup = 1.0;
    stats.average_speedup = 1.0;
    stats.peak_memory_usage_mb = 0.0;
    stats.gpu_stage_ops = 0.0;
    auto result = validateMetrics(stats);
    if (!result.isError())
        return false;
    return true;
}

bool check_validate_metrics_negative_rejected()
{
    AcceleratedPerformanceStats stats;
    stats.total_time_ms = -1.0;
    stats.gpu_time_ms = 0.0;
    stats.cpu_time_ms = 0.0;
    stats.memory_usage_mb = 0.0;
    stats.gpu_utilization = 0.0;
    stats.speedup = 1.0;
    stats.average_speedup = 1.0;
    stats.peak_memory_usage_mb = 0.0;
    stats.gpu_stage_ops = 0.0;
    auto result = validateMetrics(stats);
    if (!result.isError())
        return false;
    return true;
}

bool check_validate_pairs_valid()
{
    std::vector<Pair> pairs;
    pairs.push_back({0.0, 1.0, 0});
    pairs.push_back({0.5, 2.0, 1});
    Pair inf_pair;
    inf_pair.birth = 0.0;
    inf_pair.death = std::numeric_limits<Field>::infinity();
    inf_pair.dimension = 0;
    pairs.push_back(inf_pair);
    auto result = validatePairs(pairs);
    if (result.isError())
        return false;
    return true;
}

bool check_validate_pairs_invalid_birth()
{
    std::vector<Pair> pairs;
    Pair p;
    p.birth = std::numeric_limits<Field>::quiet_NaN();
    p.death = 1.0;
    p.dimension = 0;
    pairs.push_back(p);
    auto result = validatePairs(pairs);
    if (!result.isError())
        return false;
    return true;
}

bool check_validate_pairs_death_less_than_birth()
{
    std::vector<Pair> pairs;
    Pair p;
    p.birth = 5.0;
    p.death = 2.0;
    p.dimension = 0;
    pairs.push_back(p);
    auto result = validatePairs(pairs);
    if (!result.isError())
        return false;
    return true;
}

bool check_circuit_breaker_construction()
{
    CircuitBreakerConfig cfg;
    CircuitBreaker cb(cfg);
    if (cb.getState() != CircuitBreaker::State::CLOSED)
        return false;
    if (cb.isTripped())
        return false;
    return true;
}

bool check_circuit_breaker_operations()
{
    CircuitBreakerConfig cfg;
    cfg.failure_threshold = 3;
    CircuitBreaker cb(cfg);
    if (!cb.shouldAllowOperation("op1"))
        return false;
    cb.recordFailure("op1", ErrorCode::GPU_OOM);
    if (!cb.shouldAllowOperation("op1"))
        return false;
    cb.recordFailure("op1", ErrorCode::GPU_OOM);
    if (!cb.shouldAllowOperation("op1"))
        return false;
    cb.recordFailure("op1", ErrorCode::GPU_OOM);
    if (cb.shouldAllowOperation("op1"))
        return false;
    if (cb.getState() != CircuitBreaker::State::OPEN)
        return false;
    cb.reset();
    if (cb.getState() != CircuitBreaker::State::CLOSED)
        return false;
    return true;
}

bool check_error_observability_construction()
{
    ErrorObservability::ObservabilityConfig cfg;
    ErrorObservability obs(cfg);
    auto stats = obs.getStats();
    if (stats.total_error_events != 0)
        return false;
    return true;
}

bool check_error_observability_logging()
{
    ErrorObservability::ObservabilityConfig cfg;
    cfg.enable_structured_logging = false;
    cfg.enable_metric_increment = false;
    cfg.enable_error_histograms = false;
    ErrorObservability obs(cfg);
    ErrorEvent event;
    event.error_code = ErrorCode::GPU_OOM;
    event.operation_name = "test_op";
    event.duration_ms = 1.0;
    obs.logErrorEvent(event);
    auto stats = obs.getStats();
    if (stats.total_error_events != 1)
        return false;
    return true;
}

bool check_retry_backoff_construction()
{
    RetryBackoffManager::RetryConfig cfg;
    RetryBackoffManager mgr(cfg);
    auto stats = mgr.getStats();
    if (stats.total_retries != 0)
        return false;
    if (!mgr.isTransientError(ErrorCode::GPU_OOM))
        return false;
    if (mgr.isPermanentError(ErrorCode::GPU_OOM))
        return false;
    return true;
}

bool check_error_handling_manager()
{
    auto &mgr = ErrorHandlingManager::instance();
    auto stats = mgr.getStats();
    static_cast<void>(stats);
    if (!mgr.isHealthy())
        return false;
    return true;
}

} // namespace

int main()
{
    if (!check_validate_distribution_valid())
    {
        std::cerr << "FAIL: validate distribution valid\n";
        return 1;
    }
    if (!check_validate_distribution_mismatch())
    {
        std::cerr << "FAIL: validate distribution mismatch\n";
        return 1;
    }
    if (!check_validate_distribution_oob())
    {
        std::cerr << "FAIL: validate distribution oob\n";
        return 1;
    }
    if (!check_validate_metrics_valid())
    {
        std::cerr << "FAIL: validate metrics valid\n";
        return 1;
    }
    if (!check_validate_metrics_nan_rejected())
    {
        std::cerr << "FAIL: validate metrics nan rejected\n";
        return 1;
    }
    if (!check_validate_metrics_time_mismatch())
    {
        std::cerr << "FAIL: validate metrics time mismatch\n";
        return 1;
    }
    if (!check_validate_metrics_negative_rejected())
    {
        std::cerr << "FAIL: validate metrics negative rejected\n";
        return 1;
    }
    if (!check_validate_pairs_valid())
    {
        std::cerr << "FAIL: validate pairs valid\n";
        return 1;
    }
    if (!check_validate_pairs_invalid_birth())
    {
        std::cerr << "FAIL: validate pairs invalid birth\n";
        return 1;
    }
    if (!check_validate_pairs_death_less_than_birth())
    {
        std::cerr << "FAIL: validate pairs death less than birth\n";
        return 1;
    }
    if (!check_circuit_breaker_construction())
    {
        std::cerr << "FAIL: circuit breaker construction\n";
        return 1;
    }
    if (!check_circuit_breaker_operations())
    {
        std::cerr << "FAIL: circuit breaker operations\n";
        return 1;
    }
    if (!check_error_observability_construction())
    {
        std::cerr << "FAIL: error observability construction\n";
        return 1;
    }
    if (!check_error_observability_logging())
    {
        std::cerr << "FAIL: error observability logging\n";
        return 1;
    }
    if (!check_retry_backoff_construction())
    {
        std::cerr << "FAIL: retry backoff construction\n";
        return 1;
    }
    if (!check_error_handling_manager())
    {
        std::cerr << "FAIL: error handling manager\n";
        return 1;
    }
    return 0;
}

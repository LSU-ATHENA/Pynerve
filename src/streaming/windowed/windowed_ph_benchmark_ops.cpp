
#include "nerve/streaming/windowed_ph.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>

namespace nerve::streaming
{

StreamingPHPerformanceHarness::StreamingPHPerformanceHarness(const BenchmarkConfig &config)
    : config_(config)
{}

StreamingPHPerformanceHarness::StreamingPHPerformanceHarness()
    : StreamingPHPerformanceHarness(BenchmarkConfig{})
{}

StreamingPHPerformanceHarness::BenchmarkResults StreamingPHPerformanceHarness::benchmarkWindowedPh(
    const AcceleratedWindowedPH::OptimizationConfig &config)
{
    BenchmarkResults results{};
    AcceleratedWindowedPH ph(config);

    for (std::size_t i = 0; i < config_.num_iterations; ++i)
    {
        const std::size_t window_size = config_.window_sizes[i % config_.window_sizes.size()];
        ph.setWindowSize(window_size);
        const auto simplices = generateTestComplex(std::max<std::size_t>(8, window_size / 8));
        const double elapsed_ms = measureWindowUpdateTime(ph, simplices);
        results.update_times_ms.push_back(elapsed_ms);
        results.memory_usage_mb.push_back(measureMemoryUsage(ph));
        results.hot_path_violations.push_back(!ph.checkHotPathInvariants());
    }

    if (!results.update_times_ms.empty())
    {
        const double sum =
            std::accumulate(results.update_times_ms.begin(), results.update_times_ms.end(), 0.0);
        results.average_update_time_ms = sum / static_cast<double>(results.update_times_ms.size());
        results.max_update_time_ms =
            *std::max_element(results.update_times_ms.begin(), results.update_times_ms.end());
    }
    if (!results.memory_usage_mb.empty())
    {
        const double avg_memory =
            std::accumulate(results.memory_usage_mb.begin(), results.memory_usage_mb.end(), 0.0) /
            static_cast<double>(results.memory_usage_mb.size());
        results.memory_efficiency_score = avg_memory > 0.0 ? (1.0 / avg_memory) : 1.0;
    }
    results.meets_performance_targets = validatePerformanceTargets(results);
    return results;
}

StreamingPHPerformanceHarness::BenchmarkResults
StreamingPHPerformanceHarness::benchmarkWithBaseline(
    const AcceleratedWindowedPH::OptimizationConfig &config)
{
    AcceleratedWindowedPH::OptimizationConfig baseline_config = config;
    baseline_config.enable_numa_optimization = false;
    baseline_config.enable_parallel_processing = false;
    baseline_config.enable_batch_processing = false;

    BenchmarkResults baseline = benchmarkWindowedPh(baseline_config);
    BenchmarkResults accelerated = benchmarkWindowedPh(config);
    if (!baseline.update_times_ms.empty() && !accelerated.update_times_ms.empty())
    {
        const double baseline_avg = baseline.average_update_time_ms;
        const double accelerated_avg = accelerated.average_update_time_ms;
        if (accelerated_avg > 0.0)
        {
            accelerated.memory_efficiency_score = baseline_avg / accelerated_avg;
        }
    }
    return accelerated;
}

bool StreamingPHPerformanceHarness::validateHotPathInvariants(const BenchmarkResults &results)
{
    return std::none_of(results.hot_path_violations.begin(), results.hot_path_violations.end(),
                        [](bool violated) { return violated; });
}

bool StreamingPHPerformanceHarness::validateMemoryEfficiency(const BenchmarkResults &results)
{
    return results.memory_efficiency_score > 0.0 && std::isfinite(results.memory_efficiency_score);
}

bool StreamingPHPerformanceHarness::validatePerformanceTargets(const BenchmarkResults &results)
{
    if (results.update_times_ms.empty())
    {
        return true;
    }
    return results.average_update_time_ms <= config_.performance_target_ms &&
           validateHotPathInvariants(results);
}

StreamingPHPerformanceHarness::BenchmarkResults StreamingPHPerformanceHarness::compareBeforeAfter(
    const AcceleratedWindowedPH::OptimizationConfig &before_config,
    const AcceleratedWindowedPH::OptimizationConfig &after_config)
{
    BenchmarkResults before = benchmarkWindowedPh(before_config);
    BenchmarkResults after = benchmarkWindowedPh(after_config);
    if (after.average_update_time_ms > 0.0 && before.average_update_time_ms > 0.0)
    {
        after.memory_efficiency_score =
            before.average_update_time_ms / after.average_update_time_ms;
    }
    return after;
}

std::vector<algebra::Simplex> StreamingPHPerformanceHarness::generateTestComplex(std::size_t size)
{
    std::vector<algebra::Simplex> simplices;
    simplices.reserve(size * 2);
    for (std::size_t i = 0; i < size; ++i)
    {
        simplices.emplace_back(std::vector<Index>{static_cast<Index>(i)});
        if (i > 0)
        {
            simplices.emplace_back(
                std::vector<Index>{static_cast<Index>(i - 1), static_cast<Index>(i)});
        }
    }
    return simplices;
}

std::vector<std::vector<algebra::Simplex>>
StreamingPHPerformanceHarness::generateWindowSequence(std::size_t window_size,
                                                      std::size_t num_windows)
{
    std::vector<std::vector<algebra::Simplex>> windows;
    windows.reserve(num_windows);
    for (std::size_t i = 0; i < num_windows; ++i)
    {
        windows.push_back(generateTestComplex(window_size + i));
    }
    return windows;
}

double StreamingPHPerformanceHarness::measureWindowUpdateTime(
    AcceleratedWindowedPH &ph, const std::vector<algebra::Simplex> &simplices)
{
    const auto start = std::chrono::high_resolution_clock::now();
    ph.addSimplicesBatch(simplices);
    const auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

double StreamingPHPerformanceHarness::measureMemoryUsage(AcceleratedWindowedPH &ph)
{
    return static_cast<double>(ph.getMemoryUsage()) / (1024.0 * 1024.0);
}

} // namespace nerve::streaming

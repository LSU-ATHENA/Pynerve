#include "nerve/persistence/utils/incremental_updates.hpp"
#include "nerve/validation/ph5_ph6_microbenchmarks.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <string>
#include <utility>
#include <vector>
namespace nerve::validation
{
namespace
{
using Engine = nerve::persistence::PH5PH6Engine<std::vector<double>, double>;
constexpr std::size_t kBytesPerMb = 1024ULL * 1024ULL;
PH5PH6MicrobenchmarkResult makeBaseResult(std::string benchmark_name, std::string algorithm_type,
                                          const std::size_t point_count,
                                          const std::size_t max_dimension)
{
    PH5PH6MicrobenchmarkResult result{};
    result.benchmark_name = std::move(benchmark_name);
    result.algorithm_type = std::move(algorithm_type);
    result.point_count = point_count;
    result.max_dimension = max_dimension;
    result.success = false;
    result.mean_runtime_ms = 0.0;
    result.p50_runtime_ms = 0.0;
    result.p95_runtime_ms = 0.0;
    result.p99_runtime_ms = 0.0;
    result.peak_memory_mb = 0;
    result.num_simplices = 0;
    result.failure_rate = 0.0;
    result.stability_score = 0.0;
    result.condition_estimate = 1.0;
    result.precision_events = 0;
    result.start_time = std::chrono::steady_clock::now();
    result.end_time = result.start_time;
    return result;
}
std::size_t bytesToMb(const std::size_t bytes)
{
    if (bytes == 0)
    {
        return 0;
    }
    return (bytes + kBytesPerMb - 1ULL) / kBytesPerMb;
}
std::vector<double> asDoubleVector(const std::vector<std::size_t> &values)
{
    std::vector<double> out;
    out.reserve(values.size());
    for (const std::size_t value : values)
    {
        out.push_back(static_cast<double>(value));
    }
    return out;
}
double localMean(const std::vector<double> &values)
{
    if (values.empty())
    {
        return 0.0;
    }
    return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}
double localPercentile(const std::vector<double> &values, const double percentile)
{
    if (values.empty())
    {
        return 0.0;
    }
    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    const std::size_t index =
        static_cast<std::size_t>((percentile / 100.0) * static_cast<double>(sorted.size() - 1U));
    return sorted[index];
}
double finiteConditionEstimate(const double p99_runtime_ms, const double p50_runtime_ms)
{
    if (!std::isfinite(p99_runtime_ms) || !std::isfinite(p50_runtime_ms) || p99_runtime_ms < 0.0 ||
        p50_runtime_ms < 0.0)
    {
        return 1.0;
    }
    const double estimate = p99_runtime_ms / std::max(1e-9, p50_runtime_ms);
    if (!std::isfinite(estimate) || estimate < 1.0)
    {
        return 1.0;
    }
    return estimate;
}
void fillRuntimeStats(PH5PH6MicrobenchmarkResult &result, const std::vector<double> &runtimes_ms)
{
    if (runtimes_ms.empty())
    {
        return;
    }
    result.mean_runtime_ms = localMean(runtimes_ms);
    result.p50_runtime_ms = localPercentile(runtimes_ms, 50.0);
    result.p95_runtime_ms = localPercentile(runtimes_ms, 95.0);
    result.p99_runtime_ms = localPercentile(runtimes_ms, 99.0);
}
} // namespace
PH5PH6Microbenchmark::PH5PH6Microbenchmark(const PH5PH6MicrobenchmarkConfig &config)
    : config_(config)
{}
std::vector<PH5PH6MicrobenchmarkResult> PH5PH6Microbenchmark::runAllBenchmarks()
{
    std::vector<PH5PH6MicrobenchmarkResult> results;
    if (config_.enable_detailed_logging)
    {
        std::cout << "Starting PH5/PH6 microbenchmark suite...\n";
    }
    for (const std::size_t point_count : config_.point_counts)
    {
        for (const std::size_t max_dimension : config_.max_dimensions)
        {
            results.push_back(benchmarkPh5Performance(point_count, max_dimension));
            results.push_back(benchmarkPh6Performance(point_count, max_dimension));
            if (config_.enable_determinism_tests)
            {
                results.push_back(benchmarkDeterminismOverhead(point_count, max_dimension));
            }
            if (config_.enable_incremental_tests)
            {
                results.push_back(benchmarkIncrementalUpdates(point_count, max_dimension));
            }
            if (config_.enable_spectral_tests)
            {
                results.push_back(benchmarkSpectralIntegration(point_count, max_dimension));
            }
        }
    }
    if (config_.enable_detailed_logging)
    {
        std::cout << "Completed " << results.size() << " benchmark runs\n";
    }
    return results;
}
PH5PH6MicrobenchmarkResult
PH5PH6Microbenchmark::benchmarkPh5Performance(const std::size_t point_count,
                                              const std::size_t max_dimension)
{
    PH5PH6MicrobenchmarkResult result =
        makeBaseResult("PH5_Performance", "PH5", point_count, max_dimension);
    try
    {
        const auto points =
            generateSyntheticPoints(point_count, max_dimension, config_.random_seed);
        Engine::Config engine_config{};
        engine_config.enable_checksum_validation = false;
        Engine ph5(engine_config);
        std::vector<double> runtimes_ms;
        std::vector<std::size_t> simplex_counts;
        std::size_t peak_memory_bytes = 0;
        std::size_t precision_events = 0;
        std::size_t failures = 0;
        for (std::size_t i = 0; i < config_.iterations_per_test; ++i)
        {
            const std::size_t memory_before = measureMemoryUsage();
            const auto start = std::chrono::high_resolution_clock::now();
            const auto persistence_result = ph5.computePersistenceCohomology(points, max_dimension);
            const auto end = std::chrono::high_resolution_clock::now();
            const std::size_t memory_after = measureMemoryUsage();
            const auto metrics = ph5.getComputationMetrics();
            peak_memory_bytes = std::max(peak_memory_bytes, metrics.peak_memory_bytes);
            if (memory_after > memory_before)
            {
                peak_memory_bytes = std::max(peak_memory_bytes, memory_after - memory_before);
            }
            precision_events += metrics.numerical_errors;
            if (!persistence_result)
            {
                ++failures;
                continue;
            }
            runtimes_ms.push_back(std::chrono::duration<double, std::milli>(end - start).count());
            simplex_counts.push_back(persistence_result->size());
        }
        result.failure_rate =
            config_.iterations_per_test == 0
                ? 1.0
                : static_cast<double>(failures) / static_cast<double>(config_.iterations_per_test);
        fillRuntimeStats(result, runtimes_ms);
        result.peak_memory_mb = bytesToMb(peak_memory_bytes);
        result.precision_events = precision_events;
        if (!simplex_counts.empty())
        {
            result.num_simplices =
                static_cast<std::size_t>(std::llround(computeMean(asDoubleVector(simplex_counts))));
        }
        if (point_count <= 1000)
        {
            result.stability_score = ph5.runStabilityTest(points, max_dimension, 3) ? 1.0 : 0.0;
        }
        else
        {
            result.stability_score =
                result.failure_rate <= config_.failure_rate_threshold ? 0.9 : 0.25;
        }
        result.condition_estimate =
            finiteConditionEstimate(result.p99_runtime_ms, result.p50_runtime_ms);
        result.custom_metrics["successful_runs"] = static_cast<double>(runtimes_ms.size());
        result.custom_metrics["precision_events"] = static_cast<double>(precision_events);
        result.success = !runtimes_ms.empty() && validatePerformanceSlos(result) &&
                         validateQualityMetrics(result);
        if (!result.success && result.failure_reason.empty())
        {
            result.failure_reason = "PH5 benchmark did not satisfy validation thresholds";
        }
    }
    catch (const std::exception &e)
    {
        result.success = false;
        result.failure_reason = "Exception: " + std::string(e.what());
    }
    result.end_time = std::chrono::steady_clock::now();
    return result;
}
PH5PH6MicrobenchmarkResult
PH5PH6Microbenchmark::benchmarkPh6Performance(const std::size_t point_count,
                                              const std::size_t max_dimension)
{
    PH5PH6MicrobenchmarkResult result =
        makeBaseResult("PH6_Performance", "PH6", point_count, max_dimension);
    try
    {
        const auto points =
            generateHighDimensionalPoints(point_count, max_dimension, config_.random_seed);
        Engine::Config engine_config{};
        engine_config.enable_checksum_validation = true;
        engine_config.require_bitwise_reproducibility = true;
        Engine ph6(engine_config);
        std::vector<double> runtimes_ms;
        std::vector<std::size_t> simplex_counts;
        std::size_t peak_memory_bytes = 0;
        std::size_t precision_events = 0;
        std::size_t failures = 0;
        for (std::size_t i = 0; i < config_.iterations_per_test; ++i)
        {
            const std::size_t memory_before = measureMemoryUsage();
            const auto start = std::chrono::high_resolution_clock::now();
            const auto persistence_result = ph6.computePersistenceCohomology(points, max_dimension);
            const auto end = std::chrono::high_resolution_clock::now();
            const std::size_t memory_after = measureMemoryUsage();
            const auto metrics = ph6.getComputationMetrics();
            peak_memory_bytes = std::max(peak_memory_bytes, metrics.peak_memory_bytes);
            if (memory_after > memory_before)
            {
                peak_memory_bytes = std::max(peak_memory_bytes, memory_after - memory_before);
            }
            precision_events += metrics.numerical_errors;
            if (!persistence_result)
            {
                ++failures;
                continue;
            }
            runtimes_ms.push_back(std::chrono::duration<double, std::milli>(end - start).count());
            simplex_counts.push_back(persistence_result->size());
        }
        result.failure_rate =
            config_.iterations_per_test == 0
                ? 1.0
                : static_cast<double>(failures) / static_cast<double>(config_.iterations_per_test);
        fillRuntimeStats(result, runtimes_ms);
        result.peak_memory_mb = bytesToMb(peak_memory_bytes);
        result.precision_events = precision_events;
        if (!simplex_counts.empty())
        {
            result.num_simplices =
                static_cast<std::size_t>(std::llround(computeMean(asDoubleVector(simplex_counts))));
        }
        result.stability_score = ph6.runStabilityTest(points, max_dimension, 3) ? 1.0 : 0.0;
        result.condition_estimate =
            finiteConditionEstimate(result.p99_runtime_ms, result.p50_runtime_ms);
        result.custom_metrics["successful_runs"] = static_cast<double>(runtimes_ms.size());
        result.custom_metrics["precision_events"] = static_cast<double>(precision_events);
        result.success = !runtimes_ms.empty() && validatePerformanceSlos(result) &&
                         validateQualityMetrics(result);
        if (!result.success && result.failure_reason.empty())
        {
            result.failure_reason = "PH6 benchmark did not satisfy validation thresholds";
        }
    }
    catch (const std::exception &e)
    {
        result.success = false;
        result.failure_reason = "Exception: " + std::string(e.what());
    }
    result.end_time = std::chrono::steady_clock::now();
    return result;
}
PH5PH6MicrobenchmarkResult
PH5PH6Microbenchmark::benchmarkDeterminismOverhead(const std::size_t point_count,
                                                   const std::size_t max_dimension)
{
    PH5PH6MicrobenchmarkResult result =
        makeBaseResult("Determinism_Overhead", "PH5", point_count, max_dimension);
    try
    {
        const auto points =
            generateSyntheticPoints(point_count, max_dimension, config_.random_seed);
        auto measure_profile = [&](const bool deterministic_mode) {
            Engine::Config engine_config{};
            engine_config.enable_checksum_validation = deterministic_mode;
            engine_config.require_bitwise_reproducibility = deterministic_mode;
            Engine engine(engine_config);
            std::vector<double> timings;
            timings.reserve(config_.iterations_per_test);
            for (std::size_t i = 0; i < config_.iterations_per_test; ++i)
            {
                const auto start = std::chrono::high_resolution_clock::now();
                const auto ph_result = engine.computePersistenceCohomology(points, max_dimension);
                const auto end = std::chrono::high_resolution_clock::now();
                if (ph_result)
                {
                    timings.push_back(
                        std::chrono::duration<double, std::milli>(end - start).count());
                }
            }
            return timings;
        };
        const auto baseline = measure_profile(false);
        const auto deterministic = measure_profile(true);
        if (baseline.empty() || deterministic.empty())
        {
            result.failure_reason = "Insufficient successful runs for determinism comparison";
            result.end_time = std::chrono::steady_clock::now();
            return result;
        }
        const double baseline_mean = computeMean(baseline);
        const double deterministic_mean = computeMean(deterministic);
        const double overhead =
            (deterministic_mean - baseline_mean) / std::max(1e-9, baseline_mean);
        fillRuntimeStats(result, deterministic);
        result.custom_metrics["determinism_overhead"] = overhead;
        result.custom_metrics["baseline_runtime_ms"] = baseline_mean;
        result.custom_metrics["deterministic_runtime_ms"] = deterministic_mean;
        result.failure_rate = 0.0;
        result.stability_score = overhead <= config_.determinism_overhead_threshold ? 1.0 : 0.0;
        result.condition_estimate =
            finiteConditionEstimate(result.p99_runtime_ms, result.p50_runtime_ms);
        result.success = overhead <= config_.determinism_overhead_threshold;
        if (!result.success)
        {
            result.failure_reason =
                "Determinism overhead exceeds threshold: " + std::to_string(overhead);
        }
    }
    catch (const std::exception &e)
    {
        result.success = false;
        result.failure_reason = "Exception: " + std::string(e.what());
    }
    result.end_time = std::chrono::steady_clock::now();
    return result;
}
PH5PH6MicrobenchmarkResult
PH5PH6Microbenchmark::benchmarkIncrementalUpdates(const std::size_t point_count,
                                                  const std::size_t max_dimension)
{
    PH5PH6MicrobenchmarkResult result =
        makeBaseResult("Incremental_Updates", "PH5", point_count, max_dimension);
    try
    {
        const std::size_t base_count = std::max<std::size_t>(2, point_count / 2);
        const auto base_points =
            generateSyntheticPoints(base_count, max_dimension, config_.random_seed);
        const std::size_t update_count = std::min<std::size_t>(
            config_.incremental_batch_size, std::max<std::size_t>(1, point_count - base_count));
        const auto update_points =
            generateSyntheticPoints(update_count, max_dimension, config_.random_seed + 1);
        nerve::PersistenceBudget budget{};
        budget.memory_limit_mb = std::max<std::size_t>(config_.memory_slo_mb, 64);
        budget.time_limit_ms =
            static_cast<std::size_t>(std::max(200.0, config_.runtime_slo_ms * 10.0));
        nerve::persistence::IncrementalPersistence incremental(budget);
        std::vector<std::vector<double>> working_points = base_points;
        std::vector<double> incremental_runtimes_ms;
        std::vector<double> full_recompute_runtimes_ms;
        std::size_t failures = 0;
        for (const auto &point : update_points)
        {
            const auto inc_start = std::chrono::high_resolution_clock::now();
            const auto inc_result =
                incremental.incrementalAddPoint(working_points, point, max_dimension);
            const auto inc_end = std::chrono::high_resolution_clock::now();
            if (!inc_result.ok())
            {
                ++failures;
                continue;
            }
            working_points.push_back(point);
            incremental_runtimes_ms.push_back(
                std::chrono::duration<double, std::milli>(inc_end - inc_start).count());
            Engine::Config baseline_config{};
            baseline_config.enable_checksum_validation = false;
            Engine baseline(baseline_config);
            const auto full_start = std::chrono::high_resolution_clock::now();
            const auto full_result =
                baseline.computePersistenceCohomology(working_points, max_dimension);
            const auto full_end = std::chrono::high_resolution_clock::now();
            if (full_result)
            {
                full_recompute_runtimes_ms.push_back(
                    std::chrono::duration<double, std::milli>(full_end - full_start).count());
            }
            else
            {
                ++failures;
            }
        }
        result.failure_rate =
            update_count == 0 ? 1.0
                              : static_cast<double>(failures) / static_cast<double>(update_count);
        fillRuntimeStats(result, incremental_runtimes_ms);
        result.peak_memory_mb = bytesToMb(measureMemoryUsage());
        result.num_simplices = incremental.getDiagram().size();
        result.stability_score = incremental.hasValidDiagram() ? 1.0 : 0.0;
        result.condition_estimate =
            finiteConditionEstimate(result.p99_runtime_ms, result.p50_runtime_ms);
        if (incremental_runtimes_ms.empty() || full_recompute_runtimes_ms.empty())
        {
            result.failure_reason = "Insufficient successful runs for incremental comparison";
        }
        else
        {
            const double mean_incremental = computeMean(incremental_runtimes_ms);
            const double mean_full = computeMean(full_recompute_runtimes_ms);
            const double speedup = mean_full / std::max(1e-9, mean_incremental);
            result.custom_metrics["incremental_speedup"] = speedup;
            result.custom_metrics["mean_incremental_runtime_ms"] = mean_incremental;
            result.custom_metrics["mean_full_recompute_runtime_ms"] = mean_full;
            result.success = speedup >= config_.incremental_speedup_threshold;
            if (!result.success)
            {
                result.failure_reason =
                    "Incremental speedup below threshold: " + std::to_string(speedup);
            }
        }
    }
    catch (const std::exception &e)
    {
        result.success = false;
        result.failure_reason = "Exception: " + std::string(e.what());
    }
    result.end_time = std::chrono::steady_clock::now();
    return result;
}
} // namespace nerve::validation

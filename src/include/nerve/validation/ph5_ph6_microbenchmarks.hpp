
#pragma once
#include "nerve/errors/errors.hpp"
#include "nerve/instrumentation/stability_certificates.hpp"
#include "nerve/persistence/kernels/ph5_ph6_ops.hpp"
#include "nerve/validation/microbenchmarks.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <future>
#include <memory>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

namespace nerve::validation
{
struct PH5PH6MicrobenchmarkConfig
{
    std::vector<size_t> point_counts = {100, 500, 1000, 5000};
    std::vector<size_t> max_dimensions = {2, 3, 4, 5};
    size_t iterations_per_test = 10;
    double runtime_slo_ms = 100.0;
    size_t memory_slo_mb = 100;
    double failure_rate_threshold = 0.1;
    size_t max_simplices_threshold = 100000;
    bool enable_determinism_tests = true;
    double determinism_overhead_threshold = 0.2;
    bool enable_incremental_tests = true;
    size_t incremental_batch_size = 100;
    double incremental_speedup_threshold = 2.0;
    bool enable_spectral_tests = true;
    size_t spectral_eigenpairs = 10;
    double spectral_latency_threshold_ms = 50.0;
    bool enable_detailed_logging = true;
    bool enable_memory_profiling = true;
    uint32_t random_seed = 42;
};
struct PH5PH6MicrobenchmarkResult
{
    std::string benchmark_name;
    std::string algorithm_type;
    size_t point_count;
    size_t max_dimension;
    bool success;
    std::string failure_reason;
    double mean_runtime_ms;
    double p50_runtime_ms;
    double p95_runtime_ms;
    double p99_runtime_ms;
    size_t peak_memory_mb;
    size_t num_simplices;
    double failure_rate;
    double stability_score;
    double condition_estimate;
    size_t precision_events;
    std::unordered_map<std::string, double> custom_metrics;
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point end_time;
};
class PH5PH6Microbenchmark
{
public:
    explicit PH5PH6Microbenchmark(const PH5PH6MicrobenchmarkConfig &config);
    std::vector<PH5PH6MicrobenchmarkResult> runAllBenchmarks();
    PH5PH6MicrobenchmarkResult benchmarkPh5Performance(size_t point_count, size_t max_dimension);
    PH5PH6MicrobenchmarkResult benchmarkPh6Performance(size_t point_count, size_t max_dimension);
    PH5PH6MicrobenchmarkResult benchmarkDeterminismOverhead(size_t point_count,
                                                            size_t max_dimension);
    PH5PH6MicrobenchmarkResult benchmarkIncrementalUpdates(size_t point_count,
                                                           size_t max_dimension);
    PH5PH6MicrobenchmarkResult benchmarkSpectralIntegration(size_t point_count,
                                                            size_t max_dimension);
    bool validatePerformanceSlos(const PH5PH6MicrobenchmarkResult &result) const;
    bool validateQualityMetrics(const PH5PH6MicrobenchmarkResult &result) const;
    void generateBenchmarkReport(const std::vector<PH5PH6MicrobenchmarkResult> &results) const;
    void exportMetricsCsv(const std::vector<PH5PH6MicrobenchmarkResult> &results,
                          const std::string &filename) const;

private:
    PH5PH6MicrobenchmarkConfig config_;
    std::vector<std::vector<double>> generateSyntheticPoints(size_t count, size_t dimension,
                                                             uint32_t seed) const;
    std::vector<std::vector<double>> generateHighDimensionalPoints(size_t count, size_t dimension,
                                                                   uint32_t seed) const;
    size_t measureMemoryUsage() const;
    double measureStabilityScore(const nerve::instrumentation::StabilityCertificate &cert) const;
    template <typename PointType, typename Field>
    PH5PH6MicrobenchmarkResult runPh5Benchmark(const std::vector<PointType> &points,
                                               size_t max_dimension);
    template <typename PointType, typename Field>
    PH5PH6MicrobenchmarkResult runPh6Benchmark(const std::vector<PointType> &points,
                                               size_t max_dimension);
    double computePercentile(const std::vector<double> &values, double percentile) const;
    double computeMean(const std::vector<double> &values) const;
    double computeStdDeviation(const std::vector<double> &values) const;
};
class PH5PH6BenchmarkCI
{
public:
    struct CIResult
    {
        bool all_benchmarks_passed;
        size_t total_benchmarks;
        size_t passed_benchmarks;
        std::vector<std::string> failed_benchmarks;
        std::unordered_map<std::string, double> summary_metrics;
    };
    static CIResult runCiBenchmarks(const PH5PH6MicrobenchmarkConfig &config);
    static void generateCiReport(const CIResult &result, const std::string &output_dir);
    static bool checkRegressionGates(const CIResult &result);
};
} // namespace nerve::validation

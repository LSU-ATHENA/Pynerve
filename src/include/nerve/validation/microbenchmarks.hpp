
#pragma once
#include "nerve/optimization/component_optimizations.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <future>
#include <memory>
#include <numeric>
#include <random>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace nerve::validation
{
struct MicrobenchmarkConfig
{
    size_t feature_cache_reads = 1000000;
    size_t hot_symbols = 10;
    size_t cold_symbols = 90;
    double p99_latency_threshold_us = 1.0;
    size_t compact_summary_windows = 1000;
    size_t window_size = 2000;
    double mean_latency_threshold_ms = 2.0;
    std::vector<size_t> gpu_batch_sizes = {32, 64};
    double p95_latency_threshold_ms = 10.0;
    size_t gpu_iterations = 100;
    size_t streaming_ph_points = 10000;
    size_t adversarial_bursts = 10;
    double update_time_ratio_threshold = 2.0;
    size_t incremental_updates = 20;
    double residual_threshold = 1e-6;
    size_t laplacian_matrix_size = 1000;
    bool enable_detailed_logging = true;
    bool enable_performance_profiling = true;
    size_t num_threads = 4;
    uint32_t random_seed = 42;
};
struct MicrobenchmarkResult
{
    std::string benchmark_name;
    bool success;
    std::string failure_reason;
    std::vector<double> latencies;
    double mean_latency;
    double p50_latency;
    double p95_latency;
    double p99_latency;
    double std_deviation;
    size_t sample_count;
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point end_time;
    std::unordered_map<std::string, double> metrics;
};
class FeatureCacheMicrobenchmark
{
public:
    explicit FeatureCacheMicrobenchmark(const MicrobenchmarkConfig &config);
    MicrobenchmarkResult runFeatureCacheBenchmark();
    double measureCacheMissRate();
    std::vector<double> measureLatencyDistribution();

private:
    MicrobenchmarkConfig config_;
    std::unique_ptr<optimization::AcceleratedFeatureCache> cache_;
    std::mt19937 rng_;
    std::vector<uint32_t> hot_symbols_;
    std::vector<uint32_t> cold_symbols_;
    void generateTestSymbols();
    std::vector<uint32_t> generateRandomReads(size_t count);
    optimization::AcceleratedFeatureCache::CacheEntry createTestEntry(uint32_t symbol_id);
};
class CompactSummaryMicrobenchmark
{
public:
    explicit CompactSummaryMicrobenchmark(const MicrobenchmarkConfig &config);
    MicrobenchmarkResult runCompactSummaryBenchmark();
    bool validatePerformance(const MicrobenchmarkResult &result);

private:
    MicrobenchmarkConfig config_;
    std::unique_ptr<optimization::AcceleratedCompactSummaries> summaries_;
    std::mt19937 rng_;
    std::vector<std::vector<std::vector<float>>> generateTestWindows(size_t count,
                                                                     size_t window_size);
    optimization::AcceleratedCompactSummaries::CompactSummary
    computeSummaryReference(const std::vector<std::vector<float>> &points);
};
class GPUPrimitivesMicrobenchmark
{
public:
    explicit GPUPrimitivesMicrobenchmark(const MicrobenchmarkConfig &config);
    std::vector<MicrobenchmarkResult> runGpuBenchmarks();
    bool validateGpuPerformance(const std::vector<MicrobenchmarkResult> &results);

private:
    MicrobenchmarkConfig config_;
    std::unique_ptr<optimization::AcceleratedGpuPrimitives> gpu_primitives_;
    std::mt19937 rng_;
    std::vector<std::vector<std::vector<float>>> generateTestData(size_t batch_size,
                                                                  size_t num_points);
    std::vector<float *> allocateGpuBuffers(size_t batch_size, size_t num_points);
    void deallocateGpuBuffers(const std::vector<float *> &buffers);
};
class StreamingPHMicrobenchmark
{
public:
    explicit StreamingPHMicrobenchmark(const MicrobenchmarkConfig &config);
    MicrobenchmarkResult runStreamingPhBenchmark();
    std::vector<double> testAdversarialBursts();
    double compareUpdateVsExact();

private:
    MicrobenchmarkConfig config_;
    std::unique_ptr<optimization::AcceleratedStreamingPh> streaming_ph_;
    std::mt19937 rng_;
    std::vector<std::vector<float>> generateTimeSeries(size_t num_points);
    std::vector<std::vector<float>>
    injectAdversarialBursts(const std::vector<std::vector<float>> &base_series, size_t num_bursts);
    std::vector<std::pair<float, float>>
    computeExactPh(const std::vector<std::vector<float>> &points);
};
class IncrementalLaplacianMicrobenchmark
{
public:
    explicit IncrementalLaplacianMicrobenchmark(const MicrobenchmarkConfig &config);
    MicrobenchmarkResult runIncrementalLaplacianBenchmark();
    bool validateResiduals(const std::vector<double> &residuals);

private:
    MicrobenchmarkConfig config_;
    std::unique_ptr<optimization::AcceleratedIncrementalLaplacian> incremental_laplacian_;
    std::mt19937 rng_;
    std::vector<std::vector<uint32_t>> generateTestMatrix(size_t size);
    std::vector<double> generateTestWeights(size_t size);
    std::vector<std::vector<uint32_t>>
    generateIncrementalUpdates(const std::vector<std::vector<uint32_t>> &base_matrix,
                               size_t num_updates);
};
class ChaosTesting
{
public:
    struct ChaosConfig
    {
        bool enable_gpu_oom_injection = true;
        bool enable_network_delay = true;
        double cpu_overload_threshold = 0.9;
        uint64_t network_delay_ms = 100;
        size_t chaos_test_duration_ms = 10000;
    };
    explicit ChaosTesting(const ChaosConfig &config);
    struct ChaosTestResult
    {
        std::string test_name;
        bool circuit_breaker_tripped;
        bool recovery_engaged;
        std::vector<std::string> error_events;
        double recovery_time_ms;
        bool test_passed;
    };
    std::vector<ChaosTestResult> runChaosTests();
    ChaosTestResult testGpuOomInjection();
    ChaosTestResult testNetworkDelay();
    ChaosTestResult testCpuOverload();

private:
    ChaosConfig config_;
    std::shared_ptr<void> error_manager_;
    void injectGpuOom();
    void injectNetworkDelay();
    void injectCpuOverload();
    void monitorCircuitBreakerState();
    void monitorRecoveryEngagement();
};
class FeatureLevelMLCheck
{
public:
    struct MLCheckConfig
    {
        std::vector<std::string> param_names = {"persistence_grid_size", "sigma",
                                                "time_window_size"};
        double metric_drift_threshold = 0.05;
        size_t test_samples = 1000;
        bool enable_automated_review = false;
        std::string downstream_model_type = "regression";
    };
    explicit FeatureLevelMLCheck(const MLCheckConfig &config);
    struct MLCheckResult
    {
        std::string param_name;
        double old_value;
        double new_value;
        double metric_drift;
        bool requires_human_review;
        std::string review_reason;
        bool check_passed;
    };
    std::vector<MLCheckResult>
    checkParameterChanges(const std::unordered_map<std::string, double> &old_params,
                          const std::unordered_map<std::string, double> &new_params);
    bool validateDownstreamModel(const std::unordered_map<std::string, double> &params,
                                 const std::vector<std::vector<float>> &features,
                                 const std::vector<float> &expected_outputs);

private:
    MLCheckConfig config_;
    double computeModelMetric(const std::vector<std::vector<float>> &features,
                              const std::vector<float> &outputs);
    bool isMetricDriftSignificant(double old_metric, double new_metric);
};
class MicrobenchmarkManager
{
public:
    static MicrobenchmarkManager &instance();
    void setConfig(const MicrobenchmarkConfig &config);
    MicrobenchmarkConfig getConfig() const;
    struct BenchmarkSuite
    {
        std::vector<MicrobenchmarkResult> feature_cache_results;
        std::vector<MicrobenchmarkResult> compact_summary_results;
        std::vector<MicrobenchmarkResult> gpu_primitives_results;
        std::vector<MicrobenchmarkResult> streaming_ph_results;
        std::vector<MicrobenchmarkResult> incremental_laplacian_results;
        std::vector<ChaosTesting::ChaosTestResult> chaos_test_results;
        std::vector<FeatureLevelMLCheck::MLCheckResult> ml_check_results;
        bool all_passed;
        std::vector<std::string> failed_benchmarks;
        double total_execution_time_ms;
    };
    BenchmarkSuite runAllBenchmarks();
    std::vector<MicrobenchmarkResult> runFeatureCacheBenchmarks();
    std::vector<MicrobenchmarkResult> runCompactSummaryBenchmarks();
    std::vector<MicrobenchmarkResult> runGpuBenchmarks();
    std::vector<MicrobenchmarkResult> runStreamingPhBenchmarks();
    std::vector<MicrobenchmarkResult> runIncrementalLaplacianBenchmarks();
    std::vector<ChaosTesting::ChaosTestResult> runChaosTests();
    std::vector<FeatureLevelMLCheck::MLCheckResult>
    runMlChecks(const std::unordered_map<std::string, double> &old_params,
                const std::unordered_map<std::string, double> &new_params);
    struct RegressionTestResult
    {
        bool regression_detected;
        std::vector<std::string> regressed_metrics;
        std::vector<std::string> improved_metrics;
        double regression_threshold;
        bool test_passed;
    };
    RegressionTestResult runRegressionTests(const std::string &baseline_file,
                                            const std::string &current_file);
    std::string generateBenchmarkReport(const BenchmarkSuite &suite);
    std::string exportResultsJson(const BenchmarkSuite &suite);
    void saveResultsToFile(const BenchmarkSuite &suite, const std::string &filename);
    struct PerformanceMetrics
    {
        double total_execution_time_ms;
        std::unordered_map<std::string, double> benchmark_times;
        std::unordered_map<std::string, bool> benchmark_results;
        std::unordered_map<std::string, std::vector<double>> latency_distributions;
    };
    PerformanceMetrics getPerformanceMetrics() const;
    void resetPerformanceMetrics();

private:
    MicrobenchmarkManager() = default;
    MicrobenchmarkConfig config_;
    std::unique_ptr<FeatureCacheMicrobenchmark> feature_cache_benchmark_;
    std::unique_ptr<CompactSummaryMicrobenchmark> compact_summary_benchmark_;
    std::unique_ptr<GPUPrimitivesMicrobenchmark> gpu_benchmarks_;
    std::unique_ptr<StreamingPHMicrobenchmark> streaming_ph_benchmark_;
    std::unique_ptr<IncrementalLaplacianMicrobenchmark> incremental_laplacian_benchmark_;
    std::unique_ptr<ChaosTesting> chaos_testing_;
    std::unique_ptr<FeatureLevelMLCheck> ml_check_;
    mutable std::shared_mutex mutex_;
    mutable PerformanceMetrics performance_metrics_;
    void updatePerformanceMetrics(const std::string &benchmark_name, double execution_time,
                                  bool success, const std::vector<double> &latencies = {});
    bool validateBenchmarkResult(const MicrobenchmarkResult &result);
    std::string formatBenchmarkResult(const MicrobenchmarkResult &result);
};
} // namespace nerve::validation

#include "nerve/config.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <vector>
#if HAS_EIGEN && __has_include(<Eigen/Sparse>) && __has_include(<Eigen/Dense>)
#include "nerve/spectral/persistent_laplacian.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#define NERVE_VALIDATION_HAS_SPECTRAL_EIGEN 1
#else
#define NERVE_VALIDATION_HAS_SPECTRAL_EIGEN 0
#endif
#include "nerve/validation/ph5_ph6_microbenchmarks.hpp"

namespace nerve::validation
{
namespace
{

#if NERVE_VALIDATION_HAS_SPECTRAL_EIGEN
constexpr std::size_t kBytesPerMb = 1024ULL * 1024ULL;

std::size_t bytesToMb(const std::size_t bytes)
{
    if (bytes == 0)
    {
        return 0;
    }
    return (bytes + kBytesPerMb - 1ULL) / kBytesPerMb;
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

double sanitizeConditionEstimate(const double value)
{
    if (!std::isfinite(value) || value < 1.0)
    {
        return 1.0;
    }
    return value;
}

double squaredDistance(const std::vector<double> &a, const std::vector<double> &b)
{
    const std::size_t dim = std::min(a.size(), b.size());
    double sum = 0.0;
    for (std::size_t i = 0; i < dim; ++i)
    {
        const double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

Eigen::SparseMatrix<double> buildGraphLaplacian(const std::vector<std::vector<double>> &points)
{
    const int n = static_cast<int>(points.size());
    Eigen::SparseMatrix<double> laplacian(n, n);
    if (n <= 0)
    {
        return laplacian;
    }
    std::vector<double> degree(static_cast<std::size_t>(n), 0.0);
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(static_cast<std::size_t>(n) * static_cast<std::size_t>(n) / 2U);
    for (int i = 0; i < n; ++i)
    {
        for (int j = i + 1; j < n; ++j)
        {
            const double dist_sq = squaredDistance(points[static_cast<std::size_t>(i)],
                                                   points[static_cast<std::size_t>(j)]);
            const double weight = std::exp(-dist_sq);
            if (weight <= 1e-12)
            {
                continue;
            }
            degree[static_cast<std::size_t>(i)] += weight;
            degree[static_cast<std::size_t>(j)] += weight;
            triplets.emplace_back(i, j, -weight);
            triplets.emplace_back(j, i, -weight);
        }
    }
    for (int i = 0; i < n; ++i)
    {
        triplets.emplace_back(i, i, degree[static_cast<std::size_t>(i)]);
    }
    laplacian.setFromTriplets(triplets.begin(), triplets.end());
    return laplacian;
}
#endif

} // namespace

PH5PH6MicrobenchmarkResult
PH5PH6Microbenchmark::benchmarkSpectralIntegration(const std::size_t point_count,
                                                   const std::size_t max_dimension)
{
    PH5PH6MicrobenchmarkResult result{};
    result.benchmark_name = "Spectral_Integration";
    result.algorithm_type = "PH5+Spectral";
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

#if NERVE_VALIDATION_HAS_SPECTRAL_EIGEN
    try
    {
        const std::size_t spectral_point_count = std::min<std::size_t>(point_count, 128);
        const auto points = generateHighDimensionalPoints(
            std::max<std::size_t>(2, spectral_point_count), std::max<std::size_t>(2, max_dimension),
            config_.random_seed);
        const Eigen::SparseMatrix<double> laplacian = buildGraphLaplacian(points);

        nerve::spectral::SpectralConfig spectral_config{};
        spectral_config.num_eigenpairs = std::min<std::size_t>(
            config_.spectral_eigenpairs, static_cast<std::size_t>(laplacian.rows()));
        spectral_config.convergence_tolerance = 1e-8;
        spectral_config.max_iterations = 1000;
        spectral_config.compute_harmonic = true;
        spectral_config.compute_nonharmonic = true;
        spectral_config.enable_warm_start = true;
        spectral_config.solver_type = "lanczos";

        nerve::spectral::PersistentLaplacianSolver solver(spectral_config);
        std::vector<double> runtimes_ms;
        std::vector<std::size_t> eigenpair_counts;
        std::size_t failures = 0;
        double max_condition_number = 1.0;

        for (std::size_t i = 0; i < config_.iterations_per_test; ++i)
        {
            const auto start = std::chrono::high_resolution_clock::now();
            const auto decomposition = solver.computeSpectrum(laplacian);
            const auto end = std::chrono::high_resolution_clock::now();
            if (!decomposition.converged || decomposition.eigenpairs.empty())
            {
                ++failures;
                continue;
            }
            runtimes_ms.push_back(std::chrono::duration<double, std::milli>(end - start).count());
            eigenpair_counts.push_back(decomposition.eigenpairs.size());
            max_condition_number = std::max(
                max_condition_number, sanitizeConditionEstimate(decomposition.condition_number));
        }

        result.failure_rate =
            config_.iterations_per_test == 0
                ? 1.0
                : static_cast<double>(failures) / static_cast<double>(config_.iterations_per_test);
        fillRuntimeStats(result, runtimes_ms);
        result.peak_memory_mb = bytesToMb(measureMemoryUsage());
        result.stability_score = result.failure_rate <= config_.failure_rate_threshold ? 1.0 : 0.0;
        result.condition_estimate = sanitizeConditionEstimate(max_condition_number);
        if (!eigenpair_counts.empty())
        {
            std::vector<double> eigenpair_counts_f64;
            eigenpair_counts_f64.reserve(eigenpair_counts.size());
            for (const std::size_t count : eigenpair_counts)
            {
                eigenpair_counts_f64.push_back(static_cast<double>(count));
            }
            result.num_simplices =
                static_cast<std::size_t>(std::llround(computeMean(eigenpair_counts_f64)));
        }
        result.custom_metrics["spectral_failure_rate"] = result.failure_rate;
        result.custom_metrics["effective_eigenpairs"] = static_cast<double>(result.num_simplices);
        result.success = !runtimes_ms.empty() &&
                         result.mean_runtime_ms <= config_.spectral_latency_threshold_ms &&
                         result.failure_rate <= config_.failure_rate_threshold;
        if (!result.success && result.failure_reason.empty())
        {
            result.failure_reason = "Spectral latency or failure-rate threshold not satisfied";
        }
    }
    catch (const std::exception &e)
    {
        result.success = false;
        result.failure_reason = "Exception: " + std::string(e.what());
    }
#else
    result.failure_reason = "Spectral validation benchmark requires Eigen3";
#endif

    result.end_time = std::chrono::steady_clock::now();
    return result;
}

} // namespace nerve::validation


#include "nerve/validation/ph5_ph6_microbenchmarks.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#if defined(_WIN32)
#include <psapi.h>
#include <windows.h>
#endif
#if defined(__APPLE__)
#include <mach/mach.h>
#endif
#if defined(__linux__) || defined(__APPLE__)
#include <sys/resource.h>
#endif

namespace nerve::validation
{
namespace
{
std::size_t estimateHeapUsagePortable()
{
#if defined(__linux__) || defined(__APPLE__)
    struct rusage
        usage{};
    if (getrusage(RUSAGE_SELF, &usage) == 0)
    {
        return static_cast<std::size_t>(usage.ru_maxrss) * 1024ULL;
    }
#endif
    return 0;
}
} // namespace
std::vector<std::vector<double>>
PH5PH6Microbenchmark::generateSyntheticPoints(size_t count, size_t dimension, uint32_t seed) const
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<std::vector<double>> points;
    points.reserve(count);
    for (size_t i = 0; i < count; ++i)
    {
        std::vector<double> point(dimension);
        for (size_t j = 0; j < dimension; ++j)
        {
            point[j] = dist(gen);
        }
        points.push_back(std::move(point));
    }
    return points;
}
std::vector<std::vector<double>>
PH5PH6Microbenchmark::generateHighDimensionalPoints(size_t count, size_t dimension,
                                                    uint32_t seed) const
{
    std::mt19937 gen(seed);
    std::normal_distribution<double> dist(0.0, 1.0);
    std::vector<std::vector<double>> points;
    points.reserve(count);
    for (size_t i = 0; i < count; ++i)
    {
        std::vector<double> point(dimension);
        for (size_t j = 0; j < dimension; ++j)
        {
            point[j] = dist(gen);
        }
        points.push_back(std::move(point));
    }
    return points;
}
size_t PH5PH6Microbenchmark::measureMemoryUsage() const
{
#ifdef __linux__
    std::ifstream statusFile("/proc/self/status");
    std::string line;
    while (std::getline(statusFile, line))
    {
        if (line.substr(0, 6) == "VmRSS:")
        {
            std::istringstream iss(line.substr(6));
            size_t memory_kb;
            iss >> memory_kb;
            return memory_kb * 1024;
        }
    }
#elif defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
    {
        return pmc.WorkingSetSize;
    }
#elif defined(__APPLE__)
    struct mach_task_basic_info info;
    mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &infoCount) ==
        KERN_SUCCESS)
    {
        return info.resident_size;
    }
#endif
    return estimateHeapUsagePortable();
}
double PH5PH6Microbenchmark::measureStabilityScore(
    const nerve::instrumentation::StabilityCertificate &cert) const
{
    return cert.isHighQuality() ? 1.0 : 0.5;
}
bool PH5PH6Microbenchmark::validatePerformanceSlos(const PH5PH6MicrobenchmarkResult &result) const
{
    if (result.mean_runtime_ms > config_.runtime_slo_ms)
    {
        return false;
    }
    if (result.peak_memory_mb > config_.memory_slo_mb)
    {
        return false;
    }
    if (result.failure_rate > config_.failure_rate_threshold)
    {
        return false;
    }
    if (result.num_simplices > config_.max_simplices_threshold)
    {
        return false;
    }
    return true;
}
bool PH5PH6Microbenchmark::validateQualityMetrics(const PH5PH6MicrobenchmarkResult &result) const
{
    if (result.stability_score < 0.5)
    {
        return false;
    }
    if (result.condition_estimate > 1e12)
    {
        return false;
    }
    return true;
}
double PH5PH6Microbenchmark::computePercentile(const std::vector<double> &values,
                                               double percentile) const
{
    if (values.empty())
        return 0.0;
    std::vector<double> sorted_values = values;
    std::sort(sorted_values.begin(), sorted_values.end());
    const double position = (percentile / 100.0) * static_cast<double>(sorted_values.size() - 1U);
    size_t index = static_cast<size_t>(position);
    return sorted_values[index];
}
double PH5PH6Microbenchmark::computeMean(const std::vector<double> &values) const
{
    if (values.empty())
        return 0.0;
    return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}
double PH5PH6Microbenchmark::computeStdDeviation(const std::vector<double> &values) const
{
    if (values.size() < 2)
        return 0.0;
    double mean = computeMean(values);
    double sum_sq_diff = 0.0;
    for (double value : values)
    {
        double diff = value - mean;
        sum_sq_diff += diff * diff;
    }
    return std::sqrt(sum_sq_diff / static_cast<double>(values.size() - 1U));
}
void PH5PH6Microbenchmark::generateBenchmarkReport(
    const std::vector<PH5PH6MicrobenchmarkResult> &results) const
{
    std::cout << "\n=== PH5/PH6 Microbenchmark Report ===\n";
    std::cout << "Total benchmarks: " << results.size() << '\n';
    size_t passed =
        std::count_if(results.begin(), results.end(), [](const auto &r) { return r.success; });
    const double success_rate =
        results.empty() ? 0.0
                        : 100.0 * static_cast<double>(passed) / static_cast<double>(results.size());
    std::cout << "Passed: " << passed << " (" << success_rate << "%)\n";
    std::cout << "Failed: " << (results.size() - passed) << '\n';
    std::vector<double> all_runtimes;
    std::vector<double> all_memories;
    for (const auto &result : results)
    {
        if (result.success)
        {
            all_runtimes.push_back(result.mean_runtime_ms);
            all_memories.push_back(static_cast<double>(result.peak_memory_mb));
        }
    }
    if (!all_runtimes.empty())
    {
        std::cout << "\nRuntime Statistics (ms):\n";
        std::cout << "  Mean: " << computeMean(all_runtimes) << '\n';
        std::cout << "  P50: " << computePercentile(all_runtimes, 50.0) << '\n';
        std::cout << "  P95: " << computePercentile(all_runtimes, 95.0) << '\n';
        std::cout << "  P99: " << computePercentile(all_runtimes, 99.0) << '\n';
        std::cout << "\nMemory Statistics (MB):\n";
        std::cout << "  Mean: " << computeMean(all_memories) << '\n';
        std::cout << "  Max: " << *std::max_element(all_memories.begin(), all_memories.end())
                  << '\n';
    }
    std::cout << "\nFailed Benchmarks:\n";
    for (const auto &result : results)
    {
        if (!result.success)
        {
            std::cout << "  " << result.benchmark_name << " (N=" << result.point_count
                      << ", D=" << result.max_dimension << ")" << ": " << result.failure_reason
                      << '\n';
        }
    }
}
void PH5PH6Microbenchmark::exportMetricsCsv(const std::vector<PH5PH6MicrobenchmarkResult> &results,
                                            const std::string &filename) const
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Failed to open CSV file: " << filename << '\n';
        return;
    }
    file << "benchmark_name,algorithm_type,point_count,max_dimension,success,";
    file << "mean_runtime_ms,p50_runtime_ms,p95_runtime_ms,p99_runtime_ms,";
    file << "peak_memory_mb,num_simplices,failure_rate,";
    file << "stability_score,condition_estimate,precision_events,";
    file << "failure_reason\n";
    for (const auto &result : results)
    {
        file << result.benchmark_name << "," << result.algorithm_type << "," << result.point_count
             << "," << result.max_dimension << "," << (result.success ? "true" : "false") << ","
             << result.mean_runtime_ms << "," << result.p50_runtime_ms << ","
             << result.p95_runtime_ms << "," << result.p99_runtime_ms << ","
             << result.peak_memory_mb << "," << result.num_simplices << "," << result.failure_rate
             << "," << result.stability_score << "," << result.condition_estimate << ","
             << result.precision_events << "," << "\"" << result.failure_reason << "\"\n";
    }
    file.close();
}
} // namespace nerve::validation

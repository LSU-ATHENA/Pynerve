
#include "nerve/validation/ph5_ph6_microbenchmarks.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace nerve::validation
{
PH5PH6BenchmarkCI::CIResult
PH5PH6BenchmarkCI::runCiBenchmarks(const PH5PH6MicrobenchmarkConfig &config)
{
    CIResult result;
    result.all_benchmarks_passed = true;
    try
    {
        PH5PH6Microbenchmark benchmark(config);
        auto benchmark_results = benchmark.runAllBenchmarks();
        result.total_benchmarks = benchmark_results.size();
        for (const auto &benchmark_result : benchmark_results)
        {
            if (benchmark_result.success)
            {
                result.passed_benchmarks++;
            }
            else
            {
                result.all_benchmarks_passed = false;
                result.failed_benchmarks.push_back(
                    benchmark_result.benchmark_name +
                    " (N=" + std::to_string(benchmark_result.point_count) +
                    ", D=" + std::to_string(benchmark_result.max_dimension) + ")");
            }
            std::string key =
                benchmark_result.benchmark_name + "_" + benchmark_result.algorithm_type;
            result.summary_metrics[key + "_mean_runtime_ms"] = benchmark_result.mean_runtime_ms;
            result.summary_metrics[key + "_peak_memory_mb"] =
                static_cast<double>(benchmark_result.peak_memory_mb);
            result.summary_metrics[key + "_failure_rate"] = benchmark_result.failure_rate;
        }
    }
    catch (const std::exception &e)
    {
        result.all_benchmarks_passed = false;
        result.failed_benchmarks.push_back("CI execution failed: " + std::string(e.what()));
    }
    return result;
}
void PH5PH6BenchmarkCI::generateCiReport(const CIResult &result, const std::string &output_dir)
{
    std::filesystem::create_directories(output_dir);
    std::ofstream reportFile(output_dir + "/ph5_ph6_ci_report.md");
    if (!reportFile.is_open())
    {
        std::cerr << "Failed to create CI report file\n";
        return;
    }
    const double success_rate = result.total_benchmarks == 0
                                    ? 0.0
                                    : 100.0 * static_cast<double>(result.passed_benchmarks) /
                                          static_cast<double>(result.total_benchmarks);
    reportFile << "# PH5/PH6 Microbenchmark CI Report\n\n";
    reportFile << "## Summary\n\n";
    reportFile << "- **Total Benchmarks**: " << result.total_benchmarks << "\n";
    reportFile << "- **Passed**: " << result.passed_benchmarks << "\n";
    reportFile << "- **Failed**: " << (result.total_benchmarks - result.passed_benchmarks) << "\n";
    reportFile << "- **Success Rate**: " << success_rate << "%\n";
    reportFile << "- **Overall Status**: " << (result.all_benchmarks_passed ? "PASSED" : "FAILED")
               << "\n\n";
    if (!result.failed_benchmarks.empty())
    {
        reportFile << "## Failed Benchmarks\n\n";
        for (const auto &failed : result.failed_benchmarks)
        {
            reportFile << "- FAILED: " << failed << "\n";
        }
        reportFile << "\n";
    }
    reportFile << "## Performance Metrics\n\n";
    reportFile << "| Benchmark | Mean Runtime (ms) | Peak Memory (MB) | Failure Rate |\n";
    reportFile << "|-----------|-------------------|------------------|---------------|\n";
    for (const auto &[key, value] : result.summary_metrics)
    {
        if (key.find("_mean_runtime_ms") != std::string::npos)
        {
            std::string benchmark_name = key.substr(0, key.length() - 16);
            double runtime = value;
            double memory = result.summary_metrics.at(benchmark_name + "_peak_memory_mb");
            double failure = result.summary_metrics.at(benchmark_name + "_failure_rate");
            reportFile << "| " << benchmark_name << " | " << std::fixed << std::setprecision(2)
                       << runtime << " | " << memory << " | " << std::setprecision(3) << failure
                       << " |\n";
        }
    }
    reportFile.close();
}
bool PH5PH6BenchmarkCI::checkRegressionGates(const CIResult &result)
{
    const double max_failure_rate = 0.15;
    for (const auto &[key, value] : result.summary_metrics)
    {
        if (key.find("_mean_runtime_ms") != std::string::npos)
        {
            std::string benchmark_name = key.substr(0, key.length() - 16);
            double runtime = value;
            if (runtime > 1000.0)
            {
                return false;
            }
        }
        if (key.find("_failure_rate") != std::string::npos)
        {
            double failure = value;
            if (failure > max_failure_rate)
            {
                return false;
            }
        }
    }
    return result.all_benchmarks_passed;
}
} // namespace nerve::validation

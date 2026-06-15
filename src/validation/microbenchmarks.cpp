#include "nerve/validation/microbenchmarks.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include <utility>

namespace nerve::validation
{
namespace
{

constexpr std::size_t kPointDim = 3;

optimization::CallContract makeContract(const std::string &name)
{
    optimization::CallContract c{};
    c.time_budget_ms = 0.0;
    c.strict_time_enforcement = false;
    c.operation_name = name;
    c.params_hash = 0;
    c.window_start_ns = 0;
    c.window_end_ns = 0;
    return c;
}

void fillLatencyStats(MicrobenchmarkResult &out)
{
    out.sample_count = out.latencies.size();
    if (out.latencies.empty())
    {
        out.mean_latency = 0.0;
        out.p50_latency = 0.0;
        out.p95_latency = 0.0;
        out.p99_latency = 0.0;
        out.std_deviation = 0.0;
        return;
    }

    std::vector<double> sorted = out.latencies;
    std::sort(sorted.begin(), sorted.end());
    const auto pick = [&sorted](double q) -> double {
        const std::size_t idx = static_cast<std::size_t>(std::clamp(q, 0.0, 1.0) *
                                                         static_cast<double>(sorted.size() - 1));
        return sorted[idx];
    };

    out.mean_latency =
        std::accumulate(sorted.begin(), sorted.end(), 0.0) / static_cast<double>(sorted.size());
    out.p50_latency = pick(0.50);
    out.p95_latency = pick(0.95);
    out.p99_latency = pick(0.99);

    double sq = 0.0;
    for (const double v : sorted)
    {
        const double d = v - out.mean_latency;
        sq += d * d;
    }
    out.std_deviation = std::sqrt(sq / static_cast<double>(sorted.size()));
}

MicrobenchmarkResult makeResult(const std::string &name, std::vector<double> latencies,
                                std::chrono::steady_clock::time_point start,
                                std::chrono::steady_clock::time_point end, bool success = true,
                                std::string failure_reason = {})
{
    MicrobenchmarkResult out{};
    out.benchmark_name = name;
    out.success = success;
    out.failure_reason = std::move(failure_reason);
    out.latencies = std::move(latencies);
    out.start_time = start;
    out.end_time = end;
    fillLatencyStats(out);
    out.metrics["duration_ms"] = std::chrono::duration<double, std::milli>(end - start).count();
    return out;
}

std::vector<uint32_t> makeRandomColumn(std::mt19937 &rng, std::size_t max_value)
{
    std::uniform_int_distribution<uint32_t> dist(0, static_cast<uint32_t>(max_value));
    const std::size_t target = 16 + (rng() % 64);
    std::vector<uint32_t> col;
    col.reserve(target);
    for (std::size_t i = 0; i < target; ++i)
    {
        col.push_back(dist(rng));
    }
    std::sort(col.begin(), col.end());
    col.erase(std::unique(col.begin(), col.end()), col.end());
    return col;
}

} // namespace
// clang-format off
#include "microbenchmarks_gpu_compact.inl"
#include "microbenchmarks_streaming_laplacian.inl"
// clang-format on

} // namespace nerve::validation

#include "nerve/persistence/approximate/perfect_hash.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>

namespace nerve::persistence::perfect
{
namespace
{

double finiteBenchmarkRatio(double numerator, double denominator)
{
    if (!std::isfinite(numerator) || numerator < 0.0 || !std::isfinite(denominator) ||
        denominator <= 0.0)
    {
        return 1.0;
    }
    const double ratio = numerator / denominator;
    return std::isfinite(ratio) && ratio >= 0.0 ? ratio : 1.0;
}

} // namespace

PerfectHashBenchmark benchmarkPerfectHash(const std::vector<int> &keys, int lookup_iterations)
{
    PerfectHashBenchmark bench{};
    if (keys.empty() || lookup_iterations <= 0)
        return bench;
    if (keys.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
        return bench;

    std::vector<int> values(keys.size());
    for (size_t i = 0; i < keys.size(); ++i)
        values[i] = static_cast<int>(i);

    std::unordered_map<int, int> std_map;
    std_map.reserve(keys.size() * 2);
    for (size_t i = 0; i < keys.size(); ++i)
        std_map[keys[i]] = values[i];

    FastStaticMap static_map;
    const bool static_built = static_map.build(keys, values);
    PerfectPivotMap perfect_map;
    const bool perfect_built = perfect_map.build(keys, values);
    bench.std_memory_bytes = std_map.size() * (sizeof(int) * 2 + 16);
    bench.robin_hood_memory_bytes = static_built ? static_map.memoryUsage() : 0;
    bench.perfect_hash_memory_bytes = perfect_built ? perfect_map.memoryUsage() : 0;
    if (!static_built || !perfect_built)
        return bench;

    auto absentKey = [&std_map](int salt) {
        uint64_t state = 1469598103934665603ULL ^ static_cast<uint64_t>(salt);
        for (;;)
        {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            const int candidate = static_cast<int>((state >> 1) & 0x7fffffffULL);
            if (std_map.find(candidate) == std_map.end())
                return candidate;
        }
    };

    std::vector<int> lookup_keys;
    lookup_keys.reserve(static_cast<size_t>(lookup_iterations));
    for (int i = 0; i < lookup_iterations; ++i)
    {
        lookup_keys.push_back((i % 4 == 0) ? absentKey(i)
                                           : keys[static_cast<size_t>(i) % keys.size()]);
    }

    auto start_std = std::chrono::high_resolution_clock::now();
    int64_t sum_std = 0;
    for (int key : lookup_keys)
    {
        auto it = std_map.find(key);
        if (it != std_map.end())
            sum_std += it->second;
    }
    auto end_std = std::chrono::high_resolution_clock::now();
    bench.std_unordered_map_time_ms =
        std::chrono::duration<double, std::milli>(end_std - start_std).count();

    auto start_static = std::chrono::high_resolution_clock::now();
    int64_t sum_static = 0;
    for (int key : lookup_keys)
    {
        auto val = static_map.find(key);
        if (val)
            sum_static += *val;
    }
    auto end_static = std::chrono::high_resolution_clock::now();
    bench.robin_hood_time_ms =
        std::chrono::duration<double, std::milli>(end_static - start_static).count();

    auto start_perfect = std::chrono::high_resolution_clock::now();
    int64_t sum_perfect = 0;
    for (int key : lookup_keys)
    {
        int val = perfect_map.lookup(key);
        if (val >= 0)
            sum_perfect += val;
    }
    auto end_perfect = std::chrono::high_resolution_clock::now();
    bench.perfect_hash_time_ms =
        std::chrono::duration<double, std::milli>(end_perfect - start_perfect).count();
    if (sum_std != sum_static || sum_std != sum_perfect)
        return bench;

    bench.speedup_vs_std =
        finiteBenchmarkRatio(bench.std_unordered_map_time_ms, bench.perfect_hash_time_ms);
    bench.speedup_vs_robin_hood =
        finiteBenchmarkRatio(bench.robin_hood_time_ms, bench.perfect_hash_time_ms);
    bench.memory_reduction_vs_std =
        finiteBenchmarkRatio(static_cast<double>(bench.std_memory_bytes),
                             static_cast<double>(bench.perfect_hash_memory_bytes));
    bench.memory_reduction_vs_robin_hood =
        finiteBenchmarkRatio(static_cast<double>(bench.robin_hood_memory_bytes),
                             static_cast<double>(bench.perfect_hash_memory_bytes));
    return bench;
}

PerfectHashConfig getOptimalPerfectHashConfig(size_t num_keys)
{
    PerfectHashConfig config;
    config.num_buckets_factor = num_keys < 1000 ? 4 : (num_keys < 100000 ? 2 : 1);
    config.max_pilot_value = 255;
    config.verbose = false;
    return config;
}

PerfectHashEstimate estimatePerfectHashBenefit(size_t num_keys, int num_lookups_expected)
{
    PerfectHashEstimate estimate{};
    if (num_keys < 100)
    {
        estimate.recommended = false;
        estimate.query_speedup = 1.0;
        estimate.memory_reduction = 1.0;
        estimate.build_time_ms = 0.0;
        return estimate;
    }
    estimate.query_speedup = 1.3;
    estimate.memory_reduction = 24.0 / 9.0;
    estimate.build_time_ms = static_cast<double>(num_keys) / 1000.0;
    const size_t expected_lookups =
        num_lookups_expected > 0 ? static_cast<size_t>(num_lookups_expected) : 0;
    estimate.recommended =
        num_keys <= std::numeric_limits<size_t>::max() / 10 && expected_lookups > num_keys * 10;
    return estimate;
}

} // namespace nerve::persistence::perfect

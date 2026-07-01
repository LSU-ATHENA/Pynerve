#include "nerve/common/accelerated_types.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core_types.hpp"
#include "nerve/persistence/core/core_types.hpp"
#include "nerve/persistence/utils/bit_parallel_z2.hpp"
#include "nerve/persistence/utils/cpp20_parallel_ph.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <random>
#include <tuple>
#include <vector>

namespace
{

using nerve::persistence::Pair;
using nerve::persistence::cpp20::BitColumn;
using nerve::persistence::cpp20::ParallelConfig;
using nerve::persistence::cpp20::RangeConfig;
using nerve::persistence::cpp20::TaskConfig;

bool check_parallel_config_defaults()
{
    ParallelConfig config;
    if (config.chunk_size != 64)
    {
        std::cerr << "default chunk_size expected 64, got " << config.chunk_size << "\n";
        return false;
    }
    if (!config.use_parallel)
    {
        std::cerr << "default use_parallel should be true\n";
        return false;
    }
    return true;
}

bool check_task_config_defaults()
{
    TaskConfig config;
    if (!config.work_stealing)
    {
        std::cerr << "default work_stealing should be true\n";
        return false;
    }
    if (!config.dynamic_scheduling)
    {
        std::cerr << "default dynamic_scheduling should be true\n";
        return false;
    }
    return true;
}

bool check_range_config_defaults()
{
    RangeConfig config;
    if (config.max_columns != 0)
    {
        std::cerr << "default max_columns expected 0, got " << config.max_columns << "\n";
        return false;
    }
    if (config.batch_size != 1000)
    {
        std::cerr << "default batch_size expected 1000, got " << config.batch_size << "\n";
        return false;
    }
    return true;
}

bool check_optimal_parallel_config()
{
    auto config = nerve::persistence::cpp20::getOptimalParallelConfig(1000, 64);
    if (config.chunk_size == 0)
    {
        std::cerr << "optimal chunk_size should be positive\n";
        return false;
    }
    return true;
}

bool check_optimal_task_config()
{
    auto config = nerve::persistence::cpp20::getOptimalTaskConfig(1000);
    if (config.num_threads == 0)
    {
        std::cerr << "optimal num_threads should be auto-detected (0)\n";
    }
    return true;
}

bool check_parallel_speedup_estimate()
{
    auto estimate = nerve::persistence::cpp20::estimateParallelSpeedup(1000, 4);
    if (estimate.total_speedup <= 0.0)
    {
        std::cerr << "speedup estimate should be positive\n";
        return false;
    }
    return true;
}

bool check_should_use_parallel()
{
    bool use_small = nerve::persistence::cpp20::shouldUseParallel(100);
    bool use_large = nerve::persistence::cpp20::shouldUseParallel(1000);

    if (use_small)
    {
        std::cerr << "shouldUseParallel(100) expected false\n";
        return false;
    }
    if (!use_large)
    {
        std::cerr << "shouldUseParallel(1000) expected true\n";
        return false;
    }
    return true;
}

bool check_parallel_reduction_result_default()
{
    nerve::persistence::cpp20::ParallelReductionResult result;
    if (result.reduction_time_ms != 0.0)
    {
        std::cerr << "default reduction_time_ms should be 0.0\n";
        return false;
    }
    if (result.num_columns_processed != 0)
    {
        std::cerr << "default num_columns_processed should be 0\n";
        return false;
    }
    return true;
}

bool check_task_reduction_result_default()
{
    nerve::persistence::cpp20::TaskBasedReductionResult result;
    if (result.reduction_time_ms != 0.0)
    {
        std::cerr << "default reduction_time_ms should be 0.0\n";
        return false;
    }
    if (result.num_threads_used != 0)
    {
        std::cerr << "default num_threads_used should be 0\n";
        return false;
    }
    return true;
}

bool check_parallel_benchmark_default()
{
    nerve::persistence::cpp20::ParallelBenchmark bench;
    if (bench.speedup != 1.0)
    {
        std::cerr << "default speedup should be 1.0\n";
        return false;
    }
    if (bench.num_threads != 1)
    {
        std::cerr << "default num_threads should be 1\n";
        return false;
    }
    return true;
}

} // namespace

int main()
{
    if (!check_parallel_config_defaults())
    {
        std::cerr << "FAIL: parallel config defaults\n";
        return 1;
    }
    if (!check_task_config_defaults())
    {
        std::cerr << "FAIL: task config defaults\n";
        return 1;
    }
    if (!check_range_config_defaults())
    {
        std::cerr << "FAIL: range config defaults\n";
        return 1;
    }
    if (!check_optimal_parallel_config())
    {
        std::cerr << "FAIL: optimal parallel config\n";
        return 1;
    }
    if (!check_optimal_task_config())
    {
        std::cerr << "FAIL: optimal task config\n";
        return 1;
    }
    if (!check_parallel_speedup_estimate())
    {
        std::cerr << "FAIL: parallel speedup estimate\n";
        return 1;
    }
    if (!check_should_use_parallel())
    {
        std::cerr << "FAIL: should use parallel\n";
        return 1;
    }
    if (!check_parallel_reduction_result_default())
    {
        std::cerr << "FAIL: parallel reduction result default\n";
        return 1;
    }
    if (!check_task_reduction_result_default())
    {
        std::cerr << "FAIL: task reduction result default\n";
        return 1;
    }
    if (!check_parallel_benchmark_default())
    {
        std::cerr << "FAIL: parallel benchmark default\n";
        return 1;
    }
    return 0;
}

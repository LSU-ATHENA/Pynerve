#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core_types.hpp"
#include "nerve/persistence/core/core_types.hpp"
#include "nerve/persistence/core/per_dimension_exact.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

namespace
{

using nerve::Dimension;
using nerve::Index;
using nerve::Size;
using nerve::persistence::perdim::H0Result;
using nerve::persistence::perdim::H1Result;
using nerve::persistence::perdim::H2Result;
using nerve::persistence::perdim::PerDimensionConfig;
using nerve::persistence::perdim::PerDimensionResult;

bool check_per_dim_config_defaults()
{
    PerDimensionConfig config;
    if (config.max_dim != 6)
    {
        std::cerr << "default max_dim expected 6, got " << config.max_dim << "\n";
        return false;
    }
    if (!config.compute_h0 || !config.compute_h1 || !config.compute_h2)
    {
        std::cerr << "default should have h0, h1, h2 enabled\n";
        return false;
    }
    return true;
}

bool check_per_dim_config_h0_only()
{
    PerDimensionConfig config;
    config.compute_h1 = false;
    config.compute_h2 = false;
    config.compute_h3 = false;
    config.compute_h4 = false;
    config.compute_h5 = false;
    config.compute_h6 = false;

    if (!config.compute_h0)
    {
        std::cerr << "h0 should be enabled\n";
        return false;
    }
    if (config.compute_h1)
    {
        std::cerr << "h1 should be disabled\n";
        return false;
    }
    return true;
}

bool check_per_dim_config_h1_only()
{
    PerDimensionConfig config;
    config.compute_h0 = false;
    config.compute_h2 = false;
    config.compute_h3 = false;
    config.compute_h4 = false;
    config.compute_h5 = false;
    config.compute_h6 = false;

    if (!config.compute_h1)
    {
        std::cerr << "h1 should be enabled\n";
        return false;
    }
    if (config.compute_h0)
    {
        std::cerr << "h0 should be disabled\n";
        return false;
    }
    return true;
}

bool check_per_dim_config_h2_only()
{
    PerDimensionConfig config;
    config.compute_h0 = false;
    config.compute_h1 = false;
    config.compute_h3 = false;
    config.compute_h4 = false;
    config.compute_h5 = false;
    config.compute_h6 = false;

    if (!config.compute_h2)
    {
        std::cerr << "h2 should be enabled\n";
        return false;
    }
    if (config.compute_h0)
    {
        std::cerr << "h0 should be disabled\n";
        return false;
    }
    return true;
}

bool check_optimal_per_dim_config()
{
    auto config = nerve::persistence::perdim::getOptimalPerDimensionConfig(100, 2, 2);
    if (config.max_dim < 2)
    {
        std::cerr << "optimal max_dim should be >= 2\n";
        return false;
    }
    return true;
}

bool check_estimate_per_dim_speedup()
{
    double speedup = nerve::persistence::perdim::estimatePerDimensionSpeedup(100, 2, 2);
    if (speedup <= 0.0)
    {
        std::cerr << "speedup estimate should be positive, got " << speedup << "\n";
        return false;
    }
    return true;
}

bool check_h0_result_default()
{
    H0Result result;
    if (!result.pairs.empty())
    {
        std::cerr << "default H0 pairs should be empty\n";
        return false;
    }
    if (result.time_ms != 0.0)
    {
        std::cerr << "default H0 time_ms should be 0.0\n";
        return false;
    }
    if (result.num_pairs != 0)
    {
        std::cerr << "default H0 num_pairs should be 0\n";
        return false;
    }
    return true;
}

bool check_h1_result_default()
{
    H1Result result;
    if (!result.pairs.empty())
    {
        std::cerr << "default H1 pairs should be empty\n";
        return false;
    }
    if (result.essential_count != 0)
    {
        std::cerr << "default H1 essential_count should be 0\n";
        return false;
    }
    return true;
}

bool check_h2_result_default()
{
    H2Result result;
    if (!result.pairs.empty())
    {
        std::cerr << "default H2 pairs should be empty\n";
        return false;
    }
    if (result.num_pairs != 0)
    {
        std::cerr << "default H2 num_pairs should be 0\n";
        return false;
    }
    return true;
}

bool check_per_dim_result_default()
{
    PerDimensionResult result;
    if (!result.all_pairs.empty())
    {
        std::cerr << "default all_pairs should be empty\n";
        return false;
    }
    if (result.total_time_ms != 0.0)
    {
        std::cerr << "default total_time_ms should be 0.0\n";
        return false;
    }
    return true;
}

bool check_per_dim_result_counts_default()
{
    PerDimensionResult result;
    if (result.h0_pairs != 0 || result.h1_pairs != 0 || result.h2_pairs != 0)
    {
        std::cerr << "default per-dimension pair counts should be 0\n";
        return false;
    }
    return true;
}

} // namespace

int main()
{
    if (!check_per_dim_config_defaults())
    {
        std::cerr << "FAIL: per-dim config defaults\n";
        return 1;
    }
    if (!check_per_dim_config_h0_only())
    {
        std::cerr << "FAIL: per-dim config h0 only\n";
        return 1;
    }
    if (!check_per_dim_config_h1_only())
    {
        std::cerr << "FAIL: per-dim config h1 only\n";
        return 1;
    }
    if (!check_per_dim_config_h2_only())
    {
        std::cerr << "FAIL: per-dim config h2 only\n";
        return 1;
    }
    if (!check_optimal_per_dim_config())
    {
        std::cerr << "FAIL: optimal per-dim config\n";
        return 1;
    }
    if (!check_estimate_per_dim_speedup())
    {
        std::cerr << "FAIL: estimate per-dim speedup\n";
        return 1;
    }
    if (!check_h0_result_default())
    {
        std::cerr << "FAIL: H0 result default\n";
        return 1;
    }
    if (!check_h1_result_default())
    {
        std::cerr << "FAIL: H1 result default\n";
        return 1;
    }
    if (!check_h2_result_default())
    {
        std::cerr << "FAIL: H2 result default\n";
        return 1;
    }
    if (!check_per_dim_result_default())
    {
        std::cerr << "FAIL: per-dim result default\n";
        return 1;
    }
    if (!check_per_dim_result_counts_default())
    {
        std::cerr << "FAIL: per-dim result counts default\n";
        return 1;
    }
    return 0;
}

#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core_types.hpp"
#include "nerve/persistence/core/core_types.hpp"
#include "nerve/persistence/core/flood_complex.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

namespace
{

using nerve::persistence::FloodComplexConfig;
using nerve::persistence::FloodComplexResult;
using nerve::persistence::IndexedPoint;

bool check_flood_config_defaults()
{
    FloodComplexConfig config;
    if (config.max_dim != 2)
    {
        std::cerr << "default max_dim expected 2, got " << config.max_dim << "\n";
        return false;
    }
    if (std::abs(config.max_radius - 1.0) > 1e-12)
    {
        std::cerr << "default max_radius expected 1.0, got " << config.max_radius << "\n";
        return false;
    }
    if (std::abs(config.subset_ratio - 0.05) > 1e-12)
    {
        std::cerr << "default subset_ratio expected 0.05, got " << config.subset_ratio << "\n";
        return false;
    }
    if (config.max_subset_size != 5000)
    {
        std::cerr << "default max_subset_size expected 5000, got " << config.max_subset_size
                  << "\n";
        return false;
    }
    if (!config.use_flooding)
    {
        std::cerr << "default use_flooding should be true\n";
        return false;
    }
    return true;
}

bool check_optimal_flood_config()
{
    auto config = nerve::persistence::getOptimalFloodConfig(100, 2);
    if (config.max_dim == 0)
    {
        std::cerr << "optimal flood config max_dim should be positive\n";
        return false;
    }
    return true;
}

bool check_should_use_flood_complex()
{
    bool use_large = nerve::persistence::shouldUseFloodComplex(10000, 2);
    bool use_small = nerve::persistence::shouldUseFloodComplex(100, 2);

    if (!use_large)
    {
        std::cerr << "shouldUseFloodComplex(10000,2) expected true\n";
        return false;
    }
    if (use_small)
    {
        std::cerr << "shouldUseFloodComplex(100,2) expected false\n";
        return false;
    }
    return true;
}

bool check_estimate_flood_memory()
{
    auto mem = nerve::persistence::estimateFloodComplexMemory(1000, 2, FloodComplexConfig{});
    if (mem == 0)
    {
        std::cerr << "flood memory estimate should be positive\n";
        return false;
    }
    return true;
}

bool check_indexed_point_construction()
{
    IndexedPoint pt({0.0, 0.0, 1.0, 0.0}, 0);
    if (pt.original_index != 0)
    {
        std::cerr << "indexed point original_index expected 0, got " << pt.original_index << "\n";
        return false;
    }
    if (pt.coords.size() != 4)
    {
        std::cerr << "indexed point coords size expected 4, got " << pt.coords.size() << "\n";
        return false;
    }
    return true;
}

bool check_indexed_point_initializer_list()
{
    IndexedPoint pt({1.0, 2.0, 3.0}, 5);
    if (pt.original_index != 5)
    {
        std::cerr << "initializer list original_index expected 5, got " << pt.original_index
                  << "\n";
        return false;
    }
    if (pt.coords.size() != 3)
    {
        std::cerr << "initializer list coords size expected 3, got " << pt.coords.size() << "\n";
        return false;
    }
    return true;
}

bool check_flood_complex_result_default()
{
    FloodComplexResult result;
    if (!result.pairs.empty())
    {
        std::cerr << "default pairs should be empty\n";
        return false;
    }
    if (result.total_time_ms != 0.0)
    {
        std::cerr << "default total_time_ms should be 0.0\n";
        return false;
    }
    if (result.original_points != 0)
    {
        std::cerr << "default original_points should be 0\n";
        return false;
    }
    return true;
}

bool check_flood_complex_on_simple_set()
{
    std::vector<double> points = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0};
    FloodComplexConfig config;
    config.max_dim = 1;
    config.max_radius = 1.5;
    config.subset_ratio = 1.0;
    config.max_subset_size = 100;

    auto result = nerve::persistence::computeFloodComplex(points, 2, 3, config);
    if (result.pairs.empty())
    {
        std::cerr << "flood complex on triangle should produce pairs\n";
        return false;
    }
    bool found_h0_ess = false;
    for (const auto &p : result.pairs)
    {
        if (p.dimension == 0 && p.isInfinite())
            found_h0_ess = true;
    }
    if (!found_h0_ess)
    {
        std::cerr << "flood complex on triangle should have H0 essential\n";
        return false;
    }
    if (result.original_points != 3)
    {
        std::cerr << "flood complex original_points expected 3\n";
        return false;
    }
    return true;
}

} // namespace

int main()
{
    if (!check_flood_config_defaults())
    {
        std::cerr << "FAIL: flood config defaults\n";
        return 1;
    }
    if (!check_optimal_flood_config())
    {
        std::cerr << "FAIL: optimal flood config\n";
        return 1;
    }
    if (!check_should_use_flood_complex())
    {
        std::cerr << "FAIL: should use flood complex\n";
        return 1;
    }
    if (!check_estimate_flood_memory())
    {
        std::cerr << "FAIL: estimate flood memory\n";
        return 1;
    }
    if (!check_indexed_point_construction())
    {
        std::cerr << "FAIL: indexed point construction\n";
        return 1;
    }
    if (!check_indexed_point_initializer_list())
    {
        std::cerr << "FAIL: indexed point initializer list\n";
        return 1;
    }
    if (!check_flood_complex_result_default())
    {
        std::cerr << "FAIL: flood complex result default\n";
        return 1;
    }
    if (!check_flood_complex_on_simple_set())
    {
        std::cerr << "FAIL: flood complex on simple set\n";
        return 1;
    }
    return 0;
}

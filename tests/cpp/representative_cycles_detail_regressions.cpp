
#include "nerve/persistence/adaptive_acceleration/streaming/representative_cycles.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

namespace
{

using nerve::persistence::adaptive_acceleration::representative::Cycle;
using nerve::persistence::adaptive_acceleration::representative::CycleVisualizationData;
using nerve::persistence::adaptive_acceleration::representative::RepresentativeConfig;
using nerve::persistence::adaptive_acceleration::representative::RepresentativeStats;

bool check_cycle_default_construction()
{
    Cycle c;
    if (!c.vertices.empty())
        return false;
    if (!c.coefficients.empty())
        return false;
    if (c.dimension != 0)
        return false;
    if (std::abs(c.birth_time - 0.0) > 1e-10)
        return false;
    if (std::abs(c.death_time - 0.0) > 1e-10)
        return false;
    if (c.isValid())
        return false;
    return true;
}

bool check_cycle_valid()
{
    Cycle c;
    c.vertices = {0, 1, 2};
    c.coefficients = {1.0, 1.0, 1.0};
    c.dimension = 1;
    c.birth_time = 0.0;
    c.death_time = 1.0;
    if (!c.isValid())
        return false;
    if (c.size() != 3)
        return false;
    if (std::abs(c.persistence() - 1.0) > 1e-10)
        return false;
    return true;
}

bool check_cycle_infinite_death()
{
    Cycle c;
    c.vertices = {0, 1};
    c.coefficients = {1.0, 1.0};
    c.dimension = 0;
    c.birth_time = 0.0;
    c.death_time = std::numeric_limits<double>::infinity();
    if (!c.isValid())
        return false;
    if (!std::isinf(c.persistence()))
        return false;
    return true;
}

bool check_cycle_invalid_empty_vertices()
{
    Cycle c;
    c.coefficients = {1.0};
    c.dimension = 0;
    c.birth_time = 0.0;
    c.death_time = 1.0;
    if (c.isValid())
        return false;
    return true;
}

bool check_cycle_invalid_size_mismatch()
{
    Cycle c;
    c.vertices = {0, 1, 2};
    c.coefficients = {1.0, 1.0};
    c.dimension = 1;
    c.birth_time = 0.0;
    c.death_time = 1.0;
    if (c.isValid())
        return false;
    return true;
}

bool check_cycle_death_before_birth()
{
    Cycle c;
    c.vertices = {0, 1};
    c.coefficients = {1.0, 1.0};
    c.dimension = 0;
    c.birth_time = 2.0;
    c.death_time = 1.0;
    if (c.isValid())
        return false;
    return true;
}

bool check_representative_config_default()
{
    RepresentativeConfig cfg;
    if (!cfg.use_matrix_multiplication)
        return false;
    if (cfg.use_dual_cohomology)
        return false;
    if (!cfg.enable_fast_computation)
        return false;
    if (!cfg.enable_compression)
        return false;
    if (!cfg.enable_parallel_computation)
        return false;
    if (cfg.max_cycles_per_dimension != 1000)
        return false;
    if (std::abs(cfg.min_persistence - 1e-6) > 1e-10)
        return false;
    if (!cfg.enable_visualization_data)
        return false;
    return true;
}

bool check_representative_stats_default()
{
    RepresentativeStats stats;
    if (std::abs(stats.computation_time_ms - 0.0) > 1e-10)
        return false;
    if (stats.total_cycles_computed != 0)
        return false;
    for (int i = 0; i < 4; ++i)
    {
        if (stats.cycles_per_dimension[i] != 0)
            return false;
    }
    if (std::abs(stats.average_cycle_size - 0.0) > 1e-10)
        return false;
    if (std::abs(stats.max_cycle_size - 0.0) > 1e-10)
        return false;
    if (std::abs(stats.min_cycle_size - 0.0) > 1e-10)
        return false;
    if (stats.used_matrix_multiplication)
        return false;
    if (stats.used_dual_cohomology)
        return false;
    if (!stats.computation_details.empty())
        return false;
    return true;
}

bool check_cycle_single_vertex()
{
    Cycle c;
    c.vertices = {0};
    c.coefficients = {1.0};
    c.dimension = 0;
    c.birth_time = 0.0;
    c.death_time = 0.5;
    if (!c.isValid())
        return false;
    if (c.size() != 1)
        return false;
    return true;
}

bool check_cycle_nan_coefficient()
{
    Cycle c;
    c.vertices = {0, 1};
    c.coefficients = {1.0, std::numeric_limits<double>::quiet_NaN()};
    c.dimension = 0;
    c.birth_time = 0.0;
    c.death_time = 1.0;
    if (c.isValid())
        return false;
    return true;
}

bool check_cycle_visualization_data_default()
{
    CycleVisualizationData data;
    if (!data.vertices.empty())
        return false;
    if (!data.edges.empty())
        return false;
    if (!data.edge_weights.empty())
        return false;
    if (data.dimension != 0)
        return false;
    if (std::abs(data.persistence - 0.0) > 1e-10)
        return false;
    if (data.isValid())
        return false;
    return true;
}

bool check_cycle_visualization_data_valid()
{
    CycleVisualizationData data;
    data.vertices = {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};
    data.edges = {{0, 1}, {1, 2}, {0, 2}};
    data.edge_weights = {1.0, 1.0, 1.0};
    data.dimension = 1;
    data.persistence = 1.0;
    if (!data.isValid())
        return false;
    return true;
}

} // namespace

int main()
{
    if (!check_cycle_default_construction())
    {
        std::cerr << "FAIL: cycle default construction\n";
        return 1;
    }
    if (!check_cycle_valid())
    {
        std::cerr << "FAIL: cycle valid\n";
        return 1;
    }
    if (!check_cycle_infinite_death())
    {
        std::cerr << "FAIL: cycle infinite death\n";
        return 1;
    }
    if (!check_cycle_invalid_empty_vertices())
    {
        std::cerr << "FAIL: cycle invalid empty vertices\n";
        return 1;
    }
    if (!check_cycle_invalid_size_mismatch())
    {
        std::cerr << "FAIL: cycle invalid size mismatch\n";
        return 1;
    }
    if (!check_cycle_death_before_birth())
    {
        std::cerr << "FAIL: cycle death before birth\n";
        return 1;
    }
    if (!check_representative_config_default())
    {
        std::cerr << "FAIL: representative config default\n";
        return 1;
    }
    if (!check_representative_stats_default())
    {
        std::cerr << "FAIL: representative stats default\n";
        return 1;
    }
    if (!check_cycle_single_vertex())
    {
        std::cerr << "FAIL: cycle single vertex\n";
        return 1;
    }
    if (!check_cycle_nan_coefficient())
    {
        std::cerr << "FAIL: cycle nan coefficient\n";
        return 1;
    }
    if (!check_cycle_visualization_data_default())
    {
        std::cerr << "FAIL: cycle visualization data default\n";
        return 1;
    }
    if (!check_cycle_visualization_data_valid())
    {
        std::cerr << "FAIL: cycle visualization data valid\n";
        return 1;
    }
    return 0;
}

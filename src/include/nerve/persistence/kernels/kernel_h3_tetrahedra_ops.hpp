
#pragma once

#include "nerve/core.hpp"
#include "nerve/persistence/core/core_types.hpp"

#include <chrono>
#include <vector>

namespace nerve::persistence::h3
{

struct H3Config
{
    bool use_bit_parallel = false;
    bool use_clear_compress = false;
    int chunk_size = 16384;
};

struct H3Pair
{
    int birth_index;
    int death_index;
    double birth_time;
    double death_time;
};

struct H3Result
{
    std::vector<H3Pair> pairs;

    // Metadata
    int num_tetrahedra = 0;
    int num_reductions = 0;
    bool used_bit_parallel = false;
    bool used_clear_compress = false;
    std::string algorithm_used;
    H3Config config;

    // Timing
    double enumeration_time_ms = 0.0;
    double boundary_time_ms = 0.0;
    double computation_time_ms = 0.0;
    double total_time_ms = 0.0;
};

struct H3SpeedupEstimate
{
    double tetrahedra_layout_speedup = 1.0;
    double bit_parallel_speedup = 1.0;
    double cache_speedup = 1.0;
    double total_speedup = 1.0;
};

H3Result computeH3Tetrahedra(const std::vector<std::vector<int>> &simplices,
                             const std::vector<double> &filtration_values,
                             const std::vector<int> &dimensions, const H3Config &config);

H3Config getOptimalH3Config(size_t num_tetrahedra);
H3SpeedupEstimate estimateH3Speedup(size_t num_tetrahedra);

} // namespace nerve::persistence::h3

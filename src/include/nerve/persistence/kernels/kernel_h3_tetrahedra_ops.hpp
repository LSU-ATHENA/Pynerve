#pragma once

#include "nerve/types.hpp"

#include <cstddef>
#include <string>
#include <string>
#include <vector>

namespace nerve::persistence::h3
{

struct H3Pair
{
    Index birth_index = -1;
    Index death_index = -1;
    double birth_time = 0.0;
    double death_time = 0.0;
};

struct H3Config
{
    bool use_bit_parallel = false;
    bool use_clear_compress = false;
    std::size_t chunk_size = 16384;
};

struct H3Result
{
    std::vector<H3Pair> pairs;
    double computation_time_ms = 0.0;
    double total_time_ms = 0.0;
    int num_tetrahedra = 0;
    int num_reductions = 0;
    H3Config config;
    std::string algorithm_used;
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

H3Config getOptimalH3Config(std::size_t num_tetrahedra);

H3SpeedupEstimate estimateH3Speedup(std::size_t num_tetrahedra);

} // namespace nerve::persistence::h3

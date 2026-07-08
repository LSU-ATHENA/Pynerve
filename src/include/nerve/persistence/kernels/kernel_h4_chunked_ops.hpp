#pragma once

#include "nerve/types.hpp"

#include <cstddef>
#include <vector>

namespace nerve::persistence::h4
{

struct H4Pair
{
    Index birth_index = -1;
    Index death_index = -1;
    double birth_time = 0.0;
    double death_time = 0.0;
};

struct H4Config
{
    bool use_parallel = false;
    bool use_bit_parallel = false;
    bool use_clear_compress = false;
    std::size_t chunk_size = 16384;
};

struct H4Result
{
    std::vector<H4Pair> pairs;
    double computation_time_ms = 0.0;
    H4Config config;
    std::size_t chunk_size = 0;
    int num_4simplices = 0;
    int num_columns_reduced = 0;
    bool used_chunking = false;
};

H4Result computeH4Chunked(const std::vector<std::vector<int>> &simplices,
                          const std::vector<double> &filtration_values,
                          const std::vector<int> &dimensions, const H4Config &config);

H4Config getOptimalH4Config(std::size_t num_4simplices);

} // namespace nerve::persistence::h4

#pragma once

#include "nerve/types.hpp"

#include <cstddef>
#include <vector>

namespace nerve::persistence::h6
{

struct H6Pair
{
    Index birth_index = -1;
    Index death_index = -1;
    double birth_time = 0.0;
    double death_time = 0.0;
};

struct H6Config
{
    bool use_streaming = false;
    bool use_bit_parallel = false;
    bool use_clear_compress = false;
    std::size_t chunk_size = 4096;
};

struct H6Result
{
    std::vector<H6Pair> pairs;
    double computation_time_ms = 0.0;
    H6Config config;
    std::size_t chunk_size = 0;
    int num_6simplices = 0;
    bool used_streaming = false;
};

H6Result computeH6Streaming(const std::vector<std::vector<int>> &simplices,
                            const std::vector<double> &filtration_values,
                            const std::vector<int> &dimensions, const H6Config &config);

H6Config getOptimalH6Config(std::size_t num_6simplices);

} // namespace nerve::persistence::h6

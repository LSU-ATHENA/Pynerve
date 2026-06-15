
#pragma once

#include "nerve/core.hpp"
#include "nerve/persistence/core/core_types.hpp"

#include <chrono>
#include <cstddef>
#include <vector>

namespace nerve::persistence::h5
{

struct H5Config
{
    std::size_t prefetch_distance = 8;
    std::size_t block_size = 256;
    bool use_prefetch = true;
    bool use_prefetching = true;
    bool use_bit_parallel = false;
    bool use_clear_compress = false;
};

struct H5Pair
{
    int birth_index;
    int death_index;
    double birth_time;
    double death_time;
};

struct H5Result
{
    std::vector<H5Pair> pairs;

    // Metadata
    int num_5simplices = 0;
    std::size_t num_prefetched = 0;
    bool used_prefetching = false;
    bool used_bit_parallel = false;
    H5Config config;

    // Timing
    double computation_time_ms = 0.0;
};

H5Result computeH5Prefetch(const std::vector<std::vector<int>> &simplices,
                           const std::vector<double> &filtration_values,
                           const std::vector<int> &dimensions, const H5Config &config);

H5Config getOptimalH5Config(size_t num_5simplices);

} // namespace nerve::persistence::h5

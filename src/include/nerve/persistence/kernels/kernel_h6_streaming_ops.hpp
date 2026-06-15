
#pragma once

#include "nerve/core.hpp"
#include "nerve/persistence/core/core_types.hpp"

#include <chrono>
#include <vector>

namespace nerve::persistence::h6
{

struct H6Config
{
    int chunk_size = 4096; // Small chunks for memory efficiency
    bool use_streaming = false;
    bool use_bit_parallel = false;
    bool use_clear_compress = false;
};

struct H6Pair
{
    int birth_index;
    int death_index;
    double birth_time;
    double death_time;
};

struct H6Result
{
    std::vector<H6Pair> pairs;

    // Metadata
    int num_6simplices = 0;
    bool used_streaming = false;
    bool used_bit_parallel = false;
    int chunk_size = 0;
    H6Config config;

    // Timing
    double computation_time_ms = 0.0;
};

H6Result computeH6Streaming(const std::vector<std::vector<int>> &simplices,
                            const std::vector<double> &filtration_values,
                            const std::vector<int> &dimensions, const H6Config &config);

H6Config getOptimalH6Config(size_t num_6simplices);

} // namespace nerve::persistence::h6


#pragma once

#include "nerve/core.hpp"
#include "nerve/persistence/core/core_types.hpp"

#include <chrono>
#include <cstddef>
#include <vector>

namespace nerve::persistence::h4
{

struct H4Config
{
    int chunk_size = 16384; // L2 cache optimized
    int max_memory_mb = 1024;
    bool use_parallel = false;
    bool use_bit_parallel = false;
    bool use_clear_compress = false;
};

struct H4Pair
{
    int birth_index;
    int death_index;
    double birth_time;
    double death_time;
};

struct H4Result
{
    std::vector<H4Pair> pairs;

    // Metadata
    int num_4simplices = 0;
    int num_columns_reduced = 0;
    bool used_bit_parallel = false;
    bool used_chunking = false;
    int chunk_size = 0;
    H4Config config;

    // Timing
    double computation_time_ms = 0.0;
};

H4Result computeH4Chunked(const std::vector<std::vector<int>> &simplices,
                          const std::vector<double> &filtration_values,
                          const std::vector<int> &dimensions, const H4Config &config);

H4Config getOptimalH4Config(size_t num_4simplices);

} // namespace nerve::persistence::h4

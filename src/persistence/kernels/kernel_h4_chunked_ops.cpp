// H4 exact kernel with cache-sized reduction configuration.

#include "nerve/persistence/kernels/kernel_h4_chunked_ops.hpp"
#include "nerve/persistence/kernels/simplex_reduction_utils.hpp"

#include <algorithm>
#include <chrono>

namespace nerve::persistence::h4
{

namespace
{

H4Pair toH4Pair(const Pair &pair)
{
    return H4Pair{pair.birth_index, pair.death_index, pair.birth, pair.death};
}

} // namespace

H4Result computeH4Chunked(const std::vector<std::vector<int>> &simplices,
                          const std::vector<double> &filtration_values,
                          const std::vector<int> &dimensions, const H4Config &config)
{
    H4Result result;
    result.config = config;
    result.chunk_size = config.chunk_size;

    const auto start = std::chrono::high_resolution_clock::now();
    const auto exact =
        kernels::computeExactDimensionPairs(simplices, filtration_values, dimensions, 4);

    result.num_4simplices = static_cast<int>(exact.simplex_count);
    result.num_columns_reduced = static_cast<int>(exact.reduction_operations);
    result.pairs.reserve(exact.pairs.size());
    for (const auto &pair : exact.pairs)
    {
        result.pairs.push_back(toH4Pair(pair));
    }

    const auto end = std::chrono::high_resolution_clock::now();
    result.computation_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    return result;
}

H4Config getOptimalH4Config(size_t num_4simplices)
{
    H4Config config;
    (void)num_4simplices;
    config.chunk_size = 16384;
    config.use_parallel = false;
    config.use_bit_parallel = false;
    config.use_clear_compress = false;
    return config;
}

} // namespace nerve::persistence::h4

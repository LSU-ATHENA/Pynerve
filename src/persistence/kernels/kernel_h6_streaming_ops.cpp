// H6 exact kernel for high-dimensional filtered complexes.

#include "nerve/persistence/kernels/kernel_h6_streaming_ops.hpp"
#include "nerve/persistence/kernels/simplex_reduction_utils.hpp"

#include <chrono>

namespace nerve::persistence::h6
{

namespace
{

H6Pair toH6Pair(const Pair &pair)
{
    return H6Pair{pair.birth_index, pair.death_index, pair.birth, pair.death};
}

} // namespace

H6Result computeH6Streaming(const std::vector<std::vector<int>> &simplices,
                            const std::vector<double> &filtration_values,
                            const std::vector<int> &dimensions, const H6Config &config)
{
    H6Result result;
    result.config = config;
    result.chunk_size = config.chunk_size;

    const auto start = std::chrono::high_resolution_clock::now();
    const auto exact =
        kernels::computeExactDimensionPairs(simplices, filtration_values, dimensions, 6);

    result.num_6simplices = static_cast<int>(exact.simplex_count);
    result.pairs.reserve(exact.pairs.size());
    for (const auto &pair : exact.pairs)
    {
        result.pairs.push_back(toH6Pair(pair));
    }

    const auto end = std::chrono::high_resolution_clock::now();
    result.computation_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    return result;
}

H6Config getOptimalH6Config(size_t num_6simplices)
{
    H6Config config;
    (void)num_6simplices;
    config.chunk_size = 4096;
    config.use_streaming = false;
    config.use_bit_parallel = false;
    config.use_clear_compress = false;
    return config;
}

} // namespace nerve::persistence::h6

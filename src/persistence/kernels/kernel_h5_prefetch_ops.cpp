// H5 exact kernel with software prefetch over input simplex storage.

#include "nerve/persistence/kernels/kernel_h5_prefetch_ops.hpp"
#include "nerve/persistence/kernels/simplex_reduction_utils.hpp"

#include <chrono>

namespace nerve::persistence::h5
{

namespace
{

void prefetchSimplex(const std::vector<int> &simplex)
{
    if (simplex.empty())
    {
        return;
    }
#if defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(simplex.data(), 0, 3);
#else
    (void)simplex;
#endif
}

H5Pair toH5Pair(const Pair &pair)
{
    return H5Pair{pair.birth_index, pair.death_index, pair.birth, pair.death};
}

} // namespace

H5Result computeH5Prefetch(const std::vector<std::vector<int>> &simplices,
                           const std::vector<double> &filtration_values,
                           const std::vector<int> &dimensions, const H5Config &config)
{
    H5Result result;
    result.config = config;

    const auto start = std::chrono::high_resolution_clock::now();
    const bool prefetch_enabled = config.use_prefetch || config.use_prefetching;
    if (prefetch_enabled && config.prefetch_distance > 0)
    {
        for (std::size_t i = 0; i < simplices.size(); ++i)
        {
            const std::size_t next = i + config.prefetch_distance;
            if (next < simplices.size())
            {
                prefetchSimplex(simplices[next]);
                ++result.num_prefetched;
            }
        }
        result.used_prefetching = result.num_prefetched > 0;
    }

    const auto exact =
        kernels::computeExactDimensionPairs(simplices, filtration_values, dimensions, 5);

    result.num_5simplices = static_cast<int>(exact.simplex_count);
    result.pairs.reserve(exact.pairs.size());
    for (const auto &pair : exact.pairs)
    {
        result.pairs.push_back(toH5Pair(pair));
    }

    const auto end = std::chrono::high_resolution_clock::now();
    result.computation_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    return result;
}

H5Config getOptimalH5Config(size_t num_5simplices)
{
    H5Config config;
    config.use_prefetch = (num_5simplices > 1000);
    config.use_prefetching = config.use_prefetch;
    config.prefetch_distance = 8;
    config.use_bit_parallel = false;
    config.use_clear_compress = false;
    return config;
}

} // namespace nerve::persistence::h5

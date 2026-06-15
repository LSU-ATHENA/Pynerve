// H3 exact kernel for tetrahedra-bearing filtered complexes.

#include "nerve/persistence/kernels/kernel_h3_tetrahedra_ops.hpp"
#include "nerve/persistence/kernels/simplex_reduction_utils.hpp"

#include <chrono>

namespace nerve::persistence::h3
{

namespace
{

H3Pair toH3Pair(const Pair &pair)
{
    return H3Pair{pair.birth_index, pair.death_index, pair.birth, pair.death};
}

} // namespace

H3Result computeH3Tetrahedra(const std::vector<std::vector<int>> &simplices,
                             const std::vector<double> &filtration_values,
                             const std::vector<int> &dimensions, const H3Config &config)
{
    H3Result result;
    result.config = config;
    result.algorithm_used = "H3-Exact-Z2";

    const auto start = std::chrono::high_resolution_clock::now();
    const auto exact =
        kernels::computeExactDimensionPairs(simplices, filtration_values, dimensions, 3);

    result.num_tetrahedra = static_cast<int>(exact.simplex_count);
    result.num_reductions = static_cast<int>(exact.reduction_operations);
    result.pairs.reserve(exact.pairs.size());
    for (const auto &pair : exact.pairs)
    {
        result.pairs.push_back(toH3Pair(pair));
    }

    const auto end = std::chrono::high_resolution_clock::now();
    result.computation_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.total_time_ms = result.computation_time_ms;
    return result;
}

H3Config getOptimalH3Config(size_t num_tetrahedra)
{
    H3Config config;
    config.use_bit_parallel = false;
    (void)num_tetrahedra;
    config.use_clear_compress = false;
    config.chunk_size = 16384;
    return config;
}

H3SpeedupEstimate estimateH3Speedup(size_t num_tetrahedra)
{
    H3SpeedupEstimate estimate;
    estimate.tetrahedra_layout_speedup = 1.0;
    estimate.bit_parallel_speedup = 1.0;
    estimate.cache_speedup = (num_tetrahedra > 50000) ? 1.2 : 1.0;
    estimate.total_speedup =
        estimate.tetrahedra_layout_speedup * estimate.bit_parallel_speedup * estimate.cache_speedup;
    return estimate;
}

} // namespace nerve::persistence::h3

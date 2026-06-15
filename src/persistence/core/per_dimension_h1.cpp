// H1 persistence through exact Z2 reduction.

#include "nerve/persistence/core/per_dimension_exact.hpp"
#include "nerve/persistence/kernels/simplex_reduction_utils.hpp"

#include <chrono>
#include <cmath>

namespace nerve::persistence::perdim
{

H1Result computeH1ReducedVR(const std::vector<std::vector<int>> &simplices,
                            const std::vector<double> &filtration_values,
                            const std::vector<int> &dimensions)
{
    H1Result result;
    const auto start = std::chrono::high_resolution_clock::now();

    const auto exact =
        kernels::computeExactDimensionPairs(simplices, filtration_values, dimensions, 1);

    result.pairs.reserve(exact.pairs.size());
    for (const auto &pair : exact.pairs)
    {
        result.pairs.push_back({pair.birth, pair.death, pair.dimension});
        if (std::isinf(pair.death))
        {
            ++result.essential_count;
        }
    }
    result.num_pairs = static_cast<int>(result.pairs.size());

    const auto end = std::chrono::high_resolution_clock::now();
    result.time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    return result;
}

} // namespace nerve::persistence::perdim

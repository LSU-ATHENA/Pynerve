#include "nerve/simd/simd_base.hpp"
#include "nerve/filtration/simd_filtration.hpp"

#include <algorithm>
#include <cmath>

namespace nerve::filtration
{

void simdBatchFilterValues(double *values, std::size_t n, std::size_t start_dim, std::size_t end_dim, double threshold)
{
    (void)start_dim;
    (void)end_dim;

    // Values below threshold are set to 0; those above are kept.
    for (std::size_t i = 0; i < n; ++i)
    {
        if (values[i] < threshold)
            values[i] = 0.0;
    }
}

void simdSortPairsByBirth(Pair *pairs, std::size_t n)
{
    std::sort(pairs, pairs + n,
              [](const Pair &a, const Pair &b) { return a.birth < b.birth; });
}

} // namespace nerve::filtration

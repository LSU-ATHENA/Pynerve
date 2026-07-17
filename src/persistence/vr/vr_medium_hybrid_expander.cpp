#include "detail/vr_medium_hybrid_helpers.inl"
#include "nerve/algebra/complex.hpp"
#include "vr_medium_hybrid_expander.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace nerve::persistence
{

void ParallelCliqueExpander::expand(size_t num_points, algebra::SimplicialComplex &complex,
                                    SimplexSet &seen)
{            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(num_points); ++i)
    {
        std::vector<Index> v{static_cast<Index>(i)};
        complex.addSimplexWithFiltration(algebra::Simplex(v), 0.0);
    }
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(num_points); ++i)
    {
        for (int j : neighbors_[i])
        {
            if (static_cast<size_t>(j) <= i)
            {
                continue;
            }
            double d = distance_matrix_[i][j];
            std::vector<Index> edge{static_cast<Index>(i), static_cast<Index>(j)};
            complex.addSimplexWithFiltration(algebra::Simplex(edge), d);
        }
    }

    if (num_points < 3)
    {
        return;
    }
    const size_t max_simplex_size = max_dim_ > num_points - 2 ? num_points : max_dim_ + 2;

    for (size_t simplex_size = 3; simplex_size <= max_simplex_size; ++simplex_size)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 16)
#endif
        for (size_t i = 0; i < num_points; ++i)
        {
            std::vector<int> current{static_cast<int>(i)};
            std::vector<int> candidates;
            for (int j : neighbors_[i])
            {
                if (j > static_cast<int>(i))
                {
                    candidates.push_back(j);
                }
            }
            expandCliquesThread(current, candidates, simplex_size, complex, seen);
        }
    }
}

} // namespace nerve::persistence


#include "nerve/persistence/cuda/cuda_matrix_reduction.hpp"

#include <algorithm>
#include <limits>
#include <vector>

namespace nerve::persistence::accelerated
{

void createPersistenceDiagramFromReduction(const int *col_pivot, const double *weights,
                                           Size n_columns, Size max_dim, std::vector<Pair> &diagram)
{
    diagram.clear();
    Size reserve_count = 0;
    if (detail::checkedSizeProduct(n_columns, max_dim, reserve_count))
    {
        diagram.reserve(reserve_count);
    }

    for (Size col = 0; col < n_columns; ++col)
    {
        int pivot = col_pivot[col];
        if (pivot == -1)
        {
            for (Size dim = 0; dim < max_dim; ++dim)
            {
                diagram.push_back(
                    {weights[col], std::numeric_limits<double>::infinity(), static_cast<int>(dim)});
            }
        }
        else
        {
            double birth = weights[static_cast<Size>(pivot)];
            diagram.push_back({birth, weights[col], static_cast<int>(max_dim - 1)});
        }
    }

    std::sort(diagram.begin(), diagram.end(), [](const Pair &a, const Pair &b) {
        if (a.dimension != b.dimension)
        {
            return a.dimension < b.dimension;
        }
        if (a.birth != b.birth)
        {
            return a.birth < b.birth;
        }
        return a.death < b.death;
    });
}

} // namespace nerve::persistence::accelerated

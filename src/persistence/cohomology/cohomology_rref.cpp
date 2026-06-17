#include "nerve/persistence/cohomology/cohomology_rref.hpp"

#include <algorithm>
#include <cmath>

namespace nerve::persistence
{

namespace
{
constexpr double kLinearEpsilon = 1e-12;
}

std::vector<std::vector<double>>
CohomologyRREF::computeRREF(const std::vector<std::vector<double>> &matrix) const
{
    return detail::computeRref(matrix).matrix;
}

std::vector<std::size_t>
CohomologyRREF::findPivots(const std::vector<std::vector<double>> &rref_matrix) const
{
    if (rref_matrix.empty() || rref_matrix.front().empty())
    {
        return {};
    }

    const Size rows = rref_matrix.size();
    const Size cols = rref_matrix.front().size();
    std::vector<std::size_t> pivot_columns;

    Size pivot_row = 0;
    for (Size col = 0; col < cols && pivot_row < rows; ++col)
    {
        if (std::abs(rref_matrix[pivot_row][col]) > kLinearEpsilon)
        {
            pivot_columns.push_back(col);
            ++pivot_row;
        }
    }

    return pivot_columns;
}

std::size_t CohomologyRREF::computeRank(const std::vector<std::vector<double>> &rref_matrix) const
{
    return findPivots(rref_matrix).size();
}

std::vector<std::vector<double>>
CohomologyRREF::computeNullSpace(const std::vector<std::vector<double>> &rref_matrix) const
{
    if (rref_matrix.empty())
    {
        return {};
    }

    const Size rows = rref_matrix.size();
    const Size cols = rref_matrix.front().size();
    if (cols == 0)
    {
        return {};
    }

    const auto pivot_cols = findPivots(rref_matrix);
    std::vector<bool> is_pivot(cols, false);
    for (const Size p : pivot_cols)
    {
        if (p < cols)
        {
            is_pivot[p] = true;
        }
    }

    std::vector<std::vector<double>> basis;
    for (Size free_col = 0; free_col < cols; ++free_col)
    {
        if (is_pivot[free_col])
        {
            continue;
        }

        std::vector<double> vec(cols, 0.0);
        vec[free_col] = 1.0;

        Size current_pivot = 0;
        for (Size r = 0; r < rows && current_pivot < pivot_cols.size(); ++r)
        {
            const Size pc = pivot_cols[current_pivot];
            if (pc >= cols)
            {
                break;
            }
            if (std::abs(rref_matrix[r][pc]) > kLinearEpsilon)
            {
                vec[pc] = -rref_matrix[r][free_col];
                ++current_pivot;
            }
        }

        basis.push_back(std::move(vec));
    }

    return basis;
}

} // namespace nerve::persistence


#include "nerve/algebra/cellular.hpp"
#include "nerve/core_types.hpp"
#include "nerve/persistence/cohomology/cohomology_rref.hpp"
#include "nerve/simd/simd_base.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace nerve::persistence::detail
{

namespace
{
constexpr double kLinearEpsilon = 1e-12;
}

RrefResult computeRref(std::vector<std::vector<double>> matrix)
{
    RrefResult result;
    if (matrix.empty() || matrix.front().empty())
    {
        result.matrix = std::move(matrix);
        return result;
    }

    const Size rows = matrix.size();
    const Size cols = matrix.front().size();
    Size pivot_row = 0;

    for (Size col = 0; col < cols && pivot_row < rows; ++col)
    {
        Size best_row = pivot_row;
        double best_abs = std::abs(matrix[best_row][col]);
        for (Size r = pivot_row + 1; r < rows; ++r)
        {
            const double candidate_abs = std::abs(matrix[r][col]);
            if (candidate_abs > best_abs)
            {
                best_abs = candidate_abs;
                best_row = r;
            }
        }
        if (best_abs <= kLinearEpsilon)
        {
            continue;
        }

        if (best_row != pivot_row)
        {
            std::swap(matrix[pivot_row], matrix[best_row]);
        }

        const double pivot = matrix[pivot_row][col];
        for (Size c = col; c < cols; ++c)
        {
            matrix[pivot_row][c] /= pivot;
        }

        for (Size r = 0; r < rows; ++r)
        {
            if (r == pivot_row)
            {
                continue;
            }
            const double factor = matrix[r][col];
            if (std::abs(factor) <= kLinearEpsilon)
            {
                continue;
            }
            for (Size c = col; c < cols; ++c)
            {
                matrix[r][c] -= factor * matrix[pivot_row][c];
            }
        }

        result.pivot_columns.push_back(col);
        ++pivot_row;
    }

    result.matrix = std::move(matrix);
    return result;
}

Size matrixRank(const std::vector<std::vector<double>> &matrix)
{
    return computeRref(matrix).pivot_columns.size();
}

std::vector<std::vector<double>> nullspaceBasis(const std::vector<std::vector<double>> &matrix)
{
    if (matrix.empty())
    {
        return {};
    }
    const Size cols = matrix.front().size();
    if (cols == 0)
    {
        return {};
    }

    const RrefResult rref = computeRref(matrix);
    std::vector<bool> isPivot(cols, false);
    for (const Size pivot_col : rref.pivot_columns)
    {
        if (pivot_col < cols)
        {
            isPivot[pivot_col] = true;
        }
    }

    std::vector<std::vector<double>> basis;
    for (Size free_col = 0; free_col < cols; ++free_col)
    {
        if (isPivot[free_col])
        {
            continue;
        }

        std::vector<double> vector(cols, 0.0);
        vector[free_col] = 1.0;
        for (Size pivot_row = 0; pivot_row < rref.pivot_columns.size(); ++pivot_row)
        {
            const Size pivot_col = rref.pivot_columns[pivot_row];
            vector[pivot_col] = -rref.matrix[pivot_row][free_col];
        }
        basis.push_back(std::move(vector));
    }

    if (basis.empty())
    {
        return {};
    }
    return basis;
}

std::vector<std::vector<double>> submatrix(const std::vector<std::vector<double>> &matrix,
                                           const std::vector<Index> &row_indices,
                                           const std::vector<Index> &col_indices)
{
    if (row_indices.empty() || col_indices.empty())
    {
        return {};
    }
    std::vector<std::vector<double>> out(row_indices.size(),
                                         std::vector<double>(col_indices.size(), 0.0));
    for (Size i = 0; i < row_indices.size(); ++i)
    {
        const Index r = row_indices[i];
        if (r < 0 || static_cast<Size>(r) >= matrix.size())
        {
            continue;
        }
        for (Size j = 0; j < col_indices.size(); ++j)
        {
            const Index c = col_indices[j];
            if (c < 0)
            {
                continue;
            }
            const Size cs = static_cast<Size>(c);
            if (cs >= matrix[r].size())
            {
                continue;
            }
            out[i][j] = matrix[r][cs];
        }
    }
    return out;
}

std::vector<double> multiplyMatrixVector(const std::vector<std::vector<double>> &matrix,
                                         const std::vector<double> &vector)
{
    std::vector<double> output(matrix.size(), 0.0);
    for (Size i = 0; i < matrix.size(); ++i)
    {
        const Size width = std::min(matrix[i].size(), vector.size());
        output[i] = nerve::simd::simd_dot(matrix[i].data(), vector.data(), width);
    }
    return output;
}

Index findCellIndex(const algebra::CellularComplex &complex, const algebra::Cell &cell)
{
    const Size n = complex.numCells();
    for (Size i = 0; i < n; ++i)
    {
        if (complex.getCell(static_cast<Index>(i)) == cell)
        {
            return static_cast<Index>(i);
        }
    }
    return -1;
}

} // namespace nerve::persistence::detail

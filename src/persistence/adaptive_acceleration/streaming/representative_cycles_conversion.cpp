
#include "nerve/persistence/adaptive_acceleration/streaming/representative_cycles.hpp"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace nerve::persistence::adaptive_acceleration::representative
{
namespace
{

constexpr double kLinearTolerance = 1e-12;

std::vector<std::vector<double>> denseFromSparse(const SparseMatrix &matrix)
{
    std::vector<std::vector<double>> dense(matrix.numRows(),
                                           std::vector<double>(matrix.numCols(), 0.0));
    for (std::size_t row = 0; row < matrix.numRows(); ++row)
    {
        dense[row] = matrix.getRow(row);
    }
    return dense;
}

std::vector<double> multiplyMatrixVector(const std::vector<std::vector<double>> &matrix,
                                         const std::vector<double> &vector)
{
    std::vector<double> out(matrix.size(), 0.0);
    for (std::size_t row = 0; row < matrix.size(); ++row)
    {
        const std::size_t width = std::min(matrix[row].size(), vector.size());
        for (std::size_t col = 0; col < width; ++col)
        {
            out[row] += matrix[row][col] * vector[col];
        }
    }
    return out;
}

bool isZeroVector(const std::vector<double> &vector)
{
    for (double value : vector)
    {
        if (std::abs(value) > kLinearTolerance)
        {
            return false;
        }
    }
    return true;
}

std::vector<std::vector<double>> nullspaceBasis(std::vector<std::vector<double>> matrix)
{
    if (matrix.empty() || matrix.front().empty())
    {
        return {};
    }
    const std::size_t rows = matrix.size();
    const std::size_t cols = matrix.front().size();
    std::vector<std::size_t> pivot_columns;
    std::size_t pivot_row = 0;
    for (std::size_t col = 0; col < cols && pivot_row < rows; ++col)
    {
        std::size_t best_row = pivot_row;
        double best_abs = std::abs(matrix[best_row][col]);
        for (std::size_t row = pivot_row + 1; row < rows; ++row)
        {
            const double candidate = std::abs(matrix[row][col]);
            if (candidate > best_abs)
            {
                best_abs = candidate;
                best_row = row;
            }
        }
        if (best_abs <= kLinearTolerance)
        {
            continue;
        }
        if (best_row != pivot_row)
        {
            std::swap(matrix[best_row], matrix[pivot_row]);
        }
        const double pivot = matrix[pivot_row][col];
        for (std::size_t c = col; c < cols; ++c)
        {
            matrix[pivot_row][c] /= pivot;
        }
        for (std::size_t row = 0; row < rows; ++row)
        {
            if (row == pivot_row)
            {
                continue;
            }
            const double factor = matrix[row][col];
            if (std::abs(factor) <= kLinearTolerance)
            {
                continue;
            }
            for (std::size_t c = col; c < cols; ++c)
            {
                matrix[row][c] -= factor * matrix[pivot_row][c];
            }
        }
        pivot_columns.push_back(col);
        ++pivot_row;
    }

    std::vector<bool> isPivot(cols, false);
    for (std::size_t col : pivot_columns)
    {
        if (col < cols)
        {
            isPivot[col] = true;
        }
    }

    std::vector<std::vector<double>> basis;
    for (std::size_t free_col = 0; free_col < cols; ++free_col)
    {
        if (isPivot[free_col])
        {
            continue;
        }
        std::vector<double> vector(cols, 0.0);
        vector[free_col] = 1.0;
        for (std::size_t row = 0; row < pivot_columns.size(); ++row)
        {
            const std::size_t pivot_col = pivot_columns[row];
            vector[pivot_col] = -matrix[row][free_col];
        }
        basis.push_back(std::move(vector));
    }
    return basis;
}

std::vector<std::vector<double>> orthonormalize(const std::vector<std::vector<double>> &basis)
{
    std::vector<std::vector<double>> ortho;
    for (const auto &vec : basis)
    {
        std::vector<double> candidate = vec;
        for (const auto &prior : ortho)
        {
            double dot = 0.0;
            for (std::size_t i = 0; i < std::min(candidate.size(), prior.size()); ++i)
            {
                dot += candidate[i] * prior[i];
            }
            for (std::size_t i = 0; i < std::min(candidate.size(), prior.size()); ++i)
            {
                candidate[i] -= dot * prior[i];
            }
        }
        double norm_sq = 0.0;
        for (double value : candidate)
        {
            norm_sq += value * value;
        }
        if (norm_sq <= kLinearTolerance)
        {
            continue;
        }
        const double inv_norm = 1.0 / std::sqrt(norm_sq);
        for (double &value : candidate)
        {
            value *= inv_norm;
        }
        ortho.push_back(std::move(candidate));
    }
    return ortho;
}

std::vector<double> projectToSubspace(const std::vector<double> &vector,
                                      const std::vector<std::vector<double>> &orthonormal_basis)
{
    if (orthonormal_basis.empty())
    {
        return {};
    }
    std::vector<double> projected(vector.size(), 0.0);
    for (const auto &basis_vec : orthonormal_basis)
    {
        double coeff = 0.0;
        for (std::size_t i = 0; i < std::min(vector.size(), basis_vec.size()); ++i)
        {
            coeff += vector[i] * basis_vec[i];
        }
        for (std::size_t i = 0; i < std::min(projected.size(), basis_vec.size()); ++i)
        {
            projected[i] += coeff * basis_vec[i];
        }
    }
    return projected;
}

Cycle sparseCycleFromDense(const std::vector<double> &denseCycle, Dimension dimension,
                           double birth_time, double death_time)
{
    Cycle cycle;
    cycle.dimension = dimension;
    cycle.birth_time = birth_time;
    cycle.death_time = death_time;
    for (std::size_t i = 0; i < denseCycle.size(); ++i)
    {
        if (std::abs(denseCycle[i]) <= kLinearTolerance)
        {
            continue;
        }
        cycle.vertices.push_back(static_cast<int>(i));
        cycle.coefficients.push_back(denseCycle[i]);
    }
    return cycle;
}

} // namespace

errors::ErrorResult<std::vector<Cycle>>
DualCohomologyComputer::convertToHomology(const std::vector<Cycle> &cohomology_cycles,
                                          const SparseMatrix &boundary_matrix)
{
    if (cohomology_cycles.empty())
    {
        return errors::ErrorResult<std::vector<Cycle>>::success({});
    }
    if (boundary_matrix.numCols() == 0)
    {
        return errors::ErrorResult<std::vector<Cycle>>::error(
            errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }

    const std::vector<std::vector<double>> dense_boundary = denseFromSparse(boundary_matrix);
    const std::vector<std::vector<double>> kernel_basis =
        orthonormalize(nullspaceBasis(dense_boundary));

    std::vector<Cycle> homology;
    homology.reserve(cohomology_cycles.size());
    for (const Cycle &cohomology_cycle : cohomology_cycles)
    {
        std::vector<double> denseCycle(boundary_matrix.numCols(), 0.0);
        const std::size_t width =
            std::min(cohomology_cycle.vertices.size(), cohomology_cycle.coefficients.size());
        for (std::size_t i = 0; i < width; ++i)
        {
            const int vertex = cohomology_cycle.vertices[i];
            if (vertex < 0 || static_cast<std::size_t>(vertex) >= denseCycle.size())
            {
                continue;
            }
            denseCycle[static_cast<std::size_t>(vertex)] += cohomology_cycle.coefficients[i];
        }

        std::vector<double> candidate = denseCycle;
        if (!kernel_basis.empty())
        {
            candidate = projectToSubspace(denseCycle, kernel_basis);
        }
        if (candidate.empty())
        {
            continue;
        }
        if (!isZeroVector(multiplyMatrixVector(dense_boundary, candidate)))
        {
            continue;
        }

        Cycle projected =
            sparseCycleFromDense(candidate, std::max<Dimension>(0, cohomology_cycle.dimension),
                                 cohomology_cycle.birth_time, cohomology_cycle.death_time);
        if (projected.isValid())
        {
            homology.push_back(std::move(projected));
        }
    }
    return errors::ErrorResult<std::vector<Cycle>>::success(std::move(homology));
}

} // namespace nerve::persistence::adaptive_acceleration::representative

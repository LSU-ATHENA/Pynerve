
#include "nerve/algebra/boundary.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace nerve::algebra
{

constexpr double CHAIN_COMPLEX_TOLERANCE = 1e-10;

ChainComplex::ChainComplex(const SimplicialComplex &complex)
    : max_dimension_(complex.maxDimension() < 0 ? 0 : static_cast<Size>(complex.maxDimension()))
{
    buildAllBoundaries(complex);
}

[[nodiscard]] const BoundaryMatrix &ChainComplex::boundary(Size k) const
{
    if (k >= boundary_matrices_.size())
    {
        throw std::out_of_range("Boundary operator dimension out of range");
    }
    return boundary_matrices_[k];
}

[[nodiscard]] BoundaryMatrix &ChainComplex::boundary(Size k)
{
    if (k >= boundary_matrices_.size())
    {
        throw std::out_of_range("Boundary operator dimension out of range");
    }
    return boundary_matrices_[k];
}

Size ChainComplex::rank(Size k) const
{
    if (k >= boundary_matrices_.size())
    {
        return 0;
    }
    return boundary_matrices_[k].cols();
}

[[nodiscard]] Size ChainComplex::bettiNumber(Size k) const
{
    if (k >= boundary_matrices_.size())
    {
        return 0;
    }

    const auto &boundary_k = boundary_matrices_[k];
    Size rank_boundary_k = computeMatrixRank(boundary_k);
    Size rank_boundary_k_plus_1 = 0;
    if (k + 1 < boundary_matrices_.size())
    {
        const auto &boundary_k_plus_1 = boundary_matrices_[k + 1];
        rank_boundary_k_plus_1 = computeMatrixRank(boundary_k_plus_1);
    }

    if (rank_boundary_k + rank_boundary_k_plus_1 >= boundary_k.cols())
    {
        return 0;
    }
    return boundary_k.cols() - rank_boundary_k - rank_boundary_k_plus_1;
}

Size ChainComplex::computeMatrixRank(const BoundaryMatrix &matrix) const
{
    Size rank = 0;
    Size rows = matrix.rows();
    Size cols = matrix.cols();

    std::vector<std::vector<double>> tempMatrix(rows, std::vector<double>(cols, 0.0));
    for (Size row = 0; row < rows; ++row)
    {
        for (Size col = 0; col < cols; ++col)
        {
            tempMatrix[row][col] = matrix.getMatrixEntry(row, col);
        }
    }

    Size current_row = 0;
    Size current_col = 0;
    while (current_row < rows && current_col < cols)
    {
        Size pivot_row = current_row;
        while (pivot_row < rows &&
               std::abs(tempMatrix[pivot_row][current_col]) < CHAIN_COMPLEX_TOLERANCE)
        {
            ++pivot_row;
        }

        if (pivot_row == rows)
        {
            ++current_col;
            continue;
        }

        if (pivot_row != current_row)
        {
            std::swap(tempMatrix[current_row], tempMatrix[pivot_row]);
        }

        double pivot_value = tempMatrix[current_row][current_col];
        for (Size col = current_col; col < cols; ++col)
        {
            tempMatrix[current_row][col] /= pivot_value;
        }

        for (Size row = 0; row < rows; ++row)
        {
            if (row == current_row ||
                std::abs(tempMatrix[row][current_col]) <= CHAIN_COMPLEX_TOLERANCE)
            {
                continue;
            }
            double factor = tempMatrix[row][current_col];
            for (Size col = current_col; col < cols; ++col)
            {
                tempMatrix[row][col] -= factor * tempMatrix[current_row][col];
            }
        }

        ++rank;
        ++current_row;
        ++current_col;
    }

    return rank;
}

Size ChainComplex::maxDimension() const noexcept
{
    return max_dimension_;
}

std::vector<double> ChainComplex::applyBoundary(Size k, core::BufferView<const double> chain) const
{
    return boundary(k).applyBoundary(chain);
}

std::vector<double> ChainComplex::applyCoboundary(Size k,
                                                  core::BufferView<const double> cochain) const
{
    return boundary(k).applyCoboundary(cochain);
}

std::vector<std::pair<Index, Index>> ChainComplex::compute()
{
    std::vector<std::pair<Index, Index>> all_pairs;
    for (Size k = 0; k < boundary_matrices_.size(); ++k)
    {
        auto pairs_result = boundary_matrices_[k].computePersistencePairs();
        if (pairs_result.isSuccess())
        {
            auto pairs = pairs_result.value();
            all_pairs.insert(all_pairs.end(), pairs.begin(), pairs.end());
        }
    }
    return all_pairs;
}

std::vector<Pair> ChainComplex::computePersistenceDiagram()
{
    std::vector<Pair> diagram;
    if (boundary_matrices_.empty())
    {
        return diagram;
    }

    const double inf = std::numeric_limits<Field>::infinity();

    if (max_dimension_ == 0)
    {
        const auto &boundary0 = boundary_matrices_[0];
        for (Size col = 0; col < boundary0.cols(); ++col)
        {
            diagram.push_back(Pair{boundary0.getFiltrationValue(col), inf});
        }
        return diagram;
    }

    for (Size k = 1; k < boundary_matrices_.size(); ++k)
    {
        auto &boundary_matrix = boundary_matrices_[k];
        auto pairs_result = boundary_matrix.computePersistencePairs();
        if (!pairs_result.isSuccess())
        {
            continue;
        }

        auto pairs = pairs_result.value();
        for (const auto &pair : pairs)
        {
            Index row = pair.first;
            Index col = pair.second;
            if (col != -1 && row != -1)
            {
                double birth = boundary_matrix.getRowFiltrationValue(static_cast<Size>(row));
                double death = boundary_matrix.getFiltrationValue(static_cast<Size>(col));
                diagram.push_back(Pair{birth, death});
            }
            else if (col == -1 && row != -1)
            {
                double birth = boundary_matrix.getFiltrationValue(static_cast<Size>(row));
                diagram.push_back(Pair{birth, inf});
            }
        }

        if (k == 1)
        {
            const auto &low_row_to_col = boundary_matrix.lastLowRowToCol();
            for (Size row_idx = 0; row_idx < low_row_to_col.size(); ++row_idx)
            {
                if (low_row_to_col[row_idx] == -1)
                {
                    double birth = boundary_matrix.getRowFiltrationValue(row_idx);
                    diagram.push_back(Pair{birth, inf});
                }
            }
        }
    }

    return diagram;
}

std::vector<Size> ChainComplex::computeBettiNumbers()
{
    std::vector<Size> bettiNumbers(max_dimension_ + 1);
    for (Size k = 0; k <= max_dimension_; ++k)
    {
        bettiNumbers[k] = bettiNumber(k);
    }
    return bettiNumbers;
}

void ChainComplex::buildAllBoundaries(const SimplicialComplex &complex)
{
    boundary_matrices_.clear();
    for (Size k = 0; k <= max_dimension_; ++k)
    {
        boundary_matrices_.emplace_back(complex, k);
    }
}

std::vector<Simplex> computeBoundaryFaces(const Simplex &simplex)
{
    return simplex.faces({});
}

std::vector<int> computeBoundaryCoefficients(const Simplex &simplex)
{
    std::vector<int> coefficients;
    auto faces = simplex.faces({});
    coefficients.reserve(faces.size());

    for (const auto &face : faces)
    {
        const auto &vertices = simplex.vertices();
        int coeff = 0;
        for (Size i = 0; i < vertices.size(); ++i)
        {
            std::vector<Index> temp_vertices;
            temp_vertices.reserve(vertices.size() - 1);
            for (Size j = 0; j < vertices.size(); ++j)
            {
                if (j != i)
                {
                    temp_vertices.push_back(vertices[j]);
                }
            }
            if (Simplex(temp_vertices) == face)
            {
                coeff = (i % 2 == 0) ? 1 : -1;
                break;
            }
        }
        coefficients.push_back(coeff);
    }

    return coefficients;
}

} // namespace nerve::algebra

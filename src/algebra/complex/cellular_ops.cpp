
#include "nerve/algebra/cellular.hpp"

#include <algorithm>
#include <cmath>
#include <ranges>
#include <span>
#include <stdexcept>

namespace nerve::algebra
{
namespace
{

constinit const double kRankEpsilon = 1e-12;

std::vector<std::vector<double>> selectSubmatrix(std::span<const std::vector<double>> matrix,
                                                 std::span<const Index> rows,
                                                 std::span<const Index> cols)
{
    if (rows.empty() || cols.empty())
    {
        return {};
    }
    std::vector<std::vector<double>> out(rows.size(), std::vector<double>(cols.size(), 0.0));

    for (Size i = 0; i < rows.size(); ++i)
    {
        Index row = rows[i];
        if (row < 0 || static_cast<Size>(row) >= matrix.size())
        {
            continue;
        }
        for (Size j = 0; j < cols.size(); ++j)
        {
            Index col = cols[j];
            if (col < 0 || static_cast<Size>(col) >= matrix[row].size())
            {
                continue;
            }
            out[i][j] = matrix[row][col];
        }
    }
    return out;
}

Size computeRank(std::vector<std::vector<double>> matrix)
{
    if (matrix.empty() || matrix.front().empty())
    {
        return 0;
    }
    const Size rows = matrix.size();
    const Size cols = matrix.front().size();

    Size rank = 0;
    Size pivot_row = 0;
    for (Size col = 0; col < cols && pivot_row < rows; ++col)
    {
        Size best = pivot_row;
        double best_abs = std::abs(matrix[best][col]);
        for (Size r = pivot_row + 1; r < rows; ++r)
        {
            const double cand_abs = std::abs(matrix[r][col]);
            if (cand_abs > best_abs)
            {
                best = r;
                best_abs = cand_abs;
            }
        }
        if (best_abs <= kRankEpsilon)
        {
            continue;
        }

        if (best != pivot_row)
        {
            std::swap(matrix[pivot_row], matrix[best]);
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
            if (std::abs(factor) <= kRankEpsilon)
            {
                continue;
            }
            for (Size c = col; c < cols; ++c)
            {
                matrix[r][c] -= factor * matrix[pivot_row][c];
            }
        }
        ++rank;
        ++pivot_row;
    }

    return rank;
}

std::vector<Index> remapBoundary(const std::vector<Index> &boundary, Index removed)
{
    std::vector<Index> out;
    out.reserve(boundary.size());
    for (const Index idx : boundary)
    {
        if (idx == removed)
        {
            continue;
        }
        if (idx > removed)
        {
            out.push_back(idx - 1);
        }
        else
        {
            out.push_back(idx);
        }
    }
    return out;
}

void validateCellForInsertion(const Cell &cell, Size existing_cells)
{
    if (cell.dimension() < 0)
    {
        throw std::invalid_argument("cell dimension must be non-negative");
    }
    for (const Index idx : cell.boundary())
    {
        if (idx < 0 || static_cast<Size>(idx) >= existing_cells)
        {
            throw std::invalid_argument("cell boundary index must reference an existing cell");
        }
    }
}

} // namespace

Cell::Cell(int dimension)
    : dimension_(dimension)
{
    if (dimension < 0)
    {
        throw std::invalid_argument("cell dimension must be non-negative");
    }
}

Cell::Cell(int dimension, const std::vector<Index> &boundary)
    : dimension_(dimension)
    , boundary_(boundary)
{
    if (dimension < 0)
    {
        throw std::invalid_argument("cell dimension must be non-negative");
    }
    if (std::ranges::any_of(boundary_, [](Index idx) { return idx < 0; }))
    {
        throw std::invalid_argument("cell boundary indices must be non-negative");
    }
}

void Cell::addBoundaryCell(Index cell_index)
{
    if (cell_index < 0)
    {
        throw std::invalid_argument("cell boundary index must be non-negative");
    }
    boundary_.push_back(cell_index);
}

void Cell::removeBoundaryCell(Index cell_index)
{
    boundary_.erase(std::remove(boundary_.begin(), boundary_.end(), cell_index), boundary_.end());
}

void Cell::clearBoundary()
{
    boundary_.clear();
}

Size Cell::Hash::operator()(const Cell &cell) const noexcept
{
    Size seed = static_cast<Size>(cell.dimension_);
    for (const Index idx : cell.boundary_)
    {
        seed ^= static_cast<Size>(idx) + 0x9e3779b9U + (seed << 6U) + (seed >> 2U);
    }
    return seed;
}

Index CellularComplex::addCell(const Cell &cell)
{
    validateCellForInsertion(cell, cells_.size());
    const auto it = cell_to_index_.find(cell);
    if (it != cell_to_index_.end())
    {
        return it->second;
    }
    const Index index = static_cast<Index>(cells_.size());
    cells_.push_back(cell);
    cell_to_index_[cell] = index;
    invalidateCoboundaryCache();
    return index;
}

void CellularComplex::removeCell(Index cell_index)
{
    if (cell_index < 0 || static_cast<Size>(cell_index) >= cells_.size())
    {
        throw std::out_of_range("Cell index out of range");
    }

    cells_.erase(cells_.begin() + static_cast<std::ptrdiff_t>(cell_index));
    for (Cell &cell : cells_)
    {
        const std::vector<Index> remapped = remapBoundary(cell.boundary(), cell_index);
        cell.clearBoundary();
        for (const Index idx : remapped)
        {
            cell.addBoundaryCell(idx);
        }
    }

    cell_to_index_.clear();
    for (Size i = 0; i < cells_.size(); ++i)
    {
        cell_to_index_[cells_[i]] = static_cast<Index>(i);
    }
    invalidateCoboundaryCache();
}

void CellularComplex::clear()
{
    cells_.clear();
    cell_to_index_.clear();
    invalidateCoboundaryCache();
}

Size CellularComplex::numCells() const noexcept
{
    return cells_.size();
}

int CellularComplex::maxDimension() const
{
    int max_dim = -1;
    for (const Cell &cell : cells_)
    {
        max_dim = std::max(max_dim, cell.dimension());
    }
    return max_dim;
}

std::vector<Index> CellularComplex::cellsOfDimension(int dimension) const
{
    std::vector<Index> out;
    for (Size i = 0; i < cells_.size(); ++i)
    {
        if (cells_[i].dimension() == dimension)
        {
            out.push_back(static_cast<Index>(i));
        }
    }
    return out;
}

const Cell &CellularComplex::getCell(Index index) const
{
    if (index < 0 || static_cast<Size>(index) >= cells_.size())
    {
        throw std::out_of_range("Cell index out of range");
    }
    return cells_[index];
}

std::vector<Index> CellularComplex::getBoundary(const Cell &cell) const
{
    return cell.boundary();
}

std::vector<Index> CellularComplex::getCoboundary(const Cell &cell) const
{
    if (!coboundary_cache_valid_)
    {
        rebuildCoboundaryCache();
    }
    const auto it = cell_to_index_.find(cell);
    if (it == cell_to_index_.end())
    {
        return {};
    }
    return coboundary_cache_[static_cast<Size>(it->second)];
}

std::vector<Index> CellularComplex::boundary(const Cell &cell) const
{
    return getBoundary(cell);
}

std::vector<Index> CellularComplex::coboundary(const Cell &cell) const
{
    return getCoboundary(cell);
}

std::vector<std::vector<double>> CellularComplex::computeBoundaryMatrix() const
{
    const Size n = cells_.size();
    std::vector<std::vector<double>> boundary(n, std::vector<double>(n, 0.0));
    for (Size col = 0; col < n; ++col)
    {
        const std::vector<Index> &b = cells_[col].boundary();
        for (Size k = 0; k < b.size(); ++k)
        {
            const Index row = b[k];
            if (row < 0 || static_cast<Size>(row) >= n)
            {
                continue;
            }
            const double sign = (k % 2 == 0) ? 1.0 : -1.0;
            boundary[static_cast<Size>(row)][col] += sign;
        }
    }
    return boundary;
}

std::vector<std::vector<double>> CellularComplex::computeCoboundaryMatrix() const
{
    const std::vector<std::vector<double>> boundary = computeBoundaryMatrix();
    const Size n = boundary.size();
    std::vector<std::vector<double>> coboundary(n, std::vector<double>(n, 0.0));
    for (Size i = 0; i < n; ++i)
    {
        for (Size j = 0; j < n; ++j)
        {
            coboundary[i][j] = boundary[j][i];
        }
    }
    return coboundary;
}

errors::ErrorResult<std::vector<int>>
CellularComplex::computeEulerCharacteristic(const core::DeterminismContract &contract) const
{
    if (contract.level == core::DeterminismLevel::STRICT &&
        !core::DeterminismEnforcer::canSatisfyContract(contract))
    {
        return errors::ErrorResult<std::vector<int>>::error(errors::ErrorCode::E30_DET_MISMATCH);
    }

    const int max_dim = maxDimension();
    if (max_dim < 0)
    {
        return errors::ErrorResult<std::vector<int>>::success({});
    }
    std::vector<int> counts(static_cast<Size>(max_dim + 1), 0);
    for (int dim = 0; dim <= max_dim; ++dim)
    {
        counts[static_cast<Size>(dim)] = static_cast<int>(cellsOfDimension(dim).size());
    }
    return errors::ErrorResult<std::vector<int>>::success(std::move(counts));
}

errors::ErrorResult<std::vector<int>>
CellularComplex::computeBettiNumbers(const core::DeterminismContract &contract) const
{
    if (contract.level == core::DeterminismLevel::STRICT &&
        !core::DeterminismEnforcer::canSatisfyContract(contract))
    {
        return errors::ErrorResult<std::vector<int>>::error(errors::ErrorCode::E30_DET_MISMATCH);
    }

    const int max_dim = maxDimension();
    if (max_dim < 0)
    {
        return errors::ErrorResult<std::vector<int>>::success({});
    }

    const std::vector<std::vector<double>> boundary = computeBoundaryMatrix();
    std::vector<int> betti(static_cast<Size>(max_dim + 1), 0);
    for (int k = 0; k <= max_dim; ++k)
    {
        const std::vector<Index> cells_k = cellsOfDimension(k);
        const std::vector<Index> cells_k_minus =
            (k > 0) ? cellsOfDimension(k - 1) : std::vector<Index>{};
        const std::vector<Index> cells_k_plus = cellsOfDimension(k + 1);

        const Size dim_ck = cells_k.size();
        const Size rank_boundary_k = computeRank(selectSubmatrix(boundary, cells_k_minus, cells_k));
        const Size rank_boundary_k_plus =
            computeRank(selectSubmatrix(boundary, cells_k, cells_k_plus));

        const long long value = static_cast<long long>(dim_ck) -
                                static_cast<long long>(rank_boundary_k) -
                                static_cast<long long>(rank_boundary_k_plus);
        betti[static_cast<Size>(k)] = static_cast<int>(std::max<long long>(0, value));
    }

    return errors::ErrorResult<std::vector<int>>::success(std::move(betti));
}

void CellularComplex::invalidateCoboundaryCache() const
{
    coboundary_cache_valid_ = false;
    coboundary_cache_.clear();
}

void CellularComplex::rebuildCoboundaryCache() const
{
    coboundary_cache_.assign(cells_.size(), {});
    for (Size coface = 0; coface < cells_.size(); ++coface)
    {
        for (const Index face : cells_[coface].boundary())
        {
            if (face < 0 || static_cast<Size>(face) >= cells_.size())
            {
                continue;
            }
            coboundary_cache_[static_cast<Size>(face)].push_back(static_cast<Index>(coface));
        }
    }
    coboundary_cache_valid_ = true;
}

CellularChainComplex::CellularChainComplex(const CellularComplex &complex)
    : boundary_matrix_(complex.computeBoundaryMatrix())
    , coboundary_matrix_(complex.computeCoboundaryMatrix())
{}

std::vector<std::vector<double>> CellularChainComplex::getBoundaryMatrix() const
{
    return boundary_matrix_;
}

std::vector<std::vector<double>> CellularChainComplex::getCoboundaryMatrix() const
{
    return coboundary_matrix_;
}

errors::ErrorResult<std::vector<std::vector<int>>>
CellularChainComplex::computeHomology(const core::DeterminismContract &contract) const
{
    const auto betti = computeBettiNumbers(contract);
    if (betti.isError())
    {
        return errors::ErrorResult<std::vector<std::vector<int>>>::error(betti.errorCode());
    }
    std::vector<std::vector<int>> groups;
    for (const int rank : betti.value())
    {
        std::vector<int> group;
        group.reserve(static_cast<Size>(std::max(0, rank)));
        for (int i = 0; i < rank; ++i)
        {
            group.push_back(i);
        }
        groups.push_back(std::move(group));
    }
    return errors::ErrorResult<std::vector<std::vector<int>>>::success(std::move(groups));
}

errors::ErrorResult<std::vector<int>>
CellularChainComplex::computeBettiNumbers(const core::DeterminismContract &contract) const
{
    if (contract.level == core::DeterminismLevel::STRICT &&
        !core::DeterminismEnforcer::canSatisfyContract(contract))
    {
        return errors::ErrorResult<std::vector<int>>::error(errors::ErrorCode::E30_DET_MISMATCH);
    }
    const Size rank = computeRank(boundary_matrix_);
    const long long nullity =
        static_cast<long long>(boundary_matrix_.empty() ? 0 : boundary_matrix_.front().size()) -
        static_cast<long long>(rank);
    return errors::ErrorResult<std::vector<int>>::success(
        {static_cast<int>(std::max<long long>(0, nullity))});
}

} // namespace nerve::algebra

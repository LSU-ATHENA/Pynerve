#include "nerve/algebra/boundary.hpp"

#include <algorithm>
#include <cmath>
#include <ranges>
#include <span>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace nerve::algebra
{

namespace
{

constinit const double kEpsilon = 1e-10;

std::vector<Size> symmetricDifferenceSorted(std::span<const Size> a, std::span<const Size> b)
{
    std::vector<Size> result;
    result.reserve(a.size() + b.size());

    auto i = a.begin();
    auto j = b.begin();

    while (i != a.end() && j != b.end())
    {
        if (*i == *j)
        {
            ++i;
            ++j;
        }
        else if (*i < *j)
        {
            result.push_back(*i);
            ++i;
        }
        else
        {
            result.push_back(*j);
            ++j;
        }
    }

    std::ranges::copy(i, a.end(), std::back_inserter(result));
    std::ranges::copy(j, b.end(), std::back_inserter(result));

    return result;
}

inline bool isEffectivelyZero(double v) noexcept
{
    return std::abs(v) <= kEpsilon;
}

void enforceDeterminismContract(const core::DeterminismContract &contract)
{
    if (contract.level == core::DeterminismLevel::STRICT &&
        !core::DeterminismEnforcer::canSatisfyContract(contract))
    {
        throw std::runtime_error("cannot satisfy boundary matrix determinism contract");
    }
}

} // namespace

void BoundaryMatrix::reduceToReducedRowEchelon(const core::DeterminismContract &contract)
{
    enforceDeterminismContract(contract);

    Size current_row = 0;
    Size current_col = 0;

    while (current_row < rows_ && current_col < cols_)
    {
        Size pivot_row = current_row;
        while (pivot_row < rows_ && isEffectivelyZero(getEntry(pivot_row, current_col)))
        {
            ++pivot_row;
        }

        if (pivot_row == rows_)
        {
            ++current_col;
            continue;
        }

        if (pivot_row != current_row)
        {
            swapRows(current_row, pivot_row);
        }

        const double pivot_value = getEntry(current_row, current_col);
        if (isEffectivelyZero(pivot_value))
        {
            ++current_col;
            continue;
        }
        const double inv_pivot = 1.0 / pivot_value;
        for (Size col = current_col; col < cols_; ++col)
        {
            setEntry(current_row, col, getEntry(current_row, col) * inv_pivot);
        }

        for (Size row = 0; row < rows_; ++row)
        {
            if (row == current_row)
                continue;

            const double factor = getEntry(row, current_col);
            if (isEffectivelyZero(factor))
                continue;

            for (Size col = current_col; col < cols_; ++col)
            {
                setEntry(row, col, getEntry(row, col) - factor * getEntry(current_row, col));
            }
        }

        ++current_row;
        ++current_col;
    }
}

std::vector<Index> BoundaryMatrix::findPivotColumns(const core::DeterminismContract &contract)
{
    enforceDeterminismContract(contract);

    std::vector<Size> minCol(rows_, cols_);

    for (const auto &[row_col, value] : entries_)
    {
        const auto [row, col] = row_col;
        if (!isEffectivelyZero(value) && col < minCol[row])
        {
            minCol[row] = col;
        }
    }

    std::vector<Index> pivots;
    pivots.reserve(rows_);
    for (Size row = 0; row < rows_; ++row)
    {
        if (minCol[row] < cols_)
        {
            pivots.push_back(static_cast<Index>(minCol[row]));
        }
    }
    return pivots;
}

std::vector<Index> BoundaryMatrix::findPivotRows(const core::DeterminismContract &contract)
{
    enforceDeterminismContract(contract);

    std::vector<Size> minRow(cols_, rows_);

    for (const auto &[row_col, value] : entries_)
    {
        const auto [row, col] = row_col;
        if (!isEffectivelyZero(value) && row < minRow[col])
        {
            minRow[col] = row;
        }
    }

    std::vector<Index> pivots;
    pivots.reserve(cols_);
    for (Size col = 0; col < cols_; ++col)
    {
        if (minRow[col] < rows_)
        {
            pivots.push_back(static_cast<Index>(minRow[col]));
        }
    }
    return pivots;
}

errors::ErrorResult<std::vector<std::pair<Index, Index>>>
BoundaryMatrix::computePersistencePairs(const core::DeterminismContract &contract)
{
    if (contract.level == core::DeterminismLevel::STRICT &&
        !core::DeterminismEnforcer::canSatisfyContract(contract))
    {
        return errors::ErrorResult<std::vector<std::pair<Index, Index>>>::error(
            errors::ErrorCode::E30_DET_MISMATCH);
    }

    std::vector<std::vector<Size>> columns(cols_);
    for (const auto &[row_col, value] : entries_)
    {
        const auto [row, col] = row_col;
        if (!isEffectivelyZero(value))
        {
            columns[col].push_back(row);
        }
    }
    for (auto &col_vec : columns)
    {
        std::ranges::sort(col_vec);
        const auto [first, last] = std::ranges::unique(col_vec);
        col_vec.erase(first, last);
    }

    std::vector<Index> lowRowToCol(rows_, -1);
    std::vector<Index> low(cols_, -1);

    for (Size col = 0; col < cols_; ++col)
    {
        while (!columns[col].empty())
        {
            const Size pivot_row = columns[col].back();

            if (pivot_row >= rows_)
            {
                columns[col].pop_back();
                continue;
            }

            const Index existing = lowRowToCol[pivot_row];
            if (existing == -1)
            {
                lowRowToCol[pivot_row] = static_cast<Index>(col);
                low[col] = static_cast<Index>(pivot_row);
                break;
            }

            columns[col] =
                symmetricDifferenceSorted(columns[col], columns[static_cast<Size>(existing)]);
        }
    }

    last_low_row_to_col_ = lowRowToCol;

    std::unordered_set<Index> paired_indices;
    std::vector<std::pair<Index, Index>> pairs;
    pairs.reserve(cols_);

    for (Size col = 0; col < cols_; ++col)
    {
        if (low[col] != -1)
        {
            pairs.emplace_back(low[col], static_cast<Index>(col));
            paired_indices.insert(low[col]);
            paired_indices.insert(static_cast<Index>(col));
        }
    }

    const Size max_index = std::max(rows_, cols_);
    for (Size idx = 0; idx < max_index; ++idx)
    {
        if (paired_indices.find(static_cast<Index>(idx)) == paired_indices.end())
        {
            pairs.emplace_back(static_cast<Index>(idx), static_cast<Index>(-1));
        }
    }

    return errors::ErrorResult<std::vector<std::pair<Index, Index>>>::success(std::move(pairs));
}

errors::ErrorResult<std::vector<Index>>
BoundaryMatrix::findEssentialCycles(const core::DeterminismContract &contract)
{
    if (contract.level == core::DeterminismLevel::STRICT &&
        !core::DeterminismEnforcer::canSatisfyContract(contract))
    {
        return errors::ErrorResult<std::vector<Index>>::error(errors::ErrorCode::E30_DET_MISMATCH);
    }

    if (!last_low_row_to_col_.empty())
    {
        std::vector<Index> essential;
        for (Size row = 0; row < rows_; ++row)
        {
            if (last_low_row_to_col_[row] == -1)
            {
                essential.push_back(static_cast<Index>(row));
            }
        }
        return errors::ErrorResult<std::vector<Index>>::success(std::move(essential));
    }

    const auto pivot_rows = findPivotRows(contract);
    std::unordered_set<Index> pivotSet(pivot_rows.begin(), pivot_rows.end());

    std::vector<Index> essential;
    essential.reserve(rows_);
    for (Size row = 0; row < rows_; ++row)
    {
        if (pivotSet.find(static_cast<Index>(row)) == pivotSet.end())
        {
            essential.push_back(static_cast<Index>(row));
        }
    }

    return errors::ErrorResult<std::vector<Index>>::success(std::move(essential));
}

std::string BoundaryMatrix::matrixString() const
{
    std::ostringstream out;
    out << "Boundary Matrix (" << rows_ << " x " << cols_ << "):\n";
    for (Size row = 0; row < rows_; ++row)
    {
        out << "  Row " << row << ": ";
        for (Size col = 0; col < cols_; ++col)
        {
            const double value = getEntry(row, col);
            if (!isEffectivelyZero(value))
            {
                out << '[' << col << "]:" << value << ' ';
            }
        }
        out << '\n';
    }
    return out.str();
}

std::string BoundaryMatrix::nonzeroPatternString() const
{
    std::ostringstream out;
    out << "Nonzero pattern (" << numNonzeros() << " nonzeros):\n  ";
    for (const auto &[row_col, value] : entries_)
    {
        if (!isEffectivelyZero(value))
        {
            const auto [row, col] = row_col;
            out << '(' << row << ',' << col << ") ";
        }
    }
    out << '\n';
    return out.str();
}

} // namespace nerve::algebra

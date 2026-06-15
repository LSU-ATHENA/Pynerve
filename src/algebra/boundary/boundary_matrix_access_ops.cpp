
#include "nerve/algebra/boundary.hpp"

#include <cmath>
#include <span>
#include <stdexcept>

namespace nerve::algebra
{

constexpr double BOUNDARY_MATRIX_TOLERANCE = 1e-10;

[[nodiscard]] Size BoundaryMatrix::rows() const noexcept
{
    return rows_;
}

[[nodiscard]] Size BoundaryMatrix::cols() const noexcept
{
    return cols_;
}

[[nodiscard]] Size BoundaryMatrix::dimension() const noexcept
{
    return dimension_;
}

[[nodiscard]] bool BoundaryMatrix::isEmpty() const noexcept
{
    return entries_.empty();
}

std::vector<double> BoundaryMatrix::multiply(const core::BufferView<const double> &vector) const
{
    if (vector.size() != cols_)
    {
        throw nerve::errors::ShapeMismatchError(__FILE__, __LINE__, __func__)
            .setExpectedDims(rows_, cols_)
            .setActualAccess(0, static_cast<size_t>(vector.size()))
            .setDimensionMismatch(1)
            .setOperation("BoundaryMatrix::multiply")
            .addContext("vector_size", vector.size())
            .addContext("expected_cols", cols_)
            .addContext("operation", "matrix-vector multiplication")
            .addSuggestion("Ensure vector size matches matrix column count")
            .addSuggestion(
                "Check that you're multiplying in correct order (matrix * vector, not vector * "
                "matrix)");
    }

    std::vector<double> result(rows_, 0.0);
    for (const auto &entry : entries_)
    {
        auto [row_col, value] = entry;
        auto [row, col] = row_col;
        result[row] += value * vector[col];
    }
    return result;
}

std::vector<double>
BoundaryMatrix::transposeMultiply(const core::BufferView<const double> &vector) const
{
    if (vector.size() != rows_)
    {
        throw nerve::errors::ShapeMismatchError(__FILE__, __LINE__, __func__)
            .setExpectedDims(rows_, cols_)
            .setActualAccess(static_cast<size_t>(vector.size()), 0)
            .setDimensionMismatch(0)
            .setOperation("BoundaryMatrix::transposeMultiply")
            .addContext("vector_size", vector.size())
            .addContext("expected_rows", rows_)
            .addContext("operation", "matrix transpose-vector multiplication")
            .addSuggestion("Ensure vector size matches matrix row count for transpose operation")
            .addSuggestion(
                "For transpose multiply, vector size should equal matrix rows (not columns)");
    }

    std::vector<double> result(cols_, 0.0);
    for (const auto &entry : entries_)
    {
        auto [row_col, value] = entry;
        auto [row, col] = row_col;
        result[col] += value * vector[row];
    }
    return result;
}

BoundaryMatrix BoundaryMatrix::transpose() const
{
    BoundaryMatrix result;
    result.rows_ = cols_;
    result.cols_ = rows_;
    result.dimension_ = dimension_;

    for (const auto &entry : entries_)
    {
        auto [row_col, value] = entry;
        auto [row, col] = row_col;
        result.setEntry(col, row, value);
    }

    result.simplex_to_col_ = simplex_to_row_;
    result.simplex_to_row_ = simplex_to_col_;
    result.col_to_simplex_ = row_to_simplex_;
    result.row_to_simplex_ = col_to_simplex_;
    result.filtration_values_ = row_filtration_values_;
    result.row_filtration_values_ = filtration_values_;
    result.last_low_row_to_col_.clear();
    return result;
}

std::vector<double> BoundaryMatrix::applyBoundary(const core::BufferView<const double> &chain) const
{
    return multiply(chain);
}

std::vector<double>
BoundaryMatrix::applyCoboundary(const core::BufferView<const double> &cochain) const
{
    return transposeMultiply(cochain);
}

std::vector<Index> BoundaryMatrix::boundaryOfSimplex(const Simplex &simplex) const
{
    auto col_it = simplex_to_col_.find(simplex);
    if (col_it == simplex_to_col_.end())
    {
        return {};
    }

    std::vector<Index> boundary_indices;
    auto col = col_it->second;
    for (const auto &entry : entries_)
    {
        auto [row_col, value] = entry;
        auto [row, col_idx] = row_col;
        if (col_idx == static_cast<Size>(col) && std::abs(value) > BOUNDARY_MATRIX_TOLERANCE)
        {
            boundary_indices.push_back(static_cast<Index>(row));
        }
    }
    return boundary_indices;
}

std::vector<Index> BoundaryMatrix::coboundaryOfSimplex(const Simplex &simplex) const
{
    auto row_it = simplex_to_row_.find(simplex);
    if (row_it == simplex_to_row_.end())
    {
        return {};
    }

    std::vector<Index> coboundary_indices;
    auto row = row_it->second;
    for (const auto &entry : entries_)
    {
        auto [row_col, value] = entry;
        auto [row_idx, col] = row_col;
        if (static_cast<Size>(row_idx) == static_cast<Size>(row) &&
            std::abs(value) > BOUNDARY_MATRIX_TOLERANCE)
        {
            coboundary_indices.push_back(static_cast<Index>(col));
        }
    }
    return coboundary_indices;
}

std::vector<Simplex> BoundaryMatrix::simplicesInRow(Size row) const
{
    if (row >= row_to_simplex_.size())
    {
        return {};
    }

    std::vector<Simplex> result;
    for (const auto &entry : entries_)
    {
        auto [row_col, value] = entry;
        auto [row_idx, col] = row_col;
        if (row_idx == row && std::abs(value) > BOUNDARY_MATRIX_TOLERANCE &&
            col < col_to_simplex_.size())
        {
            result.push_back(col_to_simplex_[col]);
        }
    }
    return result;
}

std::vector<Simplex> BoundaryMatrix::simplicesInCol(Size col) const
{
    if (col >= col_to_simplex_.size())
    {
        return {};
    }

    std::vector<Simplex> result;
    for (const auto &entry : entries_)
    {
        auto [row_col, value] = entry;
        auto [row, col_idx] = row_col;
        if (col_idx == col && std::abs(value) > BOUNDARY_MATRIX_TOLERANCE &&
            row < row_to_simplex_.size())
        {
            result.push_back(row_to_simplex_[row]);
        }
    }
    return result;
}

Size BoundaryMatrix::numNonzeros() const noexcept
{
    return entries_.size();
}

double BoundaryMatrix::sparsityRatio() const noexcept
{
    if (rows_ == 0 || cols_ == 0)
    {
        return 0.0;
    }
    const long double total = static_cast<long double>(rows_) * static_cast<long double>(cols_);
    return 1.0 - static_cast<double>(static_cast<long double>(entries_.size()) / total);
}

double BoundaryMatrix::getCoefficient(Size row, Size col) const
{
    return getEntry(row, col);
}

void BoundaryMatrix::setEntry(Size row, Size col, double value)
{
    if (row >= rows_ || col >= cols_)
    {
        throw nerve::errors::ShapeMismatchError(__FILE__, __LINE__, __func__)
            .setExpectedDims(rows_, cols_)
            .setActualAccess(static_cast<size_t>(row), static_cast<size_t>(col))
            .setMessage("Matrix index out of bounds")
            .setOperation("BoundaryMatrix::setEntry")
            .addContext("requested_row", row)
            .addContext("max_row", rows_ > 0 ? rows_ - 1 : 0)
            .addContext("requested_col", col)
            .addContext("max_col", cols_ > 0 ? cols_ - 1 : 0)
            .addSuggestion("Check matrix dimensions before accessing elements")
            .addSuggestion("Ensure indices are within [0, rows-1] and [0, cols-1]");
    }

    const auto key = std::make_pair(row, col);
    if (std::abs(value) < BOUNDARY_MATRIX_TOLERANCE)
    {
        entries_.erase(key);
    }
    else
    {
        entries_[key] = value;
    }
}

void BoundaryMatrix::swapRows(Size row1, Size row2)
{
    if (row1 >= rows_ || row2 >= rows_)
    {
        return;
    }

    for (Size col = 0; col < cols_; ++col)
    {
        double val1 = getEntry(row1, col);
        double val2 = getEntry(row2, col);
        setEntry(row1, col, val2);
        setEntry(row2, col, val1);
    }
}

double BoundaryMatrix::getEntry(Size row, Size col) const
{
    auto it = entries_.find({row, col});
    return (it != entries_.end()) ? it->second : 0.0;
}

double BoundaryMatrix::getMatrixEntry(Size row, Size col) const
{
    return getEntry(row, col);
}

double BoundaryMatrix::getRowFiltrationValue(Size row) const
{
    if (row >= row_filtration_values_.size())
    {
        return 0.0;
    }
    return row_filtration_values_[row];
}

const std::vector<Index> &BoundaryMatrix::lastLowRowToCol() const
{
    return last_low_row_to_col_;
}

double BoundaryMatrix::getFiltrationValue(Size col) const
{
    if (col >= filtration_values_.size())
    {
        return 0.0;
    }
    return filtration_values_[col];
}

void BoundaryMatrix::setFiltrationValue(Size col, double value)
{
    if (col >= cols_)
    {
        throw std::out_of_range("BoundaryMatrix filtration column out of range");
    }
    if (!std::isfinite(value))
    {
        throw std::invalid_argument("BoundaryMatrix filtration value must be finite");
    }
    if (filtration_values_.size() < cols_)
    {
        filtration_values_.resize(cols_, 0.0);
    }
    filtration_values_[col] = value;
}

int BoundaryMatrix::getRowSimplexDimension(Size row) const
{
    if (row >= row_to_simplex_.size())
    {
        return -1;
    }
    return static_cast<int>(row_to_simplex_[row].dimension());
}

int BoundaryMatrix::getColSimplexDimension(Size col) const
{
    if (col >= col_to_simplex_.size())
    {
        return -1;
    }
    return static_cast<int>(col_to_simplex_[col].dimension());
}

Index BoundaryMatrix::getColumnIndexForRowSimplex(Size row) const
{
    if (row >= row_to_simplex_.size())
    {
        return -1;
    }
    const auto it = simplex_to_col_.find(row_to_simplex_[row]);
    if (it == simplex_to_col_.end())
    {
        return -1;
    }
    return static_cast<Index>(it->second);
}

} // namespace nerve::algebra


#include "nerve/persistence/adaptive_acceleration/sparse_matrix.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <utility>

namespace nerve::persistence::adaptive_acceleration
{
namespace
{

constexpr double kSparseTolerance = 1e-12;

bool isNonzero(double value)
{
    return std::abs(value) > kSparseTolerance;
}

bool denseElementCount(size_t n_rows, size_t n_cols, size_t &element_count)
{
    if (n_rows > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        n_cols > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        return false;
    }
    if (n_cols != 0 && n_rows > std::numeric_limits<size_t>::max() / n_cols)
    {
        return false;
    }
    element_count = n_rows * n_cols;
    return element_count <= static_cast<size_t>(std::numeric_limits<int>::max());
}

bool validateShape(size_t n_rows, size_t n_cols, size_t value_count)
{
    size_t element_count = 0;
    return denseElementCount(n_rows, n_cols, element_count) && element_count == value_count;
}

} // namespace

SparseMatrix::SparseMatrix(size_t n_rows, size_t n_cols)
    : n_rows_(n_rows)
    , n_cols_(n_cols)
    , nnz_(0)
{
    row_starts_.assign(n_rows_ + 1, 0);
}

errors::ErrorResult<SparseMatrix>
SparseMatrix::fromBoundaryMatrix(const std::vector<std::vector<int>> &boundary_matrix,
                                 size_t n_rows, size_t n_cols)
{
    if (n_rows > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        n_cols > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        return errors::ErrorResult<SparseMatrix>::error(errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }
    try
    {
        SparseMatrix matrix(n_rows, n_cols);
        matrix.initializeFromBoundary(boundary_matrix, n_rows, n_cols);
        matrix.compressStorage();
        matrix.validateStructure();
        return errors::ErrorResult<SparseMatrix>::success(std::move(matrix));
    }
    catch (...)
    {
        return errors::ErrorResult<SparseMatrix>::error(errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }
}

errors::ErrorResult<SparseMatrix>
SparseMatrix::fromDenseMatrix(const std::vector<double> &dense_matrix, size_t n_rows, size_t n_cols)
{
    if (!validateShape(n_rows, n_cols, dense_matrix.size()))
    {
        return errors::ErrorResult<SparseMatrix>::error(errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }
    for (double value : dense_matrix)
    {
        if (!std::isfinite(value))
        {
            return errors::ErrorResult<SparseMatrix>::error(
                errors::ErrorCode::E54_PH4_INVALID_INPUT);
        }
    }
    if (dense_matrix.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        return errors::ErrorResult<SparseMatrix>::error(errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }
    try
    {
        SparseMatrix matrix(n_rows, n_cols);
        matrix.initializeFromDense(dense_matrix, n_rows, n_cols);
        matrix.compressStorage();
        matrix.validateStructure();
        return errors::ErrorResult<SparseMatrix>::success(std::move(matrix));
    }
    catch (...)
    {
        return errors::ErrorResult<SparseMatrix>::error(errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }
}

double SparseMatrix::operator()(size_t row, size_t col) const
{
    if (row >= n_rows_ || col >= n_cols_)
    {
        return 0.0;
    }
    const int start = row_starts_[row];
    const int end = row_starts_[row + 1];
    for (int i = start; i < end; ++i)
    {
        if (col_indices_[static_cast<size_t>(i)] == static_cast<int>(col))
        {
            return values_[static_cast<size_t>(i)];
        }
    }
    return 0.0;
}

std::vector<double> SparseMatrix::getRow(size_t row) const
{
    std::vector<double> out(n_cols_, 0.0);
    if (row >= n_rows_)
    {
        return out;
    }
    const int start = row_starts_[row];
    const int end = row_starts_[row + 1];
    for (int i = start; i < end; ++i)
    {
        const size_t idx = static_cast<size_t>(i);
        const int col = col_indices_[idx];
        if (col >= 0 && static_cast<size_t>(col) < n_cols_)
        {
            out[static_cast<size_t>(col)] = values_[idx];
        }
    }
    return out;
}

std::vector<double> SparseMatrix::getColumn(size_t col) const
{
    std::vector<double> out(n_rows_, 0.0);
    if (col >= n_cols_)
    {
        return out;
    }
    for (size_t row = 0; row < n_rows_; ++row)
    {
        const int start = row_starts_[row];
        const int end = row_starts_[row + 1];
        for (int i = start; i < end; ++i)
        {
            const size_t idx = static_cast<size_t>(i);
            if (col_indices_[idx] == static_cast<int>(col))
            {
                out[row] = values_[idx];
                break;
            }
        }
    }
    return out;
}

errors::ErrorResult<SparseMatrix> SparseMatrix::transpose() const
{
    size_t element_count = 0;
    if (!denseElementCount(n_rows_, n_cols_, element_count))
    {
        return errors::ErrorResult<SparseMatrix>::error(errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }
    std::vector<double> dense(element_count, 0.0);
    for (size_t row = 0; row < n_rows_; ++row)
    {
        const int start = row_starts_[row];
        const int end = row_starts_[row + 1];
        for (int i = start; i < end; ++i)
        {
            const size_t idx = static_cast<size_t>(i);
            const int col = col_indices_[idx];
            if (col >= 0 && static_cast<size_t>(col) < n_cols_)
            {
                dense[static_cast<size_t>(col) * n_rows_ + row] = values_[idx];
            }
        }
    }
    return fromDenseMatrix(dense, n_cols_, n_rows_);
}

errors::ErrorResult<SparseMatrix> SparseMatrix::submatrix(size_t row_start, size_t row_end,
                                                          size_t col_start, size_t col_end) const
{
    if (row_start > row_end || col_start > col_end || row_end > n_rows_ || col_end > n_cols_)
    {
        return errors::ErrorResult<SparseMatrix>::error(errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }
    const size_t rows = row_end - row_start;
    const size_t cols = col_end - col_start;
    size_t element_count = 0;
    if (!denseElementCount(rows, cols, element_count))
    {
        return errors::ErrorResult<SparseMatrix>::error(errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }
    std::vector<double> dense(element_count, 0.0);
    for (size_t row = 0; row < rows; ++row)
    {
        for (size_t col = 0; col < cols; ++col)
        {
            dense[row * cols + col] = (*this)(row_start + row, col_start + col);
        }
    }
    return fromDenseMatrix(dense, rows, cols);
}

bool SparseMatrix::isValid() const
{
    if (row_starts_.size() != n_rows_ + 1)
    {
        return false;
    }
    if (col_indices_.size() != values_.size())
    {
        return false;
    }
    if (row_indices_.size() != values_.size())
    {
        return false;
    }
    if (nnz_ != values_.size())
    {
        return false;
    }
    if (row_starts_.empty() || row_starts_.front() != 0)
    {
        return false;
    }
    for (size_t row = 0; row < n_rows_; ++row)
    {
        if (row_starts_[row] > row_starts_[row + 1])
        {
            return false;
        }
    }
    if (row_starts_.back() != static_cast<int>(values_.size()))
    {
        return false;
    }
    for (size_t i = 0; i < col_indices_.size(); ++i)
    {
        if (col_indices_[i] < 0 || static_cast<size_t>(col_indices_[i]) >= n_cols_)
        {
            return false;
        }
        if (row_indices_[i] < 0 || static_cast<size_t>(row_indices_[i]) >= n_rows_)
        {
            return false;
        }
    }
    return true;
}

size_t SparseMatrix::memoryUsage() const
{
    return (row_indices_.size() * sizeof(int)) + (col_indices_.size() * sizeof(int)) +
           (values_.size() * sizeof(double)) + (row_starts_.size() * sizeof(int));
}

void SparseMatrix::initializeFromDense(const std::vector<double> &dense_matrix, size_t n_rows,
                                       size_t n_cols)
{
    n_rows_ = n_rows;
    n_cols_ = n_cols;
    row_indices_.clear();
    col_indices_.clear();
    values_.clear();
    row_starts_.assign(n_rows_ + 1, 0);

    for (size_t row = 0; row < n_rows_; ++row)
    {
        for (size_t col = 0; col < n_cols_; ++col)
        {
            const double value = dense_matrix[row * n_cols_ + col];
            if (!isNonzero(value))
            {
                continue;
            }
            row_indices_.push_back(static_cast<int>(row));
            col_indices_.push_back(static_cast<int>(col));
            values_.push_back(value);
        }
        row_starts_[row + 1] = static_cast<int>(values_.size());
    }
    nnz_ = values_.size();
}

void SparseMatrix::initializeFromBoundary(const std::vector<std::vector<int>> &boundary_matrix,
                                          size_t n_rows, size_t n_cols)
{
    n_rows_ = n_rows;
    n_cols_ = n_cols;
    row_indices_.clear();
    col_indices_.clear();
    values_.clear();
    row_starts_.assign(n_rows_ + 1, 0);

    std::vector<std::vector<std::pair<int, double>>> entriesByRow(n_rows_);
    const size_t columns = std::min(n_cols_, boundary_matrix.size());
    for (size_t col = 0; col < columns; ++col)
    {
        const auto &boundary = boundary_matrix[col];
        for (size_t k = 0; k < boundary.size(); ++k)
        {
            const int row = boundary[k];
            if (row < 0 || static_cast<size_t>(row) >= n_rows_)
            {
                continue;
            }
            const double sign = (k % 2 == 0) ? 1.0 : -1.0;
            entriesByRow[static_cast<size_t>(row)].push_back({static_cast<int>(col), sign});
        }
    }

    for (size_t row = 0; row < n_rows_; ++row)
    {
        auto &entries = entriesByRow[row];
        std::ranges::sort(entries, {}, &std::pair<int, double>::first);
        for (const auto &[col, value] : entries)
        {
            if (!isNonzero(value))
            {
                continue;
            }
            row_indices_.push_back(static_cast<int>(row));
            col_indices_.push_back(col);
            values_.push_back(value);
        }
        row_starts_[row + 1] = static_cast<int>(values_.size());
    }
    nnz_ = values_.size();
}

void SparseMatrix::compressStorage()
{
    std::vector<int> new_row_indices;
    std::vector<int> new_col_indices;
    std::vector<double> new_values;
    std::vector<int> newRowStarts(n_rows_ + 1, 0);
    new_row_indices.reserve(row_indices_.size());
    new_col_indices.reserve(col_indices_.size());
    new_values.reserve(values_.size());

    for (size_t row = 0; row < n_rows_; ++row)
    {
        const int start = row_starts_[row];
        const int end = row_starts_[row + 1];
        std::vector<std::pair<int, double>> row_entries;
        row_entries.reserve(static_cast<size_t>(std::max(0, end - start)));
        for (int i = start; i < end; ++i)
        {
            const size_t idx = static_cast<size_t>(i);
            row_entries.push_back({col_indices_[idx], values_[idx]});
        }
        std::ranges::sort(row_entries, {}, &std::pair<int, double>::first);

        int current_col = -1;
        double accum = 0.0;
        auto flush = [&]() {
            if (current_col < 0 || !isNonzero(accum))
            {
                return;
            }
            new_row_indices.push_back(static_cast<int>(row));
            new_col_indices.push_back(current_col);
            new_values.push_back(accum);
        };

        for (const auto &[col, value] : row_entries)
        {
            if (col != current_col)
            {
                flush();
                current_col = col;
                accum = value;
            }
            else
            {
                accum += value;
            }
        }
        flush();
        newRowStarts[row + 1] = static_cast<int>(new_values.size());
    }

    row_indices_ = std::move(new_row_indices);
    col_indices_ = std::move(new_col_indices);
    values_ = std::move(new_values);
    row_starts_ = std::move(newRowStarts);
    nnz_ = values_.size();
}

void SparseMatrix::validateStructure() const
{
    if (!isValid())
    {
        throw std::runtime_error("invalid sparse matrix structure");
    }
}

SparseMatrixRowIterator::SparseMatrixRowIterator(const SparseMatrix &matrix, size_t start_row)
    : matrix_(matrix)
    , current_row_(start_row)
{}

SparseMatrixRowIterator &SparseMatrixRowIterator::operator++()
{
    ++current_row_;
    return *this;
}

SparseMatrixRowIterator SparseMatrixRowIterator::operator++(int)
{
    SparseMatrixRowIterator copy(*this);
    ++current_row_;
    return copy;
}

std::vector<double> SparseMatrixRowIterator::operator*() const
{
    return matrix_.getRow(current_row_);
}

bool SparseMatrixRowIterator::operator!=(const SparseMatrixRowIterator &other) const
{
    return &matrix_ != &other.matrix_ || current_row_ != other.current_row_;
}

bool SparseMatrixRowIterator::operator==(const SparseMatrixRowIterator &other) const
{
    return !(*this != other);
}

SparseMatrixColumnIterator::SparseMatrixColumnIterator(const SparseMatrix &matrix, size_t start_col)
    : matrix_(matrix)
    , current_col_(start_col)
{}

SparseMatrixColumnIterator &SparseMatrixColumnIterator::operator++()
{
    ++current_col_;
    return *this;
}

SparseMatrixColumnIterator SparseMatrixColumnIterator::operator++(int)
{
    SparseMatrixColumnIterator copy(*this);
    ++current_col_;
    return copy;
}

std::vector<double> SparseMatrixColumnIterator::operator*() const
{
    return matrix_.getColumn(current_col_);
}

bool SparseMatrixColumnIterator::operator!=(const SparseMatrixColumnIterator &other) const
{
    return &matrix_ != &other.matrix_ || current_col_ != other.current_col_;
}

bool SparseMatrixColumnIterator::operator==(const SparseMatrixColumnIterator &other) const
{
    return !(*this != other);
}

} // namespace nerve::persistence::adaptive_acceleration

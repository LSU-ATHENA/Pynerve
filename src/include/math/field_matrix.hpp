
#pragma once
#include "math/finite_field.hpp"
#include "nerve/error/error_registry.hpp"

#include <algorithm>
#include <compare>
#include <concepts>
#include <cstddef>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <vector>

namespace nerve::math
{

template <typename T>
concept FieldCompatible = std::copyable<T> && requires(T f) {
    { T::zero() } -> std::same_as<T>;
    { T::one() } -> std::same_as<T>;
    { f + f } -> std::same_as<T>;
    { f - f } -> std::same_as<T>;
    { f *f } -> std::same_as<T>;
    { f / f } -> std::same_as<T>;
    { -f } -> std::same_as<T>;
    { f == f } -> std::convertible_to<bool>;
};

template <FieldCompatible Field>
class FieldMatrix
{
private:
    std::vector<std::vector<Field>> data_;
    size_t rows_;
    size_t cols_;

public:
    using field_type = Field;
    using value_type = Field;
    using size_type = size_t;

    FieldMatrix()
        : rows_(0)
        , cols_(0)
    {}
    FieldMatrix(size_t rows, size_t cols)
        : data_(rows, std::vector<Field>(cols, Field::zero()))
        , rows_(rows)
        , cols_(cols)
    {}
    FieldMatrix(size_t rows, size_t cols, Field initial_value)
        : data_(rows, std::vector<Field>(cols, initial_value))
        , rows_(rows)
        , cols_(cols)
    {}
    FieldMatrix(const std::vector<std::vector<Field>> &data)
        : data_(data)
        , rows_(data.size())
        , cols_(data.empty() ? 0 : data[0].size())
    {
        for (const auto &row : data_)
        {
            if (row.size() != cols_)
            {
                throw std::invalid_argument("FieldMatrix rows must have equal length");
            }
        }
    }

    [[nodiscard]] constexpr Field &operator()(size_t row, size_t col)
    {
        if (row >= rows_ || col >= cols_)
        {
            throw std::out_of_range("FieldMatrix index out of bounds");
        }
        return data_[row][col];
    }
    [[nodiscard]] constexpr const Field &operator()(size_t row, size_t col) const
    {
        if (row >= rows_ || col >= cols_)
        {
            throw std::out_of_range("FieldMatrix index out of bounds");
        }
        return data_[row][col];
    }
    [[nodiscard]] std::vector<Field> &getRow(size_t row)
    {
        if (row >= rows_)
        {
            throw std::out_of_range("FieldMatrix row index out of bounds");
        }
        return data_[row];
    }
    [[nodiscard]] const std::vector<Field> &getRow(size_t row) const
    {
        if (row >= rows_)
        {
            throw std::out_of_range("FieldMatrix row index out of bounds");
        }
        return data_[row];
    }
    [[nodiscard]] std::vector<Field> getColumn(size_t col) const
    {
        if (col >= cols_)
        {
            throw std::out_of_range("FieldMatrix column index out of bounds");
        }
        std::vector<Field> column(rows_);
        for (size_t i = 0; i < rows_; ++i)
        {
            column[i] = data_[i][col];
        }
        return column;
    }
    constexpr size_t rows() const { return rows_; }
    constexpr size_t cols() const { return cols_; }
    constexpr size_t size() const { return rows_ * cols_; }
    constexpr bool empty() const { return rows_ == 0 || cols_ == 0; }
    constexpr bool isSquare() const { return rows_ == cols_; }
    void swapRows(size_t i, size_t j)
    {
        if (i >= rows_ || j >= rows_)
        {
            throw std::out_of_range("Row index out of bounds");
        }
        std::swap(data_[i], data_[j]);
    }
    void swapColumns(size_t i, size_t j)
    {
        if (i >= cols_ || j >= cols_)
        {
            throw std::out_of_range("Column index out of bounds");
        }
        for (size_t row = 0; row < rows_; ++row)
        {
            std::swap(data_[row][i], data_[row][j]);
        }
    }
    void scaleRow(size_t row, Field scalar)
    {
        if (row >= rows_)
        {
            throw std::out_of_range("Row index out of bounds");
        }
        if (scalar == Field::zero())
        {
            throw std::invalid_argument("Cannot scale row by zero");
        }
        for (auto &elem : data_[row])
        {
            elem = elem * scalar;
        }
    }
    void addScaledRow(size_t dest_row, size_t src_row, Field scalar)
    {
        if (dest_row >= rows_ || src_row >= rows_)
        {
            throw std::out_of_range("Row index out of bounds");
        }
        for (size_t col = 0; col < cols_; ++col)
        {
            data_[dest_row][col] = data_[dest_row][col] + data_[src_row][col] * scalar;
        }
    }
    std::optional<size_t> findPivot(size_t col, size_t start_row = 0) const
    {
        if (col >= cols_ || start_row >= rows_)
        {
            return std::nullopt;
        }
        for (size_t row = start_row; row < rows_; ++row)
        {
            if (data_[row][col] != Field::zero())
            {
                return row;
            }
        }
        return std::nullopt;
    }
    std::optional<size_t> findPivotMax(size_t col, size_t start_row = 0) const
    {
        if (col >= cols_ || start_row >= rows_)
        {
            return std::nullopt;
        }
        size_t max_row = start_row;
        int max_abs = std::abs(data_[start_row][col].toInt());
        for (size_t row = start_row + 1; row < rows_; ++row)
        {
            int current_abs = std::abs(data_[row][col].toInt());
            if (current_abs > max_abs)
            {
                max_abs = current_abs;
                max_row = row;
            }
        }
        if (max_abs == 0)
        {
            return std::nullopt;
        }
        return max_row;
    }
    bool isZeroRow(size_t row) const
    {
        if (row >= rows_)
        {
            throw std::out_of_range("Row index out of bounds");
        }
        for (const auto &elem : data_[row])
        {
            if (elem != Field::zero())
            {
                return false;
            }
        }
        return true;
    }
    bool isZeroColumn(size_t col) const
    {
        if (col >= cols_)
        {
            throw std::out_of_range("Column index out of bounds");
        }
        for (size_t row = 0; row < rows_; ++row)
        {
            if (data_[row][col] != Field::zero())
            {
                return false;
            }
        }
        return true;
    }
    size_t rowRank() const
    {
        size_t rank = 0;
        for (size_t row = 0; row < rows_; ++row)
        {
            if (!isZeroRow(row))
            {
                ++rank;
            }
        }
        return rank;
    }
    size_t columnRank() const
    {
        size_t rank = 0;
        for (size_t col = 0; col < cols_; ++col)
        {
            if (!isZeroColumn(col))
            {
                ++rank;
            }
        }
        return rank;
    }
    FieldMatrix operator+(const FieldMatrix &other) const
    {
        if (rows_ != other.rows_ || cols_ != other.cols_)
        {
            throw std::invalid_argument("Matrix dimensions must match for addition");
        }
        FieldMatrix result(rows_, cols_);
        for (size_t i = 0; i < rows_; ++i)
        {
            for (size_t j = 0; j < cols_; ++j)
            {
                result(i, j) = data_[i][j] + other(i, j);
            }
        }
        return result;
    }
    FieldMatrix operator-(const FieldMatrix &other) const
    {
        if (rows_ != other.rows_ || cols_ != other.cols_)
        {
            throw std::invalid_argument("Matrix dimensions must match for subtraction");
        }
        FieldMatrix result(rows_, cols_);
        for (size_t i = 0; i < rows_; ++i)
        {
            for (size_t j = 0; j < cols_; ++j)
            {
                result(i, j) = data_[i][j] - other(i, j);
            }
        }
        return result;
    }
    FieldMatrix operator*(const FieldMatrix &other) const
    {
        if (cols_ != other.rows_)
        {
            throw std::invalid_argument("Inner matrix dimensions must match for multiplication");
        }
        FieldMatrix result(rows_, other.cols_);
        for (size_t i = 0; i < rows_; ++i)
        {
            for (size_t j = 0; j < other.cols_; ++j)
            {
                Field sum = Field::zero();
                for (size_t k = 0; k < cols_; ++k)
                {
                    sum = sum + data_[i][k] * other(k, j);
                }
                result(i, j) = sum;
            }
        }
        return result;
    }
    FieldMatrix operator*(Field scalar) const
    {
        FieldMatrix result(rows_, cols_);
        for (size_t i = 0; i < rows_; ++i)
        {
            for (size_t j = 0; j < cols_; ++j)
            {
                result(i, j) = data_[i][j] * scalar;
            }
        }
        return result;
    }
    FieldMatrix transpose() const
    {
        FieldMatrix result(cols_, rows_);
        for (size_t i = 0; i < rows_; ++i)
        {
            for (size_t j = 0; j < cols_; ++j)
            {
                result(j, i) = data_[i][j];
            }
        }
        return result;
    }
    std::vector<std::vector<Field>> getData() const { return data_; }
    void setData(const std::vector<std::vector<Field>> &data)
    {
        if (data.empty())
        {
            rows_ = 0;
            cols_ = 0;
            data_.clear();
            return;
        }
        size_t new_cols = data[0].size();
        for (const auto &row : data)
        {
            if (row.size() != new_cols)
            {
                throw std::invalid_argument("All rows must have the same number of columns");
            }
        }
        data_ = data;
        rows_ = data.size();
        cols_ = new_cols;
    }
    void clear()
    {
        data_.clear();
        rows_ = 0;
        cols_ = 0;
    }
    void resize(size_t new_rows, size_t new_cols)
    {
        data_.resize(new_rows);
        for (auto &row : data_)
        {
            row.resize(new_cols, Field::zero());
        }
        rows_ = new_rows;
        cols_ = new_cols;
    }
    void fill(Field value)
    {
        for (auto &row : data_)
        {
            std::fill(row.begin(), row.end(), value);
        }
    }
    bool operator==(const FieldMatrix &other) const
    {
        if (rows_ != other.rows_ || cols_ != other.cols_)
        {
            return false;
        }
        for (size_t i = 0; i < rows_; ++i)
        {
            for (size_t j = 0; j < cols_; ++j)
            {
                if (data_[i][j] != other(i, j))
                {
                    return false;
                }
            }
        }
        return true;
    }
    bool operator!=(const FieldMatrix &other) const { return !(*this == other); }
    friend std::ostream &operator<<(std::ostream &os, const FieldMatrix &matrix)
    {
        for (size_t i = 0; i < matrix.rows_; ++i)
        {
            os << "[";
            for (size_t j = 0; j < matrix.cols_; ++j)
            {
                os << matrix(i, j);
                if (j < matrix.cols_ - 1)
                {
                    os << ", ";
                }
            }
            os << "]";
            if (i < matrix.rows_ - 1)
            {
                os << "\n";
            }
        }
        return os;
    }
};
using Z2Matrix = FieldMatrix<Z2>;
using Z3Matrix = FieldMatrix<Z3>;
using Z5Matrix = FieldMatrix<Z5>;
using Z7Matrix = FieldMatrix<Z7>;
template <typename Field>
error::Result<FieldMatrix<Field>> makeFieldMatrix(size_t rows, size_t cols)
{
    try
    {
        return error::Result<FieldMatrix<Field>>::ok(FieldMatrix<Field>(rows, cols));
    }
    catch (const std::exception &e)
    {
        return error::Result<FieldMatrix<Field>>::err(
            error::TDAErrorCode::InvalidFieldOperation,
            std::string("Failed to create field matrix: ") + e.what());
    }
}
template <typename Field>
error::Result<FieldMatrix<Field>> makeFieldMatrix(size_t rows, size_t cols, Field initial_value)
{
    try
    {
        return error::Result<FieldMatrix<Field>>::ok(FieldMatrix<Field>(rows, cols, initial_value));
    }
    catch (const std::exception &e)
    {
        return error::Result<FieldMatrix<Field>>::err(
            error::TDAErrorCode::InvalidFieldOperation,
            std::string("Failed to create field matrix with initial value: ") + e.what());
    }
}
template <typename Field>
error::Result<FieldMatrix<Field>>
makeFieldMatrixFromData(const std::vector<std::vector<Field>> &data)
{
    try
    {
        return error::Result<FieldMatrix<Field>>::ok(FieldMatrix<Field>(data));
    }
    catch (const std::exception &e)
    {
        return error::Result<FieldMatrix<Field>>::err(
            error::TDAErrorCode::InvalidFieldOperation,
            std::string("Failed to create field matrix from data: ") + e.what());
    }
}
inline error::Result<Z2Matrix> makeZ2Matrix(size_t rows, size_t cols)
{
    return makeFieldMatrix<Z2>(rows, cols);
}
inline error::Result<Z3Matrix> makeZ3Matrix(size_t rows, size_t cols)
{
    return makeFieldMatrix<Z3>(rows, cols);
}
inline error::Result<Z5Matrix> makeZ5Matrix(size_t rows, size_t cols)
{
    return makeFieldMatrix<Z5>(rows, cols);
}
inline error::Result<Z7Matrix> makeZ7Matrix(size_t rows, size_t cols)
{
    return makeFieldMatrix<Z7>(rows, cols);
}
} // namespace nerve::math

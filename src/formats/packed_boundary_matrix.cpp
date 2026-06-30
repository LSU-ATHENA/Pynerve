#include "nerve/formats/packed_boundary_matrix.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace nerve::formats
{

namespace
{

bool fitsIndex(Size value) noexcept
{
    return value <= static_cast<Size>(std::numeric_limits<Index>::max());
}

} // namespace

PackedBoundaryMatrix::PackedBoundaryMatrix()
    : format_(Format::COO)
    , n_rows_(0)
    , n_cols_(0)
    , nnz_(0)
    , num_words_(0)
{}

PackedBoundaryMatrix::PackedBoundaryMatrix(Size n_rows, Size n_cols)
    : format_(Format::COO)
    , n_rows_(n_rows)
    , n_cols_(n_cols)
    , nnz_(0)
    , num_words_(wordsNeeded(n_rows))
{}

errors::ErrorResult<PackedBoundaryMatrix>
PackedBoundaryMatrix::fromCoo(Size n_rows, Size n_cols, const std::vector<Index> &row_indices,
                              const std::vector<Index> &col_indices)
{
    if (row_indices.size() != col_indices.size())
    {
        return errors::ErrorResult<PackedBoundaryMatrix>::error(
            errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }
    if (!fitsIndex(n_rows) || !fitsIndex(n_cols))
    {
        return errors::ErrorResult<PackedBoundaryMatrix>::error(
            errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }
    if (row_indices.size() > static_cast<Size>(std::numeric_limits<Index>::max()))
    {
        return errors::ErrorResult<PackedBoundaryMatrix>::error(
            errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }

    try
    {
        PackedBoundaryMatrix matrix(n_rows, n_cols);
        matrix.buildFromCooInternal(row_indices, col_indices);
        return errors::ErrorResult<PackedBoundaryMatrix>::success(std::move(matrix));
    }
    catch (...)
    {
        return errors::ErrorResult<PackedBoundaryMatrix>::error(
            errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }
}

errors::ErrorResult<PackedBoundaryMatrix>
PackedBoundaryMatrix::fromBoundaryColumns(Size n_rows,
                                          const std::vector<std::vector<Index>> &boundary_columns)
{
    if (!fitsIndex(n_rows))
    {
        return errors::ErrorResult<PackedBoundaryMatrix>::error(
            errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }
    if (boundary_columns.size() > static_cast<Size>(std::numeric_limits<Index>::max()))
    {
        return errors::ErrorResult<PackedBoundaryMatrix>::error(
            errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }

    try
    {
        PackedBoundaryMatrix matrix(n_rows, boundary_columns.size());
        matrix.buildFromColumnsInternal(boundary_columns);
        return errors::ErrorResult<PackedBoundaryMatrix>::success(std::move(matrix));
    }
    catch (...)
    {
        return errors::ErrorResult<PackedBoundaryMatrix>::error(
            errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }
}

errors::ErrorResult<PackedBoundaryMatrix>
PackedBoundaryMatrix::fromCscFormat(Size n_rows, Size n_cols, const std::vector<Index> &col_starts,
                                    const std::vector<Index> &row_indices)
{
    if (!fitsIndex(n_rows) || !fitsIndex(n_cols))
    {
        return errors::ErrorResult<PackedBoundaryMatrix>::error(
            errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }
    if (col_starts.size() != n_cols + 1)
    {
        return errors::ErrorResult<PackedBoundaryMatrix>::error(
            errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }

    try
    {
        PackedBoundaryMatrix matrix(n_rows, n_cols);
        matrix.format_ = Format::CSC;
        matrix.col_starts_ = col_starts;
        matrix.row_indices_ = row_indices;
        matrix.nnz_ = row_indices.size();

        std::vector<Index> col_idx;
        col_idx.reserve(matrix.nnz_);
        for (Size col = 0; col < n_cols; ++col)
        {
            const Index start = col_starts[col];
            const Index end = col_starts[col + 1];
            for (Index i = start; i < end; ++i)
            {
                col_idx.push_back(static_cast<Index>(col));
            }
        }
        matrix.col_indices_ = std::move(col_idx);

        return errors::ErrorResult<PackedBoundaryMatrix>::success(std::move(matrix));
    }
    catch (...)
    {
        return errors::ErrorResult<PackedBoundaryMatrix>::error(
            errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }
}

void PackedBoundaryMatrix::convertToPacked()
{
    if (format_ == Format::Packed)
    {
        return;
    }
    packColumnsInternal();
    format_ = Format::Packed;
}

void PackedBoundaryMatrix::buildFromCooInternal(const std::vector<Index> &row_indices,
                                                const std::vector<Index> &col_indices)
{
    row_indices_ = row_indices;
    col_indices_ = col_indices;
    nnz_ = row_indices.size();

    col_starts_.resize(n_cols_ + 1, 0);
    std::vector<Size> col_counts(n_cols_, 0);
    for (const Index c : col_indices)
    {
        col_counts[static_cast<Size>(c)]++;
    }
    Size pos = 0;
    for (Size col = 0; col < n_cols_; ++col)
    {
        col_starts_[col] = static_cast<Index>(pos);
        pos += col_counts[col];
    }
    col_starts_[n_cols_] = static_cast<Index>(pos);
}

void PackedBoundaryMatrix::buildFromColumnsInternal(const std::vector<std::vector<Index>> &columns)
{
    col_starts_.resize(n_cols_ + 1, 0);
    Size total = 0;
    for (Size col = 0; col < n_cols_; ++col)
    {
        col_starts_[col] = static_cast<Index>(total);
        total += columns[col].size();
    }
    col_starts_[n_cols_] = static_cast<Index>(total);
    nnz_ = total;

    row_indices_.reserve(total);
    col_indices_.reserve(total);
    for (Size col = 0; col < n_cols_; ++col)
    {
        for (const Index row : columns[col])
        {
            if (row >= 0 && static_cast<Size>(row) < n_rows_)
            {
                row_indices_.push_back(row);
                col_indices_.push_back(static_cast<Index>(col));
            }
        }
    }
    nnz_ = row_indices_.size();
}

void PackedBoundaryMatrix::packColumnsInternal()
{
    num_words_ = wordsNeeded(n_rows_);
    const Size total_words = n_cols_ * num_words_;
    packed_columns_.assign(total_words, 0);
    column_sizes_.assign(n_cols_, 0);

    if (format_ == Format::CSC || format_ == Format::COO)
    {
        for (Size col = 0; col < n_cols_; ++col)
        {
            const Index start = col_starts_[col];
            const Index end = col_starts_[col + 1];
            PackedWord *col_base = packed_columns_.data() + col * num_words_;

            for (Index i = start; i < end; ++i)
            {
                const Index row = row_indices_[static_cast<Size>(i)];
                if (row < 0 || static_cast<Size>(row) >= n_rows_)
                {
                    continue;
                }
                const Size word_idx = static_cast<Size>(row) / kBitsPerPackedWord;
                const Size bit_idx = static_cast<Size>(row) % kBitsPerPackedWord;
                col_base[word_idx] |= (static_cast<PackedWord>(1) << bit_idx);
            }

            Size word_count = 0;
            for (Size w = 0; w < num_words_; ++w)
            {
                if (col_base[w] != 0)
                {
                    word_count = w + 1;
                }
            }
            column_sizes_[col] = word_count;
        }
    }
}

errors::ErrorResult<GpuPackedLayout> PackedBoundaryMatrix::buildGpuLayout() const
{
    GpuPackedLayout layout;
    layout.num_columns = n_cols_;
    layout.num_rows = n_rows_;
    layout.max_words_per_column = num_words_;
    layout.total_packed_bytes = 0;

    Size stride = num_words_;
    if (!column_sizes_.empty())
    {
        for (Size s : column_sizes_)
        {
            stride = std::max(stride, s);
        }
    }
    if (stride == 0)
    {
        stride = num_words_;
    }

    layout.max_words_per_column = stride;

    if (n_cols_ > 0 && stride > 0)
    {
        if (stride > std::numeric_limits<Size>::max() / n_cols_)
        {
            return errors::ErrorResult<GpuPackedLayout>::error(
                errors::ErrorCode::E41_RESOURCE_LIMIT);
        }
        const Size total = n_cols_ * stride;
        layout.columns_flat.resize(total, 0);
        layout.column_offsets.resize(n_cols_ + 1, 0);

        const PackedWord *src = packed_columns_.data();
        PackedWord *dst = layout.columns_flat.data();

        for (Size col = 0; col < n_cols_; ++col)
        {
            layout.column_offsets[col] = col * stride;
            const Size words =
                column_sizes_.empty() ? num_words_ : std::min(column_sizes_[col], stride);
            std::memcpy(dst + col * stride, src + col * num_words_, words * kPackedWordBytes);
        }
        layout.column_offsets[n_cols_] = n_cols_ * stride;
    }

    layout.total_packed_bytes = layout.columns_flat.size() * kPackedWordBytes;
    return errors::ErrorResult<GpuPackedLayout>::success(std::move(layout));
}

PackedColumn PackedBoundaryMatrix::getColumn(Index col) const
{
    PackedColumn result;
    if (col < 0 || static_cast<Size>(col) >= n_cols_)
    {
        return result;
    }
    if (format_ != Format::Packed)
    {
        return result;
    }
    const Size size = column_sizes_.empty() ? num_words_ : column_sizes_[static_cast<Size>(col)];
    result.words.assign(
        packed_columns_.begin() + static_cast<Offset>(col) * static_cast<Offset>(num_words_),
        packed_columns_.begin() + static_cast<Offset>(col) * static_cast<Offset>(num_words_) +
            static_cast<Offset>(size));
    return result;
}

std::vector<Index> PackedBoundaryMatrix::getColumnIndices(Index col) const
{
    std::vector<Index> result;
    if (col < 0 || static_cast<Size>(col) >= n_cols_)
    {
        return result;
    }
    const Index start = col_starts_[static_cast<Size>(col)];
    const Index end = col_starts_[static_cast<Size>(col) + 1];
    result.reserve(static_cast<Size>(end - start));
    for (Index i = start; i < end; ++i)
    {
        result.push_back(row_indices_[static_cast<Size>(i)]);
    }
    return result;
}

Size PackedBoundaryMatrix::memoryUsage() const
{
    Size total = 0;
    total += packed_columns_.size() * kPackedWordBytes;
    total += column_sizes_.size() * sizeof(Size);
    total += col_indices_.size() * sizeof(Index);
    total += row_indices_.size() * sizeof(Index);
    total += col_starts_.size() * sizeof(Index);
    return total;
}

bool PackedBoundaryMatrix::isValid() const
{
    if (n_rows_ == 0 || n_cols_ == 0)
    {
        return nnz_ == 0;
    }
    if (format_ == Format::CSC || format_ == Format::COO)
    {
        if (col_starts_.size() != n_cols_ + 1)
        {
            return false;
        }
        if (row_indices_.size() != nnz_)
        {
            return false;
        }
    }
    if (format_ == Format::Packed)
    {
        if (!column_sizes_.empty() && column_sizes_.size() != n_cols_)
        {
            return false;
        }
        const Size expected_words = n_cols_ * num_words_;
        if (packed_columns_.size() < expected_words)
        {
            return false;
        }
    }
    return true;
}

} // namespace nerve::formats

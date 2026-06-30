#pragma once

#include "nerve/errors/errors.hpp"
#include "nerve/types.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace nerve::formats
{

struct PackedColumn
{
    std::vector<PackedWord> words;
};

struct GpuPackedLayout
{
    std::vector<PackedWord> columns_flat;
    std::vector<Size> column_offsets;
    Size num_columns;
    Size num_rows;
    Size max_words_per_column;
    Size total_packed_bytes;
};

struct GpuScanResult
{
    std::vector<Index> stable_columns;
    std::vector<Index> unstable_columns;
    std::vector<Index> leftmost_column_by_row;
    Size stable_count;
    Size unstable_count;
};

class PackedBoundaryMatrix
{
public:
    enum class Format
    {
        COO,
        CSC,
        Packed
    };

    PackedBoundaryMatrix();

    static errors::ErrorResult<PackedBoundaryMatrix> fromCoo(Size n_rows, Size n_cols,
                                                             const std::vector<Index> &row_indices,
                                                             const std::vector<Index> &col_indices);

    static errors::ErrorResult<PackedBoundaryMatrix>
    fromBoundaryColumns(Size n_rows, const std::vector<std::vector<Index>> &boundary_columns);

    static errors::ErrorResult<PackedBoundaryMatrix>
    fromCscFormat(Size n_rows, Size n_cols, const std::vector<Index> &col_starts,
                  const std::vector<Index> &row_indices);

    void convertToPacked();

    [[nodiscard]] errors::ErrorResult<GpuPackedLayout> buildGpuLayout() const;

    [[nodiscard]] PackedColumn getColumn(Index col) const;
    [[nodiscard]] std::vector<Index> getColumnIndices(Index col) const;

    [[nodiscard]] Size numRows() const { return n_rows_; }
    [[nodiscard]] Size numCols() const { return n_cols_; }
    [[nodiscard]] Size nnz() const { return nnz_; }
    [[nodiscard]] Size numWordsPerColumn() const { return num_words_; }
    [[nodiscard]] Format format() const { return format_; }

    [[nodiscard]] const std::vector<PackedWord> &packedColumns() const { return packed_columns_; }
    [[nodiscard]] const std::vector<Size> &columnSizes() const { return column_sizes_; }
    [[nodiscard]] const std::vector<Index> &colIndices() const { return col_indices_; }
    [[nodiscard]] const std::vector<Index> &rowIndices() const { return row_indices_; }
    [[nodiscard]] const std::vector<Index> &colStarts() const { return col_starts_; }

    [[nodiscard]] Size memoryUsage() const;
    [[nodiscard]] bool isValid() const;

private:
    PackedBoundaryMatrix(Size n_rows, Size n_cols);

    void buildFromCooInternal(const std::vector<Index> &row_indices,
                              const std::vector<Index> &col_indices);
    void buildFromColumnsInternal(const std::vector<std::vector<Index>> &columns);
    void packColumnsInternal();

    Format format_;

    Size n_rows_;
    Size n_cols_;
    Size nnz_;
    Size num_words_;

    std::vector<Index> col_indices_;
    std::vector<Index> row_indices_;
    std::vector<Index> col_starts_;

    std::vector<PackedWord> packed_columns_;
    std::vector<Size> column_sizes_;
};

[[nodiscard]] inline Size wordsNeeded(Size n_rows) noexcept
{
    return n_rows == 0 ? 0 : (n_rows + kBitsPerPackedWord - 1) / kBitsPerPackedWord;
}

inline void setBit(PackedWord &word, Index bit_index) noexcept
{
    word |= (static_cast<PackedWord>(1) << static_cast<Size>(bit_index));
}

[[nodiscard]] inline bool testBit(PackedWord word, Index bit_index) noexcept
{
    return (word & (static_cast<PackedWord>(1) << static_cast<Size>(bit_index))) != 0;
}

[[nodiscard]] inline Index findHighestSetBit(PackedWord word) noexcept
{
    if (word == 0)
    {
        return -1;
    }
    return static_cast<Index>(63 - __builtin_clzll(word));
}

} // namespace nerve::formats

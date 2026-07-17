#pragma once
#include "nerve/core_types.hpp"
#include "nerve/persistence/core/core_types.hpp"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef __AVX512F__
#include "nerve/cpu/x86_intrinsics.hpp"
#elif defined(__AVX2__)
#include "nerve/cpu/x86_intrinsics.hpp"
#endif

namespace nerve::algebra::compressed
{

inline constexpr size_t BITS_PER_WORD = 64;

class BitPackedZ2Matrix
{
public:
    BitPackedZ2Matrix(size_t n_rows, size_t n_cols)
        : n_rows_(n_rows)
        , n_cols_(n_cols)
    {
        if (n_rows > std::numeric_limits<size_t>::max() - (BITS_PER_WORD - 1))
        {
            throw std::overflow_error("bit-packed row count overflows word calculation");
        }
        words_per_col_ = (n_rows + BITS_PER_WORD - 1) / BITS_PER_WORD;
        if (words_per_col_ != 0 && n_cols > std::numeric_limits<size_t>::max() / words_per_col_)
        {
            throw std::overflow_error("bit-packed matrix shape overflows storage size");
        }
        const size_t word_count = words_per_col_ * n_cols;
        if (word_count > std::vector<uint64_t>().max_size())
        {
            throw std::length_error("bit-packed matrix storage exceeds vector capacity");
        }
        data_.resize(word_count, 0);
    }

    [[nodiscard]] bool get(size_t row, size_t col) const noexcept
    {
        if (row >= n_rows_ || col >= n_cols_)
            return false;
        const size_t word_idx = wordIndex(row, col);
        const size_t bit_off = row % BITS_PER_WORD;
        return (data_[word_idx] >> bit_off) & 1ULL;
    }

    void set(size_t row, size_t col) noexcept
    {
        if (row >= n_rows_ || col >= n_cols_)
            return;
        const size_t word_idx = wordIndex(row, col);
        const size_t bit_off = row % BITS_PER_WORD;
        data_[word_idx] |= (1ULL << bit_off);
    }

    void clear(size_t row, size_t col) noexcept
    {
        if (row >= n_rows_ || col >= n_cols_)
            return;
        const size_t word_idx = wordIndex(row, col);
        const size_t bit_off = row % BITS_PER_WORD;
        data_[word_idx] &= ~(1ULL << bit_off);
    }

    void xorColumns(size_t target_col, size_t source_col) noexcept
    {
        if (target_col >= n_cols_ || source_col >= n_cols_)
            return;
        size_t target_start = target_col * words_per_col_;
        size_t source_start = source_col * words_per_col_;
        for (size_t i = 0; i < words_per_col_; ++i)
        {
            data_[target_start + i] ^= data_[source_start + i];
        }
    }

    [[nodiscard]] std::ptrdiff_t findPivot(size_t col) const noexcept
    {
        if (col >= n_cols_)
            return -1;
        size_t col_start = col * words_per_col_;
        for (std::ptrdiff_t word_idx = static_cast<std::ptrdiff_t>(words_per_col_) - 1; word_idx >= 0; --word_idx)
        {
            uint64_t word = data_[col_start + word_idx];
            if (word != 0)
            {
                const size_t bit_pos = BITS_PER_WORD - 1 - std::countl_zero(word);
                const size_t row = static_cast<size_t>(word_idx) * BITS_PER_WORD + bit_pos;
                if (static_cast<size_t>(row) < n_rows_)
                {
                    return static_cast<std::ptrdiff_t>(row);
                }
            }
        }
        return -1;
    }

    [[nodiscard]] bool isColumnEmpty(size_t col) const noexcept
    {
        if (col >= n_cols_)
            return true;
        size_t col_start = col * words_per_col_;
        for (size_t i = 0; i < words_per_col_; ++i)
        {
            if (data_[col_start + i] != 0)
                return false;
        }
        return true;
    }

    void clearColumn(size_t col) noexcept
    {
        if (col >= n_cols_)
            return;
        size_t col_start = col * words_per_col_;
        for (size_t i = 0; i < words_per_col_; ++i)
        {
            data_[col_start + i] = 0;
        }
    }

    [[nodiscard]] std::vector<size_t> getColumnEntries(size_t col) const
    {
        std::vector<size_t> entries;
        if (col >= n_cols_)
            return entries;
        size_t col_start = col * words_per_col_;
        for (size_t word_idx = 0; word_idx < words_per_col_; ++word_idx)
        {
            uint64_t word = data_[col_start + word_idx];
            while (word != 0)
            {
                size_t bit_pos = std::countr_zero(word);
                size_t row = word_idx * BITS_PER_WORD + bit_pos;
                if (row < n_rows_)
                {
                    entries.push_back(row);
                }
                word &= (word - 1);
            }
        }
        return entries;
    }

    [[nodiscard]] std::vector<int> getColumn(int pivot) const
    {
        auto entries = getColumnEntries(static_cast<size_t>(pivot));
        return std::vector<int>(entries.begin(), entries.end());
    }

    [[nodiscard]] bool hasColumn(int pivot) const
    {
        return !isColumnEmpty(static_cast<size_t>(pivot));
    }

    void addColumn(size_t col_idx, const std::vector<int> &column)
    {
        for (int row : column)
        {
            if (row >= 0)
            {
                set(static_cast<size_t>(row), col_idx);
            }
        }
    }

    void xorWith(const BitPackedZ2Matrix &other) noexcept
    {
        size_t min_size = std::min(data_.size(), other.data_.size());
        for (size_t i = 0; i < min_size; ++i)
        {
            data_[i] ^= other.data_[i];
        }
    }

    void clearAll() noexcept { std::fill(data_.begin(), data_.end(), 0); }

    [[nodiscard]] size_t getNumRows() const noexcept { return n_rows_; }
    [[nodiscard]] size_t getNumCols() const noexcept { return n_cols_; }
    [[nodiscard]] size_t memoryBytes() const noexcept { return data_.size() * sizeof(uint64_t); }
    [[nodiscard]] double compressionRatio() const noexcept
    {
        if (memoryBytes() == 0)
            return 0.0;
        const long double dense_bytes = static_cast<long double>(n_rows_) *
                                        static_cast<long double>(n_cols_) *
                                        static_cast<long double>(sizeof(int32_t));
        return static_cast<double>(dense_bytes / static_cast<long double>(memoryBytes()));
    }
    [[nodiscard]] double density() const noexcept
    {
        const long double total_bits =
            static_cast<long double>(n_rows_) * static_cast<long double>(n_cols_);
        if (total_bits == 0.0L)
            return 0.0;
        size_t set_bits = 0;
        for (uint64_t word : data_)
        {
            set_bits += std::popcount(word);
        }
        return static_cast<double>(static_cast<long double>(set_bits) / total_bits);
    }
    [[nodiscard]] std::string statsString() const
    {
        std::ostringstream out;
        out << "BitPackedZ2Matrix (" << n_rows_ << 'x' << n_cols_ << ")\n";
        out << "Memory: " << memoryBytes() << " bytes\n";
        out << "Compression: " << compressionRatio() << "x\n";
        out << "Density: " << density();
        return out.str();
    }

protected:
    size_t n_rows_ = 0;
    size_t n_cols_ = 0;
    size_t words_per_col_ = 0;
    std::vector<uint64_t> data_;

    [[nodiscard]] size_t wordIndex(size_t row, size_t col) const noexcept
    {
        return col * words_per_col_ + row / BITS_PER_WORD;
    }
};

[[nodiscard]] inline BitPackedZ2Matrix toBitPacked(std::span<const std::vector<int>> dense_matrix,
                                                   size_t n_rows)
{
    size_t n_cols = dense_matrix.size();
    BitPackedZ2Matrix packed(n_rows, n_cols);
    for (size_t col = 0; col < n_cols; ++col)
    {
        for (int row : dense_matrix[col])
        {
            if (row >= 0 && static_cast<size_t>(row) < n_rows)
            {
                packed.set(static_cast<size_t>(row), col);
            }
        }
    }
    return packed;
}

[[nodiscard]] inline std::vector<std::pair<std::ptrdiff_t, size_t>>
reduceBitPacked(BitPackedZ2Matrix &matrix)
{
    std::vector<std::pair<std::ptrdiff_t, size_t>> pivots;
    pivots.reserve(matrix.getNumCols());

    std::unordered_map<std::ptrdiff_t, size_t> pivot_to_col;
    for (size_t col = 0; col < matrix.getNumCols(); ++col)
    {
        std::ptrdiff_t pivot = matrix.findPivot(col);
        while (pivot >= 0)
        {
            auto it = pivot_to_col.find(pivot);
            if (it == pivot_to_col.end())
            {
                break;
            }
            matrix.xorColumns(col, it->second);
            pivot = matrix.findPivot(col);
        }

        pivots.push_back({pivot, col});
        if (pivot >= 0)
        {
            pivot_to_col[pivot] = col;
        }
    }
    return pivots;
}

[[nodiscard]] inline int inferSimplexDimension(const std::vector<std::vector<int>> &boundary_matrix,
                                               size_t simplex_idx)
{
    if (simplex_idx >= boundary_matrix.size())
    {
        return -1;
    }
    const size_t face_count = boundary_matrix[simplex_idx].size();
    if (face_count == 0)
    {
        return 0;
    }
    if (face_count > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        return std::numeric_limits<int>::max();
    }
    return static_cast<int>(face_count) - 1;
}

[[nodiscard]] inline persistence::PersistenceDiagram
computeBitPackedPH(const std::vector<std::vector<int>> &boundary_matrix,
                   const std::vector<double> &filtration_values, int max_dim)
{
    const size_t n_simplices = boundary_matrix.size();
    persistence::PersistenceDiagram result;
    if (max_dim < 0 || filtration_values.size() < n_simplices)
    {
        return result;
    }

    BitPackedZ2Matrix packed = toBitPacked(boundary_matrix, n_simplices);
    auto pivots = reduceBitPacked(packed);
    result.pairs.reserve(pivots.size());

    std::vector<int> simplex_dimensions;
    simplex_dimensions.reserve(n_simplices);
    for (size_t simplex_idx = 0; simplex_idx < n_simplices; ++simplex_idx)
    {
        simplex_dimensions.push_back(inferSimplexDimension(boundary_matrix, simplex_idx));
    }

    for (const auto &[pivot, col] : pivots)
    {
        const size_t birth_idx = pivot >= 0 ? static_cast<size_t>(pivot) : col;
        const int dimension =
            birth_idx < simplex_dimensions.size() ? simplex_dimensions[birth_idx] : -1;
        if (dimension < 0 || dimension > max_dim)
        {
            continue;
        }
        persistence::PersistencePair pair;
        pair.death = pivot >= 0 ? filtration_values[col] : std::numeric_limits<double>::infinity();
        pair.birth = filtration_values[birth_idx];
        pair.dimension = dimension;
        result.pairs.push_back(pair);
    }
    return result;
}

} // namespace nerve::algebra::compressed

#pragma once

#include "nerve/platform.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace nerve::persistence
{

class BitTreePivotColumn
{
public:
    static constexpr int MAX_ROWS = 1 << 20;
    static constexpr int BITS_PER_WORD = 64;
    static constexpr int BITS_PER_LEVEL = 6;
    static constexpr int WORDS_PER_LEVEL = (MAX_ROWS + 63) / 64;

    explicit BitTreePivotColumn(int max_rows = 0)
        : max_rows_(max_rows)
        , num_words_((max_rows + BITS_PER_WORD - 1) / BITS_PER_WORD)
    {
        data_.resize(num_words_, 0);
        pivot_ = -1;
    }

    BitTreePivotColumn(const BitTreePivotColumn &) = default;
    BitTreePivotColumn(BitTreePivotColumn &&) noexcept = default;
    BitTreePivotColumn &operator=(const BitTreePivotColumn &) = default;
    BitTreePivotColumn &operator=(BitTreePivotColumn &&) noexcept = default;

    int pivot() const { return pivot_; }

    bool empty() const { return pivot_ < 0; }

    void set(int row)
    {
        if (row >= 0 && row < max_rows_)
        {
            size_t word = static_cast<size_t>(row) / BITS_PER_WORD;
            int bit = row % BITS_PER_WORD;
            data_[word] |= (1ULL << bit);
            updatePivot();
        }
    }

    void clear(int row)
    {
        if (row >= 0 && row < max_rows_)
        {
            size_t word = static_cast<size_t>(row) / BITS_PER_WORD;
            int bit = row % BITS_PER_WORD;
            data_[word] &= ~(1ULL << bit);
            updatePivot();
        }
    }

    bool has(int row) const
    {
        if (row < 0 || row >= max_rows_)
            return false;
        size_t word = static_cast<size_t>(row) / BITS_PER_WORD;
        int bit = row % BITS_PER_WORD;
        return (data_[word] >> bit) & 1ULL;
    }

    void add(const BitTreePivotColumn &other)
    {
        size_t n = std::min(data_.size(), other.data_.size());
        for (size_t i = 0; i < n; ++i)
        {
            data_[i] ^= other.data_[i];
        }
        for (size_t i = n; i < other.data_.size() && i < data_.size(); ++i)
        {
            data_[i] ^= other.data_[i];
        }
        updatePivot();
    }

    void scalarMultiply(int scalar)
    {
        if (scalar == 0)
        {
            std::fill(data_.begin(), data_.end(), 0);
            pivot_ = -1;
        }
    }

    void fromSparse(const std::vector<int> &indices)
    {
        std::fill(data_.begin(), data_.end(), 0);
        for (int row : indices)
        {
            if (row >= 0 && row < max_rows_)
            {
                size_t word = static_cast<size_t>(row) / BITS_PER_WORD;
                int bit = row % BITS_PER_WORD;
                data_[word] |= (1ULL << bit);
            }
        }
        updatePivot();
    }

    std::vector<int> toSparse() const
    {
        std::vector<int> result;
        for (size_t w = 0; w < data_.size(); ++w)
        {
            uint64_t word = data_[w];
            while (word)
            {
                int bit = nerve::bits::ctz64(word);
                result.push_back(static_cast<int>(w * BITS_PER_WORD + static_cast<size_t>(bit)));
                word &= (word - 1);
            }
        }
        return result;
    }

    size_t memoryBytes() const { return data_.size() * sizeof(uint64_t); }

private:
    int max_rows_;
    size_t num_words_;
    std::vector<uint64_t> data_;
    int pivot_;

    void updatePivot()
    {
        pivot_ = -1;
        for (size_t w = data_.size(); w-- > 0;)
        {
            if (data_[w] != 0)
            {
                int bit = nerve::bits::fls64(data_[w]) - 1;
                pivot_ = static_cast<int>(w * BITS_PER_WORD + static_cast<size_t>(bit));
                return;
            }
        }
    }
};

} // namespace nerve::persistence

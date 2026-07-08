#include "nerve/persistence/core/roaring_bitmap.hpp"
#include "nerve/persistence/core/roaring_bitmap_internal.hpp"
#include "nerve/platform.hpp"

namespace nerve::persistence::roaring
{

constexpr int BITSET_BITS_PER_WORD = 64;
constexpr int BITSET_BITS_PER_WORD_MINUS_1 = 63;

constexpr double sparseThreshold() noexcept
{
    return 0.1;
}

HybridColumn::HybridColumn(int max_rows)
    : max_rows_(max_rows)
    , type_(Type::SPARSE)
{}

void HybridColumn::add(int index)
{
    if (type_ == Type::SPARSE)
    {
        sparse_data_.add(index);

        // Check if should convert to dense
        double density = 1.0 - sparse_data_.sparsity();
        if (density > sparseThreshold())
        {
            optimizeStorage();
        }
    }
    else
    {
        int word_idx = index / 64;
        int bit_idx = index % 64;
        if (word_idx < static_cast<int>(dense_data_.size()))
        {
            dense_data_[word_idx] |= (1ULL << bit_idx);
        }
    }
}

void HybridColumn::xorInPlace(const HybridColumn &other)
{
    if (type_ == Type::SPARSE && other.type_ == Type::SPARSE)
    {
        sparse_data_.xorInPlace(other.sparse_data_);
    }
    else if (type_ == Type::DENSE && other.type_ == Type::DENSE)
    {
        for (size_t i = 0; i < dense_data_.size() && i < other.dense_data_.size(); ++i)
        {
            dense_data_[i] ^= other.dense_data_[i];
        }
    }
    else
    {
        // Mixed types - convert to sparse and XOR
        if (type_ == Type::DENSE)
        {
            sparse_data_ = RoaringColumn::fromBitVector(dense_data_);
            type_ = Type::SPARSE;
            dense_data_.clear();
        }
        if (other.type_ == Type::DENSE)
        {
            RoaringColumn temp = RoaringColumn::fromBitVector(other.dense_data_);
            sparse_data_.xorInPlace(temp);
        }
        else
        {
            sparse_data_.xorInPlace(other.sparse_data_);
        }
    }
}

int HybridColumn::computePivot() const
{
    if (type_ == Type::SPARSE)
    {
        return sparse_data_.computePivot();
    }
    else
    {
        // Find in dense
        for (int i = static_cast<int>(dense_data_.size()) - 1; i >= 0; --i)
        {
            if (dense_data_[i] != 0)
            {
                int bit = nerve::bits::fls64(dense_data_[i]) - 1;
                return i * 64 + bit;
            }
        }
        return -1;
    }
}

// Optimize storage based on density
void HybridColumn::optimizeStorage()
{
    if (type_ == Type::SPARSE)
    {
        double density = 1.0 - sparse_data_.sparsity();
        if (density > sparseThreshold())
        {
            // Convert to dense
            int num_words = (max_rows_ + BITSET_BITS_PER_WORD_MINUS_1) / BITSET_BITS_PER_WORD;
            dense_data_ = sparse_data_.toBitVector(num_words);
            type_ = Type::DENSE;
            sparse_data_.clear();
        }
    }
}

bool HybridColumn::isEmpty() const
{
    if (type_ == Type::SPARSE)
    {
        return sparse_data_.isEmpty();
    }
    else
    {
        // Check if all words are zero
        for (uint64_t word : dense_data_)
        {
            if (word != 0)
                return false;
        }
        return true;
    }
}

double HybridColumn::sparsity() const
{
    if (type_ == Type::SPARSE)
    {
        return sparse_data_.sparsity();
    }
    else
    {
        // Count set bits in dense representation
        int total_bits = max_rows_;
        if (total_bits == 0)
            return 1.0;
        int set_bits = 0;
        for (uint64_t word : dense_data_)
        {
            set_bits += nerve::bits::popcount64(word);
        }
        return 1.0 - (static_cast<double>(set_bits) / total_bits);
    }
}

} // namespace nerve::persistence::roaring

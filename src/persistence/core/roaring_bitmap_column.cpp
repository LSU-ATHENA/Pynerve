#include "memory/safe_memory_pool.hpp"
#include "nerve/persistence/core/roaring_bitmap.hpp"
#include "nerve/persistence/core/roaring_bitmap_internal.hpp"
#include "nerve/platform.hpp"

#include <algorithm>
#include <cstring>
#include <functional>
#include <ranges>

namespace nerve::persistence::roaring
{

namespace
{

constexpr uint32_t ARRAY_INITIAL_CAPACITY = 4;

} // namespace

RoaringColumn::RoaringColumn() = default;
RoaringColumn::~RoaringColumn() = default;

RoaringColumn::RoaringColumn(RoaringColumn &&) noexcept = default;
RoaringColumn &RoaringColumn::operator=(RoaringColumn &&) noexcept = default;

// Factory method: from sparse indices
RoaringColumn RoaringColumn::fromSparseIndices(const std::vector<int> &indices, int max_row)
{
    RoaringColumn column;

    for (int idx : indices)
    {
        if (idx >= 0 && idx < max_row)
        {
            column.add(idx);
        }
    }

    return column;
}

// Factory method: from bit-vector
RoaringColumn RoaringColumn::fromBitVector(const std::vector<uint64_t> &words)
{
    RoaringColumn column;
    column.impl_ = std::make_unique<RoaringBitmapImpl>();

    // For each word, extract set bits
    for (size_t word_idx = 0; word_idx < words.size(); ++word_idx)
    {
        uint64_t word = words[word_idx];
        if (word == 0)
            continue;

        uint16_t chunk_id = static_cast<uint16_t>(word_idx / 16);
        uint16_t base_low_bits = static_cast<uint16_t>((word_idx % 16) * 64);

        Container *container = column.impl_->getOrCreateContainer(chunk_id, ARRAY_CONTAINER);

        // Extract all set bits
        while (word != 0)
        {
            uint64_t lsb = word & -word;
            uint64_t bit_idx = static_cast<uint64_t>(nerve::bits::ctz64(word));
            uint16_t low_bits = base_low_bits + bit_idx;

            // Add to container
            if (container->type == ARRAY_CONTAINER)
            {
                auto &arr = container->array;
                if (arr.count >= arr.capacity)
                {
                    uint32_t new_cap = arr.capacity ? arr.capacity * 2 : ARRAY_INITIAL_CAPACITY;
                    auto &pool = getRoaringMemoryPool();
                    uint16_t *new_values =
                        static_cast<uint16_t *>(pool.allocate(new_cap * sizeof(uint16_t)));
                    if (arr.values)
                    {
                        std::memcpy(new_values, arr.values, arr.count * sizeof(uint16_t));
                        pool.deallocate(arr.values, arr.capacity * sizeof(uint16_t));
                    }
                    arr.values = new_values;
                    arr.capacity = new_cap;
                }
                arr.values[arr.count++] = low_bits;
            }

            word ^= lsb;
        }
    }

    // Sort all array containers
    for (const auto &c : column.impl_->containers)
    {
        if (c->type == ARRAY_CONTAINER && c->array.count > 0)
        {
            auto &arr = c->array;
            std::ranges::sort(arr.values, arr.values + arr.count);
        }
    }

    return column;
}

void RoaringColumn::add(int index)
{
    if (index < 0)
        return;

    if (!impl_)
    {
        impl_ = std::make_unique<RoaringBitmapImpl>();
    }

    uint16_t chunk_id = static_cast<uint16_t>(index >> 16);
    uint16_t low_bits = static_cast<uint16_t>(index & 0xFFFF);

    Container *container = impl_->getOrCreateContainer(chunk_id, ARRAY_CONTAINER);

    if (container->type == ARRAY_CONTAINER)
    {
        auto &arr = container->array;

        // Check if already exists (binary search since array is sorted)
        if (arr.values && arr.count > 0)
        {
            if (std::binary_search(arr.values, arr.values + arr.count, low_bits))
            {
                return; // Already exists
            }
        }

        if (arr.count >= arr.capacity)
        {
            uint32_t new_cap = arr.capacity ? arr.capacity * 2 : ARRAY_INITIAL_CAPACITY;
            auto &pool = getRoaringMemoryPool();
            uint16_t *new_values =
                static_cast<uint16_t *>(pool.allocate(new_cap * sizeof(uint16_t)));
            if (arr.values)
            {
                std::memcpy(new_values, arr.values, arr.count * sizeof(uint16_t));
                pool.deallocate(arr.values, arr.capacity * sizeof(uint16_t));
            }
            arr.values = new_values;
            arr.capacity = new_cap;
        }

        // Insert maintaining sorted order
        arr.values[arr.count++] = low_bits;
        if (arr.count > 1)
        {
            std::ranges::sort(arr.values, arr.values + arr.count);
        }
    }

    invalidatePivot();
}

void RoaringColumn::remove(int index)
{
    if (index < 0 || !impl_)
        return;

    uint16_t chunk_id = static_cast<uint16_t>(index >> 16);
    uint16_t low_bits = static_cast<uint16_t>(index & 0xFFFF);

    Container *container = impl_->findContainer(chunk_id);
    if (!container || container->type != ARRAY_CONTAINER)
        return;

    auto &arr = container->array;
    if (!arr.values || arr.count == 0)
        return;

    // Find element (binary search)
    auto *it = std::lower_bound(arr.values, arr.values + arr.count, low_bits);
    if (it != arr.values + arr.count && *it == low_bits)
    {
        // Shift remaining elements
        size_t idx = std::distance(arr.values, it);
        for (size_t i = idx; i + 1 < arr.count; ++i)
        {
            arr.values[i] = arr.values[i + 1];
        }
        arr.count--;

        invalidatePivot();
    }
}

bool RoaringColumn::contains(int index) const
{
    if (index < 0 || !impl_)
        return false;

    uint16_t chunk_id = static_cast<uint16_t>(index >> 16);
    uint16_t low_bits = static_cast<uint16_t>(index & 0xFFFF);

    Container *container = impl_->findContainer(chunk_id);
    if (!container || container->type != ARRAY_CONTAINER)
        return false;

    const auto &arr = container->array;

    // Binary search (array is sorted)
    return std::binary_search(arr.values, arr.values + arr.count, low_bits);
}

// Compute pivot (highest set bit)
int RoaringColumn::computePivot() const
{
    if (pivot_valid_)
        return pivot_;

    if (!impl_ || impl_->containers.empty())
    {
        pivot_ = -1;
        pivot_valid_ = true;
        return -1;
    }

    // Find highest chunk
    int max_chunk = -1;
    for (const auto &c : impl_->containers)
    {
        if (c->type == ARRAY_CONTAINER && c->array.count > 0)
        {
            max_chunk = std::max(max_chunk, static_cast<int>(c->chunk_id));
        }
    }

    if (max_chunk < 0)
    {
        pivot_ = -1;
        pivot_valid_ = true;
        return -1;
    }

    // Find highest value in highest chunk
    Container *container = impl_->findContainer(static_cast<uint16_t>(max_chunk));
    if (container && container->type == ARRAY_CONTAINER && container->array.count > 0)
    {
        uint16_t high_val = container->array.values[container->array.count - 1];
        pivot_ = (max_chunk << 16) | high_val;
    }
    else
    {
        pivot_ = -1;
    }

    pivot_valid_ = true;
    return pivot_;
}

size_t RoaringColumn::cardinality() const
{
    if (!impl_)
        return 0;

    size_t count = 0;
    for (const auto &c : impl_->containers)
    {
        if (c->type == ARRAY_CONTAINER)
        {
            count += c->array.count;
        }
    }
    return count;
}

// Sparsity metric (1.0 = empty, 0.0 = full)
double RoaringColumn::sparsity() const
{
    int pivot_val = computePivot();
    if (pivot_val <= 0)
        return 1.0;

    size_t card = cardinality();
    return 1.0 - (static_cast<double>(card) / (pivot_val + 1));
}

size_t RoaringColumn::memoryUsage() const
{
    if (!impl_)
        return 0;

    size_t bytes = sizeof(RoaringBitmapImpl);
    for (const auto &c : impl_->containers)
    {
        bytes += sizeof(Container);
        if (c->type == ARRAY_CONTAINER && c->array.values)
        {
            bytes += c->array.capacity * sizeof(uint16_t);
        }
        else if (c->type == BITSET_CONTAINER && c->bitset.words)
        {
            bytes += BITSET_WORDS_COUNT * sizeof(uint64_t);
        }
    }
    return bytes;
}

// Invalidate cached pivot
void RoaringColumn::invalidatePivot()
{
    pivot_ = -1;
    pivot_valid_ = false;
}

bool RoaringColumn::isEmpty() const
{
    return cardinality() == 0;
}

void RoaringColumn::clear()
{
    if (impl_)
    {
        impl_->containers.clear();
    }
    invalidatePivot();
}

} // namespace nerve::persistence::roaring

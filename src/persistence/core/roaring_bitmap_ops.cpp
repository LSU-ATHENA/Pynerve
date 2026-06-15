#include "memory/safe_memory_pool.hpp"
#include "nerve/persistence/core/roaring_bitmap.hpp"
#include "nerve/persistence/core/roaring_bitmap_internal.hpp"

#include <algorithm>
#include <cstring>
#include <functional>
#include <ranges>
#include <vector>

namespace nerve::persistence::roaring
{

// XOR in-place (Z2 addition)
void RoaringColumn::xorInPlace(const RoaringColumn &other)
{
    if (!impl_ || !other.impl_)
        return;

    // For each container in other, XOR with ours
    for (const auto &other_container : other.impl_->containers)
    {
        if (other_container->type != ARRAY_CONTAINER)
            continue;

        uint16_t chunk_id = other_container->chunk_id;
        Container *our_container = impl_->findContainer(chunk_id);

        const auto &other_arr = other_container->array;

        if (!our_container)
        {
            // We don't have this chunk, copy it
            for (uint32_t i = 0; i < other_arr.count; ++i)
            {
                int idx = (chunk_id << 16) | other_arr.values[i];
                add(idx);
            }
        }
        else if (our_container->type == ARRAY_CONTAINER)
        {
            auto &our_arr = our_container->array;

            // XOR: symmetric difference of sorted arrays
            std::vector<uint16_t> result;
            result.reserve(our_arr.count + other_arr.count);

            uint32_t i = 0, j = 0;
            while (i < our_arr.count && j < other_arr.count)
            {
                if (our_arr.values[i] < other_arr.values[j])
                {
                    result.push_back(our_arr.values[i++]);
                }
                else if (our_arr.values[i] > other_arr.values[j])
                {
                    result.push_back(other_arr.values[j++]);
                }
                else
                {
                    // Equal - skip (XOR eliminates common elements)
                    i++;
                    j++;
                }
            }
            while (i < our_arr.count)
            {
                result.push_back(our_arr.values[i++]);
            }
            while (j < other_arr.count)
            {
                result.push_back(other_arr.values[j++]);
            }

            if (result.size() > our_arr.capacity)
            {
                auto &pool = getRoaringMemoryPool();
                pool.deallocate(our_arr.values, our_arr.capacity * sizeof(uint16_t));
                our_arr.capacity = static_cast<uint32_t>(static_cast<double>(result.size()) * 1.5);
                our_arr.values =
                    static_cast<uint16_t *>(pool.allocate(our_arr.capacity * sizeof(uint16_t)));
            }
            our_arr.count = static_cast<uint32_t>(result.size());
            std::memcpy(our_arr.values, result.data(), result.size() * sizeof(uint16_t));
        }
    }

    invalidatePivot();
}

// XOR copy
RoaringColumn RoaringColumn::xorCopy(const RoaringColumn &other) const
{
    // Create result by moving from a new default-constructed column
    // then XOR with both *this and other
    RoaringColumn result;
    // Copy data from *this via add operations
    if (impl_)
    {
        for (const auto &c : impl_->containers)
        {
            if (c->type == ARRAY_CONTAINER && c->array.values)
            {
                for (uint32_t i = 0; i < c->array.count; ++i)
                {
                    int idx = (c->chunk_id << 16) | c->array.values[i];
                    result.add(idx);
                }
            }
        }
    }
    result.xorInPlace(other);
    return result;
}

// Convert to sparse indices
std::vector<int> RoaringColumn::toSparseIndices() const
{
    std::vector<int> result;
    if (!impl_)
        return result;

    for (const auto &c : impl_->containers)
    {
        if (c->type == ARRAY_CONTAINER)
        {
            for (uint32_t i = 0; i < c->array.count; ++i)
            {
                int idx = (c->chunk_id << 16) | c->array.values[i];
                result.push_back(idx);
            }
        }
    }

    std::ranges::sort(result);
    return result;
}

// Convert to bit-vector
std::vector<uint64_t> RoaringColumn::toBitVector(int num_words) const
{
    std::vector<uint64_t> result(num_words, 0);
    if (!impl_)
        return result;

    for (const auto &c : impl_->containers)
    {
        if (c->type == ARRAY_CONTAINER)
        {
            for (uint32_t i = 0; i < c->array.count; ++i)
            {
                int idx = (c->chunk_id << 16) | c->array.values[i];
                int word_idx = idx / 64;
                int bit_idx = idx % 64;
                if (word_idx < num_words)
                {
                    result[word_idx] |= (1ULL << bit_idx);
                }
            }
        }
    }

    return result;
}

// Iterate over all set bits
void RoaringColumn::forEach(std::function<void(int)> callback) const
{
    if (!impl_)
        return;

    for (const auto &c : impl_->containers)
    {
        if (c->type == ARRAY_CONTAINER)
        {
            for (uint32_t i = 0; i < c->array.count; ++i)
            {
                int idx = (c->chunk_id << 16) | c->array.values[i];
                callback(idx);
            }
        }
    }
}

// Comparison operators
bool RoaringColumn::operator==(const RoaringColumn &other) const
{
    return toSparseIndices() == other.toSparseIndices();
}

bool RoaringColumn::operator!=(const RoaringColumn &other) const
{
    return !(*this == other);
}

} // namespace nerve::persistence::roaring

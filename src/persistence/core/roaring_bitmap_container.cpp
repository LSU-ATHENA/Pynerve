#include "memory/safe_memory_pool.hpp"
#include "nerve/persistence/core/roaring_bitmap.hpp"
#include "nerve/persistence/core/roaring_bitmap_internal.hpp"

#include <algorithm>
#include <cstring>
#include <functional>
#include <ranges>

namespace nerve::persistence::roaring
{

namespace
{

constexpr uint32_t ARRAY_INITIAL_CAPACITY = 4;
constexpr uint32_t RUN_INITIAL_CAPACITY = 4;

nerve::memory::RawArrayPool &roaringPool()
{
    static nerve::memory::RawArrayPool pool(static_cast<size_t>(16) * 1024 * 1024);
    return pool;
}

} // namespace

nerve::memory::RawArrayPool &getRoaringMemoryPool()

{
    return roaringPool();
}

// Container struct implementation
Container::Container(uint16_t t, uint16_t chunk)
    : type(t)
    , chunk_id(chunk)
{
    if (type == ARRAY_CONTAINER)
    {
        array.values = nullptr;
        array.count = 0;
        array.capacity = 0;
    }
    else if (type == BITSET_CONTAINER)
    {
        array.values = nullptr;
    }
    else if (type == RUN_CONTAINER)
    {
        run.runs = nullptr;
        run.count = 0;
    }
}

Container::~Container()
{
    if (type == ARRAY_CONTAINER && array.values)
    {
        roaringPool().deallocate(array.values, array.capacity * sizeof(uint16_t));
    }
    else if (type == BITSET_CONTAINER && bitset.words)
    {
        roaringPool().deallocate(bitset.words, BITSET_WORDS_COUNT * sizeof(uint64_t));
    }
    else if (type == RUN_CONTAINER && run.runs)
    {
        roaringPool().deallocate(run.runs, static_cast<size_t>(run.count) * 2 * sizeof(uint16_t));
    }
}

// RoaringBitmapImpl implementation
Container *RoaringBitmapImpl::findContainer(uint16_t chunk_id) const
{
    for (const auto &c : containers)
    {
        if (c->chunk_id == chunk_id)
        {
            return c.get();
        }
    }
    return nullptr;
}

Container *RoaringBitmapImpl::getOrCreateContainer(uint16_t chunk_id, uint16_t preferred_type)
{
    Container *existing = findContainer(chunk_id);
    if (existing)
    {
        return existing;
    }

    auto new_container = std::make_unique<Container>(preferred_type, chunk_id);
    Container *ptr = new_container.get();

    if (preferred_type == ARRAY_CONTAINER)
    {
        new_container->array.capacity = ARRAY_INITIAL_CAPACITY;
        new_container->array.values = static_cast<uint16_t *>(
            roaringPool().allocate(ARRAY_INITIAL_CAPACITY * sizeof(uint16_t)));
    }
    else if (preferred_type == BITSET_CONTAINER)
    {
        new_container->bitset.words =
            static_cast<uint64_t *>(roaringPool().allocate(BITSET_WORDS_COUNT * sizeof(uint64_t)));
        std::memset(new_container->bitset.words, 0, BITSET_WORDS_COUNT * sizeof(uint64_t));
    }
    else if (preferred_type == RUN_CONTAINER)
    {
        new_container->run.runs = static_cast<uint16_t *>(
            roaringPool().allocate(RUN_INITIAL_CAPACITY * sizeof(uint16_t)));
    }

    containers.push_back(std::move(new_container));
    return ptr;
}

// Helper for bit manipulation operations
void setBitInContainer(Container *container, uint16_t low_bits)
{
    if (!container || container->type != BITSET_CONTAINER)
        return;

    uint64_t word_idx = low_bits >> 6;  // / 64
    uint64_t bit_idx = low_bits & 0x3F; // % 64
    container->bitset.words[word_idx] |= (1ULL << bit_idx);
}

void clearBitInContainer(Container *container, uint16_t low_bits)
{
    if (!container || container->type != BITSET_CONTAINER)
        return;

    uint64_t word_idx = low_bits >> 6;
    uint64_t bit_idx = low_bits & 0x3F;
    container->bitset.words[word_idx] &= ~(1ULL << bit_idx);
}

bool testBitInContainer(const Container *container, uint16_t low_bits)
{
    if (!container || container->type != BITSET_CONTAINER)
        return false;

    uint64_t word_idx = low_bits >> 6;
    uint64_t bit_idx = low_bits & 0x3F;
    return (container->bitset.words[word_idx] >> bit_idx) & 1;
}

} // namespace nerve::persistence::roaring

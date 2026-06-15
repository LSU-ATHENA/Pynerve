#pragma once

#include "memory/safe_memory_pool.hpp"
#include "nerve/core_types.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace nerve::persistence::roaring
{

using RawArrayPool = nerve::memory::RawArrayPool;

constexpr uint16_t ARRAY_CONTAINER = 1;
constexpr uint16_t BITSET_CONTAINER = 2;
constexpr uint16_t RUN_CONTAINER = 3;

// Array container - sorted array of 16-bit values
struct ArrayContainer
{
    uint16_t *values = nullptr;
    uint32_t count = 0;
    uint32_t capacity = 0;
};

// Bitset container - 65536 bits = 1024 64-bit words
struct BitsetContainer
{
    uint64_t *words = nullptr;
};

// Run container - RLE encoding
struct RunContainer
{
    uint16_t *runs = nullptr; // Pairs: (start, length)
    uint16_t count = 0;
};

// Generic container that can be any type
struct Container
{
    uint16_t type = ARRAY_CONTAINER;
    uint16_t chunk_id = 0;
    union
    {
        ArrayContainer array;
        BitsetContainer bitset;
        RunContainer run;
    };

    Container(uint16_t t, uint16_t chunk);
    ~Container();
};

// Implementation structure - collection of containers
struct RoaringBitmapImpl
{
    std::vector<std::unique_ptr<Container>> containers;

    Container *findContainer(uint16_t chunk_id) const;
    Container *getOrCreateContainer(uint16_t chunk_id, uint16_t preferred_type);
};

constexpr uint32_t BITSET_WORDS_COUNT = 1024;

RawArrayPool &getRoaringMemoryPool();

} // namespace nerve::persistence::roaring

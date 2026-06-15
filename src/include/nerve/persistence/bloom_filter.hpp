#pragma once

// Compatibility bloom-filter facade for simplex vertex keys.

#include "nerve/persistence/approximate/bloom_filter.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace nerve::persistence
{

// Simplex-oriented adapter around the integer-key bloom implementation.
class BloomFilter
{
public:
    explicit BloomFilter(std::size_t expected_elements = 10000, double false_positive_rate = 0.01)
        : bloom_(expected_elements, false_positive_rate)
    {}

    void add(const std::vector<int> &simplex_vertices)
    {
        bloom_.add(hashSimplex(simplex_vertices));
    }

    [[nodiscard]] bool possiblyContains(const std::vector<int> &simplex_vertices) const
    {
        return bloom_.mightContain(hashSimplex(simplex_vertices));
    }

    void clear() { bloom_.clear(); }

private:
    static int hashSimplex(const std::vector<int> &simplex_vertices)
    {
        std::vector<int> canonical = simplex_vertices;
        std::sort(canonical.begin(), canonical.end());

        std::uint64_t hash = 1469598103934665603ULL;
        for (int v : canonical)
        {
            hash ^= static_cast<std::uint64_t>(static_cast<std::uint32_t>(v));
            hash *= 1099511628211ULL;
        }
        return static_cast<int>(hash & 0x7fffffffULL);
    }

    bloom::BloomFilter bloom_;
};

} // namespace nerve::persistence

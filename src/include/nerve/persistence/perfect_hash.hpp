#pragma once

// Compatibility perfect-hash facade for simplex vertex keys.

#include "nerve/persistence/approximate/perfect_hash.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace nerve::persistence
{

class PerfectHashFunction
{
public:
    template <typename SimplexRange>
    void build(const SimplexRange &simplices)
    {
        key_to_index_.clear();
        int index = 0;
        for (const auto &simplex : simplices)
        {
            std::vector<int> verts(simplex.begin(), simplex.end());
            key_to_index_[hash(verts)] = index++;
        }
    }

    [[nodiscard]] std::size_t hash(const std::vector<int> &vertices) const
    {
        std::vector<int> canonical = vertices;
        std::sort(canonical.begin(), canonical.end());

        std::uint64_t h = 1469598103934665603ULL;
        for (int v : canonical)
        {
            h ^= static_cast<std::uint64_t>(static_cast<std::uint32_t>(v));
            h *= 1099511628211ULL;
        }
        return static_cast<std::size_t>(h);
    }

    [[nodiscard]] int lookup(const std::vector<int> &vertices) const
    {
        const std::size_t key = hash(vertices);
        auto it = key_to_index_.find(key);
        if (it == key_to_index_.end())
        {
            return -1;
        }
        return it->second;
    }

private:
    std::unordered_map<std::size_t, int> key_to_index_;
};

} // namespace nerve::persistence

#pragma once

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <vector>

namespace nerve::persistence::reduction
{

// Union-Find (Disjoint Set Union) for connectivity tracking.
class UnionFind
{
public:
    explicit UnionFind(std::size_t n)
        : parent_(n)
        , rank_(n, 0)
    {
        std::iota(parent_.begin(), parent_.end(), 0);
    }

    // Find representative of set containing x.
    [[nodiscard]] std::size_t find(std::size_t x)
    {
        if (parent_[x] != x)
        {
            parent_[x] = find(parent_[x]);
        }
        return parent_[x];
    }

    // Union two sets.
    void unite(std::size_t x, std::size_t y)
    {
        x = find(x);
        y = find(y);
        if (x == y)
        {
            return;
        }

        if (rank_[x] < rank_[y])
        {
            std::swap(x, y);
        }
        parent_[y] = x;
        if (rank_[x] == rank_[y])
        {
            ++rank_[x];
        }
    }

    // Check if two elements are in the same set.
    [[nodiscard]] bool connected(std::size_t x, std::size_t y) { return find(x) == find(y); }

    // Count number of disjoint sets.
    [[nodiscard]] std::size_t countSets() const
    {
        std::size_t count = 0;
        for (std::size_t i = 0; i < parent_.size(); ++i)
        {
            if (parent_[i] == i)
            {
                ++count;
            }
        }
        return count;
    }

private:
    std::vector<std::size_t> parent_;
    std::vector<std::size_t> rank_;
};

} // namespace nerve::persistence::reduction

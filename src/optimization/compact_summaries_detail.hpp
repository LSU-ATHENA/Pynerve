#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace nerve::optimization::compact_summary_detail
{

struct UnionFind
{
    std::vector<int> parent;
    std::vector<int> rank;

    explicit UnionFind(int n)
        : parent(static_cast<size_t>(n))
        , rank(static_cast<size_t>(n), 0)
    {
        std::iota(parent.begin(), parent.end(), 0);
    }

    int find(int x)
    {
        if (parent[static_cast<size_t>(x)] != x)
        {
            parent[static_cast<size_t>(x)] = find(parent[static_cast<size_t>(x)]);
        }
        return parent[static_cast<size_t>(x)];
    }

    void unite(int x, int y)
    {
        x = find(x);
        y = find(y);
        if (x == y)
            return;
        if (rank[static_cast<size_t>(x)] < rank[static_cast<size_t>(y)])
        {
            std::swap(x, y);
        }
        parent[static_cast<size_t>(y)] = x;
        if (rank[static_cast<size_t>(x)] == rank[static_cast<size_t>(y)])
        {
            rank[static_cast<size_t>(x)]++;
        }
    }
};

inline double pointDistanceSquared(const std::vector<float> &a, const std::vector<float> &b)
{
    double sum = 0.0;
    for (size_t d = 0; d < a.size(); ++d)
    {
        const double delta = static_cast<double>(a[d]) - static_cast<double>(b[d]);
        sum += delta * delta;
    }
    return sum;
}

inline size_t checkedPairCount(size_t num_points)
{
    if (num_points < 2)
        return 0;
    if (num_points > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        throw std::overflow_error("compact summary point count exceeds supported index range");
    }
    return (num_points * (num_points - 1)) / 2;
}

inline size_t saturatingMultiply(size_t lhs, size_t rhs)
{
    if (rhs != 0 && lhs > std::numeric_limits<size_t>::max() / rhs)
    {
        return std::numeric_limits<size_t>::max();
    }
    return lhs * rhs;
}

inline size_t saturatingAdd(size_t lhs, size_t rhs)
{
    if (lhs > std::numeric_limits<size_t>::max() - rhs)
    {
        return std::numeric_limits<size_t>::max();
    }
    return lhs + rhs;
}

inline size_t alignUp(size_t value, size_t alignment)
{
    const size_t remainder = value % alignment;
    return remainder == 0 ? value : value + (alignment - remainder);
}

inline void validatePointCloud(const std::vector<std::vector<float>> &points)
{
    checkedPairCount(points.size());
    if (points.empty())
        return;
    const size_t dimension = points.front().size();
    if (dimension == 0)
    {
        throw std::invalid_argument("compact summaries require non-empty point coordinates");
    }
    for (const auto &point : points)
    {
        if (point.size() != dimension)
        {
            throw std::invalid_argument("compact summaries require consistent point dimensions");
        }
        if (!std::all_of(point.begin(), point.end(),
                         [](float value) { return std::isfinite(value); }))
        {
            throw std::invalid_argument("compact summaries require finite point coordinates");
        }
    }
}

} // namespace nerve::optimization::compact_summary_detail

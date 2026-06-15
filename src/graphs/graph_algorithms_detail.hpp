#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>

namespace nerve::graphs::gpu::detail
{

inline constexpr int kInfDistance = std::numeric_limits<int>::max();

inline std::size_t checkedSquareCount(int n, const char *context)
{
    if (n < 0)
    {
        throw std::invalid_argument(context);
    }
    const std::size_t count = static_cast<std::size_t>(n);
    if (count != 0 && count > std::numeric_limits<std::size_t>::max() / count)
    {
        throw std::length_error(context);
    }
    return count * count;
}

inline int checkedIntSize(std::size_t value, const char *context)
{
    if (value > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::length_error(context);
    }
    return static_cast<int>(value);
}

inline void validateVertex(int vertex, int num_vertices, const char *name)
{
    if (vertex < 0 || vertex >= num_vertices)
    {
        throw std::out_of_range(std::string(name) + " is outside the graph vertex range");
    }
}

inline void validateCsr(int num_vertices, int edge_capacity, const std::vector<int> &row_ptr,
                        const std::vector<int> &col_idx, const std::vector<float> &weights)
{
    if (num_vertices < 0 || edge_capacity < 0)
    {
        throw std::invalid_argument("graph dimensions must be non-negative");
    }
    if (row_ptr.size() != static_cast<std::size_t>(num_vertices) + 1)
    {
        throw std::invalid_argument("CSR row pointer size must be num_vertices + 1");
    }
    if (!row_ptr.empty() && row_ptr.front() != 0)
    {
        throw std::invalid_argument("CSR row pointer must start at zero");
    }
    if (col_idx.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::length_error("CSR column count exceeds int range");
    }
    if (!row_ptr.empty() && row_ptr.back() != static_cast<int>(col_idx.size()))
    {
        throw std::invalid_argument("CSR row pointer end must match column count");
    }
    if (col_idx.size() > static_cast<std::size_t>(edge_capacity))
    {
        throw std::invalid_argument("CSR column count exceeds the graph edge capacity");
    }
    if (!weights.empty() && weights.size() != col_idx.size())
    {
        throw std::invalid_argument("CSR weights must be empty or match column count");
    }
    for (std::size_t i = 1; i < row_ptr.size(); ++i)
    {
        if (row_ptr[i] < row_ptr[i - 1])
        {
            throw std::invalid_argument("CSR row pointer must be nondecreasing");
        }
    }
    for (int vertex : col_idx)
    {
        validateVertex(vertex, num_vertices, "CSR neighbor");
    }
    for (float weight : weights)
    {
        if (!std::isfinite(weight))
        {
            throw std::invalid_argument("CSR weights must be finite");
        }
    }
}

inline std::vector<int> cpuBfs(const std::vector<int> &row_ptr, const std::vector<int> &col_idx,
                               int source)
{
    if (row_ptr.empty())
    {
        return {};
    }
    const int n = checkedIntSize(row_ptr.size() - 1, "CSR row count exceeds int range");
    std::vector<int> distances(n, kInfDistance);
    std::queue<int> queue;
    distances[source] = 0;
    queue.push(source);

    while (!queue.empty())
    {
        const int u = queue.front();
        queue.pop();
        for (int edge = row_ptr[u]; edge < row_ptr[u + 1]; ++edge)
        {
            const int v = col_idx[edge];
            if (distances[v] == kInfDistance)
            {
                distances[v] = distances[u] + 1;
                queue.push(v);
            }
        }
    }
    return distances;
}

inline std::vector<int> cpuConnectedComponents(const std::vector<int> &row_ptr,
                                               const std::vector<int> &col_idx)
{
    if (row_ptr.empty())
    {
        return {};
    }
    const int n = checkedIntSize(row_ptr.size() - 1, "CSR row count exceeds int range");
    std::vector<int> parent(n);
    std::vector<int> rank(n, 0);
    std::iota(parent.begin(), parent.end(), 0);

    auto find_root = [&](int x) {
        while (parent[x] != x)
        {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };
    auto unite = [&](int a, int b) {
        a = find_root(a);
        b = find_root(b);
        if (a == b)
        {
            return;
        }
        if (rank[a] < rank[b])
        {
            std::swap(a, b);
        }
        parent[b] = a;
        if (rank[a] == rank[b])
        {
            ++rank[a];
        }
    };

    for (int u = 0; u < n; ++u)
    {
        for (int edge = row_ptr[u]; edge < row_ptr[u + 1]; ++edge)
        {
            unite(u, col_idx[edge]);
        }
    }
    for (int i = 0; i < n; ++i)
    {
        parent[i] = find_root(i);
    }
    return parent;
}

inline std::vector<float> buildDistanceMatrix(int n, const std::vector<int> &row_ptr,
                                              const std::vector<int> &col_idx,
                                              const std::vector<float> &weights)
{
    const std::size_t matrix_size = checkedSquareCount(n, "graph distance matrix size overflows");
    const std::size_t width = static_cast<std::size_t>(n);
    std::vector<float> dist(matrix_size, std::numeric_limits<float>::infinity());
    for (int i = 0; i < n; ++i)
    {
        dist[static_cast<std::size_t>(i) * width + static_cast<std::size_t>(i)] = 0.0f;
        for (int edge = row_ptr[i]; edge < row_ptr[i + 1]; ++edge)
        {
            const float weight = weights.empty() ? 1.0f : weights[edge];
            const std::size_t offset =
                static_cast<std::size_t>(i) * width + static_cast<std::size_t>(col_idx[edge]);
            dist[offset] = std::min(dist[offset], weight);
        }
    }
    return dist;
}

} // namespace nerve::graphs::gpu::detail

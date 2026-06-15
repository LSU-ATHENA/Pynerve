#include "nerve/persistence/core/per_dimension_exact.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <numeric>
#include <ranges>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace nerve::persistence::perdim
{

namespace
{

constexpr uint64_t LOWER_32_BIT_MASK = 0xFFFFFFFF;

} // namespace

H0Result computeH0UnionFind(const std::vector<std::vector<int>> &simplices,
                            const std::vector<double> &filtration_values)
{
    H0Result result;
    auto start = std::chrono::high_resolution_clock::now();
    if (simplices.size() != filtration_values.size())
    {
        throw std::invalid_argument("simplices and filtration values must have equal size");
    }

    // Extract edges (1-simplices) from the simplices list
    std::vector<std::pair<int, int>> edges;
    std::vector<double> edge_weights;

    // Find unique edges and their minimum filtration values
    std::unordered_map<uint64_t, double> edge_filtration;
    std::unordered_map<int, double> vertex_filtration;

    for (size_t i = 0; i < simplices.size(); ++i)
    {
        const auto &simplex = simplices[i];
        for (int vertex : simplex)
        {
            auto [it, inserted] = vertex_filtration.emplace(vertex, filtration_values[i]);
            if (!inserted)
            {
                it->second = std::min(it->second, filtration_values[i]);
            }
        }
        if (simplex.size() == 2)
        {
            int u = simplex[0];
            int v = simplex[1];
            uint64_t key = (static_cast<uint64_t>(std::min(u, v)) << 32) |
                           static_cast<uint64_t>(std::max(u, v));
            double filt = filtration_values[i];

            auto it = edge_filtration.find(key);
            if (it == edge_filtration.end() || filt < it->second)
            {
                edge_filtration[key] = filt;
            }
        }
    }

    // Convert to vectors
    for (const auto &[key, filt] : edge_filtration)
    {
        int u = static_cast<int>(key >> 32);
        int v = static_cast<int>(key & LOWER_32_BIT_MASK);
        edges.push_back({u, v});
        edge_weights.push_back(filt);
    }

    // Sort edges by filtration value
    std::vector<size_t> sorted_idx(edges.size());
    std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
    std::ranges::sort(sorted_idx, {}, [&edge_weights](size_t idx) { return edge_weights[idx]; });

    std::vector<std::pair<int, int>> sorted_edges;
    std::vector<double> sorted_weights;
    for (size_t idx : sorted_idx)
    {
        sorted_edges.push_back(edges[idx]);
        sorted_weights.push_back(edge_weights[idx]);
    }

    std::vector<int> vertices;
    vertices.reserve(vertex_filtration.size());
    for (const auto &[vertex, _] : vertex_filtration)
    {
        vertices.push_back(vertex);
    }
    std::ranges::sort(vertices);

    std::unordered_map<int, int> dense_index;
    dense_index.reserve(vertices.size());
    for (size_t i = 0; i < vertices.size(); ++i)
    {
        dense_index[vertices[i]] = static_cast<int>(i);
    }

    // Run Union-Find on sorted edges
    const int num_vertices = static_cast<int>(vertices.size());
    std::vector<int> parent(num_vertices);
    std::iota(parent.begin(), parent.end(), 0);

    auto find = [&parent](int x) {
        while (parent[x] != x)
        {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };

    auto unite = [&parent, &find](int x, int y) {
        x = find(x);
        y = find(y);
        if (x != y)
        {
            parent[x] = y;
        }
    };

    // Track component creation times
    std::vector<double> birth_time(num_vertices, 0.0);
    for (size_t i = 0; i < vertices.size(); ++i)
    {
        birth_time[i] = vertex_filtration[vertices[i]];
    }

    for (size_t i = 0; i < sorted_edges.size(); ++i)
    {
        const auto u_it = dense_index.find(sorted_edges[i].first);
        const auto v_it = dense_index.find(sorted_edges[i].second);
        if (u_it == dense_index.end() || v_it == dense_index.end())
        {
            continue;
        }
        int u = u_it->second;
        int v = v_it->second;
        double filt = sorted_weights[i];

        int root_u = find(u);
        int root_v = find(v);

        if (root_u != root_v)
        {
            if (birth_time[root_u] > birth_time[root_v] ||
                (birth_time[root_u] == birth_time[root_v] && root_u > root_v))
            {
                result.pairs.push_back({birth_time[root_u], filt, 0});
                unite(root_u, root_v);
            }
            else
            {
                result.pairs.push_back({birth_time[root_v], filt, 0});
                unite(root_v, root_u);
            }
        }
    }

    // Count remaining components (essential classes)
    const double infinity = std::numeric_limits<double>::infinity();
    result.essential_count = 0;
    std::unordered_set<int> roots;
    for (int i = 0; i < num_vertices; ++i)
    {
        const int root = find(i);
        if (roots.insert(root).second)
        {
            result.essential_count++;
            result.pairs.push_back({birth_time[root], infinity, 0});
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.num_pairs = static_cast<int>(result.pairs.size());

    return result;
}

} // namespace nerve::persistence::perdim

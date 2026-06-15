// Union-Find based fast persistence computation for D0 and D(d-1)
// Based on Discrete Morse Sandwich approach (Tierny et al. 2022, 2025)
// D0 (connected components) and D(d-1) (voids) can be computed in O(alpha(n))
// instead of matrix reduction

#include "nerve/algebra/complex.hpp"
#include "nerve/algebra/simplex.hpp"
#include "nerve/persistence/reduction/reduction_union_find_ops.hpp"
#include "nerve/persistence/utils/exact_engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <ranges>
#include <tuple>
#include <unordered_map>

namespace nerve::persistence
{

namespace
{

// Union-Find with path compression and union by rank
class UnionFind
{
public:
    explicit UnionFind(int n)
        : parent(n)
        , rank(n, 0)
    {
        std::iota(parent.begin(), parent.end(), 0);
    }

    int find(int x)
    {
        if (parent[x] != x)
        {
            parent[x] = find(parent[x]); // Path compression
        }
        return parent[x];
    }

    void unite(int x, int y)
    {
        x = find(x);
        y = find(y);
        if (x == y)
            return;

        // Union by rank
        if (rank[x] < rank[y])
        {
            std::swap(x, y);
        }
        parent[y] = x;
        if (rank[x] == rank[y])
        {
            ++rank[x];
        }
    }

    bool connected(int x, int y) { return find(x) == find(y); }

private:
    std::vector<int> parent;
    std::vector<int> rank;
};

using TopDimFiltrationMap = std::unordered_map<algebra::Simplex, double, algebra::Simplex::Hash>;

void insertSimplexClosure(const algebra::Simplex &simplex, double filtration,
                          TopDimFiltrationMap *filtrations)
{
    if (filtrations == nullptr)
    {
        return;
    }
    const auto [it, inserted] = filtrations->insert({simplex, filtration});
    if (!inserted && filtration >= it->second)
    {
        return;
    }
    if (!inserted)
    {
        it->second = filtration;
    }
    if (simplex.dimension() == 0)
    {
        return;
    }
    for (const auto &face : simplex.faces(core::DeterminismContract{}))
    {
        insertSimplexClosure(face, filtration, filtrations);
    }
}

} // namespace

// Fast D0 persistence using Union-Find (connected components)
// This is O(n alpha(n)) instead of O(n^3) for matrix reduction
std::vector<Pair> computeD0PersistenceUnionFind(
    const std::vector<double> &points, size_t point_dim, size_t num_points,
    const std::vector<std::pair<int, int>> &edges, // (from, to) sorted by weight
    const std::vector<double> &edge_weights)
{
    if (num_points == 0)
    {
        return {};
    }
    if (point_dim == 0 || points.size() < num_points * point_dim)
    {
        return {};
    }
    for (size_t i = 0; i < num_points * point_dim; ++i)
    {
        if (!std::isfinite(points[i]))
        {
            return {};
        }
    }

    std::vector<Pair> persistence_pairs;
    persistence_pairs.reserve(num_points);

    // Union-Find structure
    UnionFind uf(static_cast<int>(num_points));

    // Track birth times of components
    std::unordered_map<int, double> component_birth;
    for (size_t i = 0; i < num_points; ++i)
    {
        component_birth[static_cast<int>(i)] = 0.0; // All vertices born at 0
    }

    const std::size_t edge_count = std::min(edges.size(), edge_weights.size());
    std::vector<size_t> edge_order(edge_count);
    std::iota(edge_order.begin(), edge_order.end(), 0);
    std::ranges::sort(edge_order, {}, [&edge_weights](size_t idx) { return edge_weights[idx]; });

    for (size_t idx : edge_order)
    {
        if (!std::isfinite(edge_weights[idx]))
        {
            continue;
        }
        int u = edges[idx].first;
        int v = edges[idx].second;
        if (u < 0 || v < 0 || static_cast<size_t>(u) >= num_points ||
            static_cast<size_t>(v) >= num_points)
        {
            continue;
        }
        double weight = edge_weights[idx];

        int comp_u = uf.find(u);
        int comp_v = uf.find(v);

        if (comp_u != comp_v)
        {
            // Two components merge - the younger one dies
            double birth_u = component_birth[comp_u];
            double birth_v = component_birth[comp_v];

            // Elder rule: younger component dies; break ties deterministically by root id.
            const bool u_survives = (birth_u < birth_v) || (birth_u == birth_v && comp_u < comp_v);
            const double dying_birth = u_survives ? birth_v : birth_u;
            persistence_pairs.push_back({dying_birth, weight, 0});

            // Union the components
            uf.unite(u, v);
            int new_comp = uf.find(u);

            // The new component has the earlier birth time
            component_birth[new_comp] = std::min(birth_u, birth_v);

            // Clean up old component entries
            if (new_comp != comp_u)
                component_birth.erase(comp_u);
            if (new_comp != comp_v)
                component_birth.erase(comp_v);
        }
    }

    // Add infinite persistence for remaining components
    for (const auto &[comp, birth] : component_birth)
    {
        persistence_pairs.push_back({birth, std::numeric_limits<double>::infinity(), 0});
    }

    std::ranges::sort(persistence_pairs, [](const Pair &lhs, const Pair &rhs) {
        return std::tie(lhs.birth, lhs.death) < std::tie(rhs.birth, rhs.death);
    });

    return persistence_pairs;
}

// Fast computation of D(d-1) persistence (voids in d-dimensional space)
// This processes stable manifolds of (d-1)-saddles
std::vector<Pair> computeTopDimensionalPersistence(
    const std::vector<double> &points, size_t point_dim, size_t num_points,
    const std::vector<std::vector<int>> &simplices, // d-dimensional simplices
    const std::vector<double> &simplex_weights, Dimension dim)
{
    std::vector<Pair> persistence_pairs;
    if (simplices.empty())
    {
        return persistence_pairs;
    }
    if (dim < 0)
    {
        return persistence_pairs;
    }
    if (num_points == 0 && point_dim > 0)
    {
        num_points = points.size() / point_dim;
    }

    TopDimFiltrationMap filtrations;
    filtrations.reserve(simplices.size() * 4);
    for (size_t i = 0; i < simplices.size(); ++i)
    {
        const std::vector<int> &simplex_vertices = simplices[i];
        if (simplex_vertices.empty())
        {
            continue;
        }
        const double filtration = i < simplex_weights.size()
                                      ? simplex_weights[i]
                                      : static_cast<double>(simplex_vertices.size() - 1);
        if (!std::isfinite(filtration))
        {
            continue;
        }
        std::vector<Index> vertices;
        vertices.reserve(simplex_vertices.size());
        bool valid_simplex = true;
        for (const int vertex : simplex_vertices)
        {
            if (vertex < 0 || (num_points != 0 && static_cast<size_t>(vertex) >= num_points))
            {
                valid_simplex = false;
                break;
            }
            vertices.push_back(static_cast<Index>(vertex));
        }
        if (!valid_simplex || vertices.empty())
        {
            continue;
        }
        insertSimplexClosure(algebra::Simplex(vertices), filtration, &filtrations);
    }

    algebra::SimplicialComplex complex;
    for (const auto &[simplex, filtration] : filtrations)
    {
        complex.addSimplexWithFiltration(simplex, filtration);
    }

    if (complex.size() == 0)
    {
        return persistence_pairs;
    }

    const auto exact =
        computeExactPersistenceZ2(complex, static_cast<Size>(std::max<Dimension>(0, dim)));
    for (const auto &pair : exact.pairs)
    {
        if (pair.dimension == dim)
        {
            persistence_pairs.push_back(pair);
        }
    }
    std::ranges::sort(persistence_pairs, [](const Pair &lhs, const Pair &rhs) {
        return std::tie(lhs.birth, lhs.death, lhs.birth_index, lhs.death_index) <
               std::tie(rhs.birth, rhs.death, rhs.birth_index, rhs.death_index);
    });

    return persistence_pairs;
}

// Combined fast computation using Union-Find for D0 and D(d-1)
// This is the "sandwich" approach - fast ends, matrix reduction for middle
FastPersistenceResult computeFastPersistenceSandwich(const std::vector<double> &points,
                                                     size_t point_dim, size_t num_points,
                                                     const std::vector<std::pair<int, int>> &edges,
                                                     const std::vector<double> &edge_weights,
                                                     Dimension max_dim)
{
    FastPersistenceResult result;

    // Always compute D0 with Union-Find (extremely fast)
    auto start_d0 = std::chrono::high_resolution_clock::now();
    auto d0_pairs =
        computeD0PersistenceUnionFind(points, point_dim, num_points, edges, edge_weights);
    auto end_d0 = std::chrono::high_resolution_clock::now();

    result.d0_pairs = std::move(d0_pairs);
    result.d0_time_ms = std::chrono::duration<double, std::milli>(end_d0 - start_d0).count();

    if (max_dim >= 1 && num_points > 0)
    {
        auto start_cycles = std::chrono::high_resolution_clock::now();
        const std::size_t edge_count = std::min(edges.size(), edge_weights.size());
        std::vector<size_t> edge_order(edge_count);
        std::iota(edge_order.begin(), edge_order.end(), 0);
        std::ranges::sort(edge_order, {},
                          [&edge_weights](size_t idx) { return edge_weights[idx]; });

        UnionFind uf(static_cast<int>(num_points));
        std::vector<Pair> cycle_pairs;
        cycle_pairs.reserve(edge_count);
        for (const size_t idx : edge_order)
        {
            if (!std::isfinite(edge_weights[idx]))
            {
                continue;
            }
            const int u = edges[idx].first;
            const int v = edges[idx].second;
            if (u < 0 || v < 0 || static_cast<size_t>(u) >= num_points ||
                static_cast<size_t>(v) >= num_points)
            {
                continue;
            }
            if (uf.connected(u, v))
            {
                cycle_pairs.push_back(
                    {edge_weights[idx], std::numeric_limits<double>::infinity(), 1});
                continue;
            }
            uf.unite(u, v);
        }

        // For graph filtrations in 2D, H1 is top-dimensional.
        if (point_dim == 2 && max_dim >= 1)
        {
            result.top_dim_pairs = std::move(cycle_pairs);
        }
        else
        {
            result.middle_pairs = std::move(cycle_pairs);
        }
        auto end_cycles = std::chrono::high_resolution_clock::now();
        result.top_dim_time_ms =
            std::chrono::duration<double, std::milli>(end_cycles - start_cycles).count();
    }

    // Middle dimensions use standard reduction if needed
    // Union-Find provides D0 and D(d-1); middle dimensions via standard methods
    result.middle_time_ms = 0.0;

    result.total_pairs =
        result.d0_pairs.size() + result.top_dim_pairs.size() + result.middle_pairs.size();

    return result;
}

// Statistics about Union-Find computation
UnionFindStats getUnionFindStats(size_t num_points, size_t num_edges, double computation_time_ms)
{
    UnionFindStats stats;

    // Theoretical complexity: O(n alpha(n)) where alpha is inverse Ackermann
    // For practical purposes, alpha(n) < 5 for all reasonable n
    stats.theoretical_complexity = "O(n alpha(n))";
    stats.inverse_ackermann_estimate = 4; // Conservative upper bound

    // Compare to matrix reduction: O(n^3) worst case
    const double matrix_reduction_estimate =
        static_cast<double>(num_points) * static_cast<double>(num_points) *
        static_cast<double>(num_points) / 1e6; // Rough estimate in ms
    const double denominator = (std::isfinite(computation_time_ms) && computation_time_ms > 0.0)
                                   ? computation_time_ms
                                   : 1.0;
    const double speedup = matrix_reduction_estimate / denominator;

    stats.estimated_speedup_vs_matrix = std::isfinite(speedup) ? speedup : 1.0;
    stats.num_find_operations = num_edges > std::numeric_limits<size_t>::max() / 2U
                                    ? std::numeric_limits<size_t>::max()
                                    : num_edges * 2; // 2 finds per edge
    stats.num_unite_operations =
        num_points > 0 ? num_points - 1 : 0; // At most n-1 successful unions

    return stats;
}

} // namespace nerve::persistence

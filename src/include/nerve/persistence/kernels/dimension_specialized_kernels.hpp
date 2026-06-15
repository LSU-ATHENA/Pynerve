#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nerve::persistence
{
namespace kernels
{

namespace detail
{

class UnionFind
{
public:
    explicit UnionFind(std::size_t n)
        : parent_(n)
        , rank_(n, 0)
    {
        for (std::size_t i = 0; i < n; ++i)
        {
            parent_[i] = i;
        }
    }

    std::size_t find(std::size_t x)
    {
        if (parent_[x] != x)
        {
            parent_[x] = find(parent_[x]);
        }
        return parent_[x];
    }

    bool unite(std::size_t a, std::size_t b)
    {
        a = find(a);
        b = find(b);
        if (a == b)
        {
            return false;
        }
        if (rank_[a] < rank_[b])
        {
            std::swap(a, b);
        }
        parent_[b] = a;
        if (rank_[a] == rank_[b])
        {
            ++rank_[a];
        }
        return true;
    }

    std::size_t countSets()
    {
        std::size_t count = 0;
        for (std::size_t i = 0; i < parent_.size(); ++i)
        {
            if (find(i) == i)
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

using BoundaryColumn = std::vector<std::size_t>;

inline void xorSorted(BoundaryColumn &lhs, const BoundaryColumn &rhs)
{
    BoundaryColumn result;
    result.reserve(lhs.size() + rhs.size());
    std::set_symmetric_difference(lhs.begin(), lhs.end(), rhs.begin(), rhs.end(),
                                  std::back_inserter(result));
    lhs = std::move(result);
}

inline std::size_t rankMod2(std::vector<BoundaryColumn> columns)
{
    std::unordered_map<std::size_t, BoundaryColumn> basis_by_pivot;
    std::size_t rank = 0;

    for (auto &column : columns)
    {
        std::sort(column.begin(), column.end());
        column.erase(std::unique(column.begin(), column.end()), column.end());

        while (!column.empty())
        {
            const std::size_t pivot = column.back();
            const auto basis = basis_by_pivot.find(pivot);
            if (basis == basis_by_pivot.end())
            {
                basis_by_pivot.emplace(pivot, column);
                ++rank;
                break;
            }
            xorSorted(column, basis->second);
        }
    }

    return rank;
}

inline std::vector<std::uint32_t> makeSimplexKey(std::vector<std::uint32_t> simplex)
{
    std::sort(simplex.begin(), simplex.end());
    simplex.erase(std::unique(simplex.begin(), simplex.end()), simplex.end());
    return simplex;
}

inline void insertFaces(const std::vector<std::uint32_t> &simplex,
                        std::set<std::vector<std::uint32_t>> *vertices,
                        std::set<std::vector<std::uint32_t>> *edges,
                        std::set<std::vector<std::uint32_t>> *triangles)
{
    if (vertices == nullptr || edges == nullptr || triangles == nullptr)
    {
        return;
    }
    for (std::uint32_t v : simplex)
    {
        vertices->insert({v});
    }
    if (simplex.size() >= 2)
    {
        for (std::size_t i = 0; i < simplex.size(); ++i)
        {
            for (std::size_t j = i + 1; j < simplex.size(); ++j)
            {
                edges->insert(makeSimplexKey({simplex[i], simplex[j]}));
            }
        }
    }
    if (simplex.size() >= 3)
    {
        for (std::size_t i = 0; i < simplex.size(); ++i)
        {
            for (std::size_t j = i + 1; j < simplex.size(); ++j)
            {
                for (std::size_t k = j + 1; k < simplex.size(); ++k)
                {
                    triangles->insert(makeSimplexKey({simplex[i], simplex[j], simplex[k]}));
                }
            }
        }
    }
}

} // namespace detail

// Specialized kernel for H0 (connected components).
struct H0Kernel
{
    template <typename EdgeRange>
    [[nodiscard]] std::size_t countComponents(const EdgeRange &edges,
                                              std::size_t num_vertices) const
    {
        detail::UnionFind uf(num_vertices);
        for (const auto &e : edges)
        {
            const std::size_t u = static_cast<std::size_t>(e.first);
            const std::size_t v = static_cast<std::size_t>(e.second);
            if (u < num_vertices && v < num_vertices)
            {
                uf.unite(u, v);
            }
        }
        return uf.countSets();
    }
};

// H1 persistence requires reduction over the filtered complex.
struct H1Kernel
{
    [[nodiscard]] std::size_t
    countCycles(const std::vector<std::vector<std::uint32_t>> &triangles) const
    {
        std::set<std::vector<std::uint32_t>> vertices;
        std::set<std::vector<std::uint32_t>> edges;
        std::set<std::vector<std::uint32_t>> triangle_set;
        for (const auto &triangle : triangles)
        {
            const auto key = detail::makeSimplexKey(triangle);
            if (key.size() == 3)
            {
                detail::insertFaces(key, &vertices, &edges, &triangle_set);
            }
        }
        if (edges.empty())
        {
            return 0;
        }

        std::map<std::vector<std::uint32_t>, std::size_t> edge_rows;
        std::size_t row = 0;
        for (const auto &edge : edges)
        {
            edge_rows.emplace(edge, row++);
        }

        std::vector<detail::BoundaryColumn> boundary2;
        boundary2.reserve(triangle_set.size());
        for (const auto &triangle : triangle_set)
        {
            detail::BoundaryColumn column;
            column.reserve(3);
            for (std::size_t i = 0; i < triangle.size(); ++i)
            {
                std::vector<std::uint32_t> face;
                face.reserve(2);
                for (std::size_t j = 0; j < triangle.size(); ++j)
                {
                    if (i != j)
                    {
                        face.push_back(triangle[j]);
                    }
                }
                column.push_back(edge_rows.at(detail::makeSimplexKey(std::move(face))));
            }
            boundary2.push_back(std::move(column));
        }

        const std::size_t rank_boundary2 = detail::rankMod2(std::move(boundary2));
        return triangle_set.size() > rank_boundary2 ? triangle_set.size() - rank_boundary2 : 0;
    }
};

// H2 persistence requires reduction over the filtered complex.
struct H2Kernel
{
    [[nodiscard]] std::size_t
    countVoids(const std::vector<std::vector<std::uint32_t>> &tetrahedra) const
    {
        std::set<std::vector<std::uint32_t>> vertices;
        std::set<std::vector<std::uint32_t>> edges;
        std::set<std::vector<std::uint32_t>> triangles;
        std::set<std::vector<std::uint32_t>> tetrahedron_set;
        for (const auto &tetrahedron : tetrahedra)
        {
            const auto key = detail::makeSimplexKey(tetrahedron);
            if (key.size() == 4)
            {
                detail::insertFaces(key, &vertices, &edges, &triangles);
                tetrahedron_set.insert(key);
            }
        }
        if (triangles.empty())
        {
            return 0;
        }

        std::map<std::vector<std::uint32_t>, std::size_t> triangle_rows;
        std::size_t row = 0;
        for (const auto &triangle : triangles)
        {
            triangle_rows.emplace(triangle, row++);
        }

        std::vector<detail::BoundaryColumn> boundary3;
        boundary3.reserve(tetrahedron_set.size());
        for (const auto &tetrahedron : tetrahedron_set)
        {
            detail::BoundaryColumn column;
            column.reserve(4);
            for (std::size_t i = 0; i < tetrahedron.size(); ++i)
            {
                std::vector<std::uint32_t> face;
                face.reserve(3);
                for (std::size_t j = 0; j < tetrahedron.size(); ++j)
                {
                    if (i != j)
                    {
                        face.push_back(tetrahedron[j]);
                    }
                }
                column.push_back(triangle_rows.at(detail::makeSimplexKey(std::move(face))));
            }
            boundary3.push_back(std::move(column));
        }

        const std::size_t rank_boundary3 = detail::rankMod2(std::move(boundary3));
        return tetrahedron_set.size() > rank_boundary3 ? tetrahedron_set.size() - rank_boundary3
                                                       : 0;
    }
};

// Configuration for dimension-specialized computation.
struct DimensionConfig
{
    int max_dimension = 6;
    bool use_bit_parallel = false;
    bool use_clear_compress = false;
    bool use_involution = true;
    bool use_cohomology = true;
    bool use_dimension_cascade = true;
    bool use_prefetching = false;
    bool use_branchless = false;
    size_t chunk_size = 1024;
    size_t num_threads = 1;
};

// Results for dimension-specialized kernels.
struct H0Result
{
    std::vector<std::pair<double, double>> persistence_pairs;
    double computation_time_ms = 0.0;
    size_t num_unions = 0;
    size_t num_find_ops = 0;
    std::string algorithm_used;
    bool used_bit_parallel = false;
    bool used_clear_compress = false;
    bool used_dimension_cascade = false;

    const std::vector<std::pair<double, double>> &pairs() const { return persistence_pairs; }
};

// Forward declaration for computeD0PersistenceUnionFind.
std::vector<std::pair<double, double>>
computeD0PersistenceUnionFind(const std::vector<double> &points, size_t point_dim,
                              size_t num_points, const std::vector<std::pair<int, int>> &edges,
                              const std::vector<double> &edge_weights);

struct H12Result
{
    std::vector<std::pair<double, double>> persistence_pairs;
    std::vector<std::pair<double, double>> all_pairs;
    std::vector<std::pair<double, double>> pairs_h1;
    std::vector<std::pair<double, double>> pairs_h2;
    double computation_time_ms = 0.0;
    size_t num_reductions = 0;
    bool used_bit_parallel = false;
    bool used_cohomology = false;
    bool used_clear_compress = false;
    std::string algorithm_used;
};

struct H36Result
{
    std::vector<std::pair<double, double>> persistence_pairs;
    std::vector<std::pair<double, double>> all_pairs;
    std::vector<std::pair<double, double>> pairs_h3;
    std::vector<std::pair<double, double>> pairs_h4;
    std::vector<std::pair<double, double>> pairs_h5;
    std::vector<std::pair<double, double>> pairs_h6;
    double computation_time_ms = 0.0;
    double bit_parallel_speedup = 1.0;
    size_t num_reductions = 0;
    bool used_involution = false;
    bool used_bit_parallel = false;
    bool used_clear_compress = false;
    std::string algorithm_used;
};

struct DimensionSpecializedResult
{
    std::vector<std::vector<std::pair<double, double>>> pairs_by_dimension;
    std::vector<std::pair<double, double>> all_pairs;
    H0Result h0;
    H12Result h12;
    H36Result h36;
    double total_time_ms = 0.0;
    double speedup_vs_standard = 1.0;
    double total_speedup_estimate = 1.0;
    int max_dim = 6;
    DimensionConfig config_used;
};

// Dimension selector for optimal kernel.
class DimensionSpecializedKernel
{
public:
    template <int Dimension, typename SimplexRange>
    [[nodiscard]] auto compute(const SimplexRange &simplices) const
    {
        static_assert(Dimension >= 0 && Dimension <= 6, "Dimension must be in [0, 6]");

        if constexpr (Dimension == 0)
        {
            std::vector<std::pair<int, int>> edges;
            std::size_t max_vertex = 0;
            for (const auto &simplex : simplices)
            {
                if (simplex.size() == 2)
                {
                    const int u = static_cast<int>(simplex[0]);
                    const int v = static_cast<int>(simplex[1]);
                    if (u >= 0 && v >= 0)
                    {
                        edges.emplace_back(u, v);
                        max_vertex = std::max(max_vertex, static_cast<std::size_t>(std::max(u, v)));
                    }
                }
            }
            const std::size_t n_components = H0Kernel{}.countComponents(edges, max_vertex + 1);
            std::vector<std::pair<double, double>> out;
            out.reserve(n_components);
            for (std::size_t i = 0; i < n_components; ++i)
            {
                out.emplace_back(0.0, std::numeric_limits<double>::infinity());
            }
            return out;
        }
        else if constexpr (Dimension == 1)
        {
            std::set<std::vector<std::uint32_t>> vertices;
            std::set<std::vector<std::uint32_t>> edges;
            std::set<std::vector<std::uint32_t>> triangles;
            for (const auto &simplex : simplices)
            {
                std::vector<std::uint32_t> key;
                key.reserve(simplex.size());
                for (const auto vertex : simplex)
                {
                    if (vertex >= 0)
                    {
                        key.push_back(static_cast<std::uint32_t>(vertex));
                    }
                }
                key = detail::makeSimplexKey(std::move(key));
                detail::insertFaces(key, &vertices, &edges, &triangles);
            }

            std::map<std::vector<std::uint32_t>, std::size_t> vertex_ids;
            std::size_t vertex_id = 0;
            for (const auto &vertex : vertices)
            {
                vertex_ids.emplace(vertex, vertex_id++);
            }
            std::vector<std::pair<int, int>> component_edges;
            component_edges.reserve(edges.size());
            for (const auto &edge : edges)
            {
                component_edges.emplace_back(static_cast<int>(vertex_ids.at({edge[0]})),
                                             static_cast<int>(vertex_ids.at({edge[1]})));
            }
            const std::size_t components =
                vertices.empty() ? 0 : H0Kernel{}.countComponents(component_edges, vertices.size());

            std::map<std::vector<std::uint32_t>, std::size_t> edge_rows;
            std::size_t row = 0;
            for (const auto &edge : edges)
            {
                edge_rows.emplace(edge, row++);
            }
            std::vector<detail::BoundaryColumn> boundary2;
            boundary2.reserve(triangles.size());
            for (const auto &triangle : triangles)
            {
                detail::BoundaryColumn column;
                column.reserve(3);
                for (std::size_t i = 0; i < triangle.size(); ++i)
                {
                    std::vector<std::uint32_t> face;
                    face.reserve(2);
                    for (std::size_t j = 0; j < triangle.size(); ++j)
                    {
                        if (i != j)
                        {
                            face.push_back(triangle[j]);
                        }
                    }
                    column.push_back(edge_rows.at(detail::makeSimplexKey(std::move(face))));
                }
                boundary2.push_back(std::move(column));
            }
            const std::size_t rank_boundary2 = detail::rankMod2(std::move(boundary2));
            const std::size_t raw_cycles = edges.size() + components >= vertices.size()
                                               ? edges.size() + components - vertices.size()
                                               : 0;
            const std::size_t cycles =
                raw_cycles > rank_boundary2 ? raw_cycles - rank_boundary2 : 0;
            std::vector<std::pair<double, double>> out;
            out.reserve(cycles);
            for (std::size_t i = 0; i < cycles; ++i)
            {
                out.emplace_back(0.0, std::numeric_limits<double>::infinity());
            }
            return out;
        }
        else if constexpr (Dimension == 2)
        {
            std::set<std::vector<std::uint32_t>> vertices;
            std::set<std::vector<std::uint32_t>> edges;
            std::set<std::vector<std::uint32_t>> triangles;
            std::set<std::vector<std::uint32_t>> tetrahedra;
            for (const auto &simplex : simplices)
            {
                std::vector<std::uint32_t> key;
                key.reserve(simplex.size());
                for (const auto vertex : simplex)
                {
                    if (vertex >= 0)
                    {
                        key.push_back(static_cast<std::uint32_t>(vertex));
                    }
                }
                key = detail::makeSimplexKey(std::move(key));
                detail::insertFaces(key, &vertices, &edges, &triangles);
                if (key.size() == 4)
                {
                    tetrahedra.insert(key);
                }
            }

            std::map<std::vector<std::uint32_t>, std::size_t> edge_rows;
            std::size_t edge_row = 0;
            for (const auto &edge : edges)
            {
                edge_rows.emplace(edge, edge_row++);
            }
            std::vector<detail::BoundaryColumn> boundary2;
            boundary2.reserve(triangles.size());
            for (const auto &triangle : triangles)
            {
                detail::BoundaryColumn column;
                column.reserve(3);
                for (std::size_t i = 0; i < triangle.size(); ++i)
                {
                    std::vector<std::uint32_t> face;
                    face.reserve(2);
                    for (std::size_t j = 0; j < triangle.size(); ++j)
                    {
                        if (i != j)
                        {
                            face.push_back(triangle[j]);
                        }
                    }
                    column.push_back(edge_rows.at(detail::makeSimplexKey(std::move(face))));
                }
                boundary2.push_back(std::move(column));
            }

            std::map<std::vector<std::uint32_t>, std::size_t> triangle_rows;
            std::size_t triangle_row = 0;
            for (const auto &triangle : triangles)
            {
                triangle_rows.emplace(triangle, triangle_row++);
            }
            std::vector<detail::BoundaryColumn> boundary3;
            boundary3.reserve(tetrahedra.size());
            for (const auto &tetrahedron : tetrahedra)
            {
                detail::BoundaryColumn column;
                column.reserve(4);
                for (std::size_t i = 0; i < tetrahedron.size(); ++i)
                {
                    std::vector<std::uint32_t> face;
                    face.reserve(3);
                    for (std::size_t j = 0; j < tetrahedron.size(); ++j)
                    {
                        if (i != j)
                        {
                            face.push_back(tetrahedron[j]);
                        }
                    }
                    column.push_back(triangle_rows.at(detail::makeSimplexKey(std::move(face))));
                }
                boundary3.push_back(std::move(column));
            }

            const std::size_t rank_boundary2 = detail::rankMod2(std::move(boundary2));
            const std::size_t rank_boundary3 = detail::rankMod2(std::move(boundary3));
            const std::size_t kernel_dim =
                triangles.size() > rank_boundary2 ? triangles.size() - rank_boundary2 : 0;
            const std::size_t voids = kernel_dim > rank_boundary3 ? kernel_dim - rank_boundary3 : 0;
            std::vector<std::pair<double, double>> out;
            out.reserve(voids);
            for (std::size_t i = 0; i < voids; ++i)
            {
                out.emplace_back(0.0, std::numeric_limits<double>::infinity());
            }
            return out;
        }

        return std::vector<std::pair<double, double>>{};
    }
};

} // namespace kernels

// Alias for specialized namespace used by kernel_dimension_specialized_ops.cpp.
namespace specialized
{
using kernels::DimensionConfig;
using kernels::DimensionSpecializedResult;
using kernels::H0Result;
using kernels::H12Result;
using kernels::H36Result;
} // namespace specialized

} // namespace nerve::persistence

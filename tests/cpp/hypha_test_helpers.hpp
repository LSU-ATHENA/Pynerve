// Shared test helpers for Vietoris-Rips filtration benchmarks and correctness
// tests.  Extracted from hypha_gpu_benchmark.cpp and hypha_correctness_quantitative.cpp
// to avoid code duplication.
//
// These are free-standing, GPU-agnostic utilities that generate random point
// clouds, build VR complexes, and convert boundary matrices to the lockfree
// reduction format.

#pragma once

#include "nerve/algebra/boundary.hpp"
#include "nerve/algebra/complex.hpp"
#include "nerve/algebra/simplex.hpp"
#include "nerve/persistence/reduction/reduction_lockfree_ops.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <map>
#include <random>
#include <set>
#include <vector>

namespace nerve::test::hypha
{

// Generate n random points in [0, 1]^3.
inline std::vector<std::vector<float>> random_point_cloud(int n, unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::vector<std::vector<float>> points(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
        points[static_cast<std::size_t>(i)] = {dist(rng), dist(rng), dist(rng)};
    return points;
}

// Build a Vietoris-Rips filtration complex up to dimension 2 from a point cloud.
inline nerve::algebra::SimplicialComplex build_vr_complex(
    const std::vector<std::vector<float>> &points, float threshold)
{
    nerve::algebra::SimplicialComplex complex;
    int n = static_cast<int>(points.size());

    for (int i = 0; i < n; ++i)
        complex.addSimplexWithFiltration(nerve::algebra::Simplex({i}), 0.0);

    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
        {
            float d = std::sqrt(
                (points[static_cast<std::size_t>(i)][0] - points[static_cast<std::size_t>(j)][0]) *
                    (points[static_cast<std::size_t>(i)][0] - points[static_cast<std::size_t>(j)][0]) +
                (points[static_cast<std::size_t>(i)][1] - points[static_cast<std::size_t>(j)][1]) *
                    (points[static_cast<std::size_t>(i)][1] - points[static_cast<std::size_t>(j)][1]) +
                (points[static_cast<std::size_t>(i)][2] - points[static_cast<std::size_t>(j)][2]) *
                    (points[static_cast<std::size_t>(i)][2] - points[static_cast<std::size_t>(j)][2]));
            if (d <= threshold)
                complex.addSimplexWithFiltration(nerve::algebra::Simplex({i, j}),
                                                  static_cast<double>(d));
        }

    auto all_simplices = complex.getSimplices();
    std::vector<nerve::algebra::Simplex> edges;
    for (const auto &s : all_simplices)
        if (s.dimension() == 1)
            edges.push_back(s);

    for (std::size_t ei = 0; ei < edges.size(); ++ei)
    {
        auto v0 = edges[ei].vertices()[0];
        auto v1 = edges[ei].vertices()[1];
        for (std::size_t ej = ei + 1; ej < edges.size(); ++ej)
        {
            auto u0 = edges[ej].vertices()[0];
            auto u1 = edges[ej].vertices()[1];

            int common = -1, a = -1, b = -1;
            if (v0 == u0 || v0 == u1)
            {
                common = v0;
                a = v1;
                b = (u0 == v0) ? u1 : u0;
            }
            else if (v1 == u0 || v1 == u1)
            {
                common = v1;
                a = v0;
                b = (u0 == v1) ? u1 : u0;
            }
            if (common < 0 || a < 0 || b < 0)
                continue;

            bool edge_ab = false;
            for (const auto &e : edges)
            {
                auto verts = e.vertices();
                if ((verts[0] == a && verts[1] == b) ||
                    (verts[0] == b && verts[1] == a))
                {
                    edge_ab = true;
                    break;
                }
            }
            if (!edge_ab)
                continue;

            float da = std::sqrt(
                (points[static_cast<std::size_t>(a)][0] - points[static_cast<std::size_t>(common)][0]) *
                    (points[static_cast<std::size_t>(a)][0] - points[static_cast<std::size_t>(common)][0]) +
                (points[static_cast<std::size_t>(a)][1] - points[static_cast<std::size_t>(common)][1]) *
                    (points[static_cast<std::size_t>(a)][1] - points[static_cast<std::size_t>(common)][1]) +
                (points[static_cast<std::size_t>(a)][2] - points[static_cast<std::size_t>(common)][2]) *
                    (points[static_cast<std::size_t>(a)][2] - points[static_cast<std::size_t>(common)][2]));
            float db = std::sqrt(
                (points[static_cast<std::size_t>(b)][0] - points[static_cast<std::size_t>(common)][0]) *
                    (points[static_cast<std::size_t>(b)][0] - points[static_cast<std::size_t>(common)][0]) +
                (points[static_cast<std::size_t>(b)][1] - points[static_cast<std::size_t>(common)][1]) *
                    (points[static_cast<std::size_t>(b)][1] - points[static_cast<std::size_t>(common)][1]) +
                (points[static_cast<std::size_t>(b)][2] - points[static_cast<std::size_t>(common)][2]) *
                    (points[static_cast<std::size_t>(b)][2] - points[static_cast<std::size_t>(common)][2]));

            float dab = std::sqrt(
                (points[static_cast<std::size_t>(a)][0] - points[static_cast<std::size_t>(b)][0]) *
                    (points[static_cast<std::size_t>(a)][0] - points[static_cast<std::size_t>(b)][0]) +
                (points[static_cast<std::size_t>(a)][1] - points[static_cast<std::size_t>(b)][1]) *
                    (points[static_cast<std::size_t>(a)][1] - points[static_cast<std::size_t>(b)][1]) +
                (points[static_cast<std::size_t>(a)][2] - points[static_cast<std::size_t>(b)][2]) *
                    (points[static_cast<std::size_t>(a)][2] - points[static_cast<std::size_t>(b)][2]));

            float tri_filtration = std::max({da, db, dab});
            if (tri_filtration <= threshold)
            {
                int verts[] = {a, b, common};
                if (verts[0] > verts[1])
                    std::swap(verts[0], verts[1]);
                if (verts[1] > verts[2])
                    std::swap(verts[1], verts[2]);
                if (verts[0] > verts[1])
                    std::swap(verts[0], verts[1]);
                complex.addSimplexWithFiltration(
                    nerve::algebra::Simplex({verts[0], verts[1], verts[2]}),
                    static_cast<double>(tri_filtration));
            }
        }
    }
    return complex;
}

// Convert a boundary matrix to the sparse CSC format required by
// reduceMatrixLockfree / reduceMatrixLockfreeProfiled.
// Convert a boundary matrix to the sparse CSC format required by
// reduceMatrixLockfree / reduceMatrixLockfreeProfiled.
//
// NOTE: filtration_values is indexed by COLUMN index (death simplex).
// row_filtration_values is indexed by ROW index (birth simplex).
// For k-dimensional boundary matrices (buildKDimensional), rows and columns
// index different simplex sets, so separate arrays are required for correct
// birth/death value extraction.
inline void to_lockfree_format(const nerve::algebra::BoundaryMatrix &bm,
                               std::vector<std::vector<int>> &boundary,
                               std::vector<double> &filtration_values,
                               std::vector<double> &row_filtration_values,
                               std::vector<nerve::Dimension> &dims)
{
    nerve::Size n_cols = bm.cols();
    nerve::Size n_rows = bm.rows();
    boundary.resize(static_cast<std::size_t>(n_cols));
    filtration_values.resize(static_cast<std::size_t>(n_cols));
    row_filtration_values.resize(static_cast<std::size_t>(n_rows));
    dims.resize(static_cast<std::size_t>(n_cols));
    for (nerve::Size row = 0; row < n_rows; ++row)
        row_filtration_values[static_cast<std::size_t>(row)] = bm.getRowFiltrationValue(row);
    for (nerve::Size col = 0; col < n_cols; ++col)
    {
        std::vector<int> sparse_col;
        for (nerve::Size row = 0; row < n_rows; ++row)
            if (bm.getMatrixEntry(row, col) != 0.0)
                sparse_col.push_back(static_cast<int>(row));
        boundary[static_cast<std::size_t>(col)] = std::move(sparse_col);
        filtration_values[static_cast<std::size_t>(col)] = bm.getFiltrationValue(col);
        dims[static_cast<std::size_t>(col)] =
            static_cast<nerve::Dimension>(bm.getColSimplexDimension(col));
    }
}

// Sequential ground truth (fast sparse reduction, no SIMD)
// Canonical shared copy (callers handle their own timing).
//
// Runs standard column-by-column elimination on CSC boundary data:
//   for each column j: XOR with pivot(j).column until column is empty
//                     or its MSB is unclaimed (pivot found).
// Returns the full pair set with birth/death values.
inline std::vector<nerve::Pair> reduce_sequential_fast(
    const std::vector<std::vector<int>> &boundary,
    const std::vector<double> &filtration_values,
    const std::vector<double> &row_filtration_values,
    const std::vector<nerve::Dimension> &dims,
    int n_rows)
{
    int n_cols = static_cast<int>(boundary.size());
    std::vector<int> pivot_to_column(
        static_cast<std::size_t>(n_rows > 0 ? n_rows : 1), -1);
    std::vector<int> column_pivot(static_cast<std::size_t>(n_cols), -1);

    for (int j = 0; j < n_cols; ++j)
    {
        auto col = boundary[static_cast<std::size_t>(j)];
        while (!col.empty())
        {
            int p = col.back();
            std::size_t pu = static_cast<std::size_t>(p);
            if (pu >= pivot_to_column.size())
                break;
            int k = pivot_to_column[pu];
            if (k < 0)
            {
                pivot_to_column[pu] = j;
                column_pivot[static_cast<std::size_t>(j)] = p;
                break;
            }
            const auto &other = boundary[static_cast<std::size_t>(k)];
            std::vector<int> result;
            result.reserve(col.size() + other.size());
            std::size_t i1 = 0, i2 = 0;
            while (i1 < col.size() && i2 < other.size())
            {
                if (col[i1] < other[i2])
                    result.push_back(col[i1++]);
                else if (col[i1] > other[i2])
                    result.push_back(other[i2++]);
                else
                { ++i1; ++i2; }
            }
            while (i1 < col.size()) result.push_back(col[i1++]);
            while (i2 < other.size()) result.push_back(other[i2++]);
            col = std::move(result);
        }
    }
    std::vector<nerve::Pair> pairs;
    pairs.reserve(static_cast<std::size_t>(n_cols));
    for (int j = 0; j < n_cols; ++j)
    {
        int p = column_pivot[static_cast<std::size_t>(j)];
        if (p < 0) continue;
        nerve::Pair pair{};
        pair.dimension = static_cast<std::size_t>(j) < dims.size()
            ? dims[static_cast<std::size_t>(j)] : 0;
        std::size_t pu = static_cast<std::size_t>(p);
        pair.birth = pu < row_filtration_values.size()
            ? row_filtration_values[pu]
            : filtration_values[static_cast<std::size_t>(j)];
        pair.death = static_cast<std::size_t>(j) < filtration_values.size()
            ? filtration_values[static_cast<std::size_t>(j)] : 0.0;
        pairs.push_back(pair);
    }
    return pairs;
}

// Extend a VR complex with tetrahedra (dim-3 simplices)
// Given a complex that already contains vertices, edges, and triangles,
// enumerate tetrahedra via edge-to-triangle adjacency.
//
// Algorithm: For each pair of triangles sharing an edge, check whether
// the other two faces (the triangles spanned by the shared edge and each
// rim vertex) also exist.  If all 4 faces exist, the tetrahedron is valid.
//
// This is O(T * avg_triangles_per_edge) instead of O(V^4), making it
// practical for up to ~50 vertices (larger counts become slow due to
// combinatorial explosion of the VR triangle count).
//
// Filtration of the tetrahedron = max of its 4 face filtrations (which
// themselves encode the max edge length of each face).
inline void build_tetrahedra(nerve::algebra::SimplicialComplex &complex)
{
    auto all_simplices = complex.getSimplices();

    // Use std::pair<int,int> for edge keys (has built-in operator< for map)
    auto edge_pair = [](int u, int v) -> std::pair<int,int> {
        return u < v ? std::pair<int,int>{u, v} : std::pair<int,int>{v, u};
    };

    // Collect triangles keyed by sorted vertex triplet;
    // also build edge -> list of triangles that contain that edge
    struct TriKey { int a, b, c; };

    auto cmp_tri = [](const TriKey &x, const TriKey &y) {
        return x.a < y.a || (x.a == y.a && (x.b < y.b || (x.b == y.b && x.c < y.c)));
    };
    std::map<TriKey, double, decltype(cmp_tri)> triangle_map(cmp_tri);
    std::map<std::pair<int,int>, std::vector<TriKey>> edge_to_triangles;
    std::set<std::array<int,4>> seen_tets;

    for (const auto &s : all_simplices)
    {
        if (s.dimension() != 2) continue;
        double filt = complex.getFiltration(s);

        auto v = s.vertices();
        TriKey tk{v[0], v[1], v[2]};
        triangle_map[tk] = filt;

        // Map each edge of this triangle -> triangle key
        edge_to_triangles[edge_pair(v[0], v[1])].push_back(tk);
        edge_to_triangles[edge_pair(v[0], v[2])].push_back(tk);
        edge_to_triangles[edge_pair(v[1], v[2])].push_back(tk);
    }

    // For each edge with 2+ incident triangles, try each pair
    for (const auto &[edge_pair_val, tri_list] : edge_to_triangles)
    {
        int e0 = edge_pair_val.first;
        int e1 = edge_pair_val.second;
        if (tri_list.size() < 2) continue;

        for (std::size_t ti = 0; ti < tri_list.size(); ++ti)
        {
            const auto &t1 = tri_list[ti];
            // Identify the rim vertex (the one NOT in the shared edge)
            auto rim_a = [&]() -> int {
                if (t1.a != e0 && t1.a != e1) return t1.a;
                if (t1.b != e0 && t1.b != e1) return t1.b;
                return t1.c;
            }();

            for (std::size_t tj = ti + 1; tj < tri_list.size(); ++tj)
            {
                const auto &t2 = tri_list[tj];
                auto rim_b = [&]() -> int {
                    if (t2.a != e0 && t2.a != e1) return t2.a;
                    if (t2.b != e0 && t2.b != e1) return t2.b;
                    return t2.c;
                }();

                if (rim_a == rim_b) continue; // same triangle, skip

                // The tetrahedron vertices: e0, e1, rim_a, rim_b
                // Remaining faces to check: (e0, rim_a, rim_b) and (e1, rim_a, rim_b)
                int verts_3a[] = {e0, rim_a, rim_b};
                int verts_3b[] = {e1, rim_a, rim_b};

                auto sort3 = [](int v[3]) {
                    if (v[0] > v[1]) std::swap(v[0], v[1]);
                    if (v[1] > v[2]) std::swap(v[1], v[2]);
                    if (v[0] > v[1]) std::swap(v[0], v[1]);
                };
                sort3(verts_3a);
                sort3(verts_3b);

                TriKey f3a{verts_3a[0], verts_3a[1], verts_3a[2]};
                TriKey f3b{verts_3b[0], verts_3b[1], verts_3b[2]};

                auto it_a = triangle_map.find(f3a);
                auto it_b = triangle_map.find(f3b);
                if (it_a == triangle_map.end() || it_b == triangle_map.end())
                    continue;

                // All 4 faces exist -> valid tetrahedron
                double tetra_filt = std::max({
                    triangle_map[t1],      // face (e0, e1, rim_a)
                    triangle_map[t2],      // face (e0, e1, rim_b)
                    it_a->second,          // face (e0, rim_a, rim_b)
                    it_b->second           // face (e1, rim_a, rim_b)
                });

                // Sort all 4 vertices for the tetrahedron;
                // deduplicate: same tetrahedron will be found through each of its 6 edges
                int tetra_verts[4] = {e0, e1, rim_a, rim_b};
                std::sort(std::begin(tetra_verts), std::end(tetra_verts));
                std::array<int,4> tet_key{tetra_verts[0], tetra_verts[1],
                                          tetra_verts[2], tetra_verts[3]};
                if (seen_tets.insert(tet_key).second)
                {
                    complex.addSimplexWithFiltration(
                        nerve::algebra::Simplex({tetra_verts[0], tetra_verts[1],
                                                  tetra_verts[2], tetra_verts[3]}),
                        tetra_filt);
                }
            }
        }
    }
}

} // namespace nerve::test::hypha

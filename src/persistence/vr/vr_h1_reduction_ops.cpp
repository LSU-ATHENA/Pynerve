#include "nerve/persistence/vr/vr_h1_reduction_ops.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <ranges>
#include <unordered_map>
#include <unordered_set>

namespace nerve::persistence::reduced
{

namespace
{

#include "detail/vr_h1_reduction_helpers.inl"

} // namespace

// Build reduced VR complex optimized for H1
ReducedVRH1Result buildReducedVRForH1(const std::vector<double> &points_data, size_t point_dim,
                                      size_t num_points, const ReducedVRH1Config &config)
{
    ReducedVRH1Result result;
    result.config = config;

    auto start = std::chrono::high_resolution_clock::now();

    // Convert to Point structures
    std::vector<Point> points;
    points.reserve(num_points);
    for (size_t i = 0; i < num_points; ++i)
    {
        Point p;
        p.id = static_cast<int>(i);
        for (size_t d = 0; d < point_dim; ++d)
        {
            p.coords.push_back(points_data[i * point_dim + d]);
        }
        points.push_back(p);
    }

    // Build all edges within radius
    auto start_edges = std::chrono::high_resolution_clock::now();
    std::vector<Edge> all_edges;

    for (size_t i = 0; i < num_points; ++i)
    {
        for (size_t j = i + 1; j < num_points; ++j)
        {
            double dist_sq = 0.0;
            for (size_t d = 0; d < point_dim; ++d)
            {
                double diff = points_data[i * point_dim + d] - points_data[j * point_dim + d];
                dist_sq += diff * diff;
            }
            double dist = std::sqrt(dist_sq);

            if (dist <= config.max_radius)
            {
                Edge e;
                e.v1 = static_cast<int>(i);
                e.v2 = static_cast<int>(j);
                e.weight = dist;
                all_edges.push_back(e);
            }
        }
    }

    // Sort edges by weight
    std::ranges::sort(all_edges, {}, &Edge::weight);

    result.original_edge_count = static_cast<int>(all_edges.size());
    auto end_edges = std::chrono::high_resolution_clock::now();
    result.edge_build_time_ms =
        std::chrono::duration<double, std::milli>(end_edges - start_edges).count();

    // Prune edges that cannot contribute to H1
    auto start_prune = std::chrono::high_resolution_clock::now();
    std::vector<Edge> pruned_edges;

    if (config.use_cycle_filter)
    {
        // Parallel pruning with thread pool
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic) if (config.num_threads > 1)
#endif
        for (size_t i = 0; i < all_edges.size(); ++i)
        {
            if (canFormH1Feature(all_edges[i], points, config.max_radius, config))
            {
#if defined(_OPENMP)
#pragma omp critical
#endif
                pruned_edges.push_back(all_edges[i]);
            }
        }
    }
    else
    {
        pruned_edges = all_edges;
    }

    // Additional connectivity-based pruning
    if (config.use_connectivity_pruning)
    {
        pruned_edges = pruneByLocalConnectivity(pruned_edges, points, config);
    }

    // Sort again after pruning
    std::ranges::sort(pruned_edges, {}, &Edge::weight);

    result.reduced_edge_count = static_cast<int>(pruned_edges.size());
    auto end_prune = std::chrono::high_resolution_clock::now();
    result.prune_time_ms =
        std::chrono::duration<double, std::milli>(end_prune - start_prune).count();

    // Build triangles from reduced edges
    auto start_tri = std::chrono::high_resolution_clock::now();
    std::vector<Triangle> triangles;

    // Build edge lookup for fast triangle detection
    auto edgeKey = [](int a, int b) -> uint64_t {
        if (a > b)
            std::swap(a, b);
        return (static_cast<uint64_t>(a) << 32) | static_cast<uint64_t>(b);
    };

    std::unordered_map<uint64_t, double> edge_weights;
    for (const auto &e : pruned_edges)
    {
        edge_weights[edgeKey(e.v1, e.v2)] = e.weight;
    }

    // Find triangles
    for (size_t i = 0; i < pruned_edges.size(); ++i)
    {
        const auto &e1 = pruned_edges[i];

        // Find edges that share a vertex with e1
        for (size_t j = i + 1; j < pruned_edges.size(); ++j)
        {
            const auto &e2 = pruned_edges[j];

            // Check if e1 and e2 share a vertex
            int shared = -1;
            int v1_other = -1, v2_other = -1;

            if (e1.v1 == e2.v1)
            {
                shared = e1.v1;
                v1_other = e1.v2;
                v2_other = e2.v2;
            }
            else if (e1.v1 == e2.v2)
            {
                shared = e1.v1;
                v1_other = e1.v2;
                v2_other = e2.v1;
            }
            else if (e1.v2 == e2.v1)
            {
                shared = e1.v2;
                v1_other = e1.v1;
                v2_other = e2.v2;
            }
            else if (e1.v2 == e2.v2)
            {
                shared = e1.v2;
                v1_other = e1.v1;
                v2_other = e2.v1;
            }

            if (shared >= 0 && v1_other != v2_other)
            {
                // Check if third edge exists
                uint64_t third_key = edgeKey(v1_other, v2_other);
                auto it = edge_weights.find(third_key);
                if (it != edge_weights.end())
                {
                    // Found triangle
                    Triangle t;
                    t.v1 = shared;
                    t.v2 = v1_other;
                    t.v3 = v2_other;
                    t.weight = std::max({e1.weight, e2.weight, it->second});
                    triangles.push_back(t);
                }
            }
        }
    }

    // Sort triangles by weight
    std::ranges::sort(triangles, {}, &Triangle::weight);

    // Filter triangles if requested
    if (config.use_triangle_filter)
    {
        triangles = filterTriangles(triangles, pruned_edges, config);
        result.filtered_triangle_count = static_cast<int>(triangles.size());
    }
    else
    {
        result.filtered_triangle_count = static_cast<int>(triangles.size());
    }
    result.triangle_count = static_cast<int>(triangles.size());

    auto end_tri = std::chrono::high_resolution_clock::now();
    result.triangle_build_time_ms =
        std::chrono::duration<double, std::milli>(end_tri - start_tri).count();

    // Store results
    result.edges = std::move(pruned_edges);
    result.triangles = std::move(triangles);

    // Compute reduction ratio
    if (result.original_edge_count > 0)
    {
        result.edge_reduction_ratio =
            static_cast<double>(result.reduced_edge_count) / result.original_edge_count;
    }
    else
    {
        result.edge_reduction_ratio = 0.0;
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.total_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    return result;
}

// Compute H1 using reduced VR complex
H1Result computeH1ReducedVR(const ReducedVRH1Result &reduced_vr, const ReducedVRH1Config &config)
{
    H1Result result;
    if (!std::isfinite(config.max_radius) || config.max_radius < 0.0)
    {
        return result;
    }

    auto start = std::chrono::high_resolution_clock::now();

    // Build boundary matrix for H1
    // Rows: edges (1-simplices)
    // Columns: triangles (2-simplices)

    int num_edges = static_cast<int>(reduced_vr.edges.size());
    int num_triangles = static_cast<int>(reduced_vr.triangles.size());

    // Edge lookup
    auto edgeKey = [](int a, int b) -> uint64_t {
        if (a > b)
            std::swap(a, b);
        return (static_cast<uint64_t>(a) << 32) | static_cast<uint64_t>(b);
    };

    std::unordered_map<uint64_t, int> edge_to_idx;
    for (int i = 0; i < num_edges; ++i)
    {
        edge_to_idx[edgeKey(reduced_vr.edges[i].v1, reduced_vr.edges[i].v2)] = i;
    }

    // Build boundary matrix
    std::vector<std::vector<int>> boundary_matrix(num_triangles);

    for (int t_idx = 0; t_idx < num_triangles; ++t_idx)
    {
        const auto &t = reduced_vr.triangles[t_idx];

        // Triangle boundary: 3 edges with alternating orientation
        // In Z2, orientation doesn't matter
        uint64_t e1 = edgeKey(t.v1, t.v2);
        uint64_t e2 = edgeKey(t.v1, t.v3);
        uint64_t e3 = edgeKey(t.v2, t.v3);

        auto it1 = edge_to_idx.find(e1);
        auto it2 = edge_to_idx.find(e2);
        auto it3 = edge_to_idx.find(e3);

        if (it1 != edge_to_idx.end())
            boundary_matrix[t_idx].push_back(it1->second);
        if (it2 != edge_to_idx.end())
            boundary_matrix[t_idx].push_back(it2->second);
        if (it3 != edge_to_idx.end())
            boundary_matrix[t_idx].push_back(it3->second);

        // Sort for consistency
        std::ranges::sort(boundary_matrix[t_idx]);
    }

    // Standard matrix reduction for H1
    std::unordered_map<int, int> pivot_to_column;

    for (int col_idx = 0; col_idx < num_triangles; ++col_idx)
    {
        auto &col = boundary_matrix[col_idx];

        // Reduce column
        while (!col.empty())
        {
            int pivot = col.back();
            auto it = pivot_to_column.find(pivot);
            if (it != pivot_to_column.end())
            {
                // XOR with existing column
                std::vector<int> new_col;
                std::set_symmetric_difference(
                    col.begin(), col.end(), boundary_matrix[it->second].begin(),
                    boundary_matrix[it->second].end(), std::back_inserter(new_col));
                col = std::move(new_col);
            }
            else
            {
                // New pivot
                pivot_to_column[pivot] = col_idx;

                // Record H1 death
                H1Pair pair;
                pair.birth_index = pivot;
                pair.death_index = col_idx;
                pair.birth_time = reduced_vr.edges[pivot].weight;
                pair.death_time = reduced_vr.triangles[col_idx].weight;
                result.pairs.push_back(pair);
                break;
            }
        }
    }

    // Essential cycles (no death)
    for (int e_idx = 0; e_idx < num_edges; ++e_idx)
    {
        if (pivot_to_column.find(e_idx) == pivot_to_column.end())
        {
            H1Pair pair;
            pair.birth_index = e_idx;
            pair.death_index = -1;
            pair.birth_time = reduced_vr.edges[e_idx].weight;
            pair.death_time = std::numeric_limits<double>::infinity();
            result.pairs.push_back(pair);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.computation_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    return result;
}

// Public helper wrappers are split into a detail include so this
// translation unit stays under the repository hard-cap target.
#include "detail/vr_h1_reduction_config.inl"

} // namespace nerve::persistence::reduced

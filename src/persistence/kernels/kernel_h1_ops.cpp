#include "nerve/persistence/kernels/kernel_h1_ops.hpp"
#include "nerve/persistence/utils/bit_parallel_z2.hpp"

#include <algorithm>
#include <chrono>
#include <ranges>
#include <unordered_map>

namespace nerve::persistence::h1
{

namespace
{

// H1-specific coboundary: edge -> triangles
// Edge-oriented layout for coboundaries.
struct H1CoboundaryColumn
{
    int edge_index;             // Index of edge (1-simplex)
    std::vector<int> triangles; // Triangles containing this edge (cofaces)
    int pivot;                  // Highest triangle index
    bool reduced;

    H1CoboundaryColumn()
        : edge_index(-1)
        , pivot(-1)
        , reduced(false)
    {}
};

// Build H1-specific coboundary matrix
// Edge i has coboundary = all triangles that contain that edge
std::vector<H1CoboundaryColumn> buildH1Coboundary(const reduced::ReducedVRH1Result &reduced_vr)
{
    const int num_edges = static_cast<int>(reduced_vr.edges.size());
    const int num_triangles = static_cast<int>(reduced_vr.triangles.size());
    std::vector<H1CoboundaryColumn> coboundary(num_edges);

    // Map edges to indices
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

    // For each triangle, add to coboundary of its 3 edges
    for (int t_idx = 0; t_idx < num_triangles; ++t_idx)
    {
        const auto &t = reduced_vr.triangles[t_idx];

        // Find the 3 edges of this triangle
        uint64_t e1 = edgeKey(t.v1, t.v2);
        uint64_t e2 = edgeKey(t.v1, t.v3);
        uint64_t e3 = edgeKey(t.v2, t.v3);

        auto it1 = edge_to_idx.find(e1);
        auto it2 = edge_to_idx.find(e2);
        auto it3 = edge_to_idx.find(e3);

        // Add triangle to coboundary of each edge
        if (it1 != edge_to_idx.end())
        {
            coboundary[it1->second].triangles.push_back(t_idx);
        }
        if (it2 != edge_to_idx.end())
        {
            coboundary[it2->second].triangles.push_back(t_idx);
        }
        if (it3 != edge_to_idx.end())
        {
            coboundary[it3->second].triangles.push_back(t_idx);
        }
    }

    // Sort coboundaries and find pivots
    for (int i = 0; i < num_edges; ++i)
    {
        auto &col = coboundary[i];
        col.edge_index = i;
        std::ranges::sort(col.triangles);
        if (!col.triangles.empty())
        {
            col.pivot = col.triangles.back(); // Highest = pivot
        }
    }

    return coboundary;
}

// H1-specific cohomology reduction over sorted Z2 coboundary columns.
H1Result reduceH1Cohomology(std::vector<H1CoboundaryColumn> &coboundary,
                            const reduced::ReducedVRH1Result &reduced_vr, const H1Config &config)
{
    H1Result result;
    result.algorithm_used = "H1-Reduced-VR-Cohomology";

    auto start = std::chrono::high_resolution_clock::now();

    // Sort coboundary columns by pivot
    std::ranges::sort(coboundary, {}, &H1CoboundaryColumn::pivot);

    // Pivot lookup: pivot -> column index
    std::unordered_map<int, int> pivot_to_col;

    // Process columns (edges) in order
    for (size_t col_idx = 0; col_idx < coboundary.size(); ++col_idx)
    {
        auto &col = coboundary[col_idx];

        // Skip if already cleared (from dimension cascade)
        if (col.reduced)
        {
            result.num_cleared++;
            continue;
        }

        // Reduce column
        while (col.pivot >= 0)
        {
            auto it = pivot_to_col.find(col.pivot);
            if (it != pivot_to_col.end())
            {
                // Add that column to this one (XOR in Z2)
                auto &other = coboundary[it->second];

                std::vector<int> new_triangles;
                std::set_symmetric_difference(col.triangles.begin(), col.triangles.end(),
                                              other.triangles.begin(), other.triangles.end(),
                                              std::back_inserter(new_triangles));

                col.triangles = std::move(new_triangles);

                if (col.triangles.empty())
                {
                    col.pivot = -1;
                }
                else
                {
                    col.pivot = col.triangles.back();
                }

                result.num_reductions++;
            }
            else
            {
                // New pivot found
                pivot_to_col[col.pivot] = static_cast<int>(col_idx);

                // Record H1 pair: edge (birth) dies at triangle (death)
                H1Pair pair;
                pair.birth_index = col.edge_index;
                pair.death_index = col.pivot; // Triangle index
                pair.birth_time = reduced_vr.edges[col.edge_index].weight;
                pair.death_time = reduced_vr.triangles[col.pivot].weight;
                result.pairs.push_back(pair);

                if (config.use_clearing)
                {
                    result.cleared_for_h2.insert(col.pivot);
                }

                break;
            }
        }

        // If pivot is -1, edge is essential (infinite persistence)
        if (col.pivot < 0 && !col.reduced)
        {
            H1Pair pair;
            pair.birth_index = col.edge_index;
            pair.death_index = -1;
            pair.birth_time = reduced_vr.edges[col.edge_index].weight;
            pair.death_time = std::numeric_limits<double>::infinity();
            result.pairs.push_back(pair);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.computation_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    return result;
}

} // namespace

// Main H1 computation.
H1Result computeH1Persistence(const std::vector<double> &points, size_t point_dim,
                              size_t num_points, const H1Config &config)
{
    H1Result result;
    result.config = config;

    auto start_total = std::chrono::high_resolution_clock::now();

    // Build Reduced VR complex (H1-specific)
    auto start_build = std::chrono::high_resolution_clock::now();

    reduced::ReducedVRH1Config rv_config;
    rv_config.max_radius = config.max_radius;
    rv_config.use_cycle_filter = config.use_cycle_filter;
    rv_config.use_connectivity_pruning = config.use_connectivity_pruning;
    rv_config.use_triangle_filter = config.use_triangle_filter;

    auto reduced_vr = reduced::buildReducedVRForH1(points, point_dim, num_points, rv_config);

    auto end_build = std::chrono::high_resolution_clock::now();
    result.build_time_ms =
        std::chrono::duration<double, std::milli>(end_build - start_build).count();

    // Build H1 coboundary
    auto start_cobound = std::chrono::high_resolution_clock::now();
    auto coboundary = buildH1Coboundary(reduced_vr);
    auto end_cobound = std::chrono::high_resolution_clock::now();
    result.coboundary_time_ms =
        std::chrono::duration<double, std::milli>(end_cobound - start_cobound).count();

    // Reduce H1 cohomology
    auto reduction_result = reduceH1Cohomology(coboundary, reduced_vr, config);
    result.pairs = std::move(reduction_result.pairs);
    result.cleared_for_h2 = std::move(reduction_result.cleared_for_h2);
    result.num_reductions = reduction_result.num_reductions;
    result.num_cleared = reduction_result.num_cleared;
    result.algorithm_used = std::move(reduction_result.algorithm_used);
    result.computation_time_ms = reduction_result.computation_time_ms;

    // Copy metadata from reduced VR
    result.num_edges = static_cast<int>(reduced_vr.edges.size());
    result.num_triangles = static_cast<int>(reduced_vr.triangles.size());
    result.edge_reduction_ratio = reduced_vr.edge_reduction_ratio;

    auto end_total = std::chrono::high_resolution_clock::now();
    result.total_time_ms =
        std::chrono::duration<double, std::milli>(end_total - start_total).count();

    return result;
}

// Get optimal H1 config
H1Config getOptimalH1Config(size_t num_points, double max_radius)
{
    H1Config config;
    config.max_radius = max_radius;

    // Use all H1-specific optimizations
    config.use_cycle_filter = true;
    config.use_connectivity_pruning = (num_points > 1000);
    config.use_triangle_filter = true;
    config.use_bit_parallel = false;
    config.use_clearing = true;

    return config;
}

// Estimate H1 speedup
H1SpeedupEstimate estimateH1Speedup(size_t num_points)
{
    H1SpeedupEstimate estimate;

    // Reduced VR gives 2-3x edge reduction
    estimate.edge_reduction = 2.5;

    // H1-specific cohomology layout gives another 1.2x
    estimate.layout_speedup = 1.2;

    // Clearing gives 1.3x
    estimate.clearing_speedup = 1.3;

    (void)num_points;
    estimate.bit_parallel_speedup = 1.0;

    // Combined
    estimate.total_speedup = estimate.edge_reduction * estimate.layout_speedup *
                             estimate.clearing_speedup * estimate.bit_parallel_speedup;

    return estimate;
}

} // namespace nerve::persistence::h1


#pragma once

#include <cstddef>
#include <utility>
#include <vector>

namespace nerve::persistence
{

/**
 * @brief Result of edge collapse preprocessing
 */
struct EdgeCollapseResult
{
    std::vector<std::pair<int, int>> collapse_sequence; // (collapsed_vertex, into_vertex)
    std::vector<bool> vertex_alive;                     // Which vertices remain
    std::vector<std::vector<int>> reduced_neighbors;    // Adjacency after collapse
    int original_vertices = 0;
    int original_edges = 0;
    int remaining_vertices = 0;
    int remaining_edges = 0;
};

/**
 * @brief Statistics about edge collapse effectiveness
 */
struct EdgeCollapseStats
{
    double vertex_reduction_ratio = 0.0;      // 0.0 to 1.0
    double edge_reduction_ratio = 0.0;        // 0.0 to 1.0
    double estimated_simplex_reduction = 0.0; // For dimension 2+
    size_t num_collapses = 0;
};

/**
 * @brief Apply edge collapse preprocessing to reduce simplex count
 *
 * The Edge Collapser identifies and removes "dominated" edges that don't
 * affect the persistence diagram. This is based on the observation that
 * in a flag complex, an edge (u,v) is dominated if link(u) subset  link(v).
 *
 * Key benefits:
 * - Reduces simplex count by 50-90% in many cases
 * - Preserves persistent homology exactly
 * - Near-linear time complexity in practice
 *
 * Usage:
 * - Build 1-skeleton (edges) from point cloud
 * - Call collapseEdges() to get reduced graph
 * - Build VR complex on reduced graph
 * - Compute persistence on the reduced complex
 *
 * @param input_neighbors Adjacency list of 1-skeleton
 * @param edge_weights Weights (distances) for each edge
 * @param max_radius Maximum distance for VR complex
 * @return Collapse result with reduced graph
 */
EdgeCollapseResult collapseEdges(const std::vector<std::vector<int>> &input_neighbors,
                                 const std::vector<double> &edge_weights, double max_radius);

/**
 * @brief Edge-collapse compatibility entry point for accelerated callers
 *
 * Returns the deterministic CPU edge-collapse result until a GPU runtime is
 * wired into this API.
 */
EdgeCollapseResult collapseEdgesGPU(const std::vector<std::vector<int>> &input_neighbors,
                                    const std::vector<double> &edge_weights, double max_radius);

/**
 * @brief Analyze effectiveness of collapse
 */
EdgeCollapseStats analyzeCollapse(const EdgeCollapseResult &result);

/**
 * @brief Check if edge collapse is beneficial for this graph
 *
 * Returns true if expected reduction > 30% (heuristic)
 */
bool shouldUseEdgeCollapse(size_t num_vertices, size_t num_edges, double density);

} // namespace nerve::persistence

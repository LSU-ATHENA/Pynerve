
// Based on Boissonnat & Pritam's "Edge Collapse and Persistence of Flag Complexes"
// This can reduce simplices by 50-90% before persistence computation

#include "nerve/persistence/reduction/reduction_edge_collapse_ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace nerve::persistence
{

namespace
{

// Check if edge (u,v) is dominated by vertex u
// Edge is dominated if link(u) subset  link(v)
// This means u can be collapsed into v
bool isEdgeDominated(int u, int v, const std::vector<std::vector<int>> &neighbors,
                     const std::vector<std::unordered_set<int>> &neighbor_sets)
{
    const auto &link_u = neighbors[u];
    const auto &set_v = neighbor_sets[v];

    // Check if all neighbors of u are also neighbors of v
    for (int w : link_u)
    {
        if (w == v)
            continue; // v is not in link(u) by definition
        if (set_v.find(w) == set_v.end())
        {
            return false;
        }
    }

    return true;
}

// Find all dominated edges in the graph
std::vector<std::pair<int, int>> findDominatedEdges(const std::vector<std::vector<int>> &neighbors,
                                                    int n_vertices)
{
    std::vector<std::pair<int, int>> dominated_edges;
    dominated_edges.reserve(n_vertices); // Estimate

    // Build neighbor sets for fast lookup
    std::vector<std::unordered_set<int>> neighbor_sets(n_vertices);
    for (int i = 0; i < n_vertices; ++i)
    {
        neighbor_sets[i].insert(neighbors[i].begin(), neighbors[i].end());
    }

    // Check each edge
    for (int u = 0; u < n_vertices; ++u)
    {
        for (int v : neighbors[u])
        {
            if (v < u)
                continue; // Only check each edge once

            // Check if (u,v) is dominated
            if (isEdgeDominated(u, v, neighbors, neighbor_sets))
            {
                dominated_edges.push_back({u, v});
            }
            else if (isEdgeDominated(v, u, neighbors, neighbor_sets))
            {
                dominated_edges.push_back({v, u});
            }
        }
    }

    return dominated_edges;
}

// Collapse an edge and update the graph
// When we collapse (u,v), we remove u and connect all its neighbors to v
void collapseEdge(int u, int v, std::vector<std::vector<int>> &neighbors,
                  std::vector<std::unordered_set<int>> &neighbor_sets, std::vector<bool> &alive,
                  std::vector<std::pair<int, int>> &collapse_sequence)
{
    if (!alive[u])
        return; // Already collapsed

    // Record the collapse
    collapse_sequence.push_back({u, v});

    // Mark u as collapsed
    alive[u] = false;

    // For each neighbor w of u, add edge (v,w) if not exists
    for (int w : neighbors[u])
    {
        if (!alive[w] || w == v)
            continue;

        // Check if edge (v,w) already exists
        if (neighbor_sets[v].find(w) == neighbor_sets[v].end())
        {
            // Add edge (v,w)
            neighbors[v].push_back(w);
            neighbors[w].push_back(v);
            neighbor_sets[v].insert(w);
            neighbor_sets[w].insert(v);
        }
    }

    // Clear neighbors of u
    neighbors[u].clear();
    neighbor_sets[u].clear();
}

} // namespace

// Main API: Apply edge collapse preprocessing
EdgeCollapseResult collapseEdges(const std::vector<std::vector<int>> &input_neighbors,
                                 const std::vector<double> &edge_weights, double max_radius)
{
    if (!std::isfinite(max_radius) || max_radius <= 0.0)
    {
        throw std::invalid_argument("Maximum radius must be finite and greater than 0");
    }
    if (std::any_of(edge_weights.begin(), edge_weights.end(),
                    [](double weight) { return !std::isfinite(weight) || weight < 0.0; }))
    {
        throw std::invalid_argument("Edge weights must be finite and non-negative");
    }
    if (input_neighbors.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        return {};
    }
    int n_vertices = static_cast<int>(input_neighbors.size());

    // Copy input
    std::vector<std::vector<int>> neighbors = input_neighbors;
    std::vector<std::unordered_set<int>> neighbor_sets(n_vertices);
    for (int i = 0; i < n_vertices; ++i)
    {
        neighbor_sets[i].insert(neighbors[i].begin(), neighbors[i].end());
    }

    std::vector<bool> alive(n_vertices, true);
    std::vector<std::pair<int, int>> collapse_sequence;
    collapse_sequence.reserve(n_vertices / 2);

    // Iteratively collapse dominated edges
    int iterations = 0;
    const int max_iterations = n_vertices; // Safety limit

    while (iterations < max_iterations)
    {
        auto dominated = findDominatedEdges(neighbors, n_vertices);

        if (dominated.empty())
            break;

        // Collapse all found edges
        for (const auto &[u, v] : dominated)
        {
            if (alive[u] && alive[v])
            {
                collapseEdge(u, v, neighbors, neighbor_sets, alive, collapse_sequence);
            }
        }

        ++iterations;
    }

    // Build result
    EdgeCollapseResult result;
    result.collapse_sequence = std::move(collapse_sequence);

    // Count remaining vertices and edges
    result.remaining_vertices = 0;
    result.remaining_edges = 0;

    for (int i = 0; i < n_vertices; ++i)
    {
        if (alive[i])
        {
            ++result.remaining_vertices;
            // Count edges to other alive vertices
            for (int v : neighbors[i])
            {
                if (alive[v] && v > i)
                {
                    ++result.remaining_edges;
                }
            }
        }
    }

    result.original_vertices = n_vertices;
    result.original_edges =
        static_cast<int>(std::count_if(edge_weights.begin(), edge_weights.end(),
                                       [max_radius](double w) { return w <= max_radius; }));
    result.vertex_alive = std::move(alive);
    result.reduced_neighbors = std::move(neighbors);

    return result;
}

// Statistics about the collapse
EdgeCollapseStats analyzeCollapse(const EdgeCollapseResult &result)
{
    EdgeCollapseStats stats;

    if (result.original_vertices > 0)
    {
        stats.vertex_reduction_ratio =
            std::clamp(1.0 - static_cast<double>(result.remaining_vertices) /
                                 static_cast<double>(result.original_vertices),
                       0.0, 1.0);
    }
    if (result.original_edges > 0)
    {
        stats.edge_reduction_ratio =
            std::clamp(1.0 - static_cast<double>(result.remaining_edges) /
                                 static_cast<double>(result.original_edges),
                       0.0, 1.0);
    }
    stats.num_collapses = result.collapse_sequence.size();

    // Estimate simplex reduction (rough approximation)
    // In VR complex, simplices grow exponentially with edge count
    // Reducing edges by x% reduces simplices by roughly (1-x%)^dim
    stats.estimated_simplex_reduction =
        1.0 - std::pow(1.0 - stats.edge_reduction_ratio, 3.0); // For dim 2

    return stats;
}

// Deterministic CPU compatibility entry point for accelerated callers.
EdgeCollapseResult collapseEdgesGPU(const std::vector<std::vector<int>> &input_neighbors,
                                    const std::vector<double> &edge_weights, double max_radius)
{
    return collapseEdges(input_neighbors, edge_weights, max_radius);
}

bool shouldUseEdgeCollapse(size_t num_vertices, size_t num_edges, double density)
{
    if (num_vertices < 100 || num_edges < num_vertices || !std::isfinite(density) ||
        density < 0.0 || density > 1.0)
    {
        return false;
    }

    const long double max_edges =
        static_cast<long double>(num_vertices) * static_cast<long double>(num_vertices - 1) * 0.5L;
    if (max_edges <= 0.0L || static_cast<long double>(num_edges) > max_edges)
    {
        return false;
    }

    const double actual_density =
        static_cast<double>(static_cast<long double>(num_edges) / max_edges);
    const double effective_density = std::min(density, actual_density);
    return effective_density > 0.30;
}

} // namespace nerve::persistence

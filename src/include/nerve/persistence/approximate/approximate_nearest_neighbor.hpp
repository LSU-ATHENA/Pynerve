
#pragma once

#include "nerve/core.hpp"
#include "nerve/persistence/core/core_types.hpp"

#include <utility>
#include <vector>

namespace nerve
{
namespace persistence
{

/**
 * @brief Configuration for ANN-based edge detection
 */
struct ANNConfig
{
    // Recall target: 0.95 = 95% of true neighbors found
    // Higher = more accurate but slower
    double recall_target = 0.95;

    // HNSW construction parameters
    int ef_construction = 150; // Size of dynamic candidate list
    int ef_search = 50;        // Size during search
};

/**
 * @brief Result of ANN-based edge detection
 */
struct ANNResult
{
    std::vector<std::pair<int, int>> edges;
    std::vector<double> edge_weights;
    size_t num_edges = 0;

    // Timing
    double index_build_time_ms = 0.0;
    double search_time_ms = 0.0;
    double total_time_ms = 0.0;

    // Statistics
    double max_radius = 0.0;
    double recall_target = 0.0;
    double estimated_speedup = 1.0; // vs brute force
};

/**
 * @brief ANN speedup estimate
 */
struct ANNSpeedupEstimate
{
    double theoretical_speedup = 1.0;
    double expected_speedup = 1.0;
    double estimated_recall = 0.0;
    bool recommended = false;
};

/**
 * @brief Approximate Nearest Neighbor for Fast Edge Detection
 *
 * Uses HNSW (Hierarchical Navigable Small World) algorithm for O(log n)
 * approximate nearest neighbor search instead of O(n^2) brute force.
 *
 * Key characteristics:
 * - HNSW graph provides approximate nearest-neighbor search with logarithmic behavior
 * - Recalls and runtime vary with graph parameters and dataset geometry
 *
 * Algorithm (HNSW):
 * - Build hierarchical proximity graph
 * - Greedy search from enter point at each level
 * - Multi-layer structure enables long-range + short-range navigation
 *
 * Complexity:
 * - Build: approximately O(n log n)
 * - Search: approximately logarithmic per query
 *
 * Tradeoff:
 * Lower recall targets reduce search effort but can miss candidate edges.
 *
 * Best For:
 * - High-dimensional point clouds (dim >= 10)
 * - 10K-1M points where O(n^2) is bottleneck
 * - When 90-98% recall is acceptable
 *
 * Note:
 * For exact computation, combine with brute-force verification of ANN results.
 *
 * @param points Point coordinates
 * @param point_dim Dimension of each point
 * @param num_points Number of points
 * @param max_radius Maximum distance for edge creation
 * @param config ANN configuration
 * @return Edge list with weights
 */
ANNResult fastEdgeDetectionANN(const std::vector<double> &points, size_t point_dim,
                               size_t num_points, double max_radius, const ANNConfig &config);

/**
 * @brief Get optimal ANN configuration
 */
ANNConfig getOptimalANNConfig(size_t num_points, size_t point_dim);

/**
 * @brief Estimate speedup vs brute force
 */
ANNSpeedupEstimate estimateANNSpeedup(size_t num_points, size_t point_dim);

/**
 * @brief Check if ANN should be used
 */
inline bool shouldUseANN(size_t num_points, size_t point_dim)
{
    // ANN beneficial for larger datasets
    // Overhead not worth it for small datasets
    return num_points >= 5000 || (num_points >= 1000 && point_dim >= 10);
}

} // namespace persistence
} // namespace nerve

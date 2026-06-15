
#pragma once

#include "nerve/core.hpp"

#include <chrono>
#include <cmath>
#include <vector>

namespace nerve::persistence::reduced
{

/**
 * @brief Configuration for reduced VR H1 computation
 */
struct ReducedVRH1Config
{
    double max_radius = 1.0;              // Maximum filtration radius
    bool use_cycle_filter = true;         // Filter edges that can't form cycles
    bool use_connectivity_pruning = true; // Prune based on connectivity
    bool use_triangle_filter = true;      // Filter triangles
    bool preserve_connectivity = true;    // Don't disconnect components
    int num_threads = 0;                  // 0 = auto
};

/**
 * @brief Point structure
 */
struct Point
{
    int id;
    std::vector<double> coords;

    double distance(const Point &other) const
    {
        double sum = 0.0;
        for (size_t i = 0; i < coords.size() && i < other.coords.size(); ++i)
        {
            double diff = coords[i] - other.coords[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }
};

/**
 * @brief Edge in VR complex
 */
struct Edge
{
    int v1, v2;
    double weight;
};

/**
 * @brief Triangle in VR complex
 */
struct Triangle
{
    int v1, v2, v3;
    double weight;
};

/**
 * @brief H1 persistence pair
 */
struct H1Pair
{
    int birth_index;
    int death_index;
    double birth_time;
    double death_time;
};

/**
 * @brief Result of reduced VR construction
 */
struct ReducedVRH1Result
{
    std::vector<Edge> edges;
    std::vector<Triangle> triangles;
    ReducedVRH1Config config;

    int original_edge_count;
    int reduced_edge_count;
    int triangle_count;
    int filtered_triangle_count;
    double edge_reduction_ratio;

    // Timing
    double edge_build_time_ms;
    double prune_time_ms;
    double triangle_build_time_ms;
    double total_time_ms;
};

/**
 * @brief Result of H1 computation
 */
struct H1Result
{
    std::vector<H1Pair> pairs;
    double computation_time_ms;
};

/**
 * @brief Speedup estimate
 */
struct ReducedVRSpeedup
{
    double edge_reduction;
    double memory_reduction;
    double computation_speedup;
    double total_speedup;
};

/**
 * @brief Reduced Vietoris-Rips Filtration for H1
 *
 * **2-3x FEWER EDGES FOR H1 COMPUTATION**
 *
 * Based on arXiv:2307.16333 (2023):
 * "Faster computation of degree-1 persistent homology using the
 *  reduced Vietoris-Rips filtration"
 *
 * Key Innovation:
 * Exploits geometric structure to build a smaller filtration specifically
 * for computing H1 (1-dimensional homology). Many edges cannot participate
 * in H1 features and can be safely pruned.
 *
 * Pruning Strategies:
 * - **Cycle Filter**: Remove edges that cannot form part of any cycle
 *    (need at least 2 paths between endpoints)
 * - **Connectivity Pruning**: Keep only edges that improve connectivity
 *    or are part of cycles
 * - **Triangle Filter**: Skip triangles whose edges were pruned
 *
 * H1-Specific Insight:
 * - H1 features are born when cycles appear (edges)
 * - H1 features die when cycles are filled (triangles)
 * - Edges that never participate in cycles can be pruned
 * - Triangles with pruned edges can be skipped
 *
 * Effects:
 * - Reduces edge and triangle candidates before H1 reduction.
 * - Decreases memory pressure for large neighborhoods.
 * - Preserves H1 correctness under the pruning criteria implemented here.
 *
 * Best For:
 * - H1 computation specifically
 * - Large point clouds with many edges
 * - Low-dimensional Euclidean spaces
 *
 * Integration:
 * Use as preprocessing step before standard H1 reduction. Combine with
 * cohomology, clearing, and bit-parallel for maximum performance.
 *
 * References:
 * - arXiv:2307.16333 - Reduced VR for H1 (2023)
 */

/**
 * @brief Build reduced VR complex optimized for H1
 *
 * Constructs smaller complex by pruning edges that cannot contribute to H1.
 *
 * @param points_data Point cloud data (flattened)
 * @param point_dim Dimension of each point
 * @param num_points Number of points
 * @param config Configuration options
 * @return Reduced complex with edges and triangles
 */
ReducedVRH1Result buildReducedVRForH1(const std::vector<double> &points_data, size_t point_dim,
                                      size_t num_points, const ReducedVRH1Config &config);

/**
 * @brief Compute H1 using reduced VR complex
 *
 * Standard persistence computation on the reduced complex.
 *
 * @param reduced_vr Reduced complex to process
 * @param config Configuration options
 * @return H1 persistence pairs
 */
H1Result computeH1ReducedVR(const ReducedVRH1Result &reduced_vr, const ReducedVRH1Config &config);

/**
 * @brief Get optimal configuration
 *
 * @param num_points Number of points
 * @param max_radius Maximum filtration radius
 * @return Optimized configuration
 */
ReducedVRH1Config getOptimalReducedVRH1Config(size_t num_points, double max_radius);

/**
 * @brief Estimate speedup for given problem
 *
 * @param num_points Number of points
 * @param max_radius Filtration radius
 * @return Speedup estimates
 */
ReducedVRSpeedup estimateReducedVRSpeedup(size_t num_points, double max_radius);

/**
 * @brief Check if reduced VR should be used for H1
 *
 * @param num_points Number of points
 * @return True if reduction is recommended
 */
inline bool shouldUseReducedVRH1(size_t num_points)
{
    return num_points > 500; // Benefit increases with size
}

} // namespace nerve::persistence::reduced

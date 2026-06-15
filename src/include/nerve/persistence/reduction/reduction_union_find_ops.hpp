
#pragma once

#include "nerve/core.hpp"
#include "nerve/persistence/core/core_types.hpp"

#include <chrono>
#include <utility>
#include <vector>

namespace nerve::persistence
{

/**
 * @brief Result of fast Union-Find based persistence computation
 *
 * The "sandwich" approach: fast D0 via Union-Find and graph-cycle
 * extraction from the 1-skeleton, with optional higher-dimensional
 * passes supplied by other reduction backends.
 */
struct FastPersistenceResult
{
    std::vector<Pair> d0_pairs;      // Dimension 0 (connected components)
    std::vector<Pair> top_dim_pairs; // Top-dimensional pairs when derivable from input
    std::vector<Pair> middle_pairs;  // Intermediate-dimension pairs

    double d0_time_ms;
    double top_dim_time_ms;
    double middle_time_ms;
    size_t total_pairs;
};

/**
 * @brief Statistics about Union-Find computation
 */
struct UnionFindStats
{
    std::string theoretical_complexity; // "O(n alpha(n))"
    int inverse_ackermann_estimate;     // Always < 5 for practical n
    double estimated_speedup_vs_matrix; // vs O(n^3) matrix reduction
    size_t num_find_operations;
    size_t num_unite_operations;
};

/**
 * @brief Fast D0 persistence using Union-Find
 *
 * Computes 0-dimensional persistence (connected components) in O(n alpha(n))
 * time instead of O(n^3) for matrix reduction.
 *
 * Algorithm:
 * - Sort edges by filtration value
 * - Process edges in order, maintaining Union-Find structure
 * - When two components merge, the younger component dies
 * - Remaining components at end have infinite persistence
 *
 * This is the standard persistence algorithm for 0D, but incredibly fast
 * due to Union-Find's near-constant time operations.
 *
 * @param points Point coordinates (not used for computation, for API consistency)
 * @param point_dim Dimension of each point
 * @param num_points Number of points
 * @param edges List of edges (from, to); invalid/out-of-range edges are ignored
 * @param edge_weights Weight (distance) for each edge
 * @return Persistence pairs for dimension 0
 */
std::vector<Pair> computeD0PersistenceUnionFind(const std::vector<double> &points, size_t point_dim,
                                                size_t num_points,
                                                const std::vector<std::pair<int, int>> &edges,
                                                const std::vector<double> &edge_weights);

/**
 * @brief Fast computation of top-dimensional persistence (voids)
 *
 * Builds a closure-complete simplicial complex from the provided top simplices
 * and computes exact Z2 persistence, then filters the requested dimension.
 *
 * @param points Point coordinates
 * @param point_dim Dimension of space
 * @param num_points Number of points
 * @param simplices d-dimensional simplices
 * @param simplex_weights Filtration values
 * @param dim Dimension to compute (should be point_dim - 1)
 * @return Persistence pairs for dimension dim
 */
std::vector<Pair> computeTopDimensionalPersistence(const std::vector<double> &points,
                                                   size_t point_dim, size_t num_points,
                                                   const std::vector<std::vector<int>> &simplices,
                                                   const std::vector<double> &simplex_weights,
                                                   Dimension dim);

/**
 * @brief Combined fast computation using Union-Find sandwich
 *
 * Computes D0 via Union-Find and, when available from the 1-skeleton,
 * extracts H1 cycle births as infinite classes. This function does not
 * synthesize unsupported higher-dimensional classes.
 *
 * @param points Point coordinates
 * @param point_dim Dimension of space
 * @param num_points Number of points
 * @param edges Edges in 1-skeleton
 * @param edge_weights Edge distances
 * @param max_dim Maximum homology dimension to compute
 * @return Persistence pairs for all dimensions
 */
FastPersistenceResult computeFastPersistenceSandwich(const std::vector<double> &points,
                                                     size_t point_dim, size_t num_points,
                                                     const std::vector<std::pair<int, int>> &edges,
                                                     const std::vector<double> &edge_weights,
                                                     Dimension max_dim);

/**
 * @brief Get statistics about Union-Find computation
 */
UnionFindStats getUnionFindStats(size_t num_points, size_t num_edges, double computation_time_ms);

} // namespace nerve::persistence

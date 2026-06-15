
#pragma once

#include "nerve/core.hpp"
#include "nerve/persistence/core/core_types.hpp"

#include <cstddef>
#include <vector>

namespace nerve::persistence
{

/**
 * @brief Configuration for Sparse Rips
 */
struct SparseRipsConfig
{
    Size max_dim = 2;
    double max_radius = 1.0;

    // Approximation parameter
    // epsilon = 0.1 gives (1.1)-approximation
    // epsilon = 0.3 gives (1.3)-approximation (faster, less accurate)
    double epsilon = 0.2;
};

/**
 * @brief Result of Sparse Rips computation
 */
struct SparseRipsResult
{
    std::vector<Pair> pairs;

    // Timing
    double permutation_time_ms = 0.0;
    double build_time_ms = 0.0;
    double persistence_time_ms = 0.0;
    double total_time_ms = 0.0;

    // Statistics
    double epsilon = 0.0;
    size_t original_points = 0;
    size_t net_size = 0;
    size_t num_simplices = 0;
    double sparse_ratio = 0.0;         // sparse / dense
    double compression_ratio = 0.0;    // 1 - sparse_ratio
    double approximation_factor = 1.0; // 1 + epsilon
    double theoretical_error_bound = 0.0;
};

/**
 * @brief Estimated memory/time savings from sparse Rips
 */
struct SparseRipsSavings
{
    double dense_simplices = 0.0;
    double sparse_simplices = 0.0;
    double memory_reduction_ratio = 0.0;
    double expected_speedup = 1.0;
    bool recommended = false;
};

/**
 * @brief Sparse Rips Filtration - Linear Size Approximation
 *
 * Based on Sheehy et al.: "Linear-Size Approximations to the Vietoris-Rips Filtration"
 * Provides (1+epsilon)-approximation with O(n) simplices instead of O(2^n).
 *
 * Key Innovation:
 * Uses greedy permutation (farthest-point sampling) to build a hierarchical net.
 * Only adds simplices when points are "close enough" relative to their insertion time.
 *
 * Theoretical Guarantees:
 * - (1+epsilon)-approximation of true persistence diagram
 * - Linear number of simplices: O(n) vs O(2^n)
 * - For fixed epsilon, size is independent of ambient dimension
 *
 * Best For:
 * - Massive point clouds (100K-10M points)
 * - When exact computation is too slow
 * - Applications accepting (1+epsilon)-approximation
 *
 * Tradeoffs:
 * - epsilon = 0.1: lower approximation error, higher compute cost
 * - epsilon = 0.3: moderate approximation error, lower compute cost
 * - epsilon = 0.5: higher approximation error, lowest compute cost
 *
 * @param points Point coordinates
 * @param point_dim Dimension of each point
 * @param num_points Number of points
 * @param config Sparse Rips configuration
 * @return Sparse Rips result with approximation guarantees
 */
SparseRipsResult computeSparseRips(const std::vector<double> &points, size_t point_dim,
                                   size_t num_points, const SparseRipsConfig &config);

/**
 * @brief Get optimal sparse Rips configuration
 */
SparseRipsConfig getOptimalSparseRipsConfig(size_t num_points, size_t point_dim);

/**
 * @brief Estimate memory/time savings
 */
SparseRipsSavings estimateSparseRipsSavings(size_t num_points, double epsilon);

/**
 * @brief Check if sparse Rips should be used
 */
bool shouldUseSparseRips(size_t num_points, double epsilon);

/**
 * @brief Get recommended epsilon for desired speedup
 */
inline double epsilonForSpeedup(double desired_speedup)
{
    // Heuristic: higher epsilon = more speedup
    if (desired_speedup >= 100.0)
        return 0.5;
    if (desired_speedup >= 10.0)
        return 0.3;
    if (desired_speedup >= 2.0)
        return 0.2;
    return 0.1;
}

} // namespace nerve::persistence

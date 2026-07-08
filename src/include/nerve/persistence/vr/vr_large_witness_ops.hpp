
#pragma once

#include "nerve/core.hpp"
#include "nerve/persistence/core/core_types.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"

namespace nerve::persistence
{

/**
 * @brief Landmark selection strategies for witness complex
 */
enum class LandmarkSelectionStrategy
{
    MAXMIN, // epsilon-net via farthest point sampling (best approximation)
    RANDOM, // Random subset (fastest)
    GRID,   // Grid-based for uniform distributions
    DENSITY // Density-weighted for non-uniform data
};

/**
 * @brief Configuration for witness complex computation
 */
struct WitnessComplexConfig
{
    size_t num_landmarks = 0; // 0 = auto-select
    LandmarkSelectionStrategy strategy = LandmarkSelectionStrategy::MAXMIN;
    double approximation_factor = 3.0; // Theoretical 3-approximation of VR
    double max_witness_distance = 0.0; // 0 = auto-compute
};

/**
 * @brief Optimized VR computation for large point sets (> 10K points)
 *
 * Uses lazy witness complex approximation with epsilon-net landmarks:
 * - Theoretical 3-approximation of true VR persistence
 * - Reduces O(n^2) complexity to O(m^2) where m << n
 * - Memory-efficient: only stores landmark-to-landmark edges
 *
 * Landmark selection algorithms:
 * - MAXMIN: Creates epsilon-net with optimal coverage (recommended)
 * - RANDOM: Fastest but less accurate
 * - DENSITY: Good for non-uniform point distributions
 *
 * Key optimizations:
 * - epsilon-net landmark selection (maxmin greedy algorithm)
 * - Lazy witness complex construction
 * - Parallel distance computations
 * - Memory-efficient representation
 *
 * Performance characteristics:
 * - 10K-50K points: < 10s with 5-10% landmarks
 * - 50K-100K points: < 30s with 5% landmarks
 * - > 100K points: Streaming mode with chunking
 *
 * Approximation bounds:
 * - Persistence diagram is 3-approximation of true VR diagram
 * - Birth/death times within factor 3 of true values
 * - Topological features preserved with high probability
 *
 * @param points Flattened point coordinates [n_points * point_dim]
 * @param point_dim Dimension of each point
 * @param config VR computation configuration
 * @param num_landmarks Number of landmarks (0 = auto-select)
 * @return Vector of persistence pairs (approximate)
 */
std::vector<Pair> computeVrPersistenceLargeWitness(core::BufferView<const double>points,
                                                   Size point_dim, const VRConfig &config,
                                                   size_t num_landmarks = 0);

/**
 * @brief Get optimal witness complex configuration for problem size
 */
WitnessComplexConfig getOptimalWitnessConfig(size_t num_points, size_t point_dim);

/**
 * @brief Compute theoretical approximation bounds
 */
struct ApproximationBounds
{
    double birth_factor;       // Multiplicative error in birth times
    double death_factor;       // Multiplicative error in death times
    double persistence_factor; // Multiplicative error in persistence
    size_t num_landmarks_used;
    double estimated_coverage; // Fraction of space covered by landmarks
};

ApproximationBounds computeWitnessApproximationBounds(size_t num_points, size_t num_landmarks,
                                                      double max_radius);

} // namespace nerve::persistence

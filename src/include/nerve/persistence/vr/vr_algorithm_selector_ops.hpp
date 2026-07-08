
#pragma once

#include "nerve/core.hpp"
#include "nerve/persistence/core/core_types.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"
#include "nerve/persistence/vr/vr_large_witness_ops.hpp"

#include <string>
#include <vector>

namespace nerve::persistence
{

/**
 * @brief Available VR computation algorithms
 */
enum class VRAlgorithm
{
    AUTO,          // Automatically select based on problem size
    FAST_SIMD,     // Small exact fast path; uses SIMD kernels when available
    MEDIUM_HYBRID, // Medium-scale tiled/parallel path for 1K-10K points
    LARGE_WITNESS, // Witness complex for >10K points
    EXACT_STANDARD // Standard exact computation (exact path)
};

/**
 * @brief Algorithm recommendation with performance estimates
 */
struct AlgorithmRecommendation
{
    VRAlgorithm recommended;
    std::string description;
    size_t problem_size;
    size_t point_dim;
    double estimated_time_seconds;
    size_t memory_estimate_mb;
    double approximation_factor; // 1.0 = exact, 3.0 = 3-approximation
};

/**
 * @brief Benchmark results for algorithm comparison
 */
struct AlgorithmBenchmark
{
    VRAlgorithm algorithm = VRAlgorithm::AUTO;
    double time_ms = 0.0;
    size_t num_pairs = 0;
    bool success = false;
    double approximation_error = -1.0;
};

/**
 * @brief Automatically select and execute optimal VR algorithm
 *
 * Analyzes problem characteristics and selects the fastest algorithm:
 * - < 1K points and dimensions <= 16: FAST_SIMD small exact path
 * - 1K-10K points: MEDIUM_HYBRID (tiled + parallel clique expansion)
 * - > 10K points: LARGE_WITNESS (epsilon-net approximation)
 *
 * Considers:
 * - Problem size (number of points)
 * - Point dimension
 * - CPU vector-kernel availability for the small fast path
 * - Memory estimates
 * - Time budget
 * - Exactness requirements
 *
 * @param points Flattened point coordinates [n_points * point_dim]
 * @param point_dim Dimension of each point
 * @param base_config VR configuration (may be modified based on selection)
 * @return Vector of persistence pairs
 */
std::vector<Pair> computeVrPersistenceAuto(core::BufferView<const double>points,
                                           Size point_dim, const VRConfig &base_config);

/**
 * @brief Execute specific VR algorithm
 *
 * Allows explicit algorithm selection with automatic routing
 * if problem size is outside recommended range.
 *
 * @param points Flattened point coordinates
 * @param point_dim Dimension of each point
 * @param config VR configuration
 * @param algorithm Specific algorithm to use
 * @return Vector of persistence pairs
 */
std::vector<Pair> computeVrPersistenceWithAlgorithm(core::BufferView<const double>points,
                                                    Size point_dim, const VRConfig &config,
                                                    VRAlgorithm algorithm);

/**
 * @brief Get algorithm recommendation for problem characteristics
 *
 * Provides detailed recommendation without executing computation.
 * Useful for previewing algorithm choice and estimated performance.
 *
 * @param num_points Number of points in dataset
 * @param point_dim Dimension of each point
 * @param max_radius Maximum VR radius
 * @param require_exact If true, never recommend approximate methods
 * @param time_budget_seconds Maximum allowed computation time (0 = unlimited)
 * @return Algorithm recommendation with performance estimates
 */
AlgorithmRecommendation recommendAlgorithm(size_t num_points, size_t point_dim, double max_radius,
                                           bool require_exact = true,
                                           double time_budget_seconds = 0.0);

/**
 * @brief Estimate memory usage for VR computation
 *
 * Provides conservative memory estimate to help with resource planning.
 *
 * @param num_points Number of points
 * @param point_dim Dimension of each point
 * @param max_radius Maximum VR radius (affects simplex count)
 * @return Estimated memory usage in megabytes
 */
size_t estimateMemoryUsage(size_t num_points, size_t point_dim, double max_radius);

/**
 * @brief Benchmark all algorithms on a dataset
 *
 * Runs all available VR algorithms and compares performance.
 * Useful for algorithm selection validation and tuning.
 *
 * @param points Flattened point coordinates
 * @param point_dim Dimension of each point
 * @param config Base VR configuration
 * @return Benchmark results for each algorithm
 */
std::vector<AlgorithmBenchmark> benchmarkAllAlgorithms(core::BufferView<const double>points,
                                                       Size point_dim, const VRConfig &config);

/**
 * @brief Check if AVX-512 is available on this CPU
 */
bool isAvx512Available();

/**
 * @brief Convert algorithm enum to string
 */
inline const char *algorithmToString(VRAlgorithm algo)
{
    switch (algo)
    {
        case VRAlgorithm::FAST_SIMD:
            return "FAST_SIMD";
        case VRAlgorithm::MEDIUM_HYBRID:
            return "MEDIUM_HYBRID";
        case VRAlgorithm::LARGE_WITNESS:
            return "LARGE_WITNESS";
        case VRAlgorithm::EXACT_STANDARD:
            return "EXACT_STANDARD";
        case VRAlgorithm::AUTO:
            return "AUTO";
        default:
            return "UNKNOWN";
    }
}

} // namespace nerve::persistence


#pragma once

#include "nerve/core.hpp"
#include "nerve/persistence/cohomology/cohomology_persistent_ops.hpp"
#include "nerve/persistence/core/core_types.hpp"

#include <unordered_map>
#include <vector>

namespace nerve::persistence
{

/**
 * @brief Configuration for involuted homology
 */
struct InvolutedConfig
{
    int max_dim = 6;
    bool use_involution = true;
    int involution_threshold_dim = 3; // Use involution for H3+
};

/**
 * @brief Result of involuted homology computation
 */
struct InvolutedResult
{
    std::vector<Pair> all_pairs;
    std::unordered_map<int, std::vector<Pair>> pairs_by_dimension;
    std::unordered_map<int, double> dimension_times_ms;

    double total_time_ms = 0.0;
    int max_dim = 0;
    int num_simplices = 0;
    bool used_involution = false;
    int involution_threshold = 0;
};

/**
 * @brief Speedup estimate from involution
 */
struct InvolutedSpeedupEstimate
{
    double involution_speedup = 1.0;        // From involution structure
    double total_speedup_vs_homology = 1.0; // Combined with cohomology
    double memory_reduction = 0.0;
};

/**
 * @brief Involuted Persistent Homology
 *
 * Uses an involution-style duality mapping for high-dimensional reduction
 * paths (typically H3-H6 in this codebase).
 *
 * Mathematical Basis:
 * - Involution: An operator J such that J^2 = id
 * - In persistence: J maps chains to cochains and vice versa
 * - Enables fast conversion between homology and cohomology perspectives
 *
 * Hybrid Strategy (Recommended):
 * - H0-H2: Cohomology with clearing
 * - H3-H6: Involuted homology
 *
 * References:
 * - Research 2021-2022 on involuted persistence (building on UCTP)
 * - Related to "Fast computation of persistent homology representatives"
 *
 * @param simplices List of simplices
 * @param filtration_values Filtration values
 * @param dimensions Simplex dimensions
 * @param max_dim Maximum dimension (0-6)
 * @param config Involuted configuration
 * @return Involuted computation result
 */
InvolutedResult computeInvolutedHomology(const std::vector<std::vector<int>> &simplices,
                                         const std::vector<double> &filtration_values,
                                         const std::vector<int> &dimensions, int max_dim,
                                         const InvolutedConfig &config);

/**
 * @brief Compute dual index for involution
 */
int computeDualIndex(const struct InvolutedSimplex &simplex,
                     const std::vector<struct InvolutedSimplex> &all_simplices);

/**
 * @brief Get optimal involuted configuration
 */
InvolutedConfig getOptimalInvolutedConfig(int max_dim, size_t num_simplices);

/**
 * @brief Estimate speedup from involution
 */
InvolutedSpeedupEstimate estimateInvolutedSpeedup(int max_dim, size_t num_high_dim_simplices);

/**
 * @brief Check if involution should be used
 */
inline bool shouldUseInvolution(int max_dim)
{
    return max_dim >= 3;
}

} // namespace nerve::persistence

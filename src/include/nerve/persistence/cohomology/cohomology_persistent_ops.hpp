
#pragma once

#include "nerve/core.hpp"
#include "nerve/persistence/cohomology/persistent_cohomology.hpp"
#include "nerve/persistence/core/core_types.hpp"

#include <chrono>
#include <vector>

namespace nerve::persistence
{

/**
 * @brief Persistent Cohomology with Clearing Optimization
 *
 * Key Innovations:
 * - Cohomology: Work with coboundaries (cofaces) instead of boundaries
 *    - Natural for VR filtrations
 *    - Smaller matrices in practice
 *
 * - Clearing Optimization: Skip columns whose birth is already known
 *    - When a simplex creates a feature, it can't create another
 *    - Dramatically reduces columns to process
 *
 * - Apparent Pairs: Identify (birth, death) pairs without reduction
 *    - Youngest cofacet + oldest facet = apparent pair
 *    - Skip these columns entirely
 *
 * - Dimension Cascade: Clear H0 -> use for H1 -> use for H2 -> ... -> H6
 *    - Each dimension benefits from previous clearing
 *    - Essential for 0-6D computation
 *
 * Mathematical Basis:
 * - Universal Coefficient Theorem: H^*(X; Z2)  ~=  Hom(H_*(X; Z2), Z2)
 * - For Z2 coefficients, persistent cohomology and homology give same barcodes
 * - Cohomology clearing corresponds to homology co-clearing
 *
 * References:
 * - Bauer: "fast: efficient computation..." (uses cohomology + clearing)
 * - Chen & Kerber: "Persistent homology computation..." (clearing)
 *
 * @param simplices List of simplices (vertex indices)
 * @param filtration_values Filtration value for each simplex
 * @param dimensions Dimension of each simplex
 * @param max_dim Maximum homology dimension to compute
 * @param config Cohomology configuration
 * @return Cohomology computation result
 */
CohomologyResult computePersistentCohomology(const std::vector<std::vector<int>> &simplices,
                                             const std::vector<double> &filtration_values,
                                             const std::vector<int> &dimensions, int max_dim,
                                             const CohomologyConfig &config);

/**
 * @brief Extract homology barcodes from cohomology result
 *
 * For Z2 coefficients, the barcodes are identical.
 */
std::vector<Pair> extractHomologyFromCohomology(const CohomologyResult &result);

/**
 * @brief Get optimal cohomology configuration
 */
CohomologyConfig getOptimalCohomologyConfig(int max_dim, size_t num_simplices);

/**
 * @brief Estimate speedup vs homology
 */
CohomologySpeedupEstimate estimateCohomologySpeedup(int max_dim, size_t num_simplices,
                                                    bool use_clearing, bool use_apparent_pairs);

/**
 * @brief Check if cohomology should be used
 */
bool shouldUseCohomology(int max_dim, size_t num_simplices, bool require_representatives);

/**
 * @brief Hybrid: Cohomology for H0-H2, Involuted for H3-H6
 *
 * Automatically selects best algorithm per dimension range.
 */
CohomologyResult computeHybridCohomologyInvoluted(const std::vector<std::vector<int>> &simplices,
                                                  const std::vector<double> &filtration_values,
                                                  const std::vector<int> &dimensions, int max_dim);

} // namespace nerve::persistence

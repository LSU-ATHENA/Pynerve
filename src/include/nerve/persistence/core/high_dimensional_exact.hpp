
#pragma once

/**
 * @brief High-dimensional exact persistent homology (0-6D)
 *
 * Combines exact algorithms for dimensions 0-6:
 * - Cohomology + Clearing for H0-H2
 * - Involuted Homology for H3-H6
 * - Dimension-cascading optimization
 * - Apparent pairs detection
 *
 * The implementation is organized by dimension-specific exact kernels.
 */

#include "nerve/persistence/cohomology/cohomology_involuted_ops.hpp"
#include "nerve/persistence/cohomology/cohomology_persistent_ops.hpp"
#include "nerve/persistence/core/core_types.hpp"

namespace nerve::persistence
{

/**
 * @brief High-dimensional exact computation configuration
 */
struct HighDimensionalExactConfig
{
    int max_dim = 6;                // Compute H0 through H6
    bool use_cohomology = true;     // Cohomology for H0-H2
    bool use_involution = true;     // Involution for H3-H6
    bool use_clearing = true;       // Dimension cascade
    bool use_apparent_pairs = true; // Skip unnecessary columns

    // Thresholds
    int involution_threshold = 3; // H3+ uses involution
    int cohomology_max_dim = 2;   // H0-H2 uses cohomology
};

/**
 * @brief Result of high-dimensional exact computation
 */
struct HighDimensionalExactResult
{
    std::vector<Pair> pairs;    // All persistence pairs
    std::vector<Pair> pairs_h0; // 0-dimensional
    std::vector<Pair> pairs_h1; // 1-dimensional
    std::vector<Pair> pairs_h2; // 2-dimensional
    std::vector<Pair> pairs_h3; // 3-dimensional
    std::vector<Pair> pairs_h4; // 4-dimensional
    std::vector<Pair> pairs_h5; // 5-dimensional
    std::vector<Pair> pairs_h6; // 6-dimensional

    // Timing
    double total_time_ms;
    double time_h0_h2_ms; // Cohomology time
    double time_h3_h6_ms; // Involuted time

    // Statistics
    int num_simplices;
    int num_columns_cleared;
    int num_apparent_pairs;

    // Algorithm info
    bool used_cohomology;
    bool used_involution;
    bool used_clearing;
};

/**
 * @brief Unified high-dimensional exact computation
 *
 * Automatically selects and combines the fastest exact algorithms:
 *
 * Algorithm Selection:
 * - H0: Union-Find style reduction path
 * - H1-H2: Cohomology-oriented reduction path
 * - H3-H6: Involution-capable high-dimensional path when enabled
 *
 * Optimizations:
 * - Dimension Cascade: H0 clears -> H1 uses it -> H2 uses it -> ... -> H6
 * - Apparent Pairs: Identify without matrix reduction
 * - Z2 Arithmetic: Fast XOR operations
 * - CPU exact reduction paths; GPU acceleration is not enabled by default
 *
 * Runtime and memory behavior depend on simplex counts, sparsity,
 * and enabled kernel families.
 *
 * @param simplices Simplicial complex (all dimensions 0-6)
 * @param filtration_values Filtration values
 * @param dimensions Simplex dimensions
 * @param config Exact computation config
 * @return Exact persistence pairs for all dimensions
 */
HighDimensionalExactResult computeHighDimensionalExact(
    const std::vector<std::vector<int>> &simplices, const std::vector<double> &filtration_values,
    const std::vector<int> &dimensions, const HighDimensionalExactConfig &config);

/**
 * @brief Convenience API with automatic configuration
 */
inline HighDimensionalExactResult computeExact0To6D(const std::vector<std::vector<int>> &simplices,
                                                    const std::vector<double> &filtration_values,
                                                    const std::vector<int> &dimensions)
{
    HighDimensionalExactConfig config;
    config.max_dim = 6;
    config.use_cohomology = true;
    config.use_involution = true;
    config.use_clearing = true;
    config.use_apparent_pairs = true;

    return computeHighDimensionalExact(simplices, filtration_values, dimensions, config);
}

/**
 * @brief Get optimal configuration for point cloud size
 */
HighDimensionalExactConfig getOptimalHighDimensionalConfig(size_t num_points, int max_dim);

/**
 * @brief Benchmark result
 */
struct HighDimensionalBenchmark
{
    double homology_time_ms;   // Standard homology
    double cohomology_time_ms; // Cohomology only
    double involuted_time_ms;  // Cohomology + Involution
    double speedup_vs_homology;
    double speedup_vs_cohomology;
};

/**
 * @brief Benchmark all approaches
 */
HighDimensionalBenchmark benchmarkHighDimensional(const std::vector<std::vector<int>> &simplices,
                                                  const std::vector<double> &filtration_values,
                                                  const std::vector<int> &dimensions, int max_dim);

/**
 * @brief Compute dimension-by-dimension breakdown
 */
struct DimensionBreakdown
{
    int dimension;
    int num_simplices;
    int num_pairs;
    double computation_time_ms;
    double percent_of_total;
    std::string algorithm_used;
};

std::vector<DimensionBreakdown> analyzeDimensionBreakdown(const HighDimensionalExactResult &result);

/**
 * @brief Validation: Verify results match homology computation
 */
bool validateExactResults(const HighDimensionalExactResult &result,
                          const std::vector<Pair> &reference_homology);

} // namespace nerve::persistence

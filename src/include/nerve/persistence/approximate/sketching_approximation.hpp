
#pragma once

#include "nerve/core.hpp"
#include "nerve/persistence/core/core_types.hpp"
#include "nerve/persistence/reduction/reduction_union_find_ops.hpp"

#include <chrono>
#include <vector>

namespace nerve::persistence
{

/**
 * @brief Configuration for sketching-based approximation
 */
struct SketchingConfig
{
    Size max_dim = 2;
    double max_radius = 1.0;

    // Edge sampling
    double edge_sampling_rate = 0.2; // 0.2 = keep 20% of edges

    // Dimensionality reduction
    bool use_dimensionality_reduction = true;
    size_t target_dimension = 50; // Project to this dimension

    // Column sampling (for matrix reduction)
    bool use_column_sampling = false;
    double column_sampling_rate = 0.5;

    // Random seed for reproducibility
    unsigned random_seed = 42;
};

/**
 * @brief Result of sketching-based approximation
 */
struct SketchingResult
{
    SketchingConfig config;
    std::vector<Pair> pairs;

    // Timing
    double projection_time_ms = 0.0;
    double build_time_ms = 0.0;
    double persistence_time_ms = 0.0;
    double total_time_ms = 0.0;

    // Statistics
    size_t num_edges_sampled = 0;
    size_t num_edges_total = 0;
    double edge_sampling_ratio = 0.0;

    // Accuracy estimates
    double estimated_accuracy = 0.0;
    double theoretical_error_bound = 1.0;
};

/**
 * @brief Approximation accuracy metrics
 */
struct ApproximationAccuracy
{
    double overall_accuracy = 0.0;       // 0.0 to 1.0
    double distance_preservation = 0.0;  // JL lemma preservation
    double jl_distortion = 0.0;          // Johnson-Lindenstrauss distortion
    double sampling_error = 1.0;         // From edge sampling
    double confidence_interval_95 = 1.0; // Statistical bound
};

/**
 * @brief Probabilistic sketching approximation for persistent homology
 *
 * Uses randomized projection and edge sampling to estimate H0 persistence for
 * exploratory or ML-oriented pipelines. Higher-dimensional persistence is left
 * to exact or dedicated approximate engines.
 *
 * Techniques:
 * - Random Projection (JL transform): Reduce dimensionality
 * - Radius-aware edge sampling: Keep nearby edges with controlled probability
 *
 * Runtime and approximation quality depend on data geometry and the selected
 * sampling and projection parameters.
 *
 * Use Cases:
 * - Machine learning (approximate PH is sufficient)
 * - Large-scale exploratory analysis
 * - Real-time applications
 * - Preprocessing before exact computation
 *
 * Guarantees:
 * - JL Lemma: Distances preserved within (1+/-epsilon)
 * - Chernoff bounds: Sampling error bounded
 * - Confidence intervals: 95% accuracy typical
 *
 * @param points Point coordinates
 * @param point_dim Dimension of each point
 * @param config Sketching configuration
 * @return Approximate H0 persistence pairs with accuracy estimates
 */
SketchingResult computeApproximatePHSketching(core::BufferView<const double> points, Size point_dim,
                                              const SketchingConfig &config);

/**
 * @brief Get optimal sketching configuration
 */
SketchingConfig getOptimalSketchingConfig(size_t num_points, size_t point_dim);

/**
 * @brief Estimate approximation accuracy
 *
 * Provides theoretical bounds on approximation quality based on
 * Johnson-Lindenstrauss lemma and Chernoff bounds.
 */
ApproximationAccuracy estimateApproximationAccuracy(const SketchingConfig &config,
                                                    size_t num_points, size_t point_dim);

/**
 * @brief Check if approximation should be used
 */
inline bool shouldUseApproximation(size_t num_points, bool require_exact)
{
    // Use approximation for large datasets when exact not required
    return !require_exact && num_points >= 10000;
}

/**
 * @brief Hybrid approach: Exact for small, approximate for large
 *
 * Automatically switches between exact and approximate based on
 * problem characteristics and accuracy requirements.
 */
template <typename ExactFunc, typename ApproxFunc>
auto hybridCompute(size_t num_points, bool require_exact, ExactFunc exact_func,
                   ApproxFunc approx_func) -> decltype(exact_func())
{
    if (require_exact || num_points < 5000)
    {
        return exact_func();
    }
    else
    {
        return approx_func();
    }
}

} // namespace nerve::persistence

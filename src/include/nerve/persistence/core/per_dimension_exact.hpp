
#pragma once

#include "nerve/core.hpp"
#include "nerve/persistence/core/core_types.hpp"
#include "nerve/persistence/kernels/kernel_h1_ops.hpp"
#include "nerve/persistence/kernels/kernel_h2_alpha_ops.hpp"
#include "nerve/persistence/kernels/kernel_h3_tetrahedra_ops.hpp"
#include "nerve/persistence/kernels/kernel_h4_chunked_ops.hpp"
#include "nerve/persistence/kernels/kernel_h5_prefetch_ops.hpp"
#include "nerve/persistence/kernels/kernel_h6_streaming_ops.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

namespace nerve::persistence::perdim
{

/**
 * @brief Per-dimension configuration
 *
 * Each dimension gets its own optimized settings
 */
struct PerDimensionConfig
{
    int max_dim = 6;
    double max_radius = 1.0;

    // Which dimensions to compute
    bool compute_h0 = true;
    bool compute_h1 = true;
    bool compute_h2 = true;
    bool compute_h3 = true;
    bool compute_h4 = true;
    bool compute_h5 = true;
    bool compute_h6 = true;

    // Dimension-specific configs
    h1::H1Config h1;
    h2::H2Config h2;
    h3::H3Config h3;
    h4::H4Config h4;
    h5::H5Config h5;
    h6::H6Config h6;
};

/**
 * @brief H0 computation result
 */
struct H0Result
{
    std::vector<PersistencePair> pairs;
    double time_ms = 0.0;
    int num_pairs = 0;
    int essential_count = 0;
};

/**
 * @brief H1 computation result
 */
struct H1Result
{
    std::vector<PersistencePair> pairs;
    double time_ms = 0.0;
    int num_pairs = 0;
    int essential_count = 0;
};

/**
 * @brief H2 computation result
 */
struct H2Result
{
    std::vector<PersistencePair> pairs;
    double time_ms = 0.0;
    int num_pairs = 0;
    int essential_count = 0;
};

/**
 * @brief Per-dimension computation result
 */
struct PerDimensionResult
{
    std::vector<PersistencePair> all_pairs;
    PerDimensionConfig config;

    // Per-dimension timing
    double h0_time_ms = 0.0;
    double h1_time_ms = 0.0;
    double h2_time_ms = 0.0;
    double h3_time_ms = 0.0;
    double h4_time_ms = 0.0;
    double h5_time_ms = 0.0;
    double h6_time_ms = 0.0;
    double total_time_ms = 0.0;

    // Per-dimension pair counts
    int h0_pairs = 0;
    int h1_pairs = 0;
    int h2_pairs = 0;
    int h3_pairs = 0;
    int h4_pairs = 0;
    int h5_pairs = 0;
    int h6_pairs = 0;

    // Optimization flags
    bool h3_used_tetrahedra_opt = false;
    bool h4_used_chunking = false;
    bool h5_used_prefetch = false;
    bool h6_used_streaming = false;
};

// Forward declarations for decomposed components
[[nodiscard]] H0Result computeH0UnionFind(const std::vector<std::vector<int>> &simplices,
                                          const std::vector<double> &filtration_values);

[[nodiscard]] H1Result computeH1ReducedVR(const std::vector<std::vector<int>> &simplices,
                                          const std::vector<double> &filtration_values,
                                          const std::vector<int> &dimensions);

[[nodiscard]] H2Result computeH2AlphaComplex(const std::vector<std::vector<int>> &simplices,
                                             const std::vector<double> &filtration_values,
                                             const std::vector<int> &dimensions);

/**
 * @brief Per-dimension benchmark results
 */
struct PerDimensionBenchmark
{
    // Timing per dimension
    double h0_time_ms;
    double h1_time_ms;
    double h2_time_ms;
    double h3_time_ms;
    double h4_time_ms;
    double h5_time_ms;
    double h6_time_ms;
    double total_time_ms;

    // Pairs per dimension
    int h0_pairs;
    int h1_pairs;
    int h2_pairs;
    int h3_pairs;
    int h4_pairs;
    int h5_pairs;
    int h6_pairs;
};

/**
 * @brief Per-dimension exact persistent homology (0-6D)
 *
 * The dispatcher routes each homology dimension to a dedicated kernel family
 * (union-find, reduced VR, alpha-complex, chunked/prefetch/streaming paths).
 * This keeps kernel logic dimension-specific while preserving exact outputs.
 */

/**
 * @brief Main dispatcher: computes each dimension with optimal kernel
 *
 * @param simplices All simplices (mixed dimensions)
 * @param filtration_values Filtration values for each simplex
 * @param dimensions Dimension of each simplex
 * @param config Per-dimension configuration
 * @return Results for all dimensions
 */
PerDimensionResult computePerDimension(const std::vector<std::vector<int>> &simplices,
                                       const std::vector<double> &filtration_values,
                                       const std::vector<int> &dimensions,
                                       const PerDimensionConfig &config);

/**
 * @brief Convenience: compute 0-6D from point cloud
 *
 * @param points Point cloud data
 * @param point_dim Dimension of each point
 * @param num_points Number of points
 * @param config Per-dimension configuration
 * @return Results for all dimensions
 */
PerDimensionResult compute0To6DPerDimension(const std::vector<double> &points, size_t point_dim,
                                            size_t num_points, const PerDimensionConfig &config);

/**
 * @brief Get optimal per-dimension configuration
 *
 * Automatically selects best settings for each dimension.
 *
 * @param num_points Number of points
 * @param point_dim Dimension of each point
 * @param max_dim Maximum homology dimension
 * @return Optimized configuration
 */
PerDimensionConfig getOptimalPerDimensionConfig(size_t num_points, size_t point_dim, int max_dim);

/**
 * @brief Benchmark per-dimension performance
 *
 * Measures time and pairs for each dimension separately.
 *
 * @param simplices All simplices
 * @param filtration_values Filtration values
 * @param dimensions Dimensions
 * @return Benchmark results per dimension
 */
PerDimensionBenchmark benchmarkPerDimension(const std::vector<std::vector<int>> &simplices,
                                            const std::vector<double> &filtration_values,
                                            const std::vector<int> &dimensions);

/**
 * @brief Estimate per-dimension speedup
 *
 * @param num_points Number of points
 * @param point_dim Point dimension
 * @param max_dim Maximum dimension
 * @return Speedup estimate per dimension
 */
inline double estimatePerDimensionSpeedup(size_t num_points, size_t point_dim, int max_dim)
{
    double speedup = 1.0;

    const double size_scale = std::clamp(
        std::log2(static_cast<double>(std::max<size_t>(2, num_points))) / 20.0, 0.25, 1.0);

    // H0: Union-Find replaces matrix reduction for connected components.
    speedup += 100.0 * size_scale;

    // H1: Reduced VR (5x)
    if (max_dim >= 1)
        speedup += 5.0 * size_scale;

    // H2: Alpha Complex (25x for 2D)
    if (max_dim >= 2 && point_dim == 2)
        speedup += 25.0 * size_scale;

    // H3-H6: Grouped contribution
    if (max_dim >= 3)
        speedup += 7.5 * size_scale;
    if (max_dim >= 4)
        speedup += 7.5 * size_scale;
    if (max_dim >= 5)
        speedup += 6.5 * size_scale;
    if (max_dim >= 6)
        speedup += 7.5 * size_scale;

    // Average (conservative)
    return speedup / (max_dim + 1);
}

} // namespace nerve::persistence::perdim

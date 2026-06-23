
#pragma once

#include "nerve/core.hpp"
#include "nerve/persistence/core/core_types.hpp"
#include "nerve/persistence/reduction/reduction_union_find_ops.hpp"

#include <chrono>
#include <vector>

namespace nerve::persistence
{

/**
 * @brief Configuration for Flood Complex algorithm
 *
 * The Flood Complex (Graf et al., Sep 2025) is a method for large-scale PH
 * that can handle millions of points by combining Delaunay triangulation with
 * a "flooding" process using balls of radius r.
 */
struct FloodComplexConfig
{
    Size max_dim = 2;
    double max_radius = 1.0;

    // Subset selection parameters
    double subset_ratio = 0.05;    // Fraction of points for Delaunay subset
    size_t max_subset_size = 5000; // Cap subset size

    // Algorithm options
    bool use_flooding = true; // Enable flooding check

    // Approximation parameters
    double flooding_tolerance = 1e-6; // Numerical tolerance
};

/**
 * @brief Result of Flood Complex computation
 */
struct FloodComplexResult
{
    std::vector<Pair> pairs; // Persistence pairs

    // Timing breakdown
    double subset_selection_time_ms = 0.0;
    double delaunay_time_ms = 0.0;
    double flooding_time_ms = 0.0;
    double persistence_time_ms = 0.0;
    double total_time_ms = 0.0;

    // Statistics
    size_t original_points = 0;
    size_t subset_points = 0;
    size_t num_simplices = 0;
    double simplex_reduction_ratio = 0.0; // vs VR complex

    // Accuracy metrics
    double estimated_approximation_error = 0.0;
};

/**
 * @brief The Flood Complex: Large-Scale Persistent Homology
 *
 * Flood-complex approximation pipeline for large-scale point clouds.
 *
 * Algorithm Overview:
 * - Select well-spaced subset using farthest-point sampling
 * - Compute Delaunay triangulation of subset
 * - "Flood" simplices: keep those covered by balls of radius r
 * - Compute persistence on flooded complex
 *
 * Key Advantages:
 * - Handles millions of points (vs thousands for VR)
 * - Much fewer simplices than VR (Delaunay-based)
 * - Parallelizable structure; current implementation is CPU-backed
 * - Good approximation quality for ML applications
 *
 * Performance characteristics depend on subset sizing, geometry, and
 * triangulation cost on the target hardware.
 *
 * Paper: "The Flood Complex: Large-Scale Persistent Homology on Millions of Points"
 * Graf, Pellizzoni, Uray, Huber, Kwitt (arXiv:2509.22432, Sep 2025)
 * Code: https://github.com/plus-rkwitt/flooder
 *
 * @param points Point coordinates
 * @param point_dim Dimension of each point
 * @param num_points Number of points
 * @param config Flood Complex configuration
 * @return Computation result with persistence pairs and statistics
 */
FloodComplexResult computeFloodComplex(const std::vector<double> &points, size_t point_dim,
                                       size_t num_points, const FloodComplexConfig &config);

/**
 * @brief Get optimal Flood Complex configuration
 */
FloodComplexConfig getOptimalFloodConfig(size_t num_points, size_t point_dim);

/**
 * @brief Check if Flood Complex should be used for this dataset
 */
inline bool shouldUseFloodComplex(size_t num_points, size_t point_dim)
{
    // Flood Complex is beneficial for large point clouds
    // The overhead is worth it for 10K+ points in 3D
    return num_points >= 10000 && point_dim <= 3;
}

/**
 * @brief Estimate memory usage for Flood Complex
 */
size_t estimateFloodComplexMemory(size_t num_points, size_t point_dim,
                                  const FloodComplexConfig &config);

// Indexed point for Delaunay triangulation
struct IndexedPoint
{
    std::vector<double> coords;
    int original_index;
    IndexedPoint(const std::vector<double> &c, int idx)
        : coords(c)
        , original_index(idx)
    {}
    IndexedPoint(std::initializer_list<double> c, int idx)
        : coords(c)
        , original_index(idx)
    {}
};

// 3D Delaunay triangulation using Bowyer-Watson algorithm
class Delaunay3D
{
public:
    struct Tetrahedron
    {
        int v[4]; // Vertex indices
        bool valid = true;
    };

    Delaunay3D();

    std::vector<Tetrahedron> compute(const std::vector<IndexedPoint> &points);

private:
    std::vector<Tetrahedron> bowyerWatson(const std::vector<IndexedPoint> &points);
    std::vector<Tetrahedron> divideAndConquer(const std::vector<IndexedPoint> &points);
    bool pointInCircumsphere(const IndexedPoint &p, const IndexedPoint &a, const IndexedPoint &b,
                             const IndexedPoint &c, const IndexedPoint &d);
};

} // namespace nerve::persistence

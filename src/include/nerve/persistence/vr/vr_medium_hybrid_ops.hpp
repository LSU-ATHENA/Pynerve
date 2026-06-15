
#pragma once

#include "nerve/core.hpp"
#include "nerve/persistence/core/core_types.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"

namespace nerve::persistence
{

/**
 * @brief Optimized VR computation for medium point sets (1K-10K points)
 *
 * CPU-first implementation with adaptive execution planning:
 * - Tiled or NUMA-aware distance matrix construction
 * - OpenMP-assisted clique expansion
 * - Automatic routing to the small exact fast path (<1K) and exact-standard (>10K)
 *
 * Key optimizations:
 * - Tiled distance matrix computation for cache efficiency
 * - NUMA-aware matrix computation for large working sets
 * - Dynamic scheduling for neighborhood-skewed clique expansion
 *
 * @param points Flattened point coordinates [n_points * point_dim]
 * @param point_dim Dimension of each point
 * @param config VR computation configuration
 * @return Vector of persistence pairs
 */
std::vector<Pair> computeVrPersistenceMediumHybrid(const core::BufferView<const double> &points,
                                                   Size point_dim, const VRConfig &config);

/**
 * @brief Compute execution hints for medium-scale VR workloads.
 *
 * @param n_points Number of points
 * @param point_dim Point dimension
 * @param available_gpu_memory_gb Reserved; ignored by the current CPU-only implementation
 * @return Execution hints for distance and clique phases
 */
struct HybridWorkDistribution
{
    /// Fraction of the full distance matrix computed on GPU via tensor-core kernels.
    /// 0.0 = CPU-only; values >0.0 dispatch that ratio of point pairs to the GPU.
    double gpu_distance_matrix_ratio;
    /// When true, clique expansion (neighbourhood exploration) runs on the GPU
    /// using existing GPU persistence / cohomology kernels.
    bool use_gpu_clique_expansion;
    size_t tile_size;
    int num_threads;
};

HybridWorkDistribution computeOptimalWorkDistribution(size_t n_points, size_t point_dim,
                                                      double available_gpu_memory_gb);

} // namespace nerve::persistence

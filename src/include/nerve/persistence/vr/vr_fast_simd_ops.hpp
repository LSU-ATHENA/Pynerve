
#pragma once

#include "nerve/core.hpp"
#include "nerve/persistence/core/core_types.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"

namespace nerve::persistence
{

/**
 * @brief Optimized VR computation for small point sets (< 1K points)
 *
 * Uses cache-friendly data structures and the best compiled distance
 * kernel active at runtime. If AVX-512 support is missing, the
 * same exact algorithm uses a scalar FMA distance kernel.
 *
 * Key optimizations:
 * - AVX-512 vectorized distance calculations when compiled and runtime-supported
 * - Scalar FMA distance implementation on other CPUs
 * - Pre-allocated adjacency structures (no dynamic allocation)
 * - Cache-aligned point storage
 * - Optimized Bron-Kerbosch clique enumeration
 * - 32-bit edge keys for memory efficiency
 *
 * Performance targets:
 * - 100-500 points: < 50ms (3D, dim 2)
 * - 500-1000 points: < 200ms (3D, dim 2)
 *
 * Falls back to standard implementation for point sets > 1024 points
 * or dimensions > 16.
 *
 * @param points Flattened point coordinates [n_points * point_dim]
 * @param point_dim Dimension of each point (e.g., 2 for 2D, 3 for 3D)
 * @param config VR computation configuration
 * @return Vector of persistence pairs
 */
std::vector<Pair> computeVrPersistenceFastSimd(core::BufferView<const double> points,
                                               Size point_dim, const VRConfig &config);

/**
 * @brief Check if AVX-512 is available on this CPU
 */
bool isAvx512Available();

/**
 * @brief Get optimal distance-kernel block size
 */
size_t getOptimalSimdBlockSize(size_t point_dim);

} // namespace nerve::persistence


#pragma once

#include <cstddef>
#include <vector>

namespace nerve::persistence
{

/**
 * @brief Cache-blocked distance matrix computation.
 *
 * @param points Flattened point coordinates [n_points * point_dim]
 * @param point_dim Dimension of each point
 * @param num_points Number of points
 * @param tile_size Tile size (0 = auto-select)
 * @return Flattened distance matrix [num_points * num_points]
 */
std::vector<double> computeDistanceMatrixTiled(const std::vector<double> &points, size_t point_dim,
                                               size_t num_points, size_t tile_size = 0);

/**
 * @brief NUMA-aware distance matrix computation
 *
 * @param points Flattened point coordinates
 * @param point_dim Dimension of each point
 * @param num_points Number of points
 * @param num_numa_nodes Number of NUMA nodes to use
 * @return Flattened distance matrix
 */
std::vector<double> computeDistanceMatrixNumaAware(const std::vector<double> &points,
                                                   size_t point_dim, size_t num_points,
                                                   int num_numa_nodes);

/**
 * @brief Hierarchical cache-blocked distance matrix
 *
 * @param points Flattened point coordinates
 * @param point_dim Dimension of each point
 * @param num_points Number of points
 * @return Flattened distance matrix
 */
std::vector<double> computeDistanceMatrixHierarchical(const std::vector<double> &points,
                                                      size_t point_dim, size_t num_points);

/**
 * @brief Get optimal tile size for current hardware
 */
size_t getOptimalTileSize();

/**
 * @brief Detect cache sizes (L1, L2, L3 in bytes)
 */
struct CacheSizes
{
    size_t l1_data;
    size_t l2;
    size_t l3;
};

CacheSizes detectCacheSizes();

std::vector<float> computeDistanceMatrixTiledF32(const std::vector<float> &points, size_t point_dim,
                                                 size_t num_points);

} // namespace nerve::persistence

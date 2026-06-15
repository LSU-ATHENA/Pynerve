
#pragma once

#include "nerve/core.hpp"
#include "nerve/persistence/core/core_types.hpp"

#include <cstddef>
#include <limits>
#include <string>
#include <vector>

namespace nerve::persistence
{

/**
 * @brief Configuration for tile-based streaming PH
 */
struct StreamingConfig
{
    Size max_dim = 2;
    double max_radius = 1.0;

    // Tile parameters
    size_t tile_size = 10000;   // Points per tile
    double overlap_ratio = 0.1; // Overlap between adjacent tiles

    // Parallelism
    size_t num_threads = 0; // 0 = auto-detect

    // Merging
    double merge_tolerance = 1e-6; // For deduplicating pairs from overlaps
};

/**
 * @brief Result of streaming PH computation
 */
struct StreamingResult
{
    std::vector<Pair> pairs;

    // Timing breakdown
    double partition_time_ms = 0.0;
    double processing_time_ms = 0.0;
    double merge_time_ms = 0.0;
    double total_time_ms = 0.0;

    // Statistics
    size_t total_points = 0;
    size_t point_dim = 0;
    size_t num_tiles = 0;
    size_t pairs_count = 0;
    double memory_reduction_ratio = 0.0; // vs loading entire dataset
};

/**
 * @brief Tile-Based Out-of-Core Persistent Homology
 *
 * For datasets larger than RAM or GPU memory.
 * Processes data in overlapping tiles with constant memory usage.
 *
 * Key Features:
 * - Memory-mapped file I/O for datasets on disk
 * - Async tile processing with thread pools
 * - Intelligent merging of results from overlapping regions
 * - Scales to arbitrarily large datasets
 *
 * Algorithm:
 * - Partition point cloud into overlapping tiles
 * - Process each tile independently (async/parallel)
 * - Merge persistence diagrams, handling overlaps
 * - Deduplicate pairs from overlapping regions
 *
 * Performance:
 * - Constant memory: O(tile_size) regardless of dataset size
 * - Linear time: O(n) with good parallel scaling
 * - I/O efficient: Sequential reads, minimal seeks
 *
 * Use Cases:
 * - Datasets larger than RAM (>128GB)
 * - Streaming from SSD/NVMe
 * - Cloud-based processing
 * - Real-time incremental updates
 *
 * @param data_path Path to binary point cloud file
 * @param point_dim Dimension of each point
 * @param num_points Total number of points
 * @param config Streaming configuration
 * @return Streaming result with persistence pairs
 */
StreamingResult computeStreamingPH(const std::string &data_path, size_t point_dim,
                                   size_t num_points, const StreamingConfig &config);

/**
 * @brief In-memory tile-based PH (for datasets that fit in RAM)
 *
 * Simpler API when data is already in memory but you want
 * tile-based processing for parallelization.
 */
StreamingResult computeTiledPH(const std::vector<double> &points, size_t point_dim,
                               const StreamingConfig &config);

/**
 * @brief Get optimal streaming configuration
 *
 * Computes optimal tile size based on available memory.
 */
StreamingConfig getOptimalStreamingConfig(size_t num_points, size_t point_dim,
                                          size_t available_memory_mb);

/**
 * @brief Check if streaming should be used
 */
inline bool shouldUseStreaming(size_t num_points, size_t point_dim, size_t available_memory_mb)
{
    if (point_dim != 0 && num_points > std::numeric_limits<size_t>::max() / point_dim)
    {
        return true;
    }
    const size_t values = num_points * point_dim;
    if (values > std::numeric_limits<size_t>::max() / sizeof(double))
    {
        return true;
    }
    size_t data_size_mb = values * sizeof(double) / (1024ULL * 1024ULL);
    return static_cast<double>(data_size_mb) > static_cast<double>(available_memory_mb) * 0.8;
}

} // namespace nerve::persistence

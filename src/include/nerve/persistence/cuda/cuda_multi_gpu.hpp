
#pragma once

#include "nerve/persistence/core/flood_complex.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <limits>
#include <string>
#include <vector>

namespace nerve
{
namespace gpu
{
namespace multi
{

// Memory Unit Constants
constexpr size_t BYTES_PER_KB = 1024ULL;
constexpr size_t BYTES_PER_MB = 1024ULL * 1024ULL;
constexpr size_t BYTES_PER_GB = 1024ULL * 1024ULL * 1024ULL;

// Default GPU Memory Assumption
constexpr size_t DEFAULT_SINGLE_GPU_MEMORY = 16ULL * BYTES_PER_GB; // 16GB

/**
 * @brief Multi-GPU result
 */
struct MultiGpuResult
{
    std::vector<persistence::Pair> pairs;

    double total_time_ms;
    int num_gpus_used;
    bool nvlink_enabled;
    size_t total_points_processed;
};

/**
 * @brief Single GPU info
 */
struct GpuInfo
{
    int id;
    std::string name;
    size_t memory_mb;
    std::string compute_capability;
};

/**
 * @brief Multi-GPU system information
 */
struct MultiGpuInfo
{
    int num_gpus;
    std::vector<GpuInfo> gpus;
    bool nvlink_available;
    std::vector<bool> p2p_matrix; // P2P[i][j] = GPU i can access GPU j
};

/**
 * @brief Multi-GPU Flood Complex computation
 *
 * Distributes point clouds across available GPUs and merges per-partition Flood
 * Complex results.
 *
 * Key Features:
 * - Automatic load balancing across GPUs
 * - P2P topology detection
 * - Per-device local Flood Complex execution
 * - Result aggregation into the standard persistence-pair format
 *
 * Hardware Requirements:
 * - 2-8 NVIDIA GPUs (RTX 3090/4090, A100, H100)
 * - NVLink or NVSwitch for best performance
 * - Without NVLink: still works via PCIe (slower)
 *
 * Performance:
 * - Multi-GPU execution with topology-aware partitioning
 * - P2P metadata is reported for topology-aware scheduling
 *
 * @param points Point coordinates
 * @param point_dim Dimension of each point
 * @param config Flood Complex configuration
 * @return Multi-GPU computation result
 */
MultiGpuResult computeMultiGpuFloodComplex(const std::vector<double> &points, size_t point_dim,
                                           const persistence::FloodComplexConfig &config);

/**
 * @brief Get multi-GPU system information
 */
MultiGpuInfo getMultiGpuInfo();

/**
 * @brief Check if multi-GPU is beneficial for this dataset
 */
bool shouldUseMultiGpu(size_t num_points, size_t point_dim);

/**
 * @brief Get recommended number of GPUs
 */
inline int recommendedNumGpus(size_t num_points, size_t point_dim)
{
    if (num_points == 0 || point_dim == 0)
    {
        return 0;
    }

    int available = 0;
    if (cudaGetDeviceCount(&available) != cudaSuccess || available <= 0)
    {
        return 0;
    }

    if (point_dim > std::numeric_limits<size_t>::max() / num_points)
    {
        return available;
    }
    const size_t elements = num_points * point_dim;
    if (elements > std::numeric_limits<size_t>::max() / sizeof(double))
    {
        return available;
    }

    const size_t data_size = elements * sizeof(double);
    const size_t doubled_data_size = data_size > std::numeric_limits<size_t>::max() / 2
                                         ? std::numeric_limits<size_t>::max()
                                         : data_size * 2;
    const size_t needed = doubled_data_size / DEFAULT_SINGLE_GPU_MEMORY + 1;
    const size_t bounded_needed =
        std::min<size_t>(needed, static_cast<size_t>(std::numeric_limits<int>::max()));
    return std::min(static_cast<int>(bounded_needed), available);
}

/**
 * @brief Check if NVLink is available
 */
bool isNvlinkAvailable();

} // namespace multi
} // namespace gpu
} // namespace nerve

// C-linkage wrappers
extern "C"
{
    int getNumGpus(void);
    int isNvlinkAvailable(void);
}

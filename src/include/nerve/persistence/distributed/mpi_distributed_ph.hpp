
#pragma once

#include "nerve/config.hpp"
#include "nerve/core.hpp"
#include "nerve/persistence/core/core_types.hpp"

#include <string>
#include <vector>

namespace nerve::persistence::distributed
{

/**
 * @brief Configuration for distributed PH computation
 */
struct DistributedConfig
{
    Size max_dim = 2;
    double max_radius = 1.0;

    // Spatial cover parameters
    double overlap_ratio = 0.1; // Overlap between cover regions

    // Hybrid parallelism
    bool use_openmp = true;
    bool use_cuda = false;
};

/**
 * @brief Result of distributed PH computation
 */
struct DistributedResult
{
    std::vector<Pair> pairs;

    // Timing breakdown
    double cover_time_ms = 0.0;
    double local_computation_time_ms = 0.0;
    double communication_time_ms = 0.0;
    double mv_spectral_time_ms = 0.0;
    double total_time_ms = 0.0;

    // MPI info
    int mpi_rank = 0;
    int mpi_size = 1;

    // Statistics
    size_t total_points = 0;
    size_t points_per_rank = 0;
    size_t total_pairs = 0;
    double estimated_speedup = 1.0;
};

/**
 * @brief Distributed system information
 */
struct DistributedSystemInfo
{
    int mpi_rank = 0;
    int mpi_size = 1;
    std::string processor_name;
    int num_threads = 1;
    bool mpi_available = false;
};

/**
 * @brief MPI Distributed Persistent Homology
 *
 * Implements a distributed persistence workflow that partitions points into
 * overlapping spatial covers and merges local results with a
 * Mayer-Vietoris-style reconciliation pass.
 *
 * Algorithm:
 * - Divide space into overlapping regions
 * - Each MPI rank computes local PH
 * - Exchange ghost point information
 * - Compute Mayer-Vietoris spectral sequence
 * - Merge into global persistence diagram
 *
 * Requirements:
 * - MPI-enabled build when multi-process execution is desired
 * - Multi-process runtime environment
 *
 * @param points Point coordinates (all ranks have full data or use MPI-IO)
 * @param point_dim Dimension of each point
 * @param config Distributed configuration
 * @return Distributed computation result
 */
DistributedResult computeDistributedPH(const std::vector<double> &points, size_t point_dim,
                                       const DistributedConfig &config);

/**
 * @brief Single-node multi-threaded version (no MPI)
 *
 * Uses same algorithms but with OpenMP instead of MPI.
 * Good for testing or single-node many-core systems.
 */
DistributedResult computeDistributedPHSingleNode(const std::vector<double> &points,
                                                 size_t point_dim, const DistributedConfig &config,
                                                 int num_threads);

/**
 * @brief Initialize MPI (call once at program start)
 */
void initializeDistributed();

/**
 * @brief Finalize MPI (call once at program end)
 */
void finalizeDistributed();

/**
 * @brief Get optimal distributed configuration
 */
DistributedConfig getOptimalDistributedConfig(size_t num_points, size_t point_dim, int num_ranks);

/**
 * @brief Check if distributed computation is beneficial
 */
bool shouldUseDistributed(size_t num_points, int available_cores);

/**
 * @brief Get distributed system information
 */
DistributedSystemInfo getDistributedSystemInfo();

/**
 * @brief Check if MPI is available at compile time
 */
#if defined(NERVE_HAS_MPI)
constexpr bool isMpiAvailable()
{
    return true;
}
#else
constexpr bool isMpiAvailable()
{
    return false;
}
#endif

} // namespace nerve::persistence::distributed

// C-linkage wrappers
extern "C"
{
    void initializeDistributed(void);
    void finalizeDistributed(void);
    int isMpiAvailable(void);
}

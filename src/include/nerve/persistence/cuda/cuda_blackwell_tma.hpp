
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <string>
#include <vector>

namespace nerve::gpu::blackwell
{

/**
 * @brief Hopper/Blackwell device capability snapshot.
 */
struct BlackwellInfo
{
    int compute_capability_major;
    int compute_capability_minor;
    std::string generation;

    // Hardware capability flags. The distance launchers below use portable
    // tiled and persistent kernels; these flags do not imply WGMMA/TMA use.
    bool supports_tma;        // Tensor Memory Accelerator
    bool supports_wgmma;      // Warp Group Matrix Multiply
    bool supports_persistent; // Persistent kernels
    bool supports_clusters;   // Thread block clusters
};

/**
 * @brief Benchmark results
 */
struct BlackwellBenchmark
{
    double ampere_time_ms;
    double blackwell_time_ms;
    double speedup;
};

/**
 * @brief Hopper/Blackwell-oriented distance matrix launch path.
 *
 * The implementation uses portable CUDA tiled and persistent kernels, with
 * runtime capability reporting for newer architectures. It does not claim
 * to issue TMA or WGMMA instructions.
 *
 * @param d_points Device points
 * @param d_distances Output distances
 * @param n_points Number of points
 * @param point_dim Dimension
 * @param max_radius Maximum radius
 * @param stream CUDA stream
 */
void launchBlackwellDistanceMatrix(const double *d_points, float *d_distances, int n_points,
                                   int point_dim, double max_radius, cudaStream_t stream);

/**
 * @brief Persistent kernel for continuous processing
 */
void launchPersistentDistanceMatrix(const double *d_points, float *d_distances, int n_points,
                                    int point_dim, double max_radius, cudaStream_t stream);

/**
 * @brief Check if Hopper/Blackwell is available
 */
bool isHopperAvailable();
bool isBlackwellAvailable();

/**
 * @brief Get GPU generation info
 */
BlackwellInfo getBlackwellInfo();

/**
 * @brief Benchmark vs Ampere
 */
BlackwellBenchmark benchmarkBlackwell(const std::vector<double> &points, size_t point_dim,
                                      size_t num_points, double max_radius);

/**
 * @brief Get recommended precision mode
 */
inline const char *getRecommendedPrecisionMode()
{
    return "FP64 input / FP32 distance output";
}

} // namespace nerve::gpu::blackwell

// C-linkage
extern "C"
{
    int isHopperOrBlackwell(void);
}

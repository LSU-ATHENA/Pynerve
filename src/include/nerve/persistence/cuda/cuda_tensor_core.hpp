#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <string>
#include <vector>

namespace nerve::gpu::tensorcore
{

/**
 * @brief Tensor Core benchmark results
 */
struct TensorCoreBenchmark
{
    size_t num_points = 0;
    size_t point_dim = 0;

    double standard_time_ms = 0.0;    // Standard CUDA kernel
    double tensor_core_time_ms = 0.0; // Tensor Core launcher
    double speedup = 1.0;

    // Accuracy metrics
    double max_relative_error = 0.0;
    double mean_relative_error = 0.0;
};

/**
 * @brief Tensor Core capability information
 */
struct TensorCoreInfo
{
    bool available = false;
    int compute_capability_major = 0;
    int compute_capability_minor = 0;
    std::string generation = "Missing";

    // Precision support
    bool supports_fp16 = false; // All Tensor Core GPUs
    bool supports_bf16 = false; // Ampere+
    bool supports_tf32 = false; // Ampere+
    bool supports_fp8 = false;  // Hopper+
};

/**
 * @brief Launch distance-matrix computation with an FP16 WMMA fast path.
 *
 * Uses the Tensor Core WMMA path only when the active device and matrix shape
 * are compatible with the 16x16 tile contract. Other valid inputs use the
 * standard CUDA implementation.
 *
 * Key Features:
 * - FP16 dot products with FP32 accumulation on aligned tiles
 * - Automatic standard-CUDA path when the WMMA path is unsafe or missing
 *
 * Hardware Support:
 * - Volta+ devices expose the FP16 WMMA path used by this launcher
 * - Newer precision modes are reported by getTensorCoreInfo() but are not used here
 *
 * Algorithm:
 * Distance computation uses: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a*b
 * The dot product a*b is computed via Tensor Core matrix multiply
 *
 * @param d_points_double Input points in double precision
 * @param d_distance_matrix Output distances in float precision
 * @param n_points Number of points
 * @param point_dim Dimension of each point
 * @param max_radius Maximum distance threshold
 * @param stream CUDA stream
 */
void launchTensorCoreDistanceMatrix(const double *d_points_double, float *d_distance_matrix,
                                    int n_points, int point_dim, double max_radius,
                                    cudaStream_t stream);

/**
 * @brief Check if Tensor Cores are available on this GPU
 */
bool areTensorCoresAvailable();

/**
 * @brief Get Tensor Core capability information
 */
TensorCoreInfo getTensorCoreInfo();

/**
 * @brief Benchmark Tensor Cores vs standard CUDA
 */
TensorCoreBenchmark benchmarkTensorCore(const std::vector<double> &points, size_t point_dim,
                                        size_t num_points, double max_radius);

/**
 * @brief Get recommended precision mode based on GPU
 */
inline const char *getRecommendedPrecisionMode()
{
    auto info = getTensorCoreInfo();
    if (!info.available)
        return "FP32 CUDA path";
    return "FP16 WMMA with FP32 accumulation";
}

} // namespace nerve::gpu::tensorcore

// C-linkage wrappers for external use
extern "C"
{
    void launchTensorCoreDistanceMatrix(const double *d_points, float *d_distance_matrix,
                                        int n_points, int point_dim, double max_radius,
                                        cudaStream_t stream);

    int areTensorCoresAvailable(void);
}

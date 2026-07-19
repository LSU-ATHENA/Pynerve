
#pragma once

#include "nerve/common/accelerated_types.hpp"
#include "nerve/core.hpp"
#include "nerve/core_types.hpp"
#include "nerve/errors/errors.hpp"
#include "nerve/persistence/cuda/cuda_safe_arithmetic.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

namespace nerve::persistence::accelerated
{

using ::nerve::common::AcceleratedPerformanceStats;
using ::nerve::common::DeviceInfo;

#ifndef NERVE_ACCELERATED_MEMORY_USAGE_STATS_DEFINED
#define NERVE_ACCELERATED_MEMORY_USAGE_STATS_DEFINED
struct MemoryUsageStats
{
    size_t total_allocated = 0;
    size_t peak_allocated = 0;
    size_t active_allocations = 0;
    size_t pool_bytes = 0;
    size_t pool_free_bytes = 0;
    double fragmentation_ratio = 0.0;
};
#endif

struct CUDADistanceMatrixConfig
{
    Size max_block_size = 256;
    Size max_grid_size = 65535;
    Size shared_memory_size = 48 * 1024; // Requested dynamic shared memory budget per block
    // Enables the four-lane unrolled kernel path (GPU scalar chunking, not CPU ISA SIMD).
    bool enable_simd = true;
    bool enable_shared_memory = true;
    bool enable_streaming = true;
    Size streaming_threshold = 1048576; // 1M elements

    /// @brief Validate configuration
    errors::ErrorResult<void> validate() const
    {
        if (max_block_size == 0 || max_block_size > 1024)
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E52_PH_CONFIG,
                                                    "Invalid block size");
        }

        if (max_grid_size == 0 || max_grid_size > 2147483647)
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E52_PH_CONFIG,
                                                    "Invalid grid size");
        }

        if (shared_memory_size == 0 || shared_memory_size > 256 * 1024)
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E52_PH_CONFIG,
                                                    "Shared memory size out of range");
        }

        return errors::ErrorResult<void>::ok();
    }
};

/// This class provides a high-level interface for GPU-accelerated
/// distance matrix computation with automatic kernel selection
/// and optimization.
class CUDADistanceMatrix
{
public:
    /// @brief Create CUDA distance matrix computer
    /// @param config Configuration for computation
    /// @return ErrorResult containing the computer or error information
    static errors::ErrorResult<std::unique_ptr<CUDADistanceMatrix>>
    create(const CUDADistanceMatrixConfig &config = {});

    ~CUDADistanceMatrix();

    /// @brief Compute distance matrix on GPU
    /// @param points Input points buffer
    /// @param distances Output distances buffer
    /// @param point_dim Dimensionality of points
    /// @param max_radius Maximum distance threshold
    /// @return ErrorResult indicating success or failure
    errors::ErrorResult<void> compute(core::BufferView<const double> points,
                                      core::BufferView<double> &distances, Size point_dim,
                                      double max_radius);

    /// @brief Compute distance matrix with streaming for large problems
    /// @param points Input points buffer
    /// @param distances Output distances buffer
    /// @param point_dim Dimensionality of points
    /// @param max_radius Maximum distance threshold
    /// @param stream_size Size of streaming chunks
    /// @return ErrorResult indicating success or failure
    errors::ErrorResult<void> computeStreaming(core::BufferView<const double> points,
                                               core::BufferView<double> &distances, Size point_dim,
                                               double max_radius, Size stream_size);

    /// @brief Batch computation for multiple problem instances
    /// @param points_batch Array of point buffers
    /// @param distances_batch Array of distance buffers
    /// @param point_dim Common dimensionality
    /// @param max_radius Common maximum distance threshold
    /// @return ErrorResult indicating success or failure
    errors::ErrorResult<void>
    computeBatch(const std::vector<core::BufferView<const double>> &points_batch,
                 std::vector<core::BufferView<double>> &distances_batch, Size point_dim,
                 double max_radius);

    /// @brief Get performance statistics
    /// @return Performance statistics from last computation
    const AcceleratedPerformanceStats &getPerformanceStats() const;

    /// @brief Get GPU memory usage information
    /// @return Memory usage statistics
    errors::ErrorResult<MemoryUsageStats> getMemoryUsage() const;

    /// @brief Check GPU availability
    /// @return True if GPU is available and functional
    bool isAvailable() const;

    /// @brief Get GPU device information
    /// @return GPU device properties
    errors::ErrorResult<DeviceInfo> getDeviceInfo() const;

private:
    explicit CUDADistanceMatrix(const CUDADistanceMatrixConfig &config);

    class Impl;
    std::unique_ptr<Impl> impl_;
};

namespace cuda_kernels
{

void computeDistanceMatrixKernel(const double *__restrict__ points, double *__restrict__ distances,
                                 Size n_points, Size point_dim, double max_radius_sq);

void computeDistanceMatrixSimdKernel(const double *__restrict__ points,
                                     double *__restrict__ distances, Size n_points, Size point_dim,
                                     double max_radius_sq);

void computeDistanceMatrixSharedKernel(const double *__restrict__ points,
                                       double *__restrict__ distances, Size n_points,
                                       Size point_dim, double max_radius_sq);

void computeDistanceMatrixStreamingKernel(const double *__restrict__ points,
                                          double *__restrict__ distances, Size n_points,
                                          Size point_dim, double max_radius_sq, Size stream_offset,
                                          Size stream_size);

} // namespace cuda_kernels

namespace cuda_host
{

errors::ErrorResult<void> launchDistanceMatrixKernel(const double *points, double *distances,
                                                     Size n_points, Size point_dim,
                                                     double max_radius,
                                                     const CUDADistanceMatrixConfig &config,
                                                     Size stream_offset = 0, Size stream_size = 0);

errors::ErrorResult<void> launchDistanceMatrixKernel(const double *points, double *distances,
                                                     Size n_points, Size point_dim,
                                                     double max_radius, Size stream_offset = 0,
                                                     Size stream_size = 0);

CUDADistanceMatrixConfig getOptimalConfig(Size n_points, Size point_dim,
                                          const CUDADistanceMatrixConfig &base_config = {});

errors::ErrorResult<void> validateLaunchParams(Size n_points, Size point_dim,
                                               const CUDADistanceMatrixConfig &config);

} // namespace cuda_host

namespace factory
{

inline errors::ErrorResult<std::unique_ptr<CUDADistanceMatrix>>
createAcceleratedDistanceMatrix(Size n_points = 0, Size point_dim = 0, double max_radius = 1.0)
{
    if (!std::isfinite(max_radius) || !(max_radius > 0.0))
    {
        return errors::ErrorResult<std::unique_ptr<CUDADistanceMatrix>>::error(
            errors::ErrorCode::E51_PH_INPUT, "Maximum radius must be greater than 0");
    }
    CUDADistanceMatrixConfig config = cuda_host::getOptimalConfig(n_points, point_dim);
    return CUDADistanceMatrix::create(config);
}
inline errors::ErrorResult<std::unique_ptr<CUDADistanceMatrix>>
createBatchDistanceMatrix(Size batch_size, Size point_dim, double max_radius = 1.0)
{
    if (batch_size == 0 || point_dim == 0 || !std::isfinite(max_radius) || !(max_radius > 0.0))
    {
        return errors::ErrorResult<std::unique_ptr<CUDADistanceMatrix>>::error(
            errors::ErrorCode::E51_PH_INPUT, "Invalid batch distance matrix factory parameters");
    }
    Size threshold = 0;
    if (!detail::checkedSizeProduct(batch_size, point_dim, threshold))
    {
        return errors::ErrorResult<std::unique_ptr<CUDADistanceMatrix>>::error(
            errors::ErrorCode::E41_RESOURCE_LIMIT, "Batch streaming threshold overflows size_t");
    }
    CUDADistanceMatrixConfig config = cuda_host::getOptimalConfig(batch_size, point_dim);
    config.enable_streaming = true;
    config.streaming_threshold = std::max<Size>(1024, threshold);
    return CUDADistanceMatrix::create(config);
}
inline errors::ErrorResult<std::unique_ptr<CUDADistanceMatrix>>
createStreamingDistanceMatrix(Size problem_size, Size point_dim, double max_radius = 1.0)
{
    if (problem_size == 0 || point_dim == 0 || !std::isfinite(max_radius) || !(max_radius > 0.0))
    {
        return errors::ErrorResult<std::unique_ptr<CUDADistanceMatrix>>::error(
            errors::ErrorCode::E51_PH_INPUT,
            "Invalid streaming distance matrix factory parameters");
    }
    CUDADistanceMatrixConfig config = cuda_host::getOptimalConfig(problem_size, point_dim);
    config.enable_streaming = true;
    config.streaming_threshold = std::max<Size>(1, problem_size / 4);
    return CUDADistanceMatrix::create(config);
}
} // namespace factory

namespace utils
{

inline Size estimateMemoryUsage(Size n_points, Size point_dim)
{
    const Size coordinate_count = detail::saturatingProduct(n_points, point_dim);
    const Size points_memory = detail::saturatingProduct(coordinate_count, sizeof(double));
    const Size distance_count = detail::saturatingProduct(n_points, n_points);
    const Size distances_memory = detail::saturatingProduct(distance_count, sizeof(double));
    return detail::saturatingAdd(points_memory, distances_memory);
}

inline double estimateComputationTime(Size n_points, Size point_dim, bool enable_gpu = true)
{
    double base_time = std::pow(static_cast<double>(n_points), 1.5) * point_dim;

    if (enable_gpu && n_points >= 1000)
    {
        const double acceleration = std::clamp(std::log2(static_cast<double>(n_points)), 1.0, 32.0);
        base_time /= acceleration;
    }

    return base_time;
}

inline errors::ErrorResult<void> validateParameters(Size n_points, Size point_dim,
                                                    double max_radius)
{
    if (n_points == 0)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E51_PH_INPUT,
                                                "Number of points must be greater than 0");
    }

    if (point_dim == 0)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E51_PH_INPUT,
                                                "Point dimension must be greater than 0");
    }

    if (!std::isfinite(max_radius) || max_radius <= 0)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E51_PH_INPUT,
                                                "Maximum radius must be greater than 0");
    }

    // Check for reasonable limits
    if (n_points > 1000000)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E51_PH_INPUT,
                                                "Number of points too large (>1M)");
    }

    if (point_dim > 100)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E51_PH_INPUT,
                                                "Point dimension too large (>100)");
    }

    return errors::ErrorResult<void>::ok();
}

inline bool shouldUseDistanceStreaming(Size n_points, Size available_gpu_memory)
{
    const Size inferred_dim = n_points >= 8192 ? 8 : (n_points >= 2048 ? 4 : 2);
    Size memory_needed = estimateMemoryUsage(n_points, inferred_dim);
    return shouldUseStreaming(memory_needed, available_gpu_memory);
}

} // namespace utils

} // namespace nerve::persistence::accelerated

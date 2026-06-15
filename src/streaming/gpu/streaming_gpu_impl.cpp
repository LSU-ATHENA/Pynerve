
#include "nerve/errors/errors.hpp"
#include "nerve/gpu/compute_manager.hpp"
#include "nerve/streaming/windowed_ph.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <string_view>
#include <vector>

namespace nerve::gpu::streaming::detail
{

// Forward declarations from cuda_streaming_persistence.cu
extern void launchWindowUpdate(const double *d_new_point, const double *d_old_point,
                               const int *d_window_indices, int *d_affected_simplices,
                               int *d_affected_count, int window_size, int point_dim,
                               double max_radius_sq, cudaStream_t stream);

extern void launchIncrementalAdd(const int *d_new_simplices, const double *d_filtration_vals,
                                 int *d_persistence_pairs, int *d_pair_count, int n_new,
                                 int max_dimension, int n_existing, cudaStream_t stream);

extern void launchAffectedRegionDetection(const double *d_points, const double *d_new_point,
                                          int *d_affected_mask, int n_points, int point_dim,
                                          double max_radius_sq, cudaStream_t stream);

extern void launchBirthDeathUpdate(const int *d_affected_simplices, int n_affected,
                                   const double *d_distance_matrix, double *d_birth_times,
                                   double *d_death_times, int n_points,
                                   const int *d_simplex_vertices, const int *d_simplex_sizes,
                                   int max_simplex_size, cudaStream_t stream);

namespace
{

bool checkedProduct(std::size_t lhs, std::size_t rhs, std::size_t &out) noexcept
{
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs)
    {
        return false;
    }
    out = lhs * rhs;
    return true;
}

bool checkedByteCount(std::size_t count, std::size_t element_size, std::size_t &out) noexcept
{
    return checkedProduct(count, element_size, out);
}

bool checkedIntSize(std::size_t value, int &out) noexcept
{
    if (value > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        return false;
    }
    out = static_cast<int>(value);
    return true;
}

errors::ErrorResult<void> invalidInput(std::string_view message)
{
    return errors::ErrorResult<void>::error(errors::ErrorCode::E51_PH_INPUT, message);
}

errors::ErrorResult<void> resourceLimit(std::string_view message)
{
    return errors::ErrorResult<void>::error(errors::ErrorCode::E41_RESOURCE_LIMIT, message);
}

bool validRadius(double max_radius) noexcept
{
    return max_radius >= 0.0 && std::isfinite(max_radius);
}

void cleanupWindowBuffers(double *d_window_points, double *d_new_point, double *d_old_point,
                          int *d_window_indices, int *d_affected_simplices,
                          int *d_affected_count) noexcept
{
    cudaFree(d_window_points);
    cudaFree(d_new_point);
    cudaFree(d_old_point);
    cudaFree(d_window_indices);
    cudaFree(d_affected_simplices);
    cudaFree(d_affected_count);
}

void cleanupAffectedRegionBuffers(double *d_points, double *d_new_point,
                                  int *d_affected_mask) noexcept
{
    cudaFree(d_points);
    cudaFree(d_new_point);
    cudaFree(d_affected_mask);
}

} // namespace

// GPU-accelerated streaming persistence engine
class GPUStreamingEngine
{
public:
    static errors::ErrorResult<void> processWindowSlide(const std::vector<double> &window_points,
                                                        const std::vector<double> &new_point,
                                                        const std::vector<double> &old_point,
                                                        int point_dim, double max_radius,
                                                        std::vector<int> &out_affected_indices)
    {
        out_affected_indices.clear();
        if (point_dim <= 0)
        {
            return invalidInput("streaming window point dimension must be positive");
        }
        const std::size_t point_dim_size = static_cast<std::size_t>(point_dim);
        if (!validRadius(max_radius) || new_point.size() != point_dim_size ||
            old_point.size() != point_dim_size || window_points.size() % point_dim_size != 0)
        {
            return invalidInput("invalid streaming window geometry");
        }
        const double max_radius_sq = max_radius * max_radius;
        if (!std::isfinite(max_radius_sq))
        {
            return resourceLimit("streaming window radius square exceeds supported range");
        }

        const std::size_t window_size_size = window_points.size() / point_dim_size;
        int window_size = 0;
        int point_dim_int = 0;
        std::size_t window_bytes = 0;
        std::size_t point_bytes = 0;
        std::size_t window_index_bytes = 0;
        if (!checkedIntSize(window_size_size, window_size) ||
            !checkedIntSize(point_dim_size, point_dim_int) ||
            !checkedByteCount(window_points.size(), sizeof(double), window_bytes) ||
            !checkedByteCount(point_dim_size, sizeof(double), point_bytes) ||
            !checkedByteCount(window_size_size, sizeof(int), window_index_bytes))
        {
            return resourceLimit("streaming window size exceeds CUDA limits");
        }
        if (window_size == 0)
        {
            return errors::ErrorResult<void>::success();
        }

        double *d_window_points = nullptr;
        double *d_new_point = nullptr;
        double *d_old_point = nullptr;
        int *d_window_indices = nullptr;
        int *d_affected_simplices = nullptr;
        int *d_affected_count = nullptr;

        cudaError_t err = cudaMalloc(&d_window_points, window_bytes);
        if (err != cudaSuccess)
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E10_GPU_OOM);
        }

        err =
            cudaMemcpy(d_window_points, window_points.data(), window_bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            cleanupWindowBuffers(d_window_points, d_new_point, d_old_point, d_window_indices,
                                 d_affected_simplices, d_affected_count);
            return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }

        err = cudaMalloc(&d_new_point, point_bytes);
        if (err != cudaSuccess)
        {
            cleanupWindowBuffers(d_window_points, d_new_point, d_old_point, d_window_indices,
                                 d_affected_simplices, d_affected_count);
            return errors::ErrorResult<void>::error(errors::ErrorCode::E10_GPU_OOM);
        }

        err = cudaMemcpy(d_new_point, new_point.data(), point_bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            cleanupWindowBuffers(d_window_points, d_new_point, d_old_point, d_window_indices,
                                 d_affected_simplices, d_affected_count);
            return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }

        err = cudaMalloc(&d_old_point, point_bytes);
        if (err != cudaSuccess)
        {
            cleanupWindowBuffers(d_window_points, d_new_point, d_old_point, d_window_indices,
                                 d_affected_simplices, d_affected_count);
            return errors::ErrorResult<void>::error(errors::ErrorCode::E10_GPU_OOM);
        }

        err = cudaMemcpy(d_old_point, old_point.data(), point_bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            cleanupWindowBuffers(d_window_points, d_new_point, d_old_point, d_window_indices,
                                 d_affected_simplices, d_affected_count);
            return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }

        err = cudaMalloc(&d_window_indices, window_index_bytes);
        if (err != cudaSuccess)
        {
            cleanupWindowBuffers(d_window_points, d_new_point, d_old_point, d_window_indices,
                                 d_affected_simplices, d_affected_count);
            return errors::ErrorResult<void>::error(errors::ErrorCode::E10_GPU_OOM);
        }

        std::vector<int> window_indices(window_size_size);
        for (int i = 0; i < window_size; ++i)
        {
            window_indices[static_cast<std::size_t>(i)] = i;
        }

        err = cudaMemcpy(d_window_indices, window_indices.data(), window_index_bytes,
                         cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            cleanupWindowBuffers(d_window_points, d_new_point, d_old_point, d_window_indices,
                                 d_affected_simplices, d_affected_count);
            return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }

        err = cudaMalloc(&d_affected_simplices, window_index_bytes);
        if (err != cudaSuccess)
        {
            cleanupWindowBuffers(d_window_points, d_new_point, d_old_point, d_window_indices,
                                 d_affected_simplices, d_affected_count);
            return errors::ErrorResult<void>::error(errors::ErrorCode::E10_GPU_OOM);
        }

        err = cudaMalloc(&d_affected_count, sizeof(int));
        if (err != cudaSuccess)
        {
            cleanupWindowBuffers(d_window_points, d_new_point, d_old_point, d_window_indices,
                                 d_affected_simplices, d_affected_count);
            return errors::ErrorResult<void>::error(errors::ErrorCode::E10_GPU_OOM);
        }

        err = cudaMemset(d_affected_count, 0, sizeof(int));
        if (err != cudaSuccess)
        {
            cleanupWindowBuffers(d_window_points, d_new_point, d_old_point, d_window_indices,
                                 d_affected_simplices, d_affected_count);
            return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }

        launchWindowUpdate(d_new_point, d_old_point, d_window_indices, d_affected_simplices,
                           d_affected_count, window_size, point_dim_int, max_radius_sq, nullptr);

        err = cudaGetLastError();
        if (err == cudaSuccess)
        {
            err = cudaDeviceSynchronize();
        }
        if (err != cudaSuccess)
        {
            cleanupWindowBuffers(d_window_points, d_new_point, d_old_point, d_window_indices,
                                 d_affected_simplices, d_affected_count);
            return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }

        int affected_count = 0;
        err = cudaMemcpy(&affected_count, d_affected_count, sizeof(int), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess || affected_count < 0 || affected_count > window_size)
        {
            cleanupWindowBuffers(d_window_points, d_new_point, d_old_point, d_window_indices,
                                 d_affected_simplices, d_affected_count);
            return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }

        out_affected_indices.resize(static_cast<std::size_t>(affected_count));
        if (affected_count > 0)
        {
            std::size_t affected_bytes = 0;
            if (!checkedByteCount(static_cast<std::size_t>(affected_count), sizeof(int),
                                  affected_bytes))
            {
                cleanupWindowBuffers(d_window_points, d_new_point, d_old_point, d_window_indices,
                                     d_affected_simplices, d_affected_count);
                return resourceLimit("streaming affected-index copy size exceeds host limits");
            }
            err = cudaMemcpy(out_affected_indices.data(), d_affected_simplices, affected_bytes,
                             cudaMemcpyDeviceToHost);
            if (err != cudaSuccess)
            {
                cleanupWindowBuffers(d_window_points, d_new_point, d_old_point, d_window_indices,
                                     d_affected_simplices, d_affected_count);
                return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
            }
        }

        cleanupWindowBuffers(d_window_points, d_new_point, d_old_point, d_window_indices,
                             d_affected_simplices, d_affected_count);
        return errors::ErrorResult<void>::success();
    }

    static errors::ErrorResult<void> detectAffectedRegion(const std::vector<double> &points,
                                                          const std::vector<double> &new_point,
                                                          int point_dim, double max_radius,
                                                          std::vector<int> &out_affected_mask)
    {
        out_affected_mask.clear();
        if (point_dim <= 0)
        {
            return invalidInput("affected-region point dimension must be positive");
        }
        const std::size_t point_dim_size = static_cast<std::size_t>(point_dim);
        if (!validRadius(max_radius) || new_point.size() != point_dim_size ||
            points.size() % point_dim_size != 0)
        {
            return invalidInput("invalid affected-region geometry");
        }
        const double max_radius_sq = max_radius * max_radius;
        if (!std::isfinite(max_radius_sq))
        {
            return resourceLimit("affected-region radius square exceeds supported range");
        }

        const std::size_t n_points_size = points.size() / point_dim_size;
        int n_points = 0;
        int point_dim_int = 0;
        std::size_t point_bytes = 0;
        std::size_t points_bytes = 0;
        std::size_t mask_bytes = 0;
        if (!checkedIntSize(n_points_size, n_points) ||
            !checkedIntSize(point_dim_size, point_dim_int) ||
            !checkedByteCount(point_dim_size, sizeof(double), point_bytes) ||
            !checkedByteCount(points.size(), sizeof(double), points_bytes) ||
            !checkedByteCount(n_points_size, sizeof(int), mask_bytes))
        {
            return resourceLimit("affected-region size exceeds CUDA limits");
        }
        if (n_points == 0)
        {
            return errors::ErrorResult<void>::success();
        }

        double *d_points = nullptr;
        double *d_new_point = nullptr;
        int *d_affected_mask = nullptr;

        cudaError_t err = cudaMalloc(&d_points, points_bytes);
        if (err != cudaSuccess)
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E10_GPU_OOM);
        }

        err = cudaMemcpy(d_points, points.data(), points_bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            cleanupAffectedRegionBuffers(d_points, d_new_point, d_affected_mask);
            return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }

        err = cudaMalloc(&d_new_point, point_bytes);
        if (err != cudaSuccess)
        {
            cleanupAffectedRegionBuffers(d_points, d_new_point, d_affected_mask);
            return errors::ErrorResult<void>::error(errors::ErrorCode::E10_GPU_OOM);
        }

        err = cudaMemcpy(d_new_point, new_point.data(), point_bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            cleanupAffectedRegionBuffers(d_points, d_new_point, d_affected_mask);
            return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }

        err = cudaMalloc(&d_affected_mask, mask_bytes);
        if (err != cudaSuccess)
        {
            cleanupAffectedRegionBuffers(d_points, d_new_point, d_affected_mask);
            return errors::ErrorResult<void>::error(errors::ErrorCode::E10_GPU_OOM);
        }

        launchAffectedRegionDetection(d_points, d_new_point, d_affected_mask, n_points,
                                      point_dim_int, max_radius_sq, nullptr);

        err = cudaGetLastError();
        if (err == cudaSuccess)
        {
            err = cudaDeviceSynchronize();
        }
        if (err != cudaSuccess)
        {
            cleanupAffectedRegionBuffers(d_points, d_new_point, d_affected_mask);
            return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }

        out_affected_mask.resize(n_points_size);
        err = cudaMemcpy(out_affected_mask.data(), d_affected_mask, mask_bytes,
                         cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            cleanupAffectedRegionBuffers(d_points, d_new_point, d_affected_mask);
            return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }

        cleanupAffectedRegionBuffers(d_points, d_new_point, d_affected_mask);
        return errors::ErrorResult<void>::success();
    }
};

} // namespace nerve::gpu::streaming::detail

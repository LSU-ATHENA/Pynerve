#include "nerve/gpu/cuda_dispatch.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <vector>

extern "C" void launchOptimizedDistanceMatrix(const double *d_points, double *d_distance_matrix,
                                              int n_points, int point_dim, double max_radius,
                                              cudaStream_t stream);

namespace nerve::persistence::accelerated
{
namespace
{

[[nodiscard]] bool checkedSizeProduct(const Size lhs, const Size rhs, Size &out)
{
    if (lhs != 0 && rhs > std::numeric_limits<Size>::max() / lhs)
    {
        return false;
    }
    out = lhs * rhs;
    return true;
}

[[nodiscard]] bool checkedByteProduct(const Size count, const std::size_t element_size,
                                      std::size_t &out)
{
    if (count > static_cast<Size>(std::numeric_limits<std::size_t>::max() / element_size))
    {
        return false;
    }
    out = static_cast<std::size_t>(count) * element_size;
    return true;
}

[[nodiscard]] bool validDistanceShape(const Size n_points, const Size point_dim,
                                      const double max_radius)
{
    Size total = 0;
    return n_points > 0 && point_dim > 0 &&
           n_points <= static_cast<Size>(std::numeric_limits<int>::max()) &&
           point_dim <= static_cast<Size>(std::numeric_limits<int>::max()) &&
           checkedSizeProduct(n_points, point_dim, total) &&
           checkedSizeProduct(n_points, n_points, total) && std::isfinite(max_radius) &&
           max_radius > 0.0 && std::isfinite(max_radius * max_radius);
}

[[nodiscard]] bool cudaDeviceReady()
{
    int device_count = 0;
    return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

[[nodiscard]] double unboundedBatchRadius()
{
    return std::sqrt(std::numeric_limits<double>::max()) * 0.5;
}

template <typename T>
[[nodiscard]] cudaError_t
launchHostRadiusDistanceMatrix(const T *points, T *distances, const Size n_points,
                               const Size point_dim, const T max_radius, const cudaStream_t stream)
{
    Size point_values = 0;
    Size matrix_values = 0;
    std::size_t point_bytes = 0;
    std::size_t matrix_bytes = 0;
    if (!checkedSizeProduct(n_points, point_dim, point_values) ||
        !checkedSizeProduct(n_points, n_points, matrix_values) ||
        !checkedByteProduct(point_values, sizeof(T), point_bytes) ||
        !checkedByteProduct(matrix_values, sizeof(T), matrix_bytes))
    {
        return cudaErrorInvalidValue;
    }

    std::vector<T> host_points;
    std::vector<T> host_distances;
    try
    {
        host_points.resize(static_cast<std::size_t>(point_values));
        host_distances.assign(static_cast<std::size_t>(matrix_values), T{0});
    }
    catch (const std::bad_alloc &)
    {
        return cudaErrorMemoryAllocation;
    }

    cudaError_t status =
        cudaMemcpyAsync(host_points.data(), points, point_bytes, cudaMemcpyDeviceToHost, stream);
    if (status != cudaSuccess)
    {
        return status;
    }
    status = cudaStreamSynchronize(stream);
    if (status != cudaSuccess)
    {
        return status;
    }

    const double max_radius_sq = static_cast<double>(max_radius) * static_cast<double>(max_radius);
    const T no_edge = std::numeric_limits<T>::infinity();
    for (Size i = 0; i < n_points; ++i)
    {
        for (Size j = i; j < n_points; ++j)
        {
            double sum_sq = 0.0;
            const Size offset_i = i * point_dim;
            const Size offset_j = j * point_dim;
            for (Size dim = 0; dim < point_dim; ++dim)
            {
                const double lhs = static_cast<double>(host_points[offset_i + dim]);
                const double rhs = static_cast<double>(host_points[offset_j + dim]);
                if (!std::isfinite(lhs) || !std::isfinite(rhs))
                {
                    return cudaErrorInvalidValue;
                }
                const double diff = lhs - rhs;
                const double contribution = diff * diff;
                const double next_sum_sq = sum_sq + contribution;
                if (!std::isfinite(diff) || !std::isfinite(contribution) ||
                    !std::isfinite(next_sum_sq))
                {
                    return cudaErrorInvalidValue;
                }
                sum_sq = next_sum_sq;
                if (sum_sq > max_radius_sq)
                {
                    break;
                }
            }
            const T value = (sum_sq <= max_radius_sq) ? static_cast<T>(std::sqrt(sum_sq)) : no_edge;
            host_distances[i * n_points + j] = value;
            host_distances[j * n_points + i] = value;
        }
    }

    status = cudaMemcpyAsync(distances, host_distances.data(), matrix_bytes, cudaMemcpyHostToDevice,
                             stream);
    if (status != cudaSuccess)
    {
        return status;
    }
    return cudaStreamSynchronize(stream);
}

} // namespace

int DistanceMatrixOptimizer::compute(const double *points, double *distances, Size n_points,
                                     Size point_dim, double max_radius, cudaStream_t stream)
{
    if (n_points == 0 || point_dim == 0)
    {
        return cudaSuccess;
    }
    if (points == nullptr || distances == nullptr ||
        !validDistanceShape(n_points, point_dim, max_radius))
    {
        return cudaErrorInvalidValue;
    }
    if (!cudaDeviceReady())
    {
        return cudaErrorNoDevice;
    }

    launchOptimizedDistanceMatrix(points, distances, static_cast<int>(n_points),
                                  static_cast<int>(point_dim), max_radius, stream);
    return cudaPeekAtLastError();
}

int DistanceMatrixOptimizer::computeBatch(const double *const *pointsBatch, double **distancesBatch,
                                          const Size *n_points, Size point_dim, Size batch_size,
                                          cudaStream_t stream)
{
    if (batch_size == 0)
    {
        return cudaSuccess;
    }
    if (pointsBatch == nullptr || distancesBatch == nullptr || n_points == nullptr)
    {
        return cudaErrorInvalidValue;
    }
    for (Size batch = 0; batch < batch_size; ++batch)
    {
        const int status = compute(pointsBatch[batch], distancesBatch[batch], n_points[batch],
                                   point_dim, unboundedBatchRadius(), stream);
        if (status != cudaSuccess)
        {
            return status;
        }
    }
    return cudaSuccess;
}

int DistanceMatrixOptimizer::computeFP16(const float *points, float *distances, Size n_points,
                                         Size point_dim, float max_radius, cudaStream_t stream)
{
    if (n_points == 0 || point_dim == 0)
    {
        return cudaSuccess;
    }
    if (points == nullptr || distances == nullptr ||
        n_points > static_cast<Size>(std::numeric_limits<std::uint32_t>::max()) ||
        point_dim > static_cast<Size>(std::numeric_limits<std::uint32_t>::max()) ||
        !std::isfinite(max_radius) || max_radius <= 0.0f)
    {
        return cudaErrorInvalidValue;
    }
    if (!cudaDeviceReady())
    {
        return cudaErrorNoDevice;
    }
    return launchHostRadiusDistanceMatrix<float>(points, distances, n_points, point_dim, max_radius,
                                                 stream);
}

} // namespace nerve::persistence::accelerated

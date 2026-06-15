#include "nerve/gpu/distance_kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <type_traits>

namespace nerve::gpu
{

namespace
{

inline constexpr int kTileSize = 16;

template <typename T>
concept DistanceScalar = std::is_same_v<T, float> || std::is_same_v<T, double>;

template <typename T>
__device__ __forceinline__ T device_sqrt(T value);

template <>
__device__ __forceinline__ float device_sqrt<float>(float value)
{
    return sqrtf(value);
}

template <>
__device__ __forceinline__ double device_sqrt<double>(double value)
{
    return sqrt(value);
}

template <typename T>
__device__ __forceinline__ T device_inf();

template <>
__device__ __forceinline__ float device_inf<float>()
{
    return __int_as_float(0x7f800000);
}

template <>
__device__ __forceinline__ double device_inf<double>()
{
    return __longlong_as_double(0x7ff0000000000000ULL);
}

template <DistanceScalar T>
__device__ __forceinline__ bool accumulate_squared_diff(T diff, T &sum)
{
    const T contribution = diff * diff;
    const T next = sum + contribution;
    if (!isfinite(diff) || !isfinite(contribution) || !isfinite(next))
    {
        return false;
    }
    sum = next;
    return true;
}

__host__ __device__ inline std::size_t row_major_idx(int row, int col, int leading_dim)
{
    return static_cast<std::size_t>(row) * static_cast<std::size_t>(leading_dim) +
           static_cast<std::size_t>(col);
}

template <DistanceScalar T>
__global__ void __launch_bounds__(256)
    pairwise_distance_radius_kernel(const T *__restrict__ points, int points_ld,
                                    T *__restrict__ output, int output_ld, int n_points,
                                    int point_dim, T max_radius, bool clip_radius)
{
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_points || col >= n_points || row > col)
    {
        return;
    }

    T sum = T{0};
    bool valid_distance = true;
    for (int dim = 0; dim < point_dim; ++dim)
    {
        const T diff =
            points[row_major_idx(row, dim, points_ld)] - points[row_major_idx(col, dim, points_ld)];
        if (!accumulate_squared_diff(diff, sum))
        {
            valid_distance = false;
            break;
        }
    }

    T distance = valid_distance ? device_sqrt(sum) : device_inf<T>();
    if (clip_radius && distance > max_radius)
    {
        distance = device_inf<T>();
    }

    output[row_major_idx(row, col, output_ld)] = distance;
    output[row_major_idx(col, row, output_ld)] = distance;
}

template <DistanceScalar T>
cudaError_t launch_pairwise_distance_radius_impl(const T *points, int points_ld, T *output,
                                                 int output_ld, int n_points, int point_dim,
                                                 T max_radius, cudaStream_t stream)
{
    if (!std::isfinite(max_radius))
    {
        return cudaErrorInvalidValue;
    }
    if (n_points == 0)
    {
        return cudaSuccess;
    }
    if (points == nullptr || output == nullptr || n_points < 0 || point_dim < 0 || point_dim == 0 ||
        points_ld < point_dim || output_ld < n_points)
    {
        return cudaErrorInvalidValue;
    }

    const dim3 block(kTileSize, kTileSize);
    const dim3 grid(static_cast<unsigned>((n_points + kTileSize - 1) / kTileSize),
                    static_cast<unsigned>((n_points + kTileSize - 1) / kTileSize));

    pairwise_distance_radius_kernel<T><<<grid, block, 0, stream>>>(
        points, points_ld, output, output_ld, n_points, point_dim, max_radius, max_radius > T{0});
    return cudaGetLastError();
}

} // namespace

cudaError_t launch_pairwise_distance_radius_f32(const float *d_points, int points_ld, float *d_out,
                                                int out_ld, int n_points, int dim, float max_radius,
                                                void *stream_handle)
{
    const cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
    return launch_pairwise_distance_radius_impl(d_points, points_ld, d_out, out_ld, n_points, dim,
                                                max_radius, stream);
}

cudaError_t launch_pairwise_distance_radius_f64(const double *d_points, int points_ld,
                                                double *d_out, int out_ld, int n_points, int dim,
                                                double max_radius, void *stream_handle)
{
    const cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
    return launch_pairwise_distance_radius_impl(d_points, points_ld, d_out, out_ld, n_points, dim,
                                                max_radius, stream);
}

} // namespace nerve::gpu

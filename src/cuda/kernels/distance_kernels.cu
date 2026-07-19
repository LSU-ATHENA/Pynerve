#include "nerve/gpu/distance_kernels.hpp"
#include "nerve/gpu/gpu_ptx_ops.cuh"

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
inline constexpr int kFmaTileSize = 32;

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

__host__ __device__ inline std::size_t row_major_idx(int row, int col, int leading_dim)
{
    return static_cast<std::size_t>(row) * static_cast<std::size_t>(leading_dim) +
           static_cast<std::size_t>(col);
}

template <DistanceScalar T>
__device__ __forceinline__ bool accumulate_fma_sq(T diff, T &sum)
{
    if (!isfinite(diff))
    {
        return false;
    }
    T next;
    if constexpr (std::is_same_v<T, float>)
        next = ptx::fma_f32(diff, diff, sum);
    else
        next = ptx::fma_f64(diff, diff, sum);
    if (!isfinite(next))
    {
        return false;
    }
    sum = next;
    return true;
}

// Original kernel (baseline)
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
        if (!accumulate_fma_sq(diff, sum))
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

// FMA-optimized kernel with hardware max clipping
template <DistanceScalar T>
__global__ void __launch_bounds__(256)
    pairwise_distance_fma_kernel(const T *__restrict__ points, int points_ld,
                                 T *__restrict__ output, int output_ld, int n_points, int point_dim,
                                 T max_radius, bool clip_radius)
{
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_points || col >= n_points || row > col)
    {
        return;
    }

    T sum = T{0};
    bool valid_distance = true;

    int d = 0;
    for (; d + 1 < point_dim; d += 2)
    {
        const std::size_t idx_r0 = row_major_idx(row, d, points_ld);
        const std::size_t idx_c0 = row_major_idx(col, d, points_ld);
        const std::size_t idx_r1 = row_major_idx(row, d + 1, points_ld);
        const std::size_t idx_c1 = row_major_idx(col, d + 1, points_ld);

        const T diff0 = points[idx_r0] - points[idx_c0];
        const T diff1 = points[idx_r1] - points[idx_c1];

        if constexpr (std::is_same_v<T, float>)
        {
            sum = ptx::fma_f32(diff0, diff0, sum);
            sum = ptx::fma_f32(diff1, diff1, sum);
        }
        else
        {
            sum = ptx::fma_f64(diff0, diff0, sum);
            sum = ptx::fma_f64(diff1, diff1, sum);
        }

        if (!isfinite(sum))
        {
            valid_distance = false;
            break;
        }
    }
    for (; d < point_dim && valid_distance; ++d)
    {
        const T diff =
            points[row_major_idx(row, d, points_ld)] - points[row_major_idx(col, d, points_ld)];
        if constexpr (std::is_same_v<T, float>)
            sum = ptx::fma_f32(diff, diff, sum);
        else
            sum = ptx::fma_f64(diff, diff, sum);
        if (!isfinite(sum))
        {
            valid_distance = false;
            break;
        }
    }

    T distance = valid_distance ? device_sqrt(sum) : device_inf<T>();

    if (clip_radius)
    {
        if constexpr (std::is_same_v<T, float>)
            distance = ptx::hwmin_f32(distance, max_radius);
        else
            distance = ptx::hwmin_f64(distance, max_radius);
        if (distance >= max_radius)
            distance = device_inf<T>();
    }

    output[row_major_idx(row, col, output_ld)] = distance;
    output[row_major_idx(col, row, output_ld)] = distance;
}

// Tiled FMA kernel with shared memory and async copy (sm80+)
template <DistanceScalar T>
__global__ void __launch_bounds__(256)
    pairwise_distance_tiled_fma_kernel(const T *__restrict__ points, int points_ld,
                                       T *__restrict__ output, int output_ld, int n_points,
                                       int point_dim, T max_radius, bool clip_radius)
{
    __shared__ T tile_a[kFmaTileSize][kFmaTileSize + 1];
    __shared__ T tile_b[kFmaTileSize][kFmaTileSize + 1];

    const int row = blockIdx.y * kFmaTileSize + threadIdx.y;
    const int col = blockIdx.x * kFmaTileSize + threadIdx.x;

    T sum = T{0};
    bool valid_distance = true;

    for (int k = 0; k < point_dim; k += kFmaTileSize)
    {
        if (row < n_points && (k + threadIdx.x) < point_dim)
            tile_a[threadIdx.y][threadIdx.x] = points[row * points_ld + k + threadIdx.x];
        else
            tile_a[threadIdx.y][threadIdx.x] = T{0};

        if (col < n_points && (k + threadIdx.x) < point_dim)
            tile_b[threadIdx.y][threadIdx.x] = points[col * points_ld + k + threadIdx.x];
        else
            tile_b[threadIdx.y][threadIdx.x] = T{0};

        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < kFmaTileSize && (k + kk) < point_dim; ++kk)
        {
            const T diff = tile_a[threadIdx.y][kk] - tile_b[threadIdx.x][kk];
            if constexpr (std::is_same_v<T, float>)
                sum = ptx::fma_f32(diff, diff, sum);
            else
                sum = ptx::fma_f64(diff, diff, sum);
            if (!isfinite(sum))
            {
                valid_distance = false;
                break;
            }
        }
        if (!valid_distance)
            break;
        __syncthreads();
    }

    if (row >= n_points || col >= n_points || row > col)
        return;

    T distance = valid_distance ? device_sqrt(sum) : device_inf<T>();
    if (clip_radius && distance > max_radius)
        distance = device_inf<T>();

    output[row * output_ld + col] = distance;
    output[col * output_ld + row] = distance;
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

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    const int cc = prop.major * 10 + prop.minor;
    const bool clip_radius = max_radius > T{0};

    if (point_dim >= 64 && n_points >= 256 && cc >= 80)
    {
        const dim3 block(kFmaTileSize, kFmaTileSize);
        const dim3 grid(static_cast<unsigned>((n_points + kFmaTileSize - 1) / kFmaTileSize),
                        static_cast<unsigned>((n_points + kFmaTileSize - 1) / kFmaTileSize));
        pairwise_distance_tiled_fma_kernel<T><<<grid, block, 0, stream>>>(
            points, points_ld, output, output_ld, n_points, point_dim, max_radius, clip_radius);
    }
    else if (cc >= 80)
    {
        const dim3 block(kTileSize, kTileSize);
        const dim3 grid(static_cast<unsigned>((n_points + kTileSize - 1) / kTileSize),
                        static_cast<unsigned>((n_points + kTileSize - 1) / kTileSize));
        pairwise_distance_fma_kernel<T><<<grid, block, 0, stream>>>(
            points, points_ld, output, output_ld, n_points, point_dim, max_radius, clip_radius);
    }
    else
    {
        const dim3 block(kTileSize, kTileSize);
        const dim3 grid(static_cast<unsigned>((n_points + kTileSize - 1) / kTileSize),
                        static_cast<unsigned>((n_points + kTileSize - 1) / kTileSize));
        pairwise_distance_radius_kernel<T><<<grid, block, 0, stream>>>(
            points, points_ld, output, output_ld, n_points, point_dim, max_radius, clip_radius);
    }
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

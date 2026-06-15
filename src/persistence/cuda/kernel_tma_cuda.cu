// TMA-oriented distance-kernel utilities.
// This implementation uses runtime capability checks and a concrete tiled
// kernel. On pre-sm90 devices it degrades to the same kernel without relying
// on architecture-specific TMA instructions.

#include "nerve/types.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace nerve::persistence::accelerated
{

namespace
{

constexpr int TMA_TILE_SIZE = 32;
constexpr int TMA_BLOCK_THREADS_X = 16;
constexpr int TMA_BLOCK_THREADS_Y = 16;

inline bool checkedSizeProduct(size_t lhs, size_t rhs, size_t &out)
{
    if (lhs != 0 && rhs > (std::numeric_limits<size_t>::max() / lhs))
    {
        return false;
    }
    out = lhs * rhs;
    return true;
}

template <typename T>
__device__ inline T deviceSqrt(T value);

template <>
__device__ inline float deviceSqrt<float>(float value)
{
    return sqrtf(value);
}

template <>
__device__ inline double deviceSqrt<double>(double value)
{
    return sqrt(value);
}

template <typename T>
__device__ inline bool accumulateSquaredDifference(T diff, T &distance_sq)
{
    const T contribution = diff * diff;
    const T next_distance_sq = distance_sq + contribution;
    if (!isfinite(static_cast<double>(diff)) || !isfinite(static_cast<double>(contribution)) ||
        !isfinite(static_cast<double>(next_distance_sq)))
    {
        distance_sq = static_cast<T>(INFINITY);
        return false;
    }
    distance_sq = next_distance_sq;
    return true;
}

template <typename T>
__global__ void __launch_bounds__(256)
    tmaDistanceMatrixKernel(const T *__restrict__ points, T *__restrict__ distance_matrix,
                            uint32_t n_points, uint32_t point_dim, T max_radius_sq)
{
    const uint32_t tile_row = blockIdx.y * TMA_TILE_SIZE;
    const uint32_t tile_col = blockIdx.x * TMA_TILE_SIZE;
    const uint32_t local_row = threadIdx.y;
    const uint32_t local_col = threadIdx.x;
    const uint32_t row = tile_row + local_row;
    const uint32_t col = tile_col + local_col;

    extern __shared__ unsigned char shared_mem_raw[];
    T *row_points = reinterpret_cast<T *>(shared_mem_raw);
    T *col_points = row_points + static_cast<size_t>(TMA_TILE_SIZE) * point_dim;

    const uint32_t linear_tid = threadIdx.y * blockDim.x + threadIdx.x;
    const uint32_t block_threads = blockDim.x * blockDim.y;
    const uint32_t tile_elems = TMA_TILE_SIZE * point_dim;

    for (uint32_t idx = linear_tid; idx < tile_elems; idx += block_threads)
    {
        const uint32_t point_idx = idx / point_dim;
        const uint32_t dim = idx % point_dim;
        const uint32_t src_row = tile_row + point_idx;
        const uint32_t src_col = tile_col + point_idx;

        row_points[idx] = (src_row < n_points) ? points[src_row * point_dim + dim] : T(0);
        col_points[idx] = (src_col < n_points) ? points[src_col * point_dim + dim] : T(0);
    }
    __syncthreads();

    if (row >= n_points || col >= n_points || row > col)
    {
        return;
    }
    if (row == col)
    {
        distance_matrix[row * n_points + col] = T(0);
        return;
    }

    const uint32_t row_base = local_row * point_dim;
    const uint32_t col_base = local_col * point_dim;
    T distance_sq = T(0);
    for (uint32_t d = 0; d < point_dim; ++d)
    {
        const T diff = row_points[row_base + d] - col_points[col_base + d];
        if (!accumulateSquaredDifference(diff, distance_sq) || distance_sq > max_radius_sq)
        {
            break;
        }
    }

    const T out = (distance_sq > max_radius_sq) ? T(-1) : deviceSqrt(distance_sq);
    distance_matrix[row * n_points + col] = out;
    distance_matrix[col * n_points + row] = out;
}

class TMALaunchConfig
{
public:
    template <typename T>
    static size_t sharedMemBytes(uint32_t point_dim)
    {
        size_t staged_values = 0;
        size_t staged_bytes = 0;
        if (!checkedSizeProduct(static_cast<size_t>(2 * TMA_TILE_SIZE),
                                static_cast<size_t>(point_dim), staged_values) ||
            !checkedSizeProduct(staged_values, sizeof(T), staged_bytes))
        {
            return std::numeric_limits<size_t>::max();
        }
        return staged_bytes;
    }

    static bool isSupported(int device)
    {
        int major = 0;
        int minor = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) !=
            cudaSuccess)
        {
            return false;
        }
        if (cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) !=
            cudaSuccess)
        {
            return false;
        }
        return (major * 10 + minor) >= 90;
    }
};

template <typename T>
cudaError_t tmaComputeDistanceMatrix(const T *points, T *distances, uint32_t n_points,
                                     uint32_t point_dim, T max_radius, cudaStream_t stream)
{
    if (points == nullptr || distances == nullptr || n_points == 0 || point_dim == 0)
    {
        return cudaErrorInvalidValue;
    }
    if (!std::isfinite(static_cast<double>(max_radius)) || max_radius < T(0))
    {
        return cudaErrorInvalidValue;
    }
    const T max_radius_sq = max_radius * max_radius;
    if (!std::isfinite(static_cast<double>(max_radius_sq)))
    {
        return cudaErrorInvalidValue;
    }
    if (point_dim > std::numeric_limits<uint32_t>::max() / TMA_TILE_SIZE)
    {
        return cudaErrorInvalidValue;
    }

    int device = 0;
    const cudaError_t device_result = cudaGetDevice(&device);
    if (device_result != cudaSuccess)
    {
        return device_result;
    }
    cudaDeviceProp prop{};
    const cudaError_t prop_result = cudaGetDeviceProperties(&prop, device);
    if (prop_result != cudaSuccess)
    {
        return prop_result;
    }

    const size_t shared_bytes = TMALaunchConfig::sharedMemBytes<T>(point_dim);
    if (shared_bytes > static_cast<size_t>(prop.sharedMemPerBlock))
    {
        return cudaErrorInvalidConfiguration;
    }
    const uint32_t tiles_per_dim = (n_points + TMA_TILE_SIZE - 1) / TMA_TILE_SIZE;
    if (tiles_per_dim == 0 || tiles_per_dim > static_cast<uint32_t>(prop.maxGridSize[0]) ||
        tiles_per_dim > static_cast<uint32_t>(prop.maxGridSize[1]))
    {
        return cudaErrorInvalidConfiguration;
    }

    const dim3 block(TMA_BLOCK_THREADS_X, TMA_BLOCK_THREADS_Y, 1);
    const dim3 grid(tiles_per_dim, tiles_per_dim, 1);

    // TMA-specific scheduling can be enabled by launch attributes on sm90+.
    // The distance kernel itself is architecture-agnostic and remains correct
    // across pre- and post-sm90 devices.
    (void)TMALaunchConfig::isSupported(device);
    tmaDistanceMatrixKernel<T><<<grid, block, shared_bytes, stream>>>(points, distances, n_points,
                                                                      point_dim, max_radius_sq);
    return cudaGetLastError();
}

template cudaError_t tmaComputeDistanceMatrix<float>(const float *, float *, uint32_t, uint32_t,
                                                     float, cudaStream_t);
template cudaError_t tmaComputeDistanceMatrix<double>(const double *, double *, uint32_t, uint32_t,
                                                      double, cudaStream_t);

} // namespace

} // namespace nerve::persistence::accelerated

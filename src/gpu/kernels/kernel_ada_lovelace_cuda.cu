// Ada Lovelace-oriented CUDA kernels for distance and persistence workloads.
// The implementation favors robust behavior on SM 8.9 while remaining valid
// on other recent architectures.

#include "nerve/gpu/nvidia_auto_tuner.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#ifdef ENABLE_CUDA_FP8
#include <cuda_fp8.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>

namespace nerve
{
namespace gpu
{
namespace ada
{

constinit const int ADA_WARP_SIZE = 32;
constinit const int ADA_THREADS_2D = 16;
constinit const int ADA_TILE_LARGE = 64;
constinit const int ADA_TILE_MEDIUM = 32;
constinit const unsigned ADA_FULL_WARP_MASK = 0xFFFFFFFFu;

__host__ __device__ inline int upperTrianglePackedIndex(int i, int j, int n_points)
{
    return i * n_points + j - ((i + 1) * (i + 2)) / 2;
}

__device__ inline bool accumulateSquaredDifference(float diff, float &dist_sq)
{
    const float contribution = diff * diff;
    const float next_dist_sq = dist_sq + contribution;
    if (!isfinite(diff) || !isfinite(contribution) || !isfinite(next_dist_sq))
    {
        dist_sq = INFINITY;
        return false;
    }
    dist_sq = next_dist_sq;
    return true;
}

template <int TileSize>
__global__ void __launch_bounds__(256)
    l2OptimizedDistanceMatrixKernel(const float *__restrict__ points, float *__restrict__ distances,
                                    int n_points, int point_dim, float max_radius_sq)
{
    const int tile_i = blockIdx.y;
    const int tile_j = blockIdx.x;
    const int i_start = tile_i * TileSize;
    const int j_start = tile_j * TileSize;
    const int local_i = threadIdx.y;
    const int local_j = threadIdx.x;
    const int global_i = i_start + local_i;
    const int global_j = j_start + local_j;

    extern __shared__ float shared_mem[];
    float *tile_i_points = shared_mem;
    float *tile_j_points = shared_mem + static_cast<size_t>(TileSize) * point_dim;

    const int linear_tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int block_threads = blockDim.x * blockDim.y;
    const int tile_elems = TileSize * point_dim;

    for (int idx = linear_tid; idx < tile_elems; idx += block_threads)
    {
        const int point_off = idx / point_dim;
        const int dim = idx % point_dim;
        const int gi = i_start + point_off;
        const int gj = j_start + point_off;

        tile_i_points[idx] = (gi < n_points) ? points[gi * point_dim + dim] : 0.0f;
        tile_j_points[idx] = (gj < n_points) ? points[gj * point_dim + dim] : 0.0f;
    }
    __syncthreads();

    if (global_i >= n_points || global_j >= n_points || global_i >= global_j)
    {
        return;
    }

    float dist_sq = 0.0f;
    const int i_base = local_i * point_dim;
    const int j_base = local_j * point_dim;
    for (int d = 0; d < point_dim; ++d)
    {
        const float diff = tile_i_points[i_base + d] - tile_j_points[j_base + d];
        if (!accumulateSquaredDifference(diff, dist_sq) || dist_sq > max_radius_sq)
        {
            const int out_idx = upperTrianglePackedIndex(global_i, global_j, n_points);
            distances[out_idx] = 0.0f;
            return;
        }
    }

    const int out_idx = upperTrianglePackedIndex(global_i, global_j, n_points);
    distances[out_idx] = sqrtf(dist_sq);
}

__device__ inline int warpMaxReduce(int value)
{
    for (int offset = ADA_WARP_SIZE / 2; offset > 0; offset /= 2)
    {
        value = max(value, __shfl_down_sync(ADA_FULL_WARP_MASK, value, offset));
    }
    return value;
}

/**
 * Warp-specialized pivot discovery kernel.
 * One warp owns one column and reports the current pivot index.
 */
__global__ void __launch_bounds__(256)
    warpSpecializedPersistenceKernel(const uint64_t *__restrict__ columns, int n_cols,
                                     int words_per_col, int *__restrict__ pivot_table,
                                     int *__restrict__ reduction_count)
{
    const int warp_id = threadIdx.x / ADA_WARP_SIZE;
    const int lane_id = threadIdx.x % ADA_WARP_SIZE;
    const int warps_per_block = blockDim.x / ADA_WARP_SIZE;
    const int col = blockIdx.x * warps_per_block + warp_id;
    if (col >= n_cols)
    {
        return;
    }

    const uint64_t *col_ptr = columns + static_cast<size_t>(col) * words_per_col;
    int local_pivot = -1;
    for (int w = words_per_col - 1 - lane_id; w >= 0; w -= ADA_WARP_SIZE)
    {
        const uint64_t word = col_ptr[w];
        if (word != 0)
        {
            local_pivot = w * 64 + (63 - __clzll(word));
            break;
        }
    }

    const int pivot = warpMaxReduce(local_pivot);
    if (lane_id == 0 && pivot >= 0)
    {
        const int old_owner = atomicCAS(&pivot_table[pivot], -1, col);
        if (old_owner >= 0 && old_owner != col)
        {
            atomicAdd(reduction_count, 1);
        }
    }
}

#ifdef ENABLE_CUDA_FP8
__global__ void __launch_bounds__(256)
    fp8DistanceMatrixKernel(const __nv_fp8_e4m3 *__restrict__ points,
                            __nv_fp8_e4m3 *__restrict__ distances, int n_points, int point_dim,
                            float max_radius_sq)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_pairs = n_points * (n_points - 1) / 2;
    if (idx >= total_pairs)
    {
        return;
    }

    int i = 0;
    int rem = idx;
    while (rem >= n_points - i - 1)
    {
        rem -= (n_points - i - 1);
        ++i;
    }
    const int j = i + 1 + rem;

    float dist_sq = 0.0f;
    for (int d = 0; d < point_dim; ++d)
    {
        const float pi = __nv_cvt_fp8_to_fp32(points[i * point_dim + d]);
        const float pj = __nv_cvt_fp8_to_fp32(points[j * point_dim + d]);
        const float diff = pi - pj;
        if (!accumulateSquaredDifference(diff, dist_sq))
        {
            break;
        }
    }

    const float value = (isfinite(dist_sq) && dist_sq <= max_radius_sq) ? sqrtf(dist_sq) : 0.0f;
    distances[idx] = __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
}

__global__ void __launch_bounds__(256)
    fp8PersistenceKernel(__nv_fp8_e4m3 *__restrict__ boundary_matrix, int *__restrict__ pivot_table,
                         int n_cols, int rows_per_col, int *__restrict__ reduction_count)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n_cols)
    {
        return;
    }

    int pivot = -1;
    for (int row = rows_per_col - 1; row >= 0; --row)
    {
        const float val = __nv_cvt_fp8_to_fp32(boundary_matrix[col * rows_per_col + row]);
        if (val > 0.5f)
        {
            pivot = row;
            break;
        }
    }

    if (pivot >= 0)
    {
        const int old_owner = atomicCAS(&pivot_table[pivot], -1, col);
        if (old_owner >= 0 && old_owner != col)
        {
            atomicAdd(reduction_count, 1);
        }
    }
}
#endif

class AdaLovelaceLauncher
{
public:
    static cudaError_t launchL2OptimizedDistanceMatrix(const float *d_points, float *d_distances,
                                                       int n_points, int point_dim,
                                                       float max_radius, cudaStream_t stream = 0)
    {
        if (d_points == nullptr || d_distances == nullptr || n_points <= 1 || point_dim <= 0 ||
            !std::isfinite(max_radius) || max_radius <= 0.0f)
        {
            return cudaErrorInvalidValue;
        }
        const float max_radius_sq = max_radius * max_radius;
        if (!std::isfinite(max_radius_sq))
        {
            return cudaErrorInvalidValue;
        }

        cudaDeviceProp prop{};
        cudaError_t status = cudaGetDeviceProperties(&prop, 0);
        if (status != cudaSuccess)
        {
            return status;
        }

        const int max_smem = prop.sharedMemPerBlock;
        const size_t smem_large =
            static_cast<size_t>(2) * ADA_TILE_LARGE * point_dim * sizeof(float);
        const size_t smem_medium =
            static_cast<size_t>(2) * ADA_TILE_MEDIUM * point_dim * sizeof(float);
        const int tile = (smem_large <= static_cast<size_t>(max_smem))    ? ADA_TILE_LARGE
                         : (smem_medium <= static_cast<size_t>(max_smem)) ? ADA_TILE_MEDIUM
                                                                          : 0;
        if (tile == 0)
        {
            return cudaErrorInvalidConfiguration;
        }

        const dim3 block(ADA_THREADS_2D, ADA_THREADS_2D);
        const dim3 grid(static_cast<unsigned>((n_points + tile - 1) / tile),
                        static_cast<unsigned>((n_points + tile - 1) / tile));
        const size_t smem_size = static_cast<size_t>(2) * tile * point_dim * sizeof(float);

        if (tile == ADA_TILE_LARGE)
        {
            l2OptimizedDistanceMatrixKernel<ADA_TILE_LARGE><<<grid, block, smem_size, stream>>>(
                d_points, d_distances, n_points, point_dim, max_radius_sq);
        }
        else
        {
            l2OptimizedDistanceMatrixKernel<ADA_TILE_MEDIUM><<<grid, block, smem_size, stream>>>(
                d_points, d_distances, n_points, point_dim, max_radius_sq);
        }
        return cudaGetLastError();
    }

#ifdef ENABLE_CUDA_FP8
    static cudaError_t launchFP8DistanceMatrix(const __nv_fp8_e4m3 *d_points,
                                               __nv_fp8_e4m3 *d_distances, int n_points,
                                               int point_dim, float max_radius,
                                               cudaStream_t stream = 0)
    {
        if (d_points == nullptr || d_distances == nullptr || n_points <= 1 || point_dim <= 0 ||
            !std::isfinite(max_radius) || max_radius <= 0.0f)
        {
            return cudaErrorInvalidValue;
        }
        const float max_radius_sq = max_radius * max_radius;
        if (!std::isfinite(max_radius_sq))
        {
            return cudaErrorInvalidValue;
        }
        const int total_pairs = n_points * (n_points - 1) / 2;
        const int block = 256;
        const int grid = (total_pairs + block - 1) / block;
        fp8DistanceMatrixKernel<<<grid, block, 0, stream>>>(d_points, d_distances, n_points,
                                                            point_dim, max_radius_sq);
        return cudaGetLastError();
    }
#endif

    static cudaError_t launchWarpSpecializedPersistence(const uint64_t *d_columns, int n_cols,
                                                        int words_per_col, int *d_pivot_table,
                                                        int *d_reduction_count,
                                                        cudaStream_t stream = 0)
    {
        if (d_columns == nullptr || d_pivot_table == nullptr || d_reduction_count == nullptr ||
            n_cols <= 0 || words_per_col <= 0)
        {
            return cudaErrorInvalidValue;
        }

        constexpr int warps_per_block = 8;
        constexpr int threads = warps_per_block * ADA_WARP_SIZE;
        const int grid = (n_cols + warps_per_block - 1) / warps_per_block;
        warpSpecializedPersistenceKernel<<<grid, threads, 0, stream>>>(
            d_columns, n_cols, words_per_col, d_pivot_table, d_reduction_count);
        return cudaGetLastError();
    }
};

inline cudaError_t rtx4090DistanceMatrix(const float *d_points, float *d_distances, int n_points,
                                         int point_dim, float max_radius, cudaStream_t stream = 0)
{
    return AdaLovelaceLauncher::launchL2OptimizedDistanceMatrix(d_points, d_distances, n_points,
                                                                point_dim, max_radius, stream);
}

[[nodiscard]] inline bool isRTX4090(int device_id = 0)
{
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device_id) != cudaSuccess)
    {
        return false;
    }
    const std::string name = prop.name;
    return name.find("4090") != std::string::npos ||
           (prop.major == 8 && prop.minor == 9 && prop.l2CacheSize >= 80 * 1024 * 1024);
}

[[nodiscard]] inline bool supportsFP8(int device_id = 0)
{
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device_id) != cudaSuccess)
    {
        return false;
    }
    return prop.major > 8 || (prop.major == 8 && prop.minor >= 9);
}

} // namespace ada
} // namespace gpu
} // namespace nerve

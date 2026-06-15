
#include "nerve/persistence/cuda/cuda_distance_matrix.hpp"
#include "nerve/persistence/cuda/cuda_error_handling.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace nerve::persistence::accelerated
{

constexpr int kTileSizeLarge = 32;
constexpr int kTileSizeMedium = 16;
constexpr int kTileSizeSmall = 8;
constexpr int DISTANCE_TILED_BATCH_THREADS = 256;
constexpr int DISTANCE_TILED_BATCH_MAX_BLOCKS = 4096;
constexpr int DISTANCE_GENERIC_THREADS = 256;
constexpr int DISTANCE_GENERIC_MAX_BLOCKS = 4096;

inline bool checkedSizeProduct(Size lhs, Size rhs, Size &out)
{
    if (lhs != 0 && rhs > (std::numeric_limits<Size>::max() / lhs))
    {
        return false;
    }
    out = lhs * rhs;
    return true;
}

inline Size tileSharedBytes(int tile_size, Size point_dim)
{
    Size staged_points = 0;
    Size staged_bytes = 0;
    if (!checkedSizeProduct(static_cast<Size>(2 * tile_size), point_dim, staged_points) ||
        !checkedSizeProduct(staged_points, static_cast<Size>(sizeof(double)), staged_bytes))
    {
        return std::numeric_limits<Size>::max();
    }
    return staged_bytes;
}

__device__ inline bool accumulateSquaredDifference(double diff, double &distance_sq)
{
    const double contribution = diff * diff;
    const double next_distance_sq = distance_sq + contribution;
    if (!isfinite(diff) || !isfinite(contribution) || !isfinite(next_distance_sq))
    {
        distance_sq = INFINITY;
        return false;
    }
    distance_sq = next_distance_sq;
    return true;
}

template <int TileSize>
__global__ void __launch_bounds__(1024)
    computeDistanceMatrixKernelTiled(const double *__restrict__ points,
                                     double *__restrict__ distances, Size n_points, Size point_dim,
                                     double max_radius_sq)
{
    const Size tile_row = static_cast<Size>(blockIdx.y) * TileSize;
    const Size tile_col = static_cast<Size>(blockIdx.x) * TileSize;
    if (tile_row > tile_col)
    {
        return;
    }

    const int local_row = threadIdx.y;
    const int local_col = threadIdx.x;
    const Size global_row = tile_row + static_cast<Size>(local_row);
    const Size global_col = tile_col + static_cast<Size>(local_col);

    extern __shared__ double shared_mem[];
    double *row_points = shared_mem;
    double *col_points = shared_mem + static_cast<Size>(TileSize) * point_dim;

    const int linear_tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int block_threads = blockDim.x * blockDim.y;
    const Size tile_elements = static_cast<Size>(TileSize) * point_dim;

    for (Size idx = static_cast<Size>(linear_tid); idx < tile_elements;
         idx += static_cast<Size>(block_threads))
    {
        const Size point_idx = idx / point_dim;
        const Size dim_idx = idx % point_dim;
        const Size row_index = tile_row + point_idx;
        const Size col_index = tile_col + point_idx;

        row_points[idx] = (row_index < n_points) ? points[row_index * point_dim + dim_idx] : 0.0;
        col_points[idx] = (col_index < n_points) ? points[col_index * point_dim + dim_idx] : 0.0;
    }
    __syncthreads();

    if (global_row >= n_points || global_col >= n_points || global_row > global_col)
    {
        return;
    }

    const Size row_offset = static_cast<Size>(local_row) * point_dim;
    const Size col_offset = static_cast<Size>(local_col) * point_dim;
    double distance_sq = 0.0;

    if (point_dim <= 4)
    {
#pragma unroll
        for (Size d = 0; d < point_dim; ++d)
        {
            const double diff = row_points[row_offset + d] - col_points[col_offset + d];
            if (!accumulateSquaredDifference(diff, distance_sq) || distance_sq > max_radius_sq)
            {
                distances[global_row * n_points + global_col] = -1.0;
                if (global_row != global_col)
                {
                    distances[global_col * n_points + global_row] = -1.0;
                }
                return;
            }
        }
    }
    else
    {
        for (Size d = 0; d < point_dim; ++d)
        {
            const double diff = row_points[row_offset + d] - col_points[col_offset + d];
            if (!accumulateSquaredDifference(diff, distance_sq) || distance_sq > max_radius_sq)
            {
                distances[global_row * n_points + global_col] = -1.0;
                if (global_row != global_col)
                {
                    distances[global_col * n_points + global_row] = -1.0;
                }
                return;
            }
        }
    }

    const double dist = sqrt(distance_sq);
    distances[global_row * n_points + global_col] = dist;
    if (global_row != global_col)
    {
        distances[global_col * n_points + global_row] = dist;
    }
}
__global__ void __launch_bounds__(256)
    computeDistanceMatrixKernelGeneric(const double *__restrict__ points,
                                       double *__restrict__ distances, Size n_points,
                                       Size point_dim, double max_radius_sq)
{
    const Size total = n_points * n_points;
    const Size start = static_cast<Size>(blockIdx.x) * blockDim.x + static_cast<Size>(threadIdx.x);
    const Size stride = static_cast<Size>(gridDim.x) * blockDim.x;

    for (Size idx = start; idx < total; idx += stride)
    {
        const Size row = idx / n_points;
        const Size col = idx % n_points;
        if (row >= n_points || col >= n_points || row > col)
        {
            continue;
        }

        double distance_sq = 0.0;
        bool clipped = false;
        for (Size d = 0; d < point_dim; ++d)
        {
            const double diff = points[row * point_dim + d] - points[col * point_dim + d];
            if (!accumulateSquaredDifference(diff, distance_sq) || distance_sq > max_radius_sq)
            {
                clipped = true;
                break;
            }
        }

        const double out = clipped ? -1.0 : sqrt(distance_sq);
        distances[row * n_points + col] = out;
        if (row != col)
        {
            distances[col * n_points + row] = out;
        }
    }
}

__device__ inline void decodeLowerTriangularIndex(Size idx, Size &row, Size &col)
{
    const double approx = (sqrt(1.0 + 8.0 * static_cast<double>(idx)) - 1.0) * 0.5;
    row = static_cast<Size>(approx);
    while ((row + 1) * (row + 2) / 2 <= idx)
    {
        ++row;
    }
    while (row * (row + 1) / 2 > idx)
    {
        --row;
    }
    col = idx - (row * (row + 1) / 2);
}

__global__ void __launch_bounds__(256)
    computeDistanceMatrixBatchPersistent(const double *const *__restrict__ points_batch,
                                         double **__restrict__ distances_batch,
                                         const Size *__restrict__ n_points, Size point_dim,
                                         Size batch_size, double max_radius_sq)
{
    for (Size matrix_idx = static_cast<Size>(blockIdx.x); matrix_idx < batch_size;
         matrix_idx += static_cast<Size>(gridDim.x))
    {
        const double *points = points_batch[matrix_idx];
        double *distances = distances_batch[matrix_idx];
        const Size n = n_points[matrix_idx];
        if (n == 0)
        {
            continue;
        }

        const Size pair_count = n * (n + 1) / 2;
        const Size start_idx =
            static_cast<Size>(threadIdx.x) + static_cast<Size>(blockIdx.y) * blockDim.x;
        const Size stride = static_cast<Size>(blockDim.x) * gridDim.y;

        for (Size idx = start_idx; idx < pair_count; idx += stride)
        {
            Size row = 0;
            Size col = 0;
            decodeLowerTriangularIndex(idx, row, col);
            if (row >= n || col > row)
            {
                continue;
            }

            double distance_sq = 0.0;
            bool clipped = false;
            for (Size d = 0; d < point_dim; ++d)
            {
                const double diff = points[row * point_dim + d] - points[col * point_dim + d];
                if (!accumulateSquaredDifference(diff, distance_sq) || distance_sq > max_radius_sq)
                {
                    clipped = true;
                    break;
                }
            }

            const double out = clipped ? -1.0 : sqrt(distance_sq);
            distances[row * n + col] = out;
            if (row != col)
            {
                distances[col * n + row] = out;
            }
        }
    }
}

inline int selectTileSize(Size point_dim, int shared_mem_per_block_bytes, int max_threads_per_block)
{
    if (shared_mem_per_block_bytes <= 0 || max_threads_per_block <= 0)
    {
        return 0;
    }
    const Size budget = static_cast<Size>(shared_mem_per_block_bytes);
    const Size bytes32 = tileSharedBytes(kTileSizeLarge, point_dim);
    if (bytes32 <= budget && kTileSizeLarge * kTileSizeLarge <= max_threads_per_block)
    {
        return kTileSizeLarge;
    }
    const Size bytes16 = tileSharedBytes(kTileSizeMedium, point_dim);
    if (bytes16 <= budget && kTileSizeMedium * kTileSizeMedium <= max_threads_per_block)
    {
        return kTileSizeMedium;
    }
    const Size bytes8 = tileSharedBytes(kTileSizeSmall, point_dim);
    if (bytes8 <= budget && kTileSizeSmall * kTileSizeSmall <= max_threads_per_block)
    {
        return kTileSizeSmall;
    }
    return 0;
}

cudaError_t launchDistanceMatrixGeneric(const double *points, double *distances, Size n_points,
                                        Size point_dim, double max_radius_sq, cudaStream_t stream)
{
    const Size total = n_points * n_points;
    const Size required_blocks = (total + static_cast<Size>(DISTANCE_GENERIC_THREADS) - 1) /
                                 static_cast<Size>(DISTANCE_GENERIC_THREADS);
    const Size capped_blocks = std::max<Size>(
        1, std::min(required_blocks, static_cast<Size>(DISTANCE_GENERIC_MAX_BLOCKS)));
    const dim3 block(static_cast<unsigned>(DISTANCE_GENERIC_THREADS), 1, 1);
    const dim3 grid(static_cast<unsigned>(capped_blocks), 1, 1);
    computeDistanceMatrixKernelGeneric<<<grid, block, 0, stream>>>(points, distances, n_points,
                                                                   point_dim, max_radius_sq);
    return cudaGetLastError();
}

cudaError_t computeDistanceMatrixOptimized(const double *points, double *distances, Size n_points,
                                           Size point_dim, double max_radius, cudaStream_t stream,
                                           Size shared_memory_budget_bytes)
{
    if (n_points == 0 || point_dim == 0)
    {
        return cudaSuccess;
    }
    if (points == nullptr || distances == nullptr)
    {
        return cudaErrorInvalidValue;
    }
    if (!std::isfinite(max_radius) || !(max_radius > 0.0))
    {
        return cudaErrorInvalidValue;
    }
    if (n_points > (std::numeric_limits<Size>::max() / n_points))
    {
        return cudaErrorInvalidValue;
    }
    Size point_elements = 0;
    if (!checkedSizeProduct(n_points, point_dim, point_elements))
    {
        return cudaErrorInvalidValue;
    }
    (void)point_elements;

    int device_id = 0;
    cudaError_t device_status = cudaGetDevice(&device_id);
    if (device_status != cudaSuccess)
    {
        return device_status;
    }
    cudaDeviceProp prop{};
    cudaError_t prop_status = cudaGetDeviceProperties(&prop, device_id);
    if (prop_status != cudaSuccess)
    {
        return prop_status;
    }

    const Size device_shared_limit = static_cast<Size>(prop.sharedMemPerBlock);
    const Size effective_shared_limit =
        (shared_memory_budget_bytes == 0)
            ? device_shared_limit
            : std::min(shared_memory_budget_bytes, device_shared_limit);

    const double max_radius_sq = max_radius * max_radius;
    if (!std::isfinite(max_radius_sq))
    {
        return cudaErrorInvalidValue;
    }
    const int tile_size = selectTileSize(point_dim, static_cast<int>(effective_shared_limit),
                                         prop.maxThreadsPerBlock);
    if (tile_size == 0)
    {
        return launchDistanceMatrixGeneric(points, distances, n_points, point_dim, max_radius_sq,
                                           stream);
    }

    const dim3 block(static_cast<unsigned int>(tile_size), static_cast<unsigned int>(tile_size), 1);
    const Size tiles_per_axis =
        (n_points + static_cast<Size>(tile_size) - 1) / static_cast<Size>(tile_size);
    if (tiles_per_axis == 0 ||
        tiles_per_axis > static_cast<Size>(std::numeric_limits<unsigned int>::max()))
    {
        return launchDistanceMatrixGeneric(points, distances, n_points, point_dim, max_radius_sq,
                                           stream);
    }
    if (tiles_per_axis > static_cast<Size>(prop.maxGridSize[0]) ||
        tiles_per_axis > static_cast<Size>(prop.maxGridSize[1]))
    {
        return launchDistanceMatrixGeneric(points, distances, n_points, point_dim, max_radius_sq,
                                           stream);
    }
    const dim3 grid(static_cast<unsigned int>(tiles_per_axis),
                    static_cast<unsigned int>(tiles_per_axis), 1);
    const Size shared_mem_bytes = tileSharedBytes(tile_size, point_dim);
    if (shared_mem_bytes > static_cast<Size>(std::numeric_limits<int>::max()))
    {
        return cudaErrorInvalidConfiguration;
    }
    const int shared_mem_size = static_cast<int>(shared_mem_bytes);
    switch (tile_size)
    {
        case kTileSizeLarge:
            computeDistanceMatrixKernelTiled<kTileSizeLarge>
                <<<grid, block, shared_mem_size, stream>>>(points, distances, n_points, point_dim,
                                                           max_radius_sq);
            break;
        case kTileSizeMedium:
            computeDistanceMatrixKernelTiled<kTileSizeMedium>
                <<<grid, block, shared_mem_size, stream>>>(points, distances, n_points, point_dim,
                                                           max_radius_sq);
            break;
        case kTileSizeSmall:
            computeDistanceMatrixKernelTiled<kTileSizeSmall>
                <<<grid, block, shared_mem_size, stream>>>(points, distances, n_points, point_dim,
                                                           max_radius_sq);
            break;
        default:
            return cudaErrorInvalidConfiguration;
    }

    return cudaGetLastError();
}

cudaError_t computeDistanceMatrixBatchOptimized(const double *const *points_batch,
                                                double **distances_batch, const Size *n_points,
                                                Size point_dim, Size batch_size,
                                                cudaStream_t stream)
{
    if (batch_size == 0)
    {
        return cudaSuccess;
    }
    if (points_batch == nullptr || distances_batch == nullptr || n_points == nullptr)
    {
        return cudaErrorInvalidValue;
    }
    if (point_dim == 0)
    {
        return cudaErrorInvalidValue;
    }

    const int threads_per_block = DISTANCE_TILED_BATCH_THREADS;
    const int num_blocks = static_cast<int>(
        std::min<Size>(batch_size, static_cast<Size>(DISTANCE_TILED_BATCH_MAX_BLOCKS)));
    const dim3 block(threads_per_block, 1, 1);
    const dim3 grid(std::max(1, num_blocks), 1, 1);
    const double max_radius_sq = std::numeric_limits<double>::infinity();

    computeDistanceMatrixBatchPersistent<<<grid, block, 0, stream>>>(
        points_batch, distances_batch, n_points, point_dim, batch_size, max_radius_sq);
    return cudaGetLastError();
}

} // namespace nerve::persistence::accelerated

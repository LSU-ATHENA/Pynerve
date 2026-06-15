
// Focuses on: occupancy, coalescing, shared memory, warp shuffle

#include "nerve/gpu/gpu_error.hpp"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace nerve
{
namespace gpu
{
namespace detail
{

// GPU Kernel Launch Constants
constexpr int DISTANCE_OPT_BLOCK_SIZE = 256;
constexpr int DISTANCE_OPT_MIN_BLOCKS = 4;
constexpr int DISTANCE_OPT_MAX_POINT_DIM = 16;     // Max dimension for register caching
constexpr int DISTANCE_OPT_MAX_GRID_BLOCKS = 1024; // Limit grid for better occupancy
constexpr int DISTANCE_OPT_SMALL_BLOCK_SIZE = 128; // For higher occupancy with smaller blocks
constexpr int DISTANCE_OPT_MIN_SHARED_BLOCK_SIZE = 32;
constexpr unsigned int DISTANCE_OPT_FULL_WARP_MASK = 0xFFFFFFFF; // All 32 threads active in warp
constexpr double DISTANCE_OPT_NO_EDGE_DISTANCE = std::numeric_limits<double>::infinity();

int ceilDivPositive(int value, int divisor)
{
    return (value / divisor) + ((value % divisor) == 0 ? 0 : 1);
}

bool checkedSharedBytes(int block_size, int point_dim, size_t &bytes)
{
    if (block_size <= 0 || point_dim <= 0)
    {
        return false;
    }
    const size_t block = static_cast<size_t>(block_size);
    const size_t dim = static_cast<size_t>(point_dim);
    if (dim > std::numeric_limits<size_t>::max() / block)
    {
        return false;
    }
    const size_t elements = block * dim;
    if (elements > std::numeric_limits<size_t>::max() / sizeof(double))
    {
        return false;
    }
    bytes = elements * sizeof(double);
    return true;
}

__device__ inline bool accumulateSquaredDifference(double diff, double &sum)
{
    const double next = fma(diff, diff, sum);
    if (!isfinite(diff) || !isfinite(next))
    {
        return false;
    }
    sum = next;
    return true;
}

// Kernel launch bounds for optimal occupancy
// 128 threads = 10 blocks per SM (max occupancy on most GPUs)
// 256 threads = 5-6 blocks per SM (good balance)
// 512 threads = 2-3 blocks per SM (for register-heavy kernels)

// Shared-memory row-tile kernel. Each block owns one contiguous group of
// output rows and walks over column tiles staged in shared memory.
__global__ void __launch_bounds__(DISTANCE_OPT_BLOCK_SIZE, DISTANCE_OPT_MIN_BLOCKS)
    distanceMatrixTiledKernel(const double *__restrict__ points,
                              double *__restrict__ distance_matrix, int n_points, int point_dim,
                              double max_radius_sq)
{
    extern __shared__ double shared_mem[];

    const int tile_size = blockDim.x;
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int i = bid * tile_size + tid;
    const bool row_active = i < n_points;

    // Keep inactive threads alive so every thread reaches the tile barriers.
    double point_i[DISTANCE_OPT_MAX_POINT_DIM];
    const int cached_dim =
        point_dim < DISTANCE_OPT_MAX_POINT_DIM ? point_dim : DISTANCE_OPT_MAX_POINT_DIM;
    const double *point_i_ptr = row_active ? points + static_cast<size_t>(i) * point_dim : points;

#pragma unroll
    for (int d = 0; d < cached_dim; ++d)
    {
        point_i[d] = row_active ? point_i_ptr[d] : 0.0;
    }

    for (int tile_j = bid; tile_j < (n_points + tile_size - 1) / tile_size; ++tile_j)
    {
        double *shared_points = shared_mem;
        const int j_base = tile_j * tile_size;

        for (int d = tid; d < point_dim; d += blockDim.x)
        {
            for (int t = 0; t < tile_size && (j_base + t) < n_points; ++t)
            {
                shared_points[t * point_dim + d] =
                    points[static_cast<size_t>(j_base + t) * point_dim + d];
            }
        }

        __syncthreads();

        for (int t = 0; t < tile_size; ++t)
        {
            int j = j_base + t;

            if (!row_active || j >= n_points || j < i)
                continue;

            double sum = 0.0;
            bool valid_distance = true;

#pragma unroll
            for (int d = 0; d < cached_dim; ++d)
            {
                const double diff = point_i[d] - shared_points[t * point_dim + d];
                if (!accumulateSquaredDifference(diff, sum))
                {
                    valid_distance = false;
                    break;
                }
            }
            for (int d = cached_dim; valid_distance && d < point_dim; ++d)
            {
                const double diff = point_i_ptr[d] - shared_points[t * point_dim + d];
                valid_distance = accumulateSquaredDifference(diff, sum);
            }

            double dist_sq = sum;

            // Store if within radius (early exit optimization)
            if (valid_distance && dist_sq <= max_radius_sq)
            {
                double dist = sqrt(dist_sq);
                distance_matrix[static_cast<size_t>(i) * n_points + j] = dist;
                distance_matrix[static_cast<size_t>(j) * n_points + i] = dist;
            }
            else
            {
                distance_matrix[static_cast<size_t>(i) * n_points + j] =
                    DISTANCE_OPT_NO_EDGE_DISTANCE;
                distance_matrix[static_cast<size_t>(j) * n_points + i] =
                    DISTANCE_OPT_NO_EDGE_DISTANCE;
            }
        }

        __syncthreads();
    }
}

// Default path for high-dimensional point clouds where shared-memory tiling
// would exceed per-block limits on the active device.
__global__ void __launch_bounds__(256)
    distanceMatrixGlobalKernel(const double *__restrict__ points,
                               double *__restrict__ distance_matrix, int n_points, int point_dim,
                               double max_radius_sq)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_points)
    {
        return;
    }

    const double *point_i = points + static_cast<size_t>(i) * point_dim;
    for (int j = i; j < n_points; ++j)
    {
        const double *point_j = points + static_cast<size_t>(j) * point_dim;
        double dist_sq = 0.0;
        bool valid_distance = true;
        for (int d = 0; d < point_dim; ++d)
        {
            const double diff = point_i[d] - point_j[d];
            if (!accumulateSquaredDifference(diff, dist_sq))
            {
                valid_distance = false;
                break;
            }
            if (dist_sq > max_radius_sq)
            {
                break;
            }
        }

        const double out = (valid_distance && dist_sq <= max_radius_sq)
                               ? sqrt(dist_sq)
                               : DISTANCE_OPT_NO_EDGE_DISTANCE;
        distance_matrix[static_cast<size_t>(i) * n_points + j] = out;
        distance_matrix[static_cast<size_t>(j) * n_points + i] = out;
    }
}

// Warp-shuffle optimized reduction for finding max edge in simplex
__device__ inline double warpReduceMax(double val)
{
    // Use warp shuffle for intra-warp reduction
    // No shared memory needed, no barriers needed
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        double other = __shfl_down_sync(DISTANCE_OPT_FULL_WARP_MASK, val, offset);
        val = fmax(val, other);
    }
    return val;
}

// Optimized simplex filtration value using warp shuffle
__global__ void __launch_bounds__(128, 8)
    simplexFiltrationKernel(const double *__restrict__ distance_matrix,
                            const int *__restrict__ simplex_vertices,
                            const int *__restrict__ simplex_sizes,
                            double *__restrict__ filtration_values, int n_simplices, int max_dim,
                            int n_points)
{
    const int simplex_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (simplex_idx >= n_simplices)
        return;

    int simplex_size = simplex_sizes[simplex_idx];
    const int *vertices = &simplex_vertices[static_cast<size_t>(simplex_idx) * (max_dim + 1)];

    // Each thread finds max edge in its assigned simplices
    double max_edge = 0.0;

    // Unrolled loop for common cases
    if (simplex_size == 2)
    {
        // Edge
        int v0 = vertices[0];
        int v1 = vertices[1];
        max_edge = distance_matrix[static_cast<size_t>(v0) * n_points + v1];
    }
    else if (simplex_size == 3)
    {
        // Triangle - 3 edges
        int v[3] = {vertices[0], vertices[1], vertices[2]};
        max_edge = fmax(distance_matrix[static_cast<size_t>(v[0]) * n_points + v[1]],
                        fmax(distance_matrix[static_cast<size_t>(v[0]) * n_points + v[2]],
                             distance_matrix[static_cast<size_t>(v[1]) * n_points + v[2]]));
    }
    else if (simplex_size == 4)
    {
        // Tetrahedron - 6 edges
        int v[4] = {vertices[0], vertices[1], vertices[2], vertices[3]};
        max_edge = 0.0;
        for (int i = 0; i < 4; ++i)
        {
            for (int j = i + 1; j < 4; ++j)
            {
                max_edge =
                    fmax(max_edge, distance_matrix[static_cast<size_t>(v[i]) * n_points + v[j]]);
            }
        }
    }
    else
    {
        // General case
        for (int i = 0; i < simplex_size; ++i)
        {
            for (int j = i + 1; j < simplex_size; ++j)
            {
                int vi = vertices[i];
                int vj = vertices[j];
                double edge = distance_matrix[static_cast<size_t>(vi) * n_points + vj];
                max_edge = fmax(max_edge, edge);
            }
        }
    }

    filtration_values[simplex_idx] = isfinite(max_edge) ? max_edge : DISTANCE_OPT_NO_EDGE_DISTANCE;
}

// Occupancy-optimized edge detection kernel
// Uses vectorized loads for better memory throughput
__global__ void __launch_bounds__(256, 4)
    edgeDetectionOptimizedKernel(const double *__restrict__ points,
                                 int *__restrict__ adjacency_matrix, int *__restrict__ edge_count,
                                 int n_points, int point_dim, double max_radius_sq)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    // Each thread processes a row (point i)
    for (int i = tid; i < n_points; i += stride)
    {
        const double *p_i = &points[static_cast<size_t>(i) * point_dim];

        int local_edge_count = 0;

        for (int j = i + 1; j < n_points; ++j)
        {
            const double *p_j = &points[static_cast<size_t>(j) * point_dim];
            double sum = 0.0;
            bool valid_distance = true;

            // Process adjacent coordinates together to reduce loop overhead
            // without relying on host-only prefetch intrinsics in device code.
            int d = 0;
            for (; d + 2 <= point_dim; d += 2)
            {
                double2 diff;
                diff.x = p_i[d] - p_j[d];
                diff.y = p_i[d + 1] - p_j[d + 1];
                if (!accumulateSquaredDifference(diff.x, sum) ||
                    !accumulateSquaredDifference(diff.y, sum))
                {
                    valid_distance = false;
                    break;
                }
            }

            // Remainder
            for (; valid_distance && d < point_dim; ++d)
            {
                double diff = p_i[d] - p_j[d];
                valid_distance = accumulateSquaredDifference(diff, sum);
            }

            // Invalid or overflowing distances are treated as absent edges.
            int is_edge = (valid_distance && sum <= max_radius_sq) ? 1 : 0;
            adjacency_matrix[static_cast<size_t>(i) * n_points + j] = is_edge;
            adjacency_matrix[static_cast<size_t>(j) * n_points + i] = is_edge;

            local_edge_count += is_edge;
        }

        // Atomic add to global edge count
        if (local_edge_count > 0)
        {
            atomicAdd(edge_count, local_edge_count);
        }
    }
}

// Stream-ordered (async) distance computation with overlap
void launchDistanceMatrixAsync(const double *d_points, double *d_distance_matrix, int n_points,
                               int point_dim, double max_radius, cudaStream_t stream)
{
    if (d_points == nullptr || d_distance_matrix == nullptr || n_points <= 0 || point_dim <= 0 ||
        !std::isfinite(max_radius) || !(max_radius > 0.0))
    {
        return;
    }
    const double max_radius_sq = max_radius * max_radius;
    if (!std::isfinite(max_radius_sq))
    {
        return;
    }

    int device_id = 0;
    if (cudaGetDevice(&device_id) != cudaSuccess)
    {
        return;
    }
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device_id) != cudaSuccess)
    {
        return;
    }

    const size_t shared_limit = static_cast<size_t>(prop.sharedMemPerBlock);
    int block_size = DISTANCE_OPT_BLOCK_SIZE;
    while (block_size >= DISTANCE_OPT_MIN_SHARED_BLOCK_SIZE)
    {
        size_t required = 0;
        if (!checkedSharedBytes(block_size, point_dim, required))
        {
            return;
        }
        if (required <= shared_limit)
        {
            break;
        }
        block_size /= 2;
    }

    if (block_size < DISTANCE_OPT_MIN_SHARED_BLOCK_SIZE)
    {
        const int generic_block = DISTANCE_OPT_SMALL_BLOCK_SIZE;
        const int generic_grid = ceilDivPositive(n_points, generic_block);
        distanceMatrixGlobalKernel<<<generic_grid, generic_block, 0, stream>>>(
            d_points, d_distance_matrix, n_points, point_dim, max_radius_sq);
        GPU_CHECK(cudaPeekAtLastError());
        return;
    }

    size_t shared_mem_size = 0;
    if (!checkedSharedBytes(block_size, point_dim, shared_mem_size))
    {
        return;
    }
    const int grid_size = ceilDivPositive(n_points, block_size);
    distanceMatrixTiledKernel<<<grid_size, block_size, shared_mem_size, stream>>>(
        d_points, d_distance_matrix, n_points, point_dim, max_radius_sq);
    GPU_CHECK(cudaPeekAtLastError());
}

void launchEdgeDetectionAsync(const double *d_points, int *d_adjacency, int *d_edge_count,
                              int n_points, int point_dim, double max_radius, cudaStream_t stream)
{
    if (d_points == nullptr || d_adjacency == nullptr || d_edge_count == nullptr || n_points <= 0 ||
        point_dim <= 0 || !std::isfinite(max_radius) || !(max_radius > 0.0))
    {
        return;
    }
    double max_radius_sq = max_radius * max_radius;
    if (!std::isfinite(max_radius_sq))
    {
        return;
    }

    const int block_size = DISTANCE_OPT_BLOCK_SIZE;
    const int max_blocks = DISTANCE_OPT_MAX_GRID_BLOCKS;
    const int grid_size = std::min(max_blocks, ceilDivPositive(n_points, block_size));

    edgeDetectionOptimizedKernel<<<grid_size, block_size, 0, stream>>>(
        d_points, d_adjacency, d_edge_count, n_points, point_dim, max_radius_sq);
    GPU_CHECK(cudaPeekAtLastError());
}

void launchSimplexFiltrationAsync(const double *d_distance_matrix, const int *d_simplex_vertices,
                                  const int *d_simplex_sizes, double *d_filtration_values,
                                  int n_simplices, int max_dim, int n_points, cudaStream_t stream)
{
    if (d_distance_matrix == nullptr || d_simplex_vertices == nullptr ||
        d_simplex_sizes == nullptr || d_filtration_values == nullptr || n_simplices <= 0 ||
        max_dim < 0 || n_points <= 0)
    {
        return;
    }
    const int block_size = DISTANCE_OPT_SMALL_BLOCK_SIZE;
    const int grid_size = ceilDivPositive(n_simplices, block_size);

    simplexFiltrationKernel<<<grid_size, block_size, 0, stream>>>(
        d_distance_matrix, d_simplex_vertices, d_simplex_sizes, d_filtration_values, n_simplices,
        max_dim, n_points);
    GPU_CHECK(cudaPeekAtLastError());
}

} // namespace detail
} // namespace gpu
} // namespace nerve

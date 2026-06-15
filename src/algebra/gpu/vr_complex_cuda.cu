
#include "nerve/gpu/gpu_error.hpp"

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace cg = cooperative_groups;

namespace nerve
{
namespace gpu
{
namespace algebra
{
namespace kernels
{

// GPU Kernel Launch Constants
constexpr int VR_BLOCK_SIZE = 256;
constexpr int VR_TILE_SIZE = 16;
constexpr int VR_MAX_POINT_DIM = 32;    // Maximum point dimension for shared memory
constexpr int VR_TILED_THRESHOLD = 512; // Point count threshold for tiled kernel
constexpr int VR_MAX_INT_PAIR_POINTS = 65536;

namespace
{

int checkedPairCount(int n_points, const char *context)
{
    if (n_points < 0 || n_points > VR_MAX_INT_PAIR_POINTS)
    {
        throw std::length_error(context);
    }
    return n_points * (n_points - 1) / 2;
}

int checkedGridSize(int count, int block_size, const char *context)
{
    if (count < 0 || block_size <= 0)
    {
        throw std::invalid_argument(context);
    }
    const int grid_size = (count + block_size - 1) / block_size;
    if (grid_size < 0)
    {
        throw std::length_error(context);
    }
    return grid_size;
}

unsigned int checkedTiledGridAxis(int n_points, const char *context)
{
    if (n_points < 0)
    {
        throw std::invalid_argument(context);
    }
    const int axis = (n_points + VR_TILE_SIZE - 1) / VR_TILE_SIZE;
    if (axis < 0)
    {
        throw std::length_error(context);
    }
    return static_cast<unsigned int>(axis);
}

size_t checkedSharedBytes(int current_dim, int block_size, const char *context)
{
    if (current_dim < 0 || block_size <= 0)
    {
        throw std::invalid_argument(context);
    }
    const size_t width = static_cast<size_t>(current_dim) + 1;
    const size_t block = static_cast<size_t>(block_size);
    if (width != 0 && block > std::numeric_limits<size_t>::max() / width)
    {
        throw std::length_error(context);
    }
    const size_t values = width * block;
    if (values > std::numeric_limits<size_t>::max() / sizeof(int))
    {
        throw std::length_error(context);
    }
    return values * sizeof(int);
}

} // namespace

// Kernel to detect edges within radius (VR graph construction)
// Each thread handles one pair of points
__global__ void __launch_bounds__(256) vrEdgeDetectionKernel(
    const double *__restrict__ points,  // [n_points x dim]
    int *__restrict__ adjacency_matrix, // [n_points x n_points] - output edges
    int n_points, int point_dim, double max_radius_sq)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pairs = n_points * (n_points - 1) / 2;

    if (idx >= total_pairs)
        return;

    // Convert linear index to (i, j) pair where i < j
    int i = 0;
    int j = 0;
    int count = 0;

    // This is O(n) but only done once per thread
    for (int row = 0; row < n_points; ++row)
    {
        int row_pairs = n_points - row - 1;
        if (idx < count + row_pairs)
        {
            i = row;
            j = row + 1 + (idx - count);
            break;
        }
        count += row_pairs;
    }

    if (i >= j)
        return; // Safety check

    double dist_sq = 0.0;
    for (int d = 0; d < point_dim; ++d)
    {
        double diff = points[i * point_dim + d] - points[j * point_dim + d];
        double contribution = diff * diff;
        double next_dist_sq = dist_sq + contribution;
        if (!isfinite(diff) || !isfinite(contribution) || !isfinite(next_dist_sq))
        {
            dist_sq = INFINITY;
            break;
        }
        dist_sq = next_dist_sq;
    }

    // Mark edge if within radius
    if (dist_sq <= max_radius_sq)
    {
        adjacency_matrix[i * n_points + j] = 1;
        adjacency_matrix[j * n_points + i] = 1; // Symmetric
    }
}

// Optimized edge detection using shared memory tiling
__global__ void __launch_bounds__(256)
    vrEdgeDetectionTiledKernel(const double *__restrict__ points,
                               int *__restrict__ adjacency_matrix, int n_points, int point_dim,
                               double max_radius_sq)
{
    constexpr int TILE_SIZE = VR_TILE_SIZE;

    __shared__ double tile_points[TILE_SIZE]
                                 [VR_MAX_POINT_DIM]; // Maximum point dimension for shared memory

    int i = blockIdx.y * TILE_SIZE + threadIdx.y;
    int j = blockIdx.x * TILE_SIZE + threadIdx.x;

    if (i >= n_points || j >= n_points || i >= j)
        return;

    double dist_sq = 0.0;

    // Process dimensions in tiles
    for (int tile = 0; tile < (point_dim + TILE_SIZE - 1) / TILE_SIZE; ++tile)
    {
        int dim_base = tile * TILE_SIZE;

        if (dim_base + threadIdx.x < point_dim && i < n_points)
        {
            tile_points[threadIdx.y][threadIdx.x] = points[i * point_dim + dim_base + threadIdx.x];
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE_SIZE && (dim_base + k) < point_dim; ++k)
        {
            if (j < n_points)
            {
                double p_i = tile_points[threadIdx.y][k];
                double p_j = points[j * point_dim + dim_base + k];
                double diff = p_i - p_j;
                double contribution = diff * diff;
                double next_dist_sq = dist_sq + contribution;
                if (!isfinite(diff) || !isfinite(contribution) || !isfinite(next_dist_sq))
                {
                    dist_sq = INFINITY;
                    break;
                }
                dist_sq = next_dist_sq;

                // Early termination if already too far
                if (dist_sq > max_radius_sq)
                    break;
            }
        }
        __syncthreads();

        if (dist_sq > max_radius_sq)
            break;
    }

    // Mark edge if within radius
    if (dist_sq <= max_radius_sq)
    {
        adjacency_matrix[i * n_points + j] = 1;
        adjacency_matrix[j * n_points + i] = 1;
    }
}

// Kernel to expand cliques and build higher-dimensional simplices
__global__ void __launch_bounds__(256)
    simplexExpansionKernel(const int *__restrict__ adjacency_matrix,  // [n_points x n_points]
                           const int *__restrict__ current_simplices, // [n_current x (dim+1)]
                           int *__restrict__ new_simplices,           // [max_new x (dim+2)] output
                           int *__restrict__ new_count,               // atomic counter
                           int n_current, int current_dim, int n_points, int max_new)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_current)
        return;

    // Each thread handles one current simplex and tries to expand it
    // A simplex can be expanded if there's a vertex connected to all its vertices

    extern __shared__ int shared_adj[]; // Shared adjacency row for each simplex vertex

    // Load vertices of current simplex
    int simplex_vertices[10]; // Assume max dimension 10
    for (int v = 0; v <= current_dim && v < 10; ++v)
    {
        simplex_vertices[v] = current_simplices[idx * (current_dim + 1) + v];
    }

    for (int new_v = 0; new_v < n_points; ++new_v)
    {
        // Check if new_v is connected to all vertices in the simplex
        bool connected = true;
        for (int v = 0; v <= current_dim && v < 10; ++v)
        {
            int sv = simplex_vertices[v];
            if (sv == new_v)
            {
                connected = false; // Can't add same vertex
                break;
            }
            if (adjacency_matrix[sv * n_points + new_v] == 0)
            {
                connected = false;
                break;
            }
        }

        if (connected)
        {
            int pos = atomicAdd(new_count, 1);
            if (pos < max_new)
            {
                for (int v = 0; v <= current_dim && v < 10; ++v)
                {
                    new_simplices[pos * (current_dim + 2) + v] = simplex_vertices[v];
                }
                new_simplices[pos * (current_dim + 2) + current_dim + 1] = new_v;
            }
        }
    }
}

// Kernel to compute filtration values (birth times) for simplices
// For VR: max edge length in the simplex
__global__ void __launch_bounds__(256)
    vrFiltrationValueKernel(const double *__restrict__ points, const int *__restrict__ simplices,
                            const int *__restrict__ simplex_sizes,
                            double *__restrict__ filtration_values, int n_simplices, int point_dim,
                            int n_points)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_simplices)
        return;

    int size = simplex_sizes[idx];
    double max_edge = 0.0;

    for (int i = 0; i < size; ++i)
    {
        for (int j = i + 1; j < size; ++j)
        {
            int v_i = simplices[idx * point_dim + i];
            int v_j = simplices[idx * point_dim + j];

            double dist_sq = 0.0;
            for (int d = 0; d < point_dim; ++d)
            {
                double diff = points[v_i * point_dim + d] - points[v_j * point_dim + d];
                double contribution = diff * diff;
                double next_dist_sq = dist_sq + contribution;
                if (!isfinite(diff) || !isfinite(contribution) || !isfinite(next_dist_sq))
                {
                    dist_sq = INFINITY;
                    break;
                }
                dist_sq = next_dist_sq;
            }
            double dist = sqrt(dist_sq);

            if (dist > max_edge)
            {
                max_edge = dist;
            }
        }
    }

    filtration_values[idx] = max_edge;
}

// Kernel for batched VR complex construction (all dimensions at once)
__global__ void __launch_bounds__(256)
    vrConstructionBatchKernel(const double *__restrict__ points,
                              int *__restrict__ edge_list, // [n_edges x 2]
                              int *__restrict__ edge_count, int n_points, int point_dim,
                              double max_radius_sq, int max_edges)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pairs = n_points * (n_points - 1) / 2;

    if (idx >= total_pairs)
        return;

    // Convert to (i, j) pair
    int i = 0, j = 0, count = 0;
    for (int row = 0; row < n_points; ++row)
    {
        int row_pairs = n_points - row - 1;
        if (idx < count + row_pairs)
        {
            i = row;
            j = row + 1 + (idx - count);
            break;
        }
        count += row_pairs;
    }

    if (i >= j)
        return;

    // Compute distance
    double dist_sq = 0.0;
    for (int d = 0; d < point_dim; ++d)
    {
        double diff = points[i * point_dim + d] - points[j * point_dim + d];
        double contribution = diff * diff;
        double next_dist_sq = dist_sq + contribution;
        if (!isfinite(diff) || !isfinite(contribution) || !isfinite(next_dist_sq))
        {
            dist_sq = INFINITY;
            break;
        }
        dist_sq = next_dist_sq;
    }

    // Add edge if within radius
    if (dist_sq <= max_radius_sq)
    {
        int pos = atomicAdd(edge_count, 1);
        if (pos < max_edges)
        {
            edge_list[pos * 2] = i;
            edge_list[pos * 2 + 1] = j;
        }
    }
}

// Host wrapper functions
void launchVrEdgeDetection(const double *d_points, int *d_adjacency, int n_points, int point_dim,
                           double max_radius_sq, cudaStream_t stream)
{
    if (d_points == nullptr || d_adjacency == nullptr || n_points <= 1 || point_dim <= 0 ||
        !std::isfinite(max_radius_sq) || max_radius_sq < 0.0)
    {
        return;
    }
    // Use tiled version for better performance with large point clouds
    if (n_points > VR_TILED_THRESHOLD)
    {
        constexpr int TILE_SIZE = VR_TILE_SIZE;
        dim3 block(TILE_SIZE, TILE_SIZE);
        const unsigned int grid_axis =
            checkedTiledGridAxis(n_points, "VR tiled edge grid exceeds CUDA limits");
        dim3 grid(grid_axis, grid_axis);

        vrEdgeDetectionTiledKernel<<<grid, block, 0, stream>>>(d_points, d_adjacency, n_points,
                                                               point_dim, max_radius_sq);
    }
    else
    {
        int total_pairs = checkedPairCount(n_points, "VR edge pair count exceeds int range");
        int block_size = VR_BLOCK_SIZE;
        int grid_size =
            checkedGridSize(total_pairs, block_size, "VR edge grid exceeds CUDA limits");

        vrEdgeDetectionKernel<<<grid_size, block_size, 0, stream>>>(d_points, d_adjacency, n_points,
                                                                    point_dim, max_radius_sq);
    }
    GPU_CHECK(cudaPeekAtLastError());
}

void launchSimplexExpansion(const int *d_adjacency, const int *d_current_simplices,
                            int *d_new_simplices, int *d_new_count, int n_current, int current_dim,
                            int n_points, int max_new, cudaStream_t stream)
{
    if (d_adjacency == nullptr || d_current_simplices == nullptr || d_new_simplices == nullptr ||
        d_new_count == nullptr || n_current <= 0 || current_dim < 0 || n_points <= 0 ||
        max_new <= 0)
    {
        return;
    }
    constexpr int EXPANSION_BLOCK_SIZE = 256;
    int block_size = EXPANSION_BLOCK_SIZE;
    int grid_size =
        checkedGridSize(n_current, block_size, "VR simplex expansion grid exceeds CUDA limits");
    size_t shared_mem =
        checkedSharedBytes(current_dim, block_size, "VR simplex expansion shared memory overflows");

    simplexExpansionKernel<<<grid_size, block_size, shared_mem, stream>>>(
        d_adjacency, d_current_simplices, d_new_simplices, d_new_count, n_current, current_dim,
        n_points, max_new);
    GPU_CHECK(cudaPeekAtLastError());
}

void launchVrFiltrationValues(const double *d_points, const int *d_simplices,
                              const int *d_simplex_sizes, double *d_filtration_values,
                              int n_simplices, int point_dim, int n_points, cudaStream_t stream)
{
    if (d_points == nullptr || d_simplices == nullptr || d_simplex_sizes == nullptr ||
        d_filtration_values == nullptr || n_simplices <= 0 || point_dim <= 0 || n_points <= 0)
    {
        return;
    }
    constexpr int FILTRATION_BLOCK_SIZE = 256;
    int block_size = FILTRATION_BLOCK_SIZE;
    int grid_size =
        checkedGridSize(n_simplices, block_size, "VR filtration grid exceeds CUDA limits");

    vrFiltrationValueKernel<<<grid_size, block_size, 0, stream>>>(
        d_points, d_simplices, d_simplex_sizes, d_filtration_values, n_simplices, point_dim,
        n_points);
    GPU_CHECK(cudaPeekAtLastError());
}

} // namespace kernels
} // namespace algebra
} // namespace gpu
} // namespace nerve

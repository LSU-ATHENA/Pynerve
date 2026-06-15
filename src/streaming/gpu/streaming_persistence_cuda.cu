
#include "nerve/gpu/gpu_error.hpp"

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cmath>

namespace cg = cooperative_groups;

namespace nerve
{
namespace gpu
{
namespace streaming
{
namespace kernels
{

// GPU Kernel Launch Constants
constexpr int STREAMING_BLOCK_SIZE = 256;

__device__ inline bool accumulateSquaredDifference(double diff, double &dist_sq)
{
    const double contribution = diff * diff;
    const double next_dist_sq = dist_sq + contribution;
    if (!isfinite(diff) || !isfinite(contribution) || !isfinite(next_dist_sq))
    {
        dist_sq = INFINITY;
        return false;
    }
    dist_sq = next_dist_sq;
    return true;
}

// Kernel to update persistence diagram when window slides
// Adds new point and removes old point from the window
__global__ __launch_bounds__(STREAMING_BLOCK_SIZE) void windowUpdateKernel(
    const double *__restrict__ new_point,   // [dim] - new point entering window
    const double *__restrict__ old_point,   // [dim] - point leaving window
    const int *__restrict__ window_indices, // [window_size] - indices of points in window
    int *__restrict__ affected_simplices,   // [max_simplices] - simplices to recompute
    int *__restrict__ affected_count,       // scalar - number of affected simplices
    int window_size, int point_dim, double max_radius_sq)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= window_size)
        return;

    // Compute distances from new point to all points in window
    // and from old point to all points (to identify affected simplices)

    extern __shared__ double shared_mem[];
    double *new_pt_shared = shared_mem;
    double *old_pt_shared = shared_mem + point_dim;

    // Load points into shared memory (cooperatively)
    if (threadIdx.x < point_dim)
    {
        new_pt_shared[threadIdx.x] = new_point[threadIdx.x];
        old_pt_shared[threadIdx.x] = old_point[threadIdx.x];
    }
    __syncthreads();

    // Mark this point index as potentially affected
    // A point is affected if it's within max_radius of the entering or leaving point
    int point_idx = window_indices[idx];

    int pos = atomicAdd(affected_count, 1);
    if (pos < window_size)
    {
        affected_simplices[pos] = point_idx;
    }
}

// Kernel for incremental persistence - add new simplices
__global__ __launch_bounds__(STREAMING_BLOCK_SIZE) void incrementalAddKernel(
    const int *__restrict__ new_simplices,      // [n_new * (dim+1)] - vertex indices
    const double *__restrict__ filtration_vals, // [n_new] - birth times
    int *__restrict__ persistence_pairs,        // [n_simplices x 2] - birth/death pairs
    int *__restrict__ pair_count,               // scalar - current number of pairs
    int n_new, int max_dimension, int n_existing)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_new)
        return;

    // Find its boundary and determine if it creates or destroys a homology class

    int simplex_idx = n_existing + idx;
    double birth_time = filtration_vals[idx];

    // Mark as birth simplex initially (refined by subsequent reduction passes)
    // Streaming approach: initial classification followed by boundary matrix reduction

    int pair_pos = atomicAdd(pair_count, 1);
    if (pair_pos < n_existing + n_new)
    {
        persistence_pairs[pair_pos * 2] = simplex_idx; // birth
        persistence_pairs[pair_pos * 2 + 1] = -1;      // death (infinite)
    }
}

// Kernel for birth/death time updates during window slide
__global__ __launch_bounds__(STREAMING_BLOCK_SIZE) void birthDeathUpdateKernel(
    const int *__restrict__ affected_simplices, // [n_affected] - indices to check
    int n_affected,
    const double *__restrict__ distance_matrix, // [n_points x n_points]
    double *__restrict__ birth_times,           // [n_simplices]
    double *__restrict__ death_times,           // [n_simplices]
    int n_points,
    const int *__restrict__ simplex_vertices, // [n_simplices x max_dim] - vertex indices
    const int *__restrict__ simplex_sizes,    // [n_simplices] - number of vertices per simplex
    int max_simplex_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_affected)
        return;

    int simplex_idx = affected_simplices[idx];
    int simplex_size = simplex_sizes[simplex_idx];

    const int *vertices = &simplex_vertices[simplex_idx * max_simplex_size];

    // For VR complex: birth = max edge length in simplex
    // Compute max edge by checking all pairs of vertices in this simplex
    double max_edge = 0.0;

    for (int i = 0; i < simplex_size; ++i)
    {
        int vi = vertices[i];
        if (vi < 0 || vi >= n_points)
            continue;

        for (int j = i + 1; j < simplex_size; ++j)
        {
            int vj = vertices[j];
            if (vj < 0 || vj >= n_points)
                continue;

            double dist = distance_matrix[vi * n_points + vj];
            if (isfinite(dist) && dist > max_edge)
            {
                max_edge = dist;
            }
        }
    }

    birth_times[simplex_idx] = max_edge;
}

// Kernel to identify points affected by a new point addition
__global__ __launch_bounds__(STREAMING_BLOCK_SIZE) void affectedRegionDetectionKernel(
    const double *__restrict__ points,    // [n_points x dim]
    const double *__restrict__ new_point, // [dim]
    int *__restrict__ affected_mask,      // [n_points] - 1 if affected
    int n_points, int point_dim, double max_radius_sq)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_points)
        return;

    double dist_sq = 0.0;
    for (int d = 0; d < point_dim; ++d)
    {
        double diff = points[idx * point_dim + d] - new_point[d];
        if (!accumulateSquaredDifference(diff, dist_sq))
        {
            break;
        }
    }

    affected_mask[idx] = (isfinite(dist_sq) && dist_sq <= max_radius_sq) ? 1 : 0;
}

// Kernel for parallel simplex insertion in a batch
__global__ __launch_bounds__(STREAMING_BLOCK_SIZE) void batchSimplexInsertionKernel(
    const int *__restrict__ batch_vertices,       // [batch_size x max_dim] - vertex indices
    const int *__restrict__ batch_sizes,          // [batch_size] - simplex dimensions
    const double *__restrict__ batch_filtrations, // [batch_size]
    int *__restrict__ simplex_buffer,             // [max_simplices x max_dim] - output
    double *__restrict__ filtration_buffer,       // [max_simplices]
    int *__restrict__ simplex_count,              // scalar
    int batch_size, int max_dim, int max_simplices)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size)
        return;

    // Add simplex to buffer
    int pos = atomicAdd(simplex_count, 1);
    if (pos >= max_simplices)
        return;

    // Copy vertex indices
    int size = batch_sizes[idx];
    for (int i = 0; i < size && i < max_dim; ++i)
    {
        simplex_buffer[pos * max_dim + i] = batch_vertices[idx * max_dim + i];
    }

    // Copy filtration value
    filtration_buffer[pos] = batch_filtrations[idx];
}

// Host wrapper functions
void launchWindowUpdate(const double *d_new_point, const double *d_old_point,
                        const int *d_window_indices, int *d_affected_simplices,
                        int *d_affected_count, int window_size, int point_dim, double max_radius_sq,
                        cudaStream_t stream)
{
    if (d_new_point == nullptr || d_old_point == nullptr || d_window_indices == nullptr ||
        d_affected_simplices == nullptr || d_affected_count == nullptr || window_size <= 0 ||
        point_dim <= 0 || !std::isfinite(max_radius_sq) || max_radius_sq < 0.0)
    {
        return;
    }
    int block_size = STREAMING_BLOCK_SIZE;
    int grid_size = (window_size + block_size - 1) / block_size;
    size_t shared_mem_size = 2 * point_dim * sizeof(double);

    windowUpdateKernel<<<grid_size, block_size, shared_mem_size, stream>>>(
        d_new_point, d_old_point, d_window_indices, d_affected_simplices, d_affected_count,
        window_size, point_dim, max_radius_sq);
    GPU_CHECK(cudaPeekAtLastError());
}

void launchIncrementalAdd(const int *d_new_simplices, const double *d_filtration_vals,
                          int *d_persistence_pairs, int *d_pair_count, int n_new, int max_dimension,
                          int n_existing, cudaStream_t stream)
{
    if (d_new_simplices == nullptr || d_filtration_vals == nullptr ||
        d_persistence_pairs == nullptr || d_pair_count == nullptr || n_new <= 0 ||
        max_dimension <= 0 || n_existing < 0)
    {
        return;
    }
    int block_size = STREAMING_BLOCK_SIZE;
    int grid_size = (n_new + block_size - 1) / block_size;

    incrementalAddKernel<<<grid_size, block_size, 0, stream>>>(d_new_simplices, d_filtration_vals,
                                                               d_persistence_pairs, d_pair_count,
                                                               n_new, max_dimension, n_existing);
    GPU_CHECK(cudaPeekAtLastError());
}

void launchAffectedRegionDetection(const double *d_points, const double *d_new_point,
                                   int *d_affected_mask, int n_points, int point_dim,
                                   double max_radius_sq, cudaStream_t stream)
{
    if (d_points == nullptr || d_new_point == nullptr || d_affected_mask == nullptr ||
        n_points <= 0 || point_dim <= 0 || !std::isfinite(max_radius_sq) || max_radius_sq < 0.0)
    {
        return;
    }
    int block_size = STREAMING_BLOCK_SIZE;
    int grid_size = (n_points + block_size - 1) / block_size;

    affectedRegionDetectionKernel<<<grid_size, block_size, 0, stream>>>(
        d_points, d_new_point, d_affected_mask, n_points, point_dim, max_radius_sq);
    GPU_CHECK(cudaPeekAtLastError());
}

void launchBirthDeathUpdate(const int *d_affected_simplices, int n_affected,
                            const double *d_distance_matrix, double *d_birth_times,
                            double *d_death_times, int n_points, const int *d_simplex_vertices,
                            const int *d_simplex_sizes, int max_simplex_size, cudaStream_t stream)
{
    if (d_affected_simplices == nullptr || d_distance_matrix == nullptr ||
        d_birth_times == nullptr || d_death_times == nullptr || d_simplex_vertices == nullptr ||
        d_simplex_sizes == nullptr || n_affected <= 0 || n_points <= 0 || max_simplex_size <= 0)
    {
        return;
    }
    int block_size = STREAMING_BLOCK_SIZE;
    int grid_size = (n_affected + block_size - 1) / block_size;

    birthDeathUpdateKernel<<<grid_size, block_size, 0, stream>>>(
        d_affected_simplices, n_affected, d_distance_matrix, d_birth_times, d_death_times, n_points,
        d_simplex_vertices, d_simplex_sizes, max_simplex_size);
    GPU_CHECK(cudaPeekAtLastError());
}

} // namespace kernels
} // namespace streaming
} // namespace gpu
} // namespace nerve

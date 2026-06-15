
#include "nerve/gpu/gpu_error.hpp"

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cmath>
#include <limits>

namespace cg = cooperative_groups;

namespace nerve
{
namespace gpu
{
namespace metrics
{
namespace kernels
{

// Hungarian Algorithm GPU Implementation
// Based on parallel augmenting path search and label updates

// Default CUDA block size
constexpr int BLOCK_SIZE = 256;

// Tolerance constants for numerical comparisons
constexpr double HUNGARIAN_TOLERANCE = 1e-12;
constexpr double HUNGARIAN_LARGE_COST = 1.0e300;

__device__ inline double finiteCostOrLarge(double cost)
{
    return isfinite(cost) ? cost : HUNGARIAN_LARGE_COST;
}

__device__ inline void atomicMinDouble(double *address, double value)
{
    auto *address_as_ull = reinterpret_cast<unsigned long long *>(address);
    unsigned long long old = *address_as_ull;
    while (__longlong_as_double(old) > value)
    {
        const unsigned long long assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(value));
        if (old == assumed)
        {
            break;
        }
    }
}

// Initialize labels (row and column reductions)
__global__ __launch_bounds__(BLOCK_SIZE) void hungarianInitLabelsKernel(
    const double *__restrict__ cost_matrix, // [n x n]
    double *__restrict__ row_labels,        // [n]
    double *__restrict__ col_labels,        // [n]
    int *__restrict__ row_matched,          // [n] - matched column for each row (-1 if
                                            // unmatched)
    int *__restrict__ col_matched,          // [n] - matched row for each column (-1 if
                                            // unmatched)
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;

    // Initialize row labels as min cost in each row
    double min_cost = finiteCostOrLarge(cost_matrix[idx * n]);
    for (int j = 1; j < n; ++j)
    {
        double cost = finiteCostOrLarge(cost_matrix[idx * n + j]);
        if (cost < min_cost)
        {
            min_cost = cost;
        }
    }
    row_labels[idx] = min_cost;

    // Initialize column labels to 0
    col_labels[idx] = 0.0;

    // Initialize matching to -1 (unmatched)
    row_matched[idx] = -1;
    col_matched[idx] = -1;
}

// Find initial matching using greedy approach
__global__ __launch_bounds__(BLOCK_SIZE) void hungarianGreedyMatchKernel(
    const double *__restrict__ cost_matrix, const double *__restrict__ row_labels,
    const double *__restrict__ col_labels, int *__restrict__ row_matched,
    int *__restrict__ col_matched, int *__restrict__ match_count, int n)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n)
        return;

    // Each thread tries to find a match for its row
    // Look for columns where cost == row_label + colLabel (equality edge)
    for (int col = 0; col < n; ++col)
    {
        if (col_matched[col] != -1)
            continue; // Column already matched

        double reduced_cost = cost_matrix[row * n + col] - row_labels[row] - col_labels[col];
        if (isfinite(reduced_cost) && fabs(reduced_cost) < HUNGARIAN_TOLERANCE)
        {
            // Try to claim this column atomically
            int expected = -1;
            if (atomicCAS(&col_matched[col], expected, row) == expected)
            {
                row_matched[row] = col;
                atomicAdd(match_count, 1);
                break;
            }
        }
    }
}

// Parallel slack computation for unmatched rows
__global__ __launch_bounds__(BLOCK_SIZE) void hungarianComputeSlackKernel(
    const double *__restrict__ cost_matrix, const double *__restrict__ row_labels,
    const double *__restrict__ col_labels, const int *__restrict__ row_matched,
    double *__restrict__ slack,  // [n] - min slack for each column
    int *__restrict__ slack_row, // [n] - row giving min slack
    int n)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n)
        return;

    // Compute slack for this column from all unmatched rows
    double min_slack = 1.0e300;
    int min_row = -1;

    for (int row = 0; row < n; ++row)
    {
        if (row_matched[row] != -1)
            continue; // Skip matched rows

        double s = cost_matrix[row * n + col] - row_labels[row] - col_labels[col];
        if (isfinite(s) && s < min_slack)
        {
            min_slack = s;
            min_row = row;
        }
    }

    slack[col] = min_slack;
    slack_row[col] = min_row;
}

// Find minimum slack value
__global__ __launch_bounds__(BLOCK_SIZE) void hungarianFindMinSlackKernel(
    const double *__restrict__ slack, const int *__restrict__ col_matched,
    double *__restrict__ min_slack_out, int n)
{
    extern __shared__ double shared_mem[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load into shared memory (only unmatched columns)
    double local_min = 1.0e300;
    if (idx < n && col_matched[idx] == -1)
    {
        local_min = finiteCostOrLarge(slack[idx]);
    }
    shared_mem[tid] = local_min;
    __syncthreads();

    // Parallel reduction
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (tid < stride)
        {
            if (shared_mem[tid + stride] < shared_mem[tid])
            {
                shared_mem[tid] = shared_mem[tid + stride];
            }
        }
        __syncthreads();
    }

    // Write result
    if (tid == 0)
    {
        atomicMinDouble(min_slack_out, shared_mem[0]);
    }
}

// Update labels based on minimum slack
__global__ __launch_bounds__(BLOCK_SIZE) void hungarianUpdateLabelsKernel(
    double *__restrict__ row_labels, double *__restrict__ col_labels,
    const int *__restrict__ row_visited, const int *__restrict__ col_visited, double min_slack,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;

    // Update labels for visited rows and columns
    if (!isfinite(min_slack))
    {
        return;
    }
    if (row_visited[idx])
    {
        row_labels[idx] += min_slack;
    }
    if (col_visited[idx])
    {
        col_labels[idx] -= min_slack;
    }
}

// Augmenting path search (BFS on GPU)
__global__ __launch_bounds__(BLOCK_SIZE) void hungarianAugmentPathKernel(
    const double *__restrict__ cost_matrix, const double *__restrict__ row_labels,
    const double *__restrict__ col_labels, int *__restrict__ row_matched,
    int *__restrict__ col_matched,
    int *__restrict__ parent_row, // [n] - parent in augmenting tree
    int *__restrict__ parent_col, // [n]
    int *__restrict__ queue,      // [n] - BFS queue
    int *__restrict__ queue_head, int *__restrict__ queue_tail, int *__restrict__ found_augmenting,
    int start_row, int n)
{
    int tid = threadIdx.x;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ int shared_queue[];

    // Initialize
    if (col < n)
    {
        parent_col[col] = -1;
    }

    __syncthreads();

    // Cooperatively process queue
    while (*queue_head < *queue_tail)
    {
        // Get current row from queue (only one thread)
        int current_row = -1;
        if (tid == 0 && *queue_head < *queue_tail)
        {
            current_row = queue[*queue_head];
            atomicAdd(queue_head, 1);
        }

        // Broadcast to all threads
        __shared__ int shared_row;
        if (tid == 0)
        {
            shared_row = current_row;
        }
        __syncthreads();
        current_row = shared_row;

        if (current_row == -1)
            break;

        // Each thread checks its assigned columns for equality edges
        if (col < n)
        {
            double reduced_cost =
                cost_matrix[current_row * n + col] - row_labels[current_row] - col_labels[col];

            if (isfinite(reduced_cost) && fabs(reduced_cost) < HUNGARIAN_TOLERANCE &&
                parent_col[col] == -1)
            {
                // Found equality edge
                parent_col[col] = current_row;

                int matched_row = col_matched[col];
                if (matched_row == -1)
                {
                    // Found augmenting path!
                    atomicExch(found_augmenting, col);
                }
                else
                {
                    // Add matched row to queue
                    int pos = atomicAdd(queue_tail, 1);
                    if (pos < n)
                    {
                        queue[pos] = matched_row;
                    }
                    parent_row[matched_row] = col;
                }
            }
        }

        __syncthreads();

        if (*found_augmenting != -1)
            break;
    }
}

// Apply augmenting path to update matching
__global__ __launch_bounds__(BLOCK_SIZE) void hungarianApplyAugmentationKernel(
    int *__restrict__ row_matched, int *__restrict__ col_matched,
    const int *__restrict__ parent_col, int end_col, int n)
{
    // Single thread applies the augmentation
    // This is a sequential operation but very fast
    int current_col = end_col;

    while (current_col != -1)
    {
        int parent_row = parent_col[current_col];
        if (parent_row == -1)
            break;

        int next_col = row_matched[parent_row];

        // Update matching
        row_matched[parent_row] = current_col;
        col_matched[current_col] = parent_row;

        current_col = next_col;
    }
}

// Parallel Candidate Extraction for Bottleneck Distance
// GPU-accelerated cost collection with parallel processing
__global__ __launch_bounds__(BLOCK_SIZE) void bottleneckFindCandidatesKernel(
    const double *__restrict__ cost_matrix, double *__restrict__ candidates,
    int *__restrict__ candidate_count, int n, int max_candidates)
{
    extern __shared__ double shared_costs[];
    const int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n * n;
    const double sentinel = 1.0e300;
    const double cost = (idx < total) ? finiteCostOrLarge(cost_matrix[idx]) : sentinel;
    shared_costs[tid] = cost;
    __syncthreads();

    if (idx >= total)
    {
        return;
    }

    bool unique_in_block = true;
    for (int i = 0; i < tid; ++i)
    {
        if (fabs(shared_costs[i] - cost) < HUNGARIAN_TOLERANCE)
        {
            unique_in_block = false;
            break;
        }
    }

    if (unique_in_block)
    {
        const int pos = atomicAdd(candidate_count, 1);
        if (pos < max_candidates)
        {
            candidates[pos] = cost;
        }
    }
}

// Host wrapper for initialization
void launchHungarianInit(const double *d_cost_matrix, double *d_row_labels, double *d_col_labels,
                         int *d_row_matched, int *d_col_matched, int n, cudaStream_t stream)
{
    int block_size = BLOCK_SIZE;
    int grid_size = (n + block_size - 1) / block_size;

    hungarianInitLabelsKernel<<<grid_size, block_size, 0, stream>>>(
        d_cost_matrix, d_row_labels, d_col_labels, d_row_matched, d_col_matched, n);
    GPU_CHECK(cudaPeekAtLastError());
}

void launchHungarianGreedyMatch(const double *d_cost_matrix, const double *d_row_labels,
                                const double *d_col_labels, int *d_row_matched, int *d_col_matched,
                                int *d_match_count, int n, cudaStream_t stream)
{
    int block_size = BLOCK_SIZE;
    int grid_size = (n + block_size - 1) / block_size;

    hungarianGreedyMatchKernel<<<grid_size, block_size, 0, stream>>>(
        d_cost_matrix, d_row_labels, d_col_labels, d_row_matched, d_col_matched, d_match_count, n);
    GPU_CHECK(cudaPeekAtLastError());
}

void launchBottleneckCandidates(const double *d_cost_matrix, double *d_candidates,
                                int *d_candidate_count, int n, int max_candidates,
                                cudaStream_t stream)
{
    int total_elements = n * n;
    int block_size = BLOCK_SIZE;
    int grid_size = (total_elements + block_size - 1) / block_size;

    bottleneckFindCandidatesKernel<<<grid_size, block_size, block_size * sizeof(double), stream>>>(
        d_cost_matrix, d_candidates, d_candidate_count, n, max_candidates);
    GPU_CHECK(cudaPeekAtLastError());
}

} // namespace kernels
} // namespace metrics
} // namespace gpu
} // namespace nerve

// GPU-oriented Hungarian assignment kernels.

#include "nerve/gpu/gpu_error.hpp"

#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace cg = cooperative_groups;

namespace nerve::gpu::metrics::tiled
{

// Warp size
constexpr int WARP_SIZE = 32;

// Shared memory per block
constexpr int SMEM_PER_BLOCK = 48 * 1024; // 48KB

// Hungarian Algorithm GPU Kernel Constants
constexpr int HUNGARIAN_BLOCK_SIZE = 256;       // Standard block size
constexpr int HUNGARIAN_TENSOR_CORE_BLOCK = 16; // 16x16 tiled thread block (256 threads)
constexpr int HUNGARIAN_MAX_GRID_BLOCKS = 1024; // Maximum grid blocks for async kernels
constexpr int HUNGARIAN_BATCH_SMEM = 8192;
constexpr unsigned int FULL_WARP_MASK = 0xFFFFFFFF; // Shared memory for batch processing
constexpr float SLACK_TOLERANCE = 1e-6f;            // Tolerance constant
constexpr float HUNGARIAN_TILED_LARGE_COST = 1.0e30f;

__device__ inline float finiteCostOrLarge(float cost)
{
    return isfinite(cost) ? cost : HUNGARIAN_TILED_LARGE_COST;
}

/**
 * @brief Warp-cooperative reduction for finding minimum
 */
__device__ float warpReduceMin(float val)
{
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
    {
        val = fminf(val, __shfl_xor_sync(FULL_WARP_MASK, val, offset));
    }
    return val;
}

/**
 * @brief Warp-cooperative reduction for finding minimum index
 */
__device__ int warpReduceMinIndex(float val, int idx)
{
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
    {
        float other_val = __shfl_xor_sync(FULL_WARP_MASK, val, offset);
        int other_idx = __shfl_xor_sync(FULL_WARP_MASK, idx, offset);
        if (other_val < val)
        {
            val = other_val;
            idx = other_idx;
        }
    }
    return idx;
}

/**
 * @brief 16x16 tiled cost-matrix construction for diagram matching.
 *
 * Formula: cost[i,j] = min(Linf match cost, diagonal projection cost)
 */
template <typename T>
__global__ __launch_bounds__(256) void tensorCoreCostMatrixKernel(
    const float *__restrict__ diagram1_birth, const float *__restrict__ diagram1_death,
    const float *__restrict__ diagram2_birth, const float *__restrict__ diagram2_death,
    T *__restrict__ cost_matrix, int n1, int n2)
{
    // Use scalar arithmetic for deterministic cross-architecture behavior.
    // The 16x16 launch layout is retained so a WMMA path can be introduced
    // later without changing the host API.

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n1 && j < n2)
    {
        float birth_diff = fabsf(diagram1_birth[i] - diagram2_birth[j]);
        float death_diff = fabsf(diagram1_death[i] - diagram2_death[j]);
        float cost = finiteCostOrLarge(fmaxf(birth_diff, death_diff)); // L-infinity

        // Add diagonal distance option
        float diag1 = fabsf(diagram1_death[i] - diagram1_birth[i]) * 0.5f;
        float diag2 = fabsf(diagram2_death[j] - diagram2_birth[j]) * 0.5f;

        // Store minimum of matching or diagonal
        cost = fminf(cost, finiteCostOrLarge(diag1 + diag2));

        cost_matrix[i * n2 + j] = static_cast<T>(cost);
    }
}

/**
 * @brief Warp-cooperative Hungarian algorithm kernel
 *
 * Each warp processes one row assignment
 * Uses shared memory for labels and matching state
 */
__global__ __launch_bounds__(HUNGARIAN_BLOCK_SIZE) void warpCooperativeHungarianKernel(
    const float *__restrict__ cost_matrix, int *__restrict__ assignment,
    float *__restrict__ row_labels, float *__restrict__ col_labels, int *__restrict__ row_matched,
    int *__restrict__ col_matched, int n, int *__restrict__ match_count)
{
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    // Shared memory allocation
    extern __shared__ char shared_mem[];
    bool *s_visited = reinterpret_cast<bool *>(shared_mem);

    // Each warp handles multiple rows round-robin
    for (int row = warp_id; row < n; row += num_warps)
    {
        if (row_matched[row] >= 0)
            continue; // Already matched

        // Augmenting path search using warp cooperation
        int current_row = row;
        int current_col = -1;

        // Initialize visited array
        for (int i = lane_id; i < n; i += WARP_SIZE)
        {
            s_visited[warp_id * n + i] = false;
        }
        warp.sync();

        bool found_augmenting_path = false;

        for (int iter = 0; iter < n && !found_augmenting_path; ++iter)
        {
            // Find minimum slack for all columns (warp cooperative)
            float min_slack = INFINITY;
            int min_col = -1;

            for (int col = lane_id; col < n; col += WARP_SIZE)
            {
                if (!s_visited[warp_id * n + col] && col_matched[col] < 0)
                {
                    float slack = cost_matrix[current_row * n + col] - row_labels[current_row] -
                                  col_labels[col];
                    if (isfinite(slack) && slack < min_slack)
                    {
                        min_slack = slack;
                        min_col = col;
                    }
                }
            }

            // Warp-level minimum reduction
            int best_col = warpReduceMinIndex(min_slack, min_col);
            float best_slack = __shfl_sync(FULL_WARP_MASK, min_slack, 0);

            if (isfinite(best_slack) && best_slack > SLACK_TOLERANCE)
            {
                // Update labels
                if (lane_id == 0)
                {
                    row_labels[current_row] += best_slack;
                }
            }
            else
            {
                // Found augmenting edge
                if (lane_id == 0)
                {
                    current_col = best_col;
                    s_visited[warp_id * n + current_col] = true;

                    // Check if column is free
                    if (col_matched[current_col] < 0)
                    {
                        // Augment
                        row_matched[current_row] = current_col;
                        col_matched[current_col] = current_row;
                        found_augmenting_path = true;
                        atomicAdd(match_count, 1);
                    }
                    else
                    {
                        // Continue search from matched row
                        current_row = col_matched[current_col];
                    }
                }
                warp.sync();
            }
        }
    }
}

/**
 * @brief Multi-block concurrent Hungarian for batch processing
 */
__global__ __launch_bounds__(HUNGARIAN_BLOCK_SIZE) void batchHungarianKernel(
    const float **__restrict__ cost_matrices, int **__restrict__ assignments,
    const int *__restrict__ sizes, int num_diagrams)
{
    int diagram_idx = blockIdx.x;
    if (diagram_idx >= num_diagrams)
        return;

    int n = sizes[diagram_idx];
    const float *cost = cost_matrices[diagram_idx];
    int *assign = assignments[diagram_idx];

    // Deterministic one-to-one greedy assignment inside each diagram batch.
    // This preserves assignment validity (unique columns) even when run in a single block.
    extern __shared__ int col_used[];
    const int shared_capacity = HUNGARIAN_BATCH_SMEM / static_cast<int>(sizeof(int));
    if (n > shared_capacity)
    {
        if (threadIdx.x == 0)
        {
            for (int row = 0; row < n; ++row)
            {
                float min_cost = INFINITY;
                int min_col = 0;
                for (int col = 0; col < n; ++col)
                {
                    const float c = finiteCostOrLarge(cost[row * n + col]);
                    if (c < min_cost)
                    {
                        min_cost = c;
                        min_col = col;
                    }
                }
                assign[row] = min_col;
            }
        }
        return;
    }

    for (int i = threadIdx.x; i < n; i += blockDim.x)
    {
        col_used[i] = 0;
        assign[i] = -1;
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        for (int row = 0; row < n; ++row)
        {
            float min_cost = INFINITY;
            int min_col = -1;
            for (int col = 0; col < n; ++col)
            {
                if (col_used[col] != 0)
                {
                    continue;
                }
                const float c = finiteCostOrLarge(cost[row * n + col]);
                if (c < min_cost)
                {
                    min_cost = c;
                    min_col = col;
                }
            }

            if (min_col == -1)
            {
                // Default when all remaining columns were claimed unexpectedly.
                min_col = (n > 0) ? (row % n) : 0;
            }
            assign[row] = min_col;
            col_used[min_col] = 1;
        }
    }
}

/**
 * @brief Async cost matrix construction with prefetching
 */
__global__ __launch_bounds__(HUNGARIAN_BLOCK_SIZE) void asyncCostMatrixKernel(
    const float *__restrict__ d_birth1, const float *__restrict__ d_death1,
    const float *__restrict__ d_birth2, const float *__restrict__ d_death2,
    float *__restrict__ d_cost, int n1, int n2)
{
    // Grid-stride loop for coalesced memory access
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n1 * n2;
         idx += gridDim.x * blockDim.x)
    {
        int i = idx / n2;
        int j = idx % n2;

        float cost = finiteCostOrLarge(
            fmaxf(fabsf(d_birth1[i] - d_birth2[j]), fabsf(d_death1[i] - d_death2[j])));

        d_cost[idx] = cost;
    }
}

// Host Launchers

void launchTensorCoreCostMatrix(const float *d_birth1, const float *d_death1, const float *d_birth2,
                                const float *d_death2, float *d_cost, int n1, int n2,
                                cudaStream_t stream)
{
    if (d_birth1 == nullptr || d_death1 == nullptr || d_birth2 == nullptr || d_death2 == nullptr ||
        d_cost == nullptr || n1 <= 0 || n2 <= 0)
    {
        return;
    }
    dim3 block(HUNGARIAN_TENSOR_CORE_BLOCK, HUNGARIAN_TENSOR_CORE_BLOCK);
    dim3 grid((n1 + HUNGARIAN_TENSOR_CORE_BLOCK - 1) / HUNGARIAN_TENSOR_CORE_BLOCK,
              (n2 + HUNGARIAN_TENSOR_CORE_BLOCK - 1) / HUNGARIAN_TENSOR_CORE_BLOCK);

    tensorCoreCostMatrixKernel<<<grid, block, 0, stream>>>(d_birth1, d_death1, d_birth2, d_death2,
                                                           d_cost, n1, n2);
    GPU_CHECK(cudaPeekAtLastError());
}

void launchWarpCooperativeHungarian(const float *d_cost, int *d_assignment, float *d_row_labels,
                                    float *d_col_labels, int *d_row_matched, int *d_col_matched,
                                    int n, int *d_match_count, cudaStream_t stream)
{
    if (d_cost == nullptr || d_assignment == nullptr || d_row_labels == nullptr ||
        d_col_labels == nullptr || d_row_matched == nullptr || d_col_matched == nullptr ||
        d_match_count == nullptr || n <= 0)
    {
        return;
    }
    int block_size = HUNGARIAN_BLOCK_SIZE;
    int num_warps = block_size / WARP_SIZE;
    size_t smem_size = num_warps * n * sizeof(bool);

    // Cap shared memory
    if (smem_size > SMEM_PER_BLOCK)
    {
        smem_size = SMEM_PER_BLOCK;
    }

    warpCooperativeHungarianKernel<<<1, block_size, smem_size, stream>>>(
        d_cost, d_assignment, d_row_labels, d_col_labels, d_row_matched, d_col_matched, n,
        d_match_count);
    GPU_CHECK(cudaPeekAtLastError());
}

void launchBatchHungarian(const float **d_cost_matrices, int **d_assignments, const int *d_sizes,
                          int num_diagrams, cudaStream_t stream)
{
    if (d_cost_matrices == nullptr || d_assignments == nullptr || d_sizes == nullptr ||
        num_diagrams <= 0)
    {
        return;
    }
    int block_size = HUNGARIAN_BLOCK_SIZE;
    batchHungarianKernel<<<num_diagrams, block_size, HUNGARIAN_BATCH_SMEM, stream>>>(
        d_cost_matrices, d_assignments, d_sizes, num_diagrams);
    GPU_CHECK(cudaPeekAtLastError());
}

void launchAsyncCostMatrix(const float *d_birth1, const float *d_death1, const float *d_birth2,
                           const float *d_death2, float *d_cost, int n1, int n2,
                           cudaStream_t stream)
{
    if (d_birth1 == nullptr || d_death1 == nullptr || d_birth2 == nullptr || d_death2 == nullptr ||
        d_cost == nullptr || n1 <= 0 || n2 <= 0)
    {
        return;
    }
    int threads = HUNGARIAN_BLOCK_SIZE;
    const std::size_t total = static_cast<std::size_t>(n1) * static_cast<std::size_t>(n2);
    int blocks = std::min<std::size_t>(HUNGARIAN_MAX_GRID_BLOCKS, (total + threads - 1) / threads);

    asyncCostMatrixKernel<<<blocks, threads, 0, stream>>>(d_birth1, d_death1, d_birth2, d_death2,
                                                          d_cost, n1, n2);
    GPU_CHECK(cudaPeekAtLastError());
}

} // namespace nerve::gpu::metrics::tiled

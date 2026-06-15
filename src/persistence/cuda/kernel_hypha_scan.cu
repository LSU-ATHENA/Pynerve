#include "nerve/gpu/gpu_error.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>

namespace nerve
{
namespace gpu
{
namespace persistence
{
namespace detail
{

namespace
{

constexpr int kScanBlockSize = 256;

int gridFor(int items)
{
    if (items <= 0)
        return 0;
    return (items / kScanBlockSize) + ((items % kScanBlockSize) == 0 ? 0 : 1);
}

// GPU boundary scan kernel (Phase 1 of HYPHA):
// For each column, determines if it's a 0-addition column (fully reduced)
// and finds the leftmost 1 position (pivot candidate).
__global__ __launch_bounds__(256) void gpuBoundaryScanKernel(const int *__restrict__ col_data,
                                                             const int *__restrict__ col_indices,
                                                             const int *__restrict__ col_starts,
                                                             int *__restrict__ stable_columns,
                                                             int *__restrict__ pivot_candidates,
                                                             int *__restrict__ zero_addition,
                                                             int n_columns, int max_height)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n_columns)
        return;

    int start = col_starts[col];
    int end = (col + 1 < n_columns) ? col_starts[col + 1] : start + max_height;
    int len = end - start;

    int pivot = -1;
    int nz_count = 0;
    int highest_row = -1;

    for (int i = start; i < end; ++i)
    {
        if (col_data[i] != 0 && col_indices[i] >= 0)
        {
            ++nz_count;
            if (col_indices[i] > highest_row)
            {
                highest_row = col_indices[i];
            }
        }
    }

    // A 0-addition column has exactly 1 non-zero entry (its pivot only)
    // It requires no further column additions during reduction.
    zero_addition[col] = (nz_count == 1) ? 1 : 0;

    // For columns with non-zero entries, the pivot candidate is the
    // highest row index (rightmost in sorted order).
    if (nz_count > 0)
    {
        pivot_candidates[col] = highest_row;
        stable_columns[col] = 0;
    }
    else
    {
        pivot_candidates[col] = -1;
        stable_columns[col] = 1;
    }
}

// Kernel to identify pivot-claimable columns:
// For each pivot candidate, attempt to claim the row in a global mapping.
// This mirrors the first pass of standard reduction on GPU.
__global__ __launch_bounds__(256) void gpuPivotClaimKernel(const int *__restrict__ pivot_candidates,
                                                           int *__restrict__ low_to_col,
                                                           int *__restrict__ claimed_pivots,
                                                           int n_columns)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n_columns)
        return;

    int pivot = pivot_candidates[col];
    if (pivot < 0)
    {
        claimed_pivots[col] = -1;
        return;
    }

    int expected = -1;
    int actual = atomicCAS(&low_to_col[pivot], -1, col);
    claimed_pivots[col] = (actual == -1 || actual == col) ? pivot : -1;
}

} // anonymous namespace

void launchGpuBoundaryScan(const int *d_col_data, const int *d_col_indices, const int *d_col_starts,
                           int *d_stable_columns, int *d_pivot_candidates, int *d_zero_addition,
                           int n_columns, int max_height, cudaStream_t stream)
{
    if (d_col_data == nullptr || d_col_indices == nullptr || d_col_starts == nullptr ||
        d_stable_columns == nullptr || d_pivot_candidates == nullptr || d_zero_addition == nullptr)
    {
        return;
    }
    int grid = gridFor(n_columns);
    if (grid == 0)
        return;

    gpuBoundaryScanKernel<<<grid, kScanBlockSize, 0, stream>>>(
        d_col_data, d_col_indices, d_col_starts, d_stable_columns, d_pivot_candidates,
        d_zero_addition, n_columns, max_height);
    GPU_CHECK(cudaPeekAtLastError());
}

void launchGpuPivotClaim(const int *d_pivot_candidates, int *d_low_to_col, int *d_claimed_pivots,
                         int n_columns, cudaStream_t stream)
{
    if (d_pivot_candidates == nullptr || d_low_to_col == nullptr || d_claimed_pivots == nullptr)
    {
        return;
    }
    int grid = gridFor(n_columns);
    if (grid == 0)
        return;

    gpuPivotClaimKernel<<<grid, kScanBlockSize, 0, stream>>>(d_pivot_candidates, d_low_to_col,
                                                             d_claimed_pivots, n_columns);
    GPU_CHECK(cudaPeekAtLastError());
}

} // namespace detail
} // namespace persistence
} // namespace gpu
} // namespace nerve

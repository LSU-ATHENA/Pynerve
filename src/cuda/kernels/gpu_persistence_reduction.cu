#include <cuda_runtime.h>

#include <climits>
#include <cstdint>

namespace nerve
{
namespace gpu
{
namespace kernels
{

__device__ __forceinline__ int deviceMax(int a, int b)
{
    return (a > b) ? a : b;
}
__device__ __forceinline__ int deviceMin(int a, int b)
{
    return (a < b) ? a : b;
}

__device__ int warpReduceMax(int val)
{
#pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        const int other = __shfl_down_sync(0xffffffff, val, offset);
        val = deviceMax(val, other);
    }
    return val;
}

__device__ int warpReduceMin(int val)
{
#pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        const int other = __shfl_down_sync(0xffffffff, val, offset);
        val = deviceMin(val, other);
    }
    return val;
}

__device__ int blockReduceMax(int val, int *shmem)
{
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    int num_warps = (blockDim.x + 31) >> 5;
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        int other = __shfl_down_sync(0xFFFFFFFFu, val, offset);
        val = (val > other) ? val : other;
    }
    if (lane == 0)
        shmem[wid] = val;
    __syncthreads();
    if (wid == 0)
    {
        val = (lane < num_warps) ? shmem[lane] : -1;
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            int other = __shfl_down_sync(0xFFFFFFFFu, val, offset);
            val = (val > other) ? val : other;
        }
    }
    __syncthreads();
    return val;
}

__device__ int blockReduceMin(int val, int *shmem)
{
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    int num_warps = (blockDim.x + 31) >> 5;
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        int other = __shfl_down_sync(0xFFFFFFFFu, val, offset);
        val = (val < other) ? val : other;
    }
    if (lane == 0)
        shmem[wid] = val;
    __syncthreads();
    if (wid == 0)
    {
        val = (lane < num_warps) ? shmem[lane] : INT_MAX;
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            int other = __shfl_down_sync(0xFFFFFFFFu, val, offset);
            val = (val < other) ? val : other;
        }
    }
    __syncthreads();
    return val;
}

__global__ __launch_bounds__(256)
    void gpuComputePivotsKernel(const int *__restrict__ col_data, const int *__restrict__ col_indices,
                           const int *__restrict__ col_starts, int *__restrict__ pivots,
                           int n_columns, int max_height)
{
    const int col = blockIdx.x;
    if (col >= n_columns)
    {
        return;
    }

    const int start = col_starts[col];
    const int end = (col + 1 < n_columns) ? col_starts[col + 1] : start + max_height;

    __shared__ int reduce_shmem[64];

    int local_max = -1;
    for (int i = start + threadIdx.x; i < end; i += blockDim.x)
    {
        if (col_data[i] != 0)
        {
            local_max = deviceMax(local_max, col_indices[i]);
        }
    }

    const int pivot = blockReduceMax(local_max, reduce_shmem);

    if (threadIdx.x == 0)
    {
        pivots[col] = pivot;
    }
}

__global__ __launch_bounds__(256)
    void gpuComputePivotsCohomologyKernel(const int *__restrict__ col_data,
                                          const int *__restrict__ col_indices,
                                          const int *__restrict__ col_starts, int *__restrict__ pivots,
                                     int n_columns, int max_height)
{
    const int col = blockIdx.x;
    if (col >= n_columns)
    {
        return;
    }

    const int start = col_starts[col];
    const int end = (col + 1 < n_columns) ? col_starts[col + 1] : start + max_height;

    __shared__ int reduce_shmem[64];

    int local_min = INT_MAX;
    for (int i = start + threadIdx.x; i < end; i += blockDim.x)
    {
        if (col_data[i] != 0)
        {
            local_min = deviceMin(local_min, col_indices[i]);
        }
    }

    const int pivot = blockReduceMin(local_min, reduce_shmem);

    if (threadIdx.x == 0)
    {
        pivots[col] = (pivot == INT_MAX) ? -1 : pivot;
    }
}

__global__ __launch_bounds__(256)
    void gpuAssignPivotOwnersKernel(const int *__restrict__ pivots, int *__restrict__ low_to_col,
                               int *__restrict__ conflict_src, char *__restrict__ has_conflict,
                               int n_columns)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n_columns)
    {
        return;
    }

    const int pivot = pivots[col];
    conflict_src[col] = -1;
    has_conflict[col] = 0;

    if (pivot < 0)
    {
        return;
    }

    const int claimed = atomicCAS(&low_to_col[pivot], -1, col);

    if (claimed == -1 || claimed == col)
    {
        return;
    }

    conflict_src[col] = claimed;
    has_conflict[col] = 1;
}

__global__ __launch_bounds__(256)
    void gpuAddColumnsKernel(int *__restrict__ col_data, int *__restrict__ col_indices,
                        const int *__restrict__ col_starts, const int *__restrict__ conflict_src,
                        const char *__restrict__ has_conflict, int n_columns, int max_height)
{
    const int dst_col = blockIdx.x;
    if (dst_col >= n_columns)
    {
        return;
    }
    if (!has_conflict[dst_col])
    {
        return;
    }

    const int src_col = conflict_src[dst_col];
    if (src_col < 0 || src_col >= n_columns || src_col == dst_col)
    {
        return;
    }

    const int dst_start = col_starts[dst_col];
    const int src_start = col_starts[src_col];
    const int src_end =
        (src_col + 1 < n_columns) ? col_starts[src_col + 1] : src_start + max_height;
    const int dst_end =
        (dst_col + 1 < n_columns) ? col_starts[dst_col + 1] : dst_start + max_height;
    const int dst_len = dst_end - dst_start;

    if (dst_len <= 0)
    {
        return;
    }

    int *const dst_data = col_data + dst_start;
    int *const dst_idx = col_indices + dst_start;

    for (int i = threadIdx.x; i < (src_end - src_start); i += blockDim.x)
    {
        if (col_data[src_start + i] == 0)
        {
            continue;
        }
        const int s_row = col_indices[src_start + i];
        if (s_row < 0)
        {
            continue;
        }

        int lo = 0;
        int hi = dst_len - 1;
        while (lo <= hi)
        {
            const int mid = lo + ((hi - lo) >> 1);
            if (dst_idx[mid] == s_row)
            {
                atomicXor(dst_data + mid, 1);
                break;
            }
            else if (dst_idx[mid] < s_row)
            {
                lo = mid + 1;
            }
            else
            {
                hi = mid - 1;
            }
        }
    }
}

__global__ __launch_bounds__(256)
    void gpuResolveConflictsKernel(int *__restrict__ col_data, int *__restrict__ col_indices,
                              const int *__restrict__ col_starts, int *__restrict__ pivots,
                              int *__restrict__ low_to_col, int *__restrict__ changed,
                              int n_columns, int max_height, int max_passes)
{
    const int col = blockIdx.x;
    if (col >= n_columns)
    {
        return;
    }

    const int start = col_starts[col];
    const int col_end = (col + 1 < n_columns) ? col_starts[col + 1] : start + max_height;
    const int col_len = col_end - start;

    if (col_len <= 0)
    {
        if (threadIdx.x == 0)
        {
            pivots[col] = -1;
        }
        return;
    }

    int *dst_data = col_data + start;
    int *dst_idx = col_indices + start;

    __shared__ int reduce_shmem[64];
    __shared__ int block_pivot;
    __shared__ int block_src;

    for (int pass = 0; pass < max_passes; ++pass)
    {
        int local_max = -1;
        for (int i = threadIdx.x; i < col_len; i += blockDim.x)
        {
            if (dst_data[i] != 0)
            {
                local_max = deviceMax(local_max, dst_idx[i]);
            }
        }

        const int pivot_candidate = blockReduceMax(local_max, reduce_shmem);

        if (threadIdx.x == 0)
        {
            block_pivot = pivot_candidate;
            block_src = (pivot_candidate >= 0) ? low_to_col[pivot_candidate] : -1;
        }
        __syncthreads();

        if (block_pivot < 0)
        {
            break;
        }

        if (threadIdx.x == 0)
        {
            if (block_src < 0)
            {
                const int claimed = atomicCAS(&low_to_col[block_pivot], -1, col);
                if (claimed == -1)
                {
                    pivots[col] = block_pivot;
                    return;
                }
                block_src = claimed;
            }
        }
        __syncthreads();

        const int src_col = block_src;
        if (src_col < 0 || src_col == col)
        {
            break;
        }

        const int src_start = col_starts[src_col];
        const int src_end =
            (src_col + 1 < n_columns) ? col_starts[src_col + 1] : src_start + max_height;

        for (int i = threadIdx.x; i < (src_end - src_start); i += blockDim.x)
        {
            if (col_data[src_start + i] == 0)
            {
                continue;
            }
            const int s_row = col_indices[src_start + i];
            if (s_row < 0)
            {
                continue;
            }

            int lo = 0;
            int hi = col_len - 1;
            while (lo <= hi)
            {
                const int mid = lo + ((hi - lo) >> 1);
                if (dst_idx[mid] == s_row)
                {
                    atomicXor(dst_data + mid, 1);
                    break;
                }
                else if (dst_idx[mid] < s_row)
                {
                    lo = mid + 1;
                }
                else
                {
                    hi = mid - 1;
                }
            }
        }
        __syncthreads();

        if (threadIdx.x == 0)
        {
            atomicExch(changed, 1);
        }
    }

    if (threadIdx.x == 0)
    {
        pivots[col] = -1;
    }
}

__global__ __launch_bounds__(256)
    void gpuCohomologyReductionKernel(int *__restrict__ cob_data, int *__restrict__ cob_indices,
                                 const int *__restrict__ cob_starts, int *__restrict__ pivots,
                                 int *__restrict__ low_to_col, int *__restrict__ changed,
                                 int n_cochains, int max_height, int max_passes)
{
    const int cochain_idx = blockIdx.x;
    if (cochain_idx >= n_cochains)
    {
        return;
    }

    const int cochain = n_cochains - 1 - cochain_idx;
    if (cochain < 0 || cochain >= n_cochains)
    {
        return;
    }

    const int start = cob_starts[cochain];
    const int col_end = (cochain + 1 < n_cochains) ? cob_starts[cochain + 1] : start + max_height;
    const int col_len = col_end - start;

    if (col_len <= 0)
    {
        if (threadIdx.x == 0)
        {
            pivots[cochain] = -1;
        }
        return;
    }

    int *dst_data = cob_data + start;
    int *dst_idx = cob_indices + start;

    __shared__ int reduce_shmem[64];
    __shared__ int block_pivot;
    __shared__ int block_src;

    for (int pass = 0; pass < max_passes; ++pass)
    {
        int local_min = INT_MAX;
        for (int i = threadIdx.x; i < col_len; i += blockDim.x)
        {
            if (dst_data[i] != 0)
            {
                local_min = deviceMin(local_min, dst_idx[i]);
            }
        }

        const int pivot_candidate = blockReduceMin(local_min, reduce_shmem);

        if (threadIdx.x == 0)
        {
            block_pivot = (pivot_candidate == INT_MAX) ? -1 : pivot_candidate;
            block_src = (block_pivot >= 0) ? low_to_col[block_pivot] : -1;
        }
        __syncthreads();

        if (block_pivot < 0)
        {
            break;
        }

        if (threadIdx.x == 0)
        {
            if (block_src < 0)
            {
                const int claimed = atomicCAS(&low_to_col[block_pivot], -1, cochain);
                if (claimed == -1)
                {
                    pivots[cochain] = block_pivot;
                    return;
                }
                block_src = claimed;
            }
        }
        __syncthreads();

        const int src_col = block_src;
        if (src_col < 0 || src_col == cochain)
        {
            break;
        }

        const int src_start = cob_starts[src_col];
        const int src_end =
            (src_col + 1 < n_cochains) ? cob_starts[src_col + 1] : src_start + max_height;

        for (int i = threadIdx.x; i < (src_end - src_start); i += blockDim.x)
        {
            if (cob_data[src_start + i] == 0)
            {
                continue;
            }
            const int s_row = cob_indices[src_start + i];
            if (s_row < 0)
            {
                continue;
            }

            int lo = 0;
            int hi = col_len - 1;
            while (lo <= hi)
            {
                const int mid = lo + ((hi - lo) >> 1);
                if (dst_idx[mid] == s_row)
                {
                    atomicXor(dst_data + mid, 1);
                    break;
                }
                else if (dst_idx[mid] < s_row)
                {
                    lo = mid + 1;
                }
                else
                {
                    hi = mid - 1;
                }
            }
        }
        __syncthreads();

        if (threadIdx.x == 0)
        {
            atomicExch(changed, 1);
        }
    }

    if (threadIdx.x == 0)
    {
        pivots[cochain] = -1;
    }
}

__global__ __launch_bounds__(256)
    void gpuClearingKernel(const int *__restrict__ pivots, const int *__restrict__ low_to_col,
                      int *__restrict__ col_data, int *__restrict__ col_indices,
                      const int *__restrict__ col_starts, int n_columns, int max_height)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n_columns)
    {
        return;
    }

    const int pivot = pivots[col];
    if (pivot < 0 || pivot >= n_columns)
    {
        return;
    }

    const int owner = low_to_col[pivot];
    if (owner < 0 || owner == col)
    {
        return;
    }

    const int start = col_starts[col];
    const int end = (col + 1 < n_columns) ? col_starts[col + 1] : start + max_height;

    for (int i = start + threadIdx.x; i < end; i += blockDim.x)
    {
        col_data[i] = 0;
        col_indices[i] = -1;
    }
}

__global__ __launch_bounds__(256)
    void gpuExtractPairsKernel(const int *__restrict__ pivots, const int *__restrict__ low_to_col,
                          int2 *__restrict__ pairs, int *__restrict__ pair_count, int n_columns)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n_columns)
    {
        return;
    }

    const int pivot = pivots[col];
    if (pivot < 0 || pivot >= n_columns)
    {
        return;
    }

    const int owner = low_to_col[pivot];
    if (owner != col)
    {
        return;
    }

    const int idx = atomicAdd(pair_count, 1);
    pairs[idx] = make_int2(pivot, col);
}

__global__ __launch_bounds__(256)
    void gpuCheckConvergenceKernel(int *__restrict__ changed, int *__restrict__ iteration_flag)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
    {
        return;
    }
    *iteration_flag = *changed;
    *changed = 0;
}

} // namespace kernels
} // namespace gpu
} // namespace nerve

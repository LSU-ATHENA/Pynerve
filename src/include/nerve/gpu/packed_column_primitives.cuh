#pragma once

#include "nerve/gpu/gpu_ptx_ops.cuh"

#include <cstdint>

namespace nerve::gpu::packed
{

static constexpr int kWarpSize = 32;
static constexpr unsigned int kFullWarpMask = 0xFFFFFFFFu;

enum class XorStrategy
{
    Auto,
    GlobalAtomics,
    DirectXor,
    SharedMemXor
};

__device__ __forceinline__ XorStrategy selectXorStrategy(int num_words, int num_columns)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (num_words <= 8 && num_columns >= 512)
    {
        return XorStrategy::DirectXor;
    }
    if (num_words <= 8)
    {
        return XorStrategy::SharedMemXor;
    }
    return XorStrategy::GlobalAtomics;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
    if (num_words <= 8)
    {
        return XorStrategy::DirectXor;
    }
    return XorStrategy::GlobalAtomics;
#else
    return XorStrategy::DirectXor;
#endif
}

__device__ __forceinline__ void packed_column_xor(uint64_t *dest, const uint64_t *src,
                                                           int num_words, int lane_id,
                                                           XorStrategy strategy = XorStrategy::Auto)
{
    if (strategy == XorStrategy::Auto)
    {
        strategy = XorStrategy::DirectXor;
    }

    switch (strategy)
    {
    case XorStrategy::GlobalAtomics:
        for (int w = lane_id; w < num_words; w += kWarpSize)
        {
            ptx::atom_xor_global_u64(&dest[w], src[w]);
        }
        break;
    case XorStrategy::DirectXor:
    default:
        for (int w = lane_id; w < num_words; w += kWarpSize)
        {
            dest[w] ^= src[w];
        }
        break;
    case XorStrategy::SharedMemXor:
        for (int w = lane_id; w < num_words; w += kWarpSize)
        {
            dest[w] ^= src[w];
        }
        break;
    }
}

__device__ __forceinline__ int packed_column_find_msb_warp(const uint64_t *col_words, int num_words,
                                                            int lane_id)
{
    int lane_pivot = -1;
    for (int w = num_words - 1 - lane_id; w >= 0; w -= kWarpSize)
    {
        uint64_t word = col_words[w];
        if (word != 0)
        {
            lane_pivot = w * 64 + ptx::find_msb_u64(word);
            break;
        }
    }
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1)
    {
        int other = __shfl_down_sync(kFullWarpMask, lane_pivot, offset);
        if (other > lane_pivot)
        {
            lane_pivot = other;
        }
    }
    int result = __shfl_sync(kFullWarpMask, lane_pivot, 0);
    return result;
}

__device__ __forceinline__ int packed_column_find_lsb_warp(const uint64_t *col_words, int num_words,
                                                            int lane_id)
{
    int lane_pivot = -1;
    for (int w = lane_id; w < num_words; w += kWarpSize)
    {
        uint64_t word = col_words[w];
        if (word != 0)
        {
            lane_pivot = w * 64 + (__ffsll(static_cast<long long>(word)) - 1);
            break;
        }
    }
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1)
    {
        int other = __shfl_down_sync(kFullWarpMask, lane_pivot, offset);
        if (lane_pivot < 0 || (other >= 0 && other < lane_pivot))
        {
            lane_pivot = other;
        }
    }
    int result = __shfl_sync(kFullWarpMask, lane_pivot, 0);
    return result;
}

__device__ __forceinline__ int packed_column_pivot_find(const uint64_t *col_words, int num_words,
                                                         int lane_id, bool use_lowest = false)
{
    if (use_lowest)
    {
        return packed_column_find_lsb_warp(col_words, num_words, lane_id);
    }
    return packed_column_find_msb_warp(col_words, num_words, lane_id);
}

__device__ __forceinline__ bool packed_column_try_claim_pivot(int *pivot_to_col, int pivot_row,
                                                               int col_idx, int lane_id)
{
    unsigned int warp_mask = kFullWarpMask;
    int selected = -1;

    if (lane_id == 0)
    {
        int expected = -1;
        int actual = atomicCAS(&pivot_to_col[pivot_row], expected, col_idx);
        if (actual == expected || actual == col_idx)
        {
            selected = col_idx;
        }
        else
        {
            selected = actual;
        }
    }
    selected = __shfl_sync(kFullWarpMask, selected, 0);

    if (selected < 0 || selected == col_idx)
    {
        if (lane_id == 0)
        {
            pivot_to_col[pivot_row] = col_idx;
        }
        return true;
    }
    return false;
}

__device__ __forceinline__ bool packed_column_try_claim_pivot_64(uint64_t *pivot_storage,
                                                                  int pivot_word_idx,
                                                                  int pivot_bit_idx, int col_idx,
                                                                  int lane_id)
{
    uint64_t bit_mask = 1ULL << pivot_bit_idx;
    bool claimed = false;

    if (lane_id == 0)
    {
        uint64_t old = pivot_storage[pivot_word_idx];
        uint64_t assumed;
        do
        {
            if (old & bit_mask)
            {
                break;
            }
            assumed = old;
            old = atomicCAS(&pivot_storage[pivot_word_idx], assumed, assumed | bit_mask);
        } while (assumed != old);
        if (!(old & bit_mask))
        {
            claimed = true;
        }
    }
    return __shfl_sync(kFullWarpMask, claimed, 0);
}

__device__ __forceinline__ void packed_column_xor_masked(uint64_t *dest, const uint64_t *src,
                                                           const uint64_t *mask, int num_words,
                                                           int lane_id)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 500
    if (num_words <= 8)
    {
        for (int w = lane_id; w < num_words; w += kWarpSize)
        {
            uint64_t d = dest[w];
            uint64_t s = src[w];
            uint64_t m = mask[w];
            uint32_t lo_dest = static_cast<uint32_t>(d & 0xFFFFFFFFu);
            uint32_t hi_dest = static_cast<uint32_t>(d >> 32);
            uint32_t lo_src = static_cast<uint32_t>(s & 0xFFFFFFFFu);
            uint32_t hi_src = static_cast<uint32_t>(s >> 32);
            uint32_t lo_mask = static_cast<uint32_t>(m & 0xFFFFFFFFu);
            uint32_t hi_mask = static_cast<uint32_t>(m >> 32);
            uint32_t lo_result = ptx::xor_and_notc(lo_dest, lo_src, lo_mask);
            uint32_t hi_result = ptx::xor_and_notc(hi_dest, hi_src, hi_mask);
            uint64_t result = (static_cast<uint64_t>(hi_result) << 32) | lo_result;
            dest[w] = result;
        }
        return;
    }
#endif
    for (int w = lane_id; w < num_words; w += kWarpSize)
    {
        dest[w] = (dest[w] ^ src[w]) & ~mask[w];
    }
}

__device__ __forceinline__ void packed_column_clear(uint64_t *col, int num_words, int lane_id)
{
    for (int w = lane_id; w < num_words; w += kWarpSize)
    {
        col[w] = 0;
    }
}

__device__ __forceinline__ int packed_column_popcount_warp(const uint64_t *col_words, int num_words,
                                                            int lane_id)
{
    int lane_sum = 0;
    for (int w = lane_id; w < num_words; w += kWarpSize)
    {
        lane_sum += static_cast<int>(ptx::popc_u64(col_words[w]));
    }
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1)
    {
        lane_sum += __shfl_down_sync(kFullWarpMask, lane_sum, offset);
    }
    return __shfl_sync(kFullWarpMask, lane_sum, 0);
}

__device__ __forceinline__ void packed_column_copy(uint64_t *dest, const uint64_t *src,
                                                    int num_words, int lane_id)
{
    for (int w = lane_id; w < num_words; w += kWarpSize)
    {
        dest[w] = src[w];
    }
}

__device__ __forceinline__ bool packed_column_is_empty_warp(const uint64_t *col_words, int num_words,
                                                             int lane_id)
{
    int lane_empty = 1;
    for (int w = lane_id; w < num_words; w += kWarpSize)
    {
        if (col_words[w] != 0)
        {
            lane_empty = 0;
            break;
        }
    }
    unsigned int ballot = __ballot_sync(kFullWarpMask, lane_empty);
    return ballot == kFullWarpMask;
}

template <typename T, int kBlockSize>
__device__ __forceinline__ T warp_block_reduce_max(T val, T *shared_mem)
{
    int lane_id = threadIdx.x & (kWarpSize - 1);
    int warp_id = threadIdx.x / kWarpSize;
    int num_warps = kBlockSize / kWarpSize;
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1)
    {
        T other = __shfl_down_sync(kFullWarpMask, val, offset);
        val = (val > other) ? val : other;
    }
    if (lane_id == 0)
    {
        shared_mem[warp_id] = val;
    }
    __syncthreads();
    if (warp_id == 0)
    {
        val = (lane_id < num_warps) ? shared_mem[lane_id] : val;
        for (int offset = kWarpSize / 2; offset > 0; offset >>= 1)
        {
            T other = __shfl_down_sync(kFullWarpMask, val, offset);
            val = (val > other) ? val : other;
        }
    }
    __syncthreads();
    return val;
}

template <typename T, int kBlockSize>
__device__ __forceinline__ T warp_block_reduce_min(T val, T *shared_mem)
{
    int lane_id = threadIdx.x & (kWarpSize - 1);
    int warp_id = threadIdx.x / kWarpSize;
    int num_warps = kBlockSize / kWarpSize;
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1)
    {
        T other = __shfl_down_sync(kFullWarpMask, val, offset);
        val = (val < other) ? val : other;
    }
    if (lane_id == 0)
    {
        shared_mem[warp_id] = val;
    }
    __syncthreads();
    if (warp_id == 0)
    {
        val = (lane_id < num_warps) ? shared_mem[lane_id] : val;
        for (int offset = kWarpSize / 2; offset > 0; offset >>= 1)
        {
            T other = __shfl_down_sync(kFullWarpMask, val, offset);
            val = (val < other) ? val : other;
        }
    }
    __syncthreads();
    return val;
}

template <typename T, int kBlockSize>
__device__ __forceinline__ T warp_block_reduce_sum(T val, T *shared_mem)
{
    int lane_id = threadIdx.x & (kWarpSize - 1);
    int warp_id = threadIdx.x / kWarpSize;
    int num_warps = kBlockSize / kWarpSize;
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1)
    {
        val += __shfl_down_sync(kFullWarpMask, val, offset);
    }
    if (lane_id == 0)
    {
        shared_mem[warp_id] = val;
    }
    __syncthreads();
    if (warp_id == 0)
    {
        val = (lane_id < num_warps) ? shared_mem[lane_id] : static_cast<T>(0);
        for (int offset = kWarpSize / 2; offset > 0; offset >>= 1)
        {
            val += __shfl_down_sync(kFullWarpMask, val, offset);
        }
    }
    __syncthreads();
    return val;
}

template <int kBlockSize>
__device__ __forceinline__ unsigned int warp_block_reduce_or(unsigned int val,
                                                              unsigned int *shared_mem)
{
    int lane_id = threadIdx.x & (kWarpSize - 1);
    int warp_id = threadIdx.x / kWarpSize;
    int num_warps = kBlockSize / kWarpSize;
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1)
    {
        val |= __shfl_down_sync(kFullWarpMask, val, offset);
    }
    if (lane_id == 0)
    {
        shared_mem[warp_id] = val;
    }
    __syncthreads();
    if (warp_id == 0)
    {
        val = (lane_id < num_warps) ? shared_mem[lane_id] : 0u;
        for (int offset = kWarpSize / 2; offset > 0; offset >>= 1)
        {
            val |= __shfl_down_sync(kFullWarpMask, val, offset);
        }
    }
    __syncthreads();
    return val;
}

__device__ __forceinline__ void async_pipeline_stage_column(const uint64_t *__restrict__ src_global,
                                                             uint64_t *__restrict__ dst_shared,
                                                             int num_words, int lane_id)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    for (int w = lane_id; w < num_words; w += kWarpSize)
    {
        ptx::cp_async_shared_global(&dst_shared[w], &src_global[w], sizeof(uint64_t), false);
    }
    ptx::cp_async_commit_group();
    ptx::cp_async_wait_group(0);
#else
    for (int w = lane_id; w < num_words; w += kWarpSize)
    {
        dst_shared[w] = src_global[w];
    }
#endif
}

__device__ __forceinline__ void async_pipeline_stage_and_xor(uint64_t *__restrict__ dest_global,
                                                              const uint64_t *__restrict__ src_global,
                                                              uint64_t *__restrict__ scratch_shared,
                                                              int num_words, int lane_id)
{
    async_pipeline_stage_column(src_global, scratch_shared, num_words, lane_id);
    __syncwarp();
    packed_column_xor(dest_global, scratch_shared, num_words, lane_id,
                               XorStrategy::DirectXor);
}

__device__ __forceinline__ int packed_column_reduce_iterative(uint64_t *col_words,
                                                               const uint64_t *__restrict__ all_columns,
                                                               const int *__restrict__ pivot_to_col,
                                                               int num_words, int num_columns,
                                                               int col_idx, int lane_id,
                                                               int iteration_limit)
{
    int pivot = packed_column_pivot_find(col_words, num_words, lane_id, false);
    const int pivot_limit = num_words * 64;

    for (int iter = 0; pivot >= 0 && iter < iteration_limit; ++iter)
    {
        if (pivot >= pivot_limit)
        {
            break;
        }
        int src_col = pivot_to_col[pivot];
        if (src_col < 0 || src_col >= num_columns)
        {
            break;
        }
        if (src_col == col_idx)
        {
            packed_column_clear(col_words, num_words, lane_id);
            pivot = -1;
            break;
        }
        const uint64_t *src_base = all_columns + static_cast<size_t>(src_col) * num_words;
        packed_column_xor(col_words, src_base, num_words, lane_id,
                                   XorStrategy::DirectXor);
        __syncwarp();
        pivot = packed_column_pivot_find(col_words, num_words, lane_id, false);
    }
    return pivot;
}

__device__ __forceinline__ void packed_column_prefetch_next(const void *next_col, int num_words)
{
    (void)num_words;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    ptx::prefetch_l2(next_col);
#endif
}

__device__ __forceinline__ bool __attribute__((always_inline))
packed_column_is_empty(const uint64_t *col_words, int num_words)
{
    for (int w = 0; w < num_words; ++w)
    {
        if (col_words[w] != 0)
        {
            return false;
        }
    }
    return true;
}

__device__ __forceinline__ int packed_column_find_pivot(const uint64_t *col_words, int num_words)
{
    for (int w = num_words - 1; w >= 0; --w)
    {
        if (col_words[w] != 0)
        {
            return w * 64 + ptx::find_msb_u64(col_words[w]);
        }
    }
    return -1;
}

__device__ __forceinline__ void packed_column_xor_hw(uint64_t *dest, const uint64_t *src,
                                                      int num_words, int lane_id)
{
    packed_column_xor(dest, src, num_words, lane_id, XorStrategy::DirectXor);
    __syncwarp();
}

__device__ __forceinline__ void packed_column_xor_atomic_hw(uint64_t *dest, const uint64_t *src,
                                                             int num_words, int lane_id)
{
    packed_column_xor(dest, src, num_words, lane_id, XorStrategy::GlobalAtomics);
    __syncwarp();
}

__device__ __forceinline__ unsigned int warp_match_any_col(uint64_t col_hash, int lane_id,
                                                            unsigned int warp_mask)
{
    (void)lane_id;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    return ptx::match_any_sync_u64(warp_mask, static_cast<unsigned long long>(col_hash));
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
    return __match_any_sync(warp_mask, static_cast<unsigned long long>(col_hash));
#else
    (void)warp_mask;
    (void)col_hash;
    return 0;
#endif
}

} // namespace nerve::gpu::packed

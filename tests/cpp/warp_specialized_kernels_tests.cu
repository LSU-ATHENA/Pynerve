// Warp-specialized kernels unit tests.
// Tests the three main kernel launchers (column add, pivot find, pipelined reduction)
// plus the benchmark, hardware-query, and config-optimization helpers from
// kernel_warp_specialized_cuda.cu.
//
// Label: persistence;reduction;gpu;cuda;warp;regression

#include "nerve/persistence/cuda/cuda_warp_specialized_kernels.hpp"

#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

namespace
{

bool check_cuda(cudaError_t code, const char *expression)
{
    if (code == cudaSuccess)
        return true;
    std::cerr << expression << " failed: " << cudaGetErrorString(code) << '\n';
    return false;
}

bool has_gpu()
{
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return err == cudaSuccess && device_count > 0;
}

// Single-warp reference reduction kernel: processes all columns SEQUENTIALLY
// using one warp, eliminating inter-warp read-write races on cross-column reads.
// Semantically equivalent to packed_column_reduce_iterative.
__global__ void singleWarpReductionRef(uint64_t *columns, const int *pivot_to_col, int num_words,
                                       int num_columns, int *new_pivots)
{
    int lane_id = threadIdx.x;
    if (lane_id >= 32)
        return;

    for (int col = 0; col < num_columns; ++col)
    {
        uint64_t *col_words = columns + (size_t)col * (size_t)num_words;

        // Find pivot (MSB) via warp reduction
        int pivot = -1;
        for (int w = num_words - 1 - lane_id; w >= 0; w -= 32)
        {
            if (col_words[w] != 0)
            {
                int msb = 63 - __clzll(col_words[w]);
                pivot = w * 64 + msb;
                break;
            }
        }
        for (int off = 16; off > 0; off >>= 1)
        {
            int other = __shfl_down_sync(0xFFFFFFFFu, pivot, off);
            if (other > pivot)
                pivot = other;
        }
        pivot = __shfl_sync(0xFFFFFFFFu, pivot, 0);

        // Iterative reduction (same logic as packed_column_reduce_iterative)
        const int pivot_limit = num_words * 64;
        const int iter_limit = max(1, num_words * 4);
        for (int it = 0; pivot >= 0 && it < iter_limit; ++it)
        {
            if (pivot >= pivot_limit)
                break;
            int src_col = pivot_to_col[pivot];
            if (src_col < 0 || src_col >= num_columns)
                break;

            if (src_col == col)
            {
                // Self-clear: zero out column
                for (int w = lane_id; w < num_words; w += 32)
                    col_words[w] = 0;
                pivot = -1;
                break;
            }

            // XOR with source column
            const uint64_t *src_base = columns + (size_t)src_col * (size_t)num_words;
            for (int w = lane_id; w < num_words; w += 32)
                col_words[w] ^= src_base[w];
            __syncwarp();

            // Find new pivot
            pivot = -1;
            for (int w = num_words - 1 - lane_id; w >= 0; w -= 32)
            {
                if (col_words[w] != 0)
                {
                    int msb = 63 - __clzll(col_words[w]);
                    pivot = w * 64 + msb;
                    break;
                }
            }
            for (int off = 16; off > 0; off >>= 1)
            {
                int other = __shfl_down_sync(0xFFFFFFFFu, pivot, off);
                if (other > pivot)
                    pivot = other;
            }
            pivot = __shfl_sync(0xFFFFFFFFu, pivot, 0);
        }

        if (lane_id == 0)
            new_pivots[col] = pivot;
    }
}

// Block-synchronized reduction reference kernel: processes all columns
// concurrently (one warp per column) with __syncthreads() barriers between
// each iteration. This eliminates inter-warp races by ensuring all warps
// complete their reads/writes before any warp proceeds to the next step.
//
// Launched as a single block with (num_columns * 32) threads; all columns
// share __syncthreads() barriers, providing a fully synchronized reference.
//
// Early-exit warps (pivot becomes unclaimed or self-cleared) use an `active`
// flag but continue looping to participate in __syncthreads() barriers.
__global__ void syncthreadsReductionRef(uint64_t *columns, const int * /*col_pivots*/,
                                        const int *pivot_to_col, int num_words, int num_columns,
                                        int *new_pivots)
{
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int col_idx = blockIdx.x * (blockDim.x / 32) + warp_id;
    if (col_idx >= num_columns)
        return;

    uint64_t *col_words = columns + (size_t)col_idx * (size_t)num_words;
    const int pivot_limit = num_words * 64;
    const int iter_limit = max(1, num_words * 4);

    // Compute initial pivot from column data (like the regular kernel)
    int pivot = -1;
    for (int w = num_words - 1 - lane_id; w >= 0; w -= 32)
    {
        if (col_words[w] != 0)
        {
            pivot = w * 64 + (63 - __clzll(col_words[w]));
            break;
        }
    }
    for (int off = 16; off > 0; off >>= 1)
    {
        int other = __shfl_down_sync(0xFFFFFFFFu, pivot, off);
        if (other > pivot)
            pivot = other;
    }
    pivot = __shfl_sync(0xFFFFFFFFu, pivot, 0);

    bool active = true;
    __syncthreads(); // Initial column data visible to all

    for (int it = 0; it < iter_limit; ++it)
    {
        // Check pivot state, handle self-clears
        if (active)
        {
            if (pivot < 0)
            {
                active = false;
            }
            else if (pivot >= pivot_limit)
            {
                active = false;
            }
            else
            {
                int src_col = pivot_to_col[pivot];
                if (src_col < 0 || src_col >= num_columns)
                {
                    active = false;
                }
                else if (src_col == col_idx)
                {
                    for (int w = lane_id; w < num_words; w += 32)
                        col_words[w] = 0;
                    pivot = -1;
                    active = false;
                }
            }
        }

        __syncthreads(); // Self-clears visible before XOR reads

        // XOR with source column, find new pivot
        if (active)
        {
            int src_col = pivot_to_col[pivot];
            const uint64_t *src_base = columns + (size_t)src_col * (size_t)num_words;

            for (int w = lane_id; w < num_words; w += 32)
                col_words[w] ^= src_base[w];
            __syncwarp();

            // Find new pivot via warp reduction
            pivot = -1;
            for (int w = num_words - 1 - lane_id; w >= 0; w -= 32)
            {
                if (col_words[w] != 0)
                {
                    pivot = w * 64 + (63 - __clzll(col_words[w]));
                    break;
                }
            }
            for (int off = 16; off > 0; off >>= 1)
            {
                int other = __shfl_down_sync(0xFFFFFFFFu, pivot, off);
                if (other > pivot)
                    pivot = other;
            }
            pivot = __shfl_sync(0xFFFFFFFFu, pivot, 0);
        }

        __syncthreads(); // XOR results visible before next iteration
    }

    if (lane_id == 0)
        new_pivots[col_idx] = pivot;
}

} // namespace

int main()
{
    if (!has_gpu())
    {
        std::cerr << "No CUDA device available -- skipping warp-specialized kernel tests\n";
        return 0;
    }

    // Column Add Basic (1 word)
    {
        constexpr int num_words = 1;
        constexpr int num_cols = 2;

        // dest (in-place): col 0 = 0b0011 (bits 0,1), col 1 = 0b1100 (bits 2,3)
        // src:              col 0 = 0b0101 (bits 0,2), col 1 = 0b1010 (bits 1,3)
        // dest ^= src:      col 0 = 0b0110,           col 1 = 0b0110
        const uint64_t h_dest[num_cols * num_words] = {0b0011ULL, 0b1100ULL};
        const uint64_t h_src[num_cols * num_words] = {0b0101ULL, 0b1010ULL};
        const int h_sizes[num_cols] = {1, 1};

        uint64_t *d_dest = nullptr;
        uint64_t *d_src = nullptr;
        int *d_sizes = nullptr;

        bool ok = check_cuda(cudaMalloc(&d_dest, sizeof(h_dest)), "malloc d_dest") &&
                  check_cuda(cudaMalloc(&d_src, sizeof(h_src)), "malloc d_src") &&
                  check_cuda(cudaMalloc(&d_sizes, sizeof(h_sizes)), "malloc d_sizes") &&
                  check_cuda(cudaMemcpy(d_dest, h_dest, sizeof(h_dest), cudaMemcpyHostToDevice),
                             "memcpy d_dest h2d") &&
                  check_cuda(cudaMemcpy(d_src, h_src, sizeof(h_src), cudaMemcpyHostToDevice),
                             "memcpy d_src h2d") &&
                  check_cuda(cudaMemcpy(d_sizes, h_sizes, sizeof(h_sizes), cudaMemcpyHostToDevice),
                             "memcpy d_sizes h2d");

        if (!ok)
        {
            cudaFree(d_dest);
            cudaFree(d_src);
            cudaFree(d_sizes);
            std::cerr << "FAIL: column add basic setup\n";
            return 1;
        }

        nerve::persistence::gpu::launchWarpSpecializedColumnAdd(d_dest, d_src, d_sizes, num_words,
                                                                num_cols, 0);
        if (!check_cuda(cudaDeviceSynchronize(), "col_add sync"))
        {
            cudaFree(d_dest);
            cudaFree(d_src);
            cudaFree(d_sizes);
            return 1;
        }

        uint64_t h_result[num_cols * num_words];
        if (!check_cuda(cudaMemcpy(h_result, d_dest, sizeof(h_result), cudaMemcpyDeviceToHost),
                        "memcpy result d2h"))
        {
            cudaFree(d_dest);
            cudaFree(d_src);
            cudaFree(d_sizes);
            return 1;
        }

        // col 0: 0b0011 ^ 0b0101 = 0b0110, col 1: 0b1100 ^ 0b1010 = 0b0110
        const uint64_t expected_col0 = 0b0011ULL ^ 0b0101ULL;
        const uint64_t expected_col1 = 0b1100ULL ^ 0b1010ULL;
        if (h_result[0] != expected_col0 || h_result[1] != expected_col1)
        {
            std::cerr << "FAIL: col_add_basic got [0x" << std::hex << h_result[0] << ", 0x"
                      << h_result[1] << std::dec << "] expected [0x" << std::hex << expected_col0
                      << ", 0x" << expected_col1 << std::dec << "]\n";
            cudaFree(d_dest);
            cudaFree(d_src);
            cudaFree(d_sizes);
            return 1;
        }

        cudaFree(d_dest);
        cudaFree(d_src);
        cudaFree(d_sizes);
    }

    // Column Add Multiple Words (2 words)
    {
        constexpr int num_words = 2;
        constexpr int num_cols = 3;

        // col 0: word[0]=0xFF00, word[1]=0x00FF
        // col 1: word[0]=0x0FF0, word[1]=0xF00F
        const uint64_t h_a[num_cols * num_words] = {0xFF00ULL, 0x00FFULL, 0x0FF0ULL,
                                                    0xF00FULL, 0x0000ULL, 0xFFFFULL};
        const uint64_t h_b[num_cols * num_words] = {0x0FF0ULL, 0xF00FULL, 0xFF00ULL,
                                                    0x00FFULL, 0xAAAAULL, 0x5555ULL};
        // All columns have 2 words
        const int h_sizes[num_cols] = {2, 2, 2};

        uint64_t *d_a = nullptr;
        uint64_t *d_b = nullptr;
        int *d_sizes = nullptr;

        bool ok =
            check_cuda(cudaMalloc(&d_a, sizeof(h_a)), "malloc d_a") &&
            check_cuda(cudaMalloc(&d_b, sizeof(h_b)), "malloc d_b") &&
            check_cuda(cudaMalloc(&d_sizes, sizeof(h_sizes)), "malloc d_sizes") &&
            check_cuda(cudaMemcpy(d_a, h_a, sizeof(h_a), cudaMemcpyHostToDevice), "memcpy d_a") &&
            check_cuda(cudaMemcpy(d_b, h_b, sizeof(h_b), cudaMemcpyHostToDevice), "memcpy d_b") &&
            check_cuda(cudaMemcpy(d_sizes, h_sizes, sizeof(h_sizes), cudaMemcpyHostToDevice),
                       "memcpy d_sizes");

        if (!ok)
        {
            cudaFree(d_a);
            cudaFree(d_b);
            cudaFree(d_sizes);
            std::cerr << "FAIL: col_add_multiword setup\n";
            return 1;
        }

        nerve::persistence::gpu::launchWarpSpecializedColumnAdd(d_a, d_b, d_sizes, num_words,
                                                                num_cols, 0);
        if (!check_cuda(cudaDeviceSynchronize(), "col_add_multiword sync"))
        {
            cudaFree(d_a);
            cudaFree(d_b);
            cudaFree(d_sizes);
            return 1;
        }

        uint64_t h_result[num_cols * num_words];
        check_cuda(cudaMemcpy(h_result, d_a, sizeof(h_result), cudaMemcpyDeviceToHost),
                   "memcpy result");

        for (int col = 0; col < num_cols; ++col)
        {
            for (int w = 0; w < num_words; ++w)
            {
                uint64_t expected = h_a[col * num_words + w] ^ h_b[col * num_words + w];
                if (h_result[col * num_words + w] != expected)
                {
                    std::cerr << "FAIL: col_add_multiword col=" << col << " word=" << w << " got 0x"
                              << std::hex << h_result[col * num_words + w] << " expected 0x"
                              << expected << std::dec << "\n";
                    cudaFree(d_a);
                    cudaFree(d_b);
                    cudaFree(d_sizes);
                    return 1;
                }
            }
        }

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_sizes);
    }

    // Column Add -- zero columns (no-op)
    {
        uint64_t *d_src = nullptr;
        uint64_t *d_dst = nullptr;
        int *d_size = nullptr;

        if (!check_cuda(cudaMalloc(&d_src, sizeof(uint64_t)), "malloc zero d_src") ||
            !check_cuda(cudaMalloc(&d_dst, sizeof(uint64_t)), "malloc zero d_dst") ||
            !check_cuda(cudaMalloc(&d_size, sizeof(int)), "malloc zero d_size"))
        {
            cudaFree(d_src);
            cudaFree(d_dst);
            cudaFree(d_size);
            std::cerr << "FAIL: col_add_zero alloc\n";
            return 1;
        }

        // Should not crash or modify anything
        nerve::persistence::gpu::launchWarpSpecializedColumnAdd(d_dst, d_src, d_size, 1, 0, 0);
        if (!check_cuda(cudaDeviceSynchronize(), "col_add_zero sync"))
        {
            cudaFree(d_src);
            cudaFree(d_dst);
            cudaFree(d_size);
            return 1;
        }

        cudaFree(d_src);
        cudaFree(d_dst);
        cudaFree(d_size);
    }

    // Column Add -- null pointers (should not crash)
    {
        nerve::persistence::gpu::launchWarpSpecializedColumnAdd(nullptr, nullptr, nullptr, 1, 1, 0);
        // No crash is the pass condition
    }

    // Pivot Find Basic (1 word)
    {
        constexpr int num_words = 1;
        constexpr int num_cols = 4;

        // Column 0: bit 63 set (MSB of uint64_t) -> pivot = 63
        // Column 1: bit 0 set                    -> pivot = 0
        // Column 2: bit 33 set                   -> pivot = 33
        // Column 3: all zeros                    -> pivot = -1
        const uint64_t h_cols[num_cols * num_words] = {(1ULL << 63), 1ULL, (1ULL << 33), 0ULL};
        const int h_sizes[num_cols] = {1, 1, 1, 1};

        uint64_t *d_cols = nullptr;
        int *d_sizes = nullptr;
        int *d_pivots = nullptr;
        int h_pivots[num_cols] = {};

        bool ok = check_cuda(cudaMalloc(&d_cols, sizeof(h_cols)), "pivot malloc cols") &&
                  check_cuda(cudaMalloc(&d_sizes, sizeof(h_sizes)), "pivot malloc sizes") &&
                  check_cuda(cudaMalloc(&d_pivots, sizeof(h_pivots)), "pivot malloc pivots") &&
                  check_cuda(cudaMemcpy(d_cols, h_cols, sizeof(h_cols), cudaMemcpyHostToDevice),
                             "pivot memcpy cols") &&
                  check_cuda(cudaMemcpy(d_sizes, h_sizes, sizeof(h_sizes), cudaMemcpyHostToDevice),
                             "pivot memcpy sizes") &&
                  check_cuda(cudaMemset(d_pivots, 0xFF, sizeof(h_pivots)), "pivot memset");

        if (!ok)
        {
            cudaFree(d_cols);
            cudaFree(d_sizes);
            cudaFree(d_pivots);
            std::cerr << "FAIL: pivot_find_basic setup\n";
            return 1;
        }

        nerve::persistence::gpu::launchWarpSpecializedPivotFind(d_cols, d_sizes, num_words,
                                                                num_cols, d_pivots, 0);
        if (!check_cuda(cudaDeviceSynchronize(), "pivot_find sync"))
        {
            cudaFree(d_cols);
            cudaFree(d_sizes);
            cudaFree(d_pivots);
            return 1;
        }

        check_cuda(cudaMemcpy(h_pivots, d_pivots, sizeof(h_pivots), cudaMemcpyDeviceToHost),
                   "pivot memcpy result");

        const int expected_pivots[num_cols] = {63, 0, 33, -1};
        for (int i = 0; i < num_cols; ++i)
        {
            if (h_pivots[i] != expected_pivots[i])
            {
                std::cerr << "FAIL: pivot_find_basic col " << i << " got " << h_pivots[i]
                          << " expected " << expected_pivots[i] << "\n";
                cudaFree(d_cols);
                cudaFree(d_sizes);
                cudaFree(d_pivots);
                return 1;
            }
        }

        cudaFree(d_cols);
        cudaFree(d_sizes);
        cudaFree(d_pivots);
    }

    // Pivot Find Multiple Words
    {
        constexpr int num_words = 3;
        constexpr int num_cols = 2;

        // Column 0: word[2] has bit 0 set -> pivot = 2*64 + 0 = 128
        // Column 1: word[1] has bit 63 set, word[2] has bit 31 set -> pivot = 2*64+31 = 159
        const uint64_t h_cols[num_cols * num_words] = {
            0ULL, 0ULL,         1ULL,          // col 0: pivot at 128
            0ULL, (1ULL << 63), (1ULL << 31)}; // col 1: pivot at 159
        const int h_sizes[num_cols] = {3, 3};

        uint64_t *d_cols = nullptr;
        int *d_sizes = nullptr;
        int *d_pivots = nullptr;
        int h_pivots[num_cols] = {};

        bool ok = check_cuda(cudaMalloc(&d_cols, sizeof(h_cols)), "pivot mw malloc cols") &&
                  check_cuda(cudaMalloc(&d_sizes, sizeof(h_sizes)), "pivot mw malloc sizes") &&
                  check_cuda(cudaMalloc(&d_pivots, sizeof(h_pivots)), "pivot mw malloc pivots") &&
                  check_cuda(cudaMemcpy(d_cols, h_cols, sizeof(h_cols), cudaMemcpyHostToDevice),
                             "pivot mw memcpy cols") &&
                  check_cuda(cudaMemcpy(d_sizes, h_sizes, sizeof(h_sizes), cudaMemcpyHostToDevice),
                             "pivot mw memcpy sizes") &&
                  check_cuda(cudaMemset(d_pivots, 0xFF, sizeof(h_pivots)), "pivot mw memset");

        if (!ok)
        {
            cudaFree(d_cols);
            cudaFree(d_sizes);
            cudaFree(d_pivots);
            std::cerr << "FAIL: pivot_find_multiword setup\n";
            return 1;
        }

        nerve::persistence::gpu::launchWarpSpecializedPivotFind(d_cols, d_sizes, num_words,
                                                                num_cols, d_pivots, 0);
        if (!check_cuda(cudaDeviceSynchronize(), "pivot mw sync"))
        {
            cudaFree(d_cols);
            cudaFree(d_sizes);
            cudaFree(d_pivots);
            return 1;
        }

        check_cuda(cudaMemcpy(h_pivots, d_pivots, sizeof(h_pivots), cudaMemcpyDeviceToHost),
                   "pivot mw result");

        const int expected_pivots[num_cols] = {128, 159};
        for (int i = 0; i < num_cols; ++i)
        {
            if (h_pivots[i] != expected_pivots[i])
            {
                std::cerr << "FAIL: pivot_find_multiword col " << i << " got " << h_pivots[i]
                          << " expected " << expected_pivots[i] << "\n";
                cudaFree(d_cols);
                cudaFree(d_sizes);
                cudaFree(d_pivots);
                return 1;
            }
        }

        cudaFree(d_cols);
        cudaFree(d_sizes);
        cudaFree(d_pivots);
    }

    // Pivot Find -- zero columns (no-op)
    {
        nerve::persistence::gpu::launchWarpSpecializedPivotFind(nullptr, nullptr, 1, 0, nullptr, 0);
    }

    // Pipelined Reduction -- self-clearing (each column claims its own pivot)
    // The kernel processes independently: pivot -> itself -> clear column, return -1.
    // No cross-column dependencies, so no race condition.
    {
        constexpr int num_words = 1;
        constexpr int num_cols = 3;

        // Column 0: word = 0b100 (bit 2), pivot = 2 -> self-claim -> clear, pivot = -1
        // Column 1: word = 0b010 (bit 1), pivot = 1 -> self-claim -> clear, pivot = -1
        // Column 2: word = 0b001 (bit 0), pivot = 0 -> self-claim -> clear, pivot = -1
        uint64_t h_cols[num_cols * num_words] = {0b100ULL, 0b010ULL, 0b001ULL};
        // Each column claims its own pivot
        int h_pivot_to_col[64];
        std::memset(h_pivot_to_col, 0xFF, sizeof(h_pivot_to_col));
        h_pivot_to_col[2] = 0;
        h_pivot_to_col[1] = 1;
        h_pivot_to_col[0] = 2;

        int h_col_pivots[num_cols] = {2, 1, 0};
        int h_new_pivots[num_cols] = {};

        uint64_t *d_cols = nullptr;
        int *d_pivot_to_col = nullptr;
        int *d_col_pivots = nullptr;
        int *d_new_pivots = nullptr;

        bool ok = check_cuda(cudaMalloc(&d_cols, sizeof(h_cols)), "reduc malloc cols") &&
                  check_cuda(cudaMalloc(&d_pivot_to_col, sizeof(h_pivot_to_col)),
                             "reduc malloc pivot_to_col") &&
                  check_cuda(cudaMalloc(&d_col_pivots, sizeof(h_col_pivots)),
                             "reduc malloc col_pivots") &&
                  check_cuda(cudaMalloc(&d_new_pivots, sizeof(int) * num_cols),
                             "reduc malloc new_pivots") &&
                  check_cuda(cudaMemcpy(d_cols, h_cols, sizeof(h_cols), cudaMemcpyHostToDevice),
                             "reduc memcpy cols") &&
                  check_cuda(cudaMemcpy(d_pivot_to_col, h_pivot_to_col, sizeof(h_pivot_to_col),
                                        cudaMemcpyHostToDevice),
                             "reduc memcpy pivot_to_col") &&
                  check_cuda(cudaMemcpy(d_col_pivots, h_col_pivots, sizeof(h_col_pivots),
                                        cudaMemcpyHostToDevice),
                             "reduc memcpy col_pivots") &&
                  check_cuda(cudaMemset(d_new_pivots, 0xFF, sizeof(int) * num_cols),
                             "reduc memset new_pivots");

        if (!ok)
        {
            cudaFree(d_cols);
            cudaFree(d_pivot_to_col);
            cudaFree(d_col_pivots);
            cudaFree(d_new_pivots);
            std::cerr << "FAIL: pipelined_reduction_self_clear setup\n";
            return 1;
        }

        nerve::persistence::gpu::launchPipelinedReduction(
            d_cols, d_col_pivots, d_pivot_to_col, num_words, num_cols, d_new_pivots, 0, false);
        if (!check_cuda(cudaDeviceSynchronize(), "reduc sync"))
        {
            cudaFree(d_cols);
            cudaFree(d_pivot_to_col);
            cudaFree(d_col_pivots);
            cudaFree(d_new_pivots);
            return 1;
        }

        check_cuda(
            cudaMemcpy(h_new_pivots, d_new_pivots, sizeof(int) * num_cols, cudaMemcpyDeviceToHost),
            "reduc result");

        // All columns should self-clear (pivot = -1)
        for (int i = 0; i < num_cols; ++i)
        {
            if (h_new_pivots[i] != -1)
            {
                std::cerr << "FAIL: pipelined_reduction_self_clear col " << i
                          << " pivot=" << h_new_pivots[i] << " expected -1\n";
                cudaFree(d_cols);
                cudaFree(d_pivot_to_col);
                cudaFree(d_col_pivots);
                cudaFree(d_new_pivots);
                return 1;
            }
        }

        // Verify all columns are zeroed
        uint64_t h_result_cols[num_cols * num_words];
        check_cuda(cudaMemcpy(h_result_cols, d_cols, sizeof(h_result_cols), cudaMemcpyDeviceToHost),
                   "reduc cols result");
        for (int i = 0; i < num_cols; ++i)
        {
            if (h_result_cols[i] != 0)
            {
                std::cerr << "FAIL: pipelined_reduction_self_clear col " << i << " data=0x"
                          << std::hex << h_result_cols[i] << std::dec << " expected 0\n";
                cudaFree(d_cols);
                cudaFree(d_pivot_to_col);
                cudaFree(d_col_pivots);
                cudaFree(d_new_pivots);
                return 1;
            }
        }

        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
    }

    // Pipelined Reduction -- unclaimed pivot (no self-claim, kernel breaks
    // immediately because pivot_to_col[pivot] == -1)
    {
        constexpr int num_words = 1;
        constexpr int num_cols = 2;

        // Both columns have unclaimed pivots -> kernel breaks immediately.
        // Column 0: word = 0b100 (bit 2), pivot = 2. pivot_to_col[2] = -1 -> break, pivot=2.
        // Column 1: word = 0b010 (bit 1), pivot = 1. pivot_to_col[1] = -1 -> break, pivot=1.
        uint64_t h_cols[num_cols * num_words] = {0b100ULL, 0b010ULL};
        int h_pivot_to_col[64];
        std::memset(h_pivot_to_col, 0xFF, sizeof(h_pivot_to_col));
        // All pivots unclaimed (initialized to -1 via memset)

        int h_col_pivots[num_cols] = {2, 1};

        uint64_t *d_cols = nullptr;
        int *d_pivot_to_col = nullptr;
        int *d_col_pivots = nullptr;
        int *d_new_pivots = nullptr;
        int h_new_pivots[num_cols] = {};

        bool ok = check_cuda(cudaMalloc(&d_cols, sizeof(h_cols)), "reduc2 malloc cols") &&
                  check_cuda(cudaMalloc(&d_pivot_to_col, sizeof(h_pivot_to_col)),
                             "reduc2 malloc pivot_to_col") &&
                  check_cuda(cudaMalloc(&d_col_pivots, sizeof(h_col_pivots)),
                             "reduc2 malloc col_pivots") &&
                  check_cuda(cudaMalloc(&d_new_pivots, sizeof(int) * num_cols),
                             "reduc2 malloc new_pivots") &&
                  check_cuda(cudaMemcpy(d_cols, h_cols, sizeof(h_cols), cudaMemcpyHostToDevice),
                             "reduc2 memcpy cols") &&
                  check_cuda(cudaMemcpy(d_pivot_to_col, h_pivot_to_col, sizeof(h_pivot_to_col),
                                        cudaMemcpyHostToDevice),
                             "reduc2 memcpy pivot_to_col") &&
                  check_cuda(cudaMemcpy(d_col_pivots, h_col_pivots, sizeof(h_col_pivots),
                                        cudaMemcpyHostToDevice),
                             "reduc2 memcpy col_pivots") &&
                  check_cuda(cudaMemset(d_new_pivots, 0xFF, sizeof(int) * num_cols),
                             "reduc2 memset new_pivots");

        if (!ok)
        {
            cudaFree(d_cols);
            cudaFree(d_pivot_to_col);
            cudaFree(d_col_pivots);
            cudaFree(d_new_pivots);
            std::cerr << "FAIL: pipelined_reduction_unclaimed setup\n";
            return 1;
        }

        nerve::persistence::gpu::launchPipelinedReduction(
            d_cols, d_col_pivots, d_pivot_to_col, num_words, num_cols, d_new_pivots, 0, false);
        if (!check_cuda(cudaDeviceSynchronize(), "reduc2 sync"))
        {
            cudaFree(d_cols);
            cudaFree(d_pivot_to_col);
            cudaFree(d_col_pivots);
            cudaFree(d_new_pivots);
            return 1;
        }

        check_cuda(
            cudaMemcpy(h_new_pivots, d_new_pivots, sizeof(int) * num_cols, cudaMemcpyDeviceToHost),
            "reduc2 result");

        // Both columns should retain their original pivots (unclaimed -> break)
        const int expected_pivots[num_cols] = {2, 1};
        for (int i = 0; i < num_cols; ++i)
        {
            if (h_new_pivots[i] != expected_pivots[i])
            {
                std::cerr << "FAIL: pipelined_reduction_unclaimed col " << i
                          << " pivot=" << h_new_pivots[i] << " expected " << expected_pivots[i]
                          << "\n";
                cudaFree(d_cols);
                cudaFree(d_pivot_to_col);
                cudaFree(d_col_pivots);
                cudaFree(d_new_pivots);
                return 1;
            }
        }

        // Verify data unchanged
        uint64_t h_result_cols[num_cols * num_words];
        check_cuda(cudaMemcpy(h_result_cols, d_cols, sizeof(h_result_cols), cudaMemcpyDeviceToHost),
                   "reduc2 cols result");
        for (int i = 0; i < num_cols; ++i)
        {
            if (h_result_cols[i] != h_cols[i])
            {
                std::cerr << "FAIL: pipelined_reduction_unclaimed col " << i << " data changed\n";
                cudaFree(d_cols);
                cudaFree(d_pivot_to_col);
                cudaFree(d_col_pivots);
                cudaFree(d_new_pivots);
                return 1;
            }
        }

        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
    }

    // Pipelined Reduction -- zero columns (no-op)
    {
        nerve::persistence::gpu::launchPipelinedReduction(nullptr, nullptr, nullptr, 1, 0, nullptr,
                                                          0, false);
    }

    // Benchmark function
    {
        // Valid call returns benchmark results with non-zero times
        auto bench = nerve::persistence::gpu::benchmarkWarpSpecialization(64, 2);
        // At least one benchmark should produce non-negative time
        if (bench.column_add_time_ms < 0.0)
        {
            std::cerr << "FAIL: benchmark column_add_time_ms=" << bench.column_add_time_ms << "\n";
            return 1;
        }
        if (bench.pivot_find_time_ms < 0.0)
        {
            std::cerr << "FAIL: benchmark pivot_find_time_ms=" << bench.pivot_find_time_ms << "\n";
            return 1;
        }
        // Speedup values should be positive and finite
        if (!std::isfinite(bench.warp_speedup) || bench.warp_speedup <= 0.0)
        {
            std::cerr << "FAIL: benchmark warp_speedup=" << bench.warp_speedup << "\n";
            return 1;
        }
        if (!std::isfinite(bench.total_speedup) || bench.total_speedup <= 0.0)
        {
            std::cerr << "FAIL: benchmark total_speedup=" << bench.total_speedup << "\n";
            return 1;
        }
    }

    // Benchmark -- zero edge case
    {
        auto bench = nerve::persistence::gpu::benchmarkWarpSpecialization(0, 0);
        // Should return default (all zeros / 1.0) without crashing
        if (bench.column_add_time_ms != 0.0)
        {
            std::cerr << "FAIL: benchmark zero case column_add_time_ms=" << bench.column_add_time_ms
                      << " expected 0\n";
            return 1;
        }
    }

    // Feature query functions
    {
        bool has_tc = nerve::persistence::gpu::hasTensorCoreSupport();
        bool has_ac = nerve::persistence::gpu::hasAsyncCopySupport();
        // These should return a boolean (no crash). On a Volta+ GPU, has_tc should be true.
        // We can't assert on the exact values since it depends on hardware, but we can
        // verify they don't crash and return a consistent result.
        if (has_tc != nerve::persistence::gpu::hasTensorCoreSupport())
        {
            std::cerr << "FAIL: hasTensorCoreSupport not deterministic\n";
            return 1;
        }
        if (has_ac != nerve::persistence::gpu::hasAsyncCopySupport())
        {
            std::cerr << "FAIL: hasAsyncCopySupport not deterministic\n";
            return 1;
        }
    }

    // Optimal config function
    {
        auto cfg = nerve::persistence::gpu::getOptimalWarpSpecializationConfig(1024, 8, true);
        if (!cfg.use_warp_specialization)
        {
            std::cerr << "FAIL: config use_warp_specialization should be true\n";
            return 1;
        }
        if (cfg.warps_per_block <= 0)
        {
            std::cerr << "FAIL: config warps_per_block=" << cfg.warps_per_block
                      << " should be > 0\n";
            return 1;
        }
        // Large problem with tensor cores available should use tensor cores
        auto cfg_large =
            nerve::persistence::gpu::getOptimalWarpSpecializationConfig(4096, 16, true);
        if (!cfg_large.use_tensor_cores)
        {
            std::cerr << "FAIL: expected tensor cores enabled for large problem\n";
            return 1;
        }
        // Small problem should not use tensor cores
        auto cfg_small = nerve::persistence::gpu::getOptimalWarpSpecializationConfig(16, 2, false);
        if (cfg_small.use_tensor_cores)
        {
            std::cerr << "FAIL: expected tensor cores disabled for small problem\n";
            return 1;
        }
    }

    // estimateWarpSpecSpeedup
    {
        double blk = nerve::persistence::gpu::estimateWarpSpecSpeedup(10, 0);
        if (blk < 3.5)
        {
            std::cerr << "FAIL: estimateWarpSpecSpeedup(10,0)=" << blk << " expected >= 3.5\n";
            return 1;
        }

        double old = nerve::persistence::gpu::estimateWarpSpecSpeedup(5, 0);
        if (old != 1.0)
        {
            std::cerr << "FAIL: estimateWarpSpecSpeedup(5,0)=" << old << " expected 1.0\n";
            return 1;
        }
    }

    // Async-copy pipelined reduction -- self-clearing (1 word, use_async_copy=true)
    // Tests warpSpecializedAsyncReductionKernel which reads col_pivots as starting
    // pivots and uses shared memory for column staging.
    // Falls back to regular pipelinedReductionKernel if CUDA < 11.0 or if the
    // shared memory requirement exceeds 64KB.
    {
        constexpr int num_words = 1;
        constexpr int num_cols = 3;

        // Column 0: word = 0b100 (bit 2), pivot = 2 -> self-claim -> clear, pivot = -1
        // Column 1: word = 0b010 (bit 1), pivot = 1 -> self-claim -> clear, pivot = -1
        // Column 2: word = 0b001 (bit 0), pivot = 0 -> self-claim -> clear, pivot = -1
        uint64_t h_cols[num_cols * num_words] = {0b100ULL, 0b010ULL, 0b001ULL};
        int h_pivot_to_col[64];
        std::memset(h_pivot_to_col, 0xFF, sizeof(h_pivot_to_col));
        h_pivot_to_col[2] = 0;
        h_pivot_to_col[1] = 1;
        h_pivot_to_col[0] = 2;

        // col_pivots is READ by the async kernel as the starting pivot
        int h_col_pivots[num_cols] = {2, 1, 0};

        uint64_t *d_cols = nullptr;
        int *d_pivot_to_col = nullptr;
        int *d_col_pivots = nullptr;
        int *d_new_pivots = nullptr;
        int h_new_pivots[num_cols] = {};

        bool ok = check_cuda(cudaMalloc(&d_cols, sizeof(h_cols)), "async malloc cols") &&
                  check_cuda(cudaMalloc(&d_pivot_to_col, sizeof(h_pivot_to_col)),
                             "async malloc pivot_to_col") &&
                  check_cuda(cudaMalloc(&d_col_pivots, sizeof(h_col_pivots)),
                             "async malloc col_pivots") &&
                  check_cuda(cudaMalloc(&d_new_pivots, sizeof(int) * num_cols),
                             "async malloc new_pivots") &&
                  check_cuda(cudaMemcpy(d_cols, h_cols, sizeof(h_cols), cudaMemcpyHostToDevice),
                             "async memcpy cols") &&
                  check_cuda(cudaMemcpy(d_pivot_to_col, h_pivot_to_col, sizeof(h_pivot_to_col),
                                        cudaMemcpyHostToDevice),
                             "async memcpy pivot_to_col") &&
                  check_cuda(cudaMemcpy(d_col_pivots, h_col_pivots, sizeof(h_col_pivots),
                                        cudaMemcpyHostToDevice),
                             "async memcpy col_pivots") &&
                  check_cuda(cudaMemset(d_new_pivots, 0xFF, sizeof(int) * num_cols),
                             "async memset new_pivots");

        if (!ok)
        {
            cudaFree(d_cols);
            cudaFree(d_pivot_to_col);
            cudaFree(d_col_pivots);
            cudaFree(d_new_pivots);
            std::cerr << "FAIL: async_self_clear setup\n";
            return 1;
        }

        // use_async_copy=true activates the async kernel variant
        nerve::persistence::gpu::launchPipelinedReduction(
            d_cols, d_col_pivots, d_pivot_to_col, num_words, num_cols, d_new_pivots, 0, true);
        if (!check_cuda(cudaDeviceSynchronize(), "async sync"))
        {
            cudaFree(d_cols);
            cudaFree(d_pivot_to_col);
            cudaFree(d_col_pivots);
            cudaFree(d_new_pivots);
            return 1;
        }

        check_cuda(
            cudaMemcpy(h_new_pivots, d_new_pivots, sizeof(int) * num_cols, cudaMemcpyDeviceToHost),
            "async result");

        // All columns should self-clear (pivot = -1)
        for (int i = 0; i < num_cols; ++i)
        {
            if (h_new_pivots[i] != -1)
            {
                std::cerr << "FAIL: async_self_clear col " << i << " pivot=" << h_new_pivots[i]
                          << " expected -1\n";
                cudaFree(d_cols);
                cudaFree(d_pivot_to_col);
                cudaFree(d_col_pivots);
                cudaFree(d_new_pivots);
                return 1;
            }
        }

        // Verify all columns are zeroed
        uint64_t h_result_cols[num_cols * num_words];
        check_cuda(cudaMemcpy(h_result_cols, d_cols, sizeof(h_result_cols), cudaMemcpyDeviceToHost),
                   "async cols result");
        for (int i = 0; i < num_cols; ++i)
        {
            if (h_result_cols[i] != 0)
            {
                std::cerr << "FAIL: async_self_clear col " << i << " data=0x" << std::hex
                          << h_result_cols[i] << std::dec << " expected 0\n";
                cudaFree(d_cols);
                cudaFree(d_pivot_to_col);
                cudaFree(d_col_pivots);
                cudaFree(d_new_pivots);
                return 1;
            }
        }

        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
    }

    // Async-copy pipelined reduction -- unclaimed pivot (use_async_copy=true)
    // All pivot_to_col entries are -1, so both async and regular paths produce
    // the same result: no reduction.
    {
        constexpr int num_words = 1;
        constexpr int num_cols = 2;

        // Column 0: word = 0b100 (bit 2), pivot = 2. pivot_to_col[2] = -1 -> break.
        // Column 1: word = 0b010 (bit 1), pivot = 1. pivot_to_col[1] = -1 -> break.
        uint64_t h_cols[num_cols * num_words] = {0b100ULL, 0b010ULL};
        int h_pivot_to_col[64];
        std::memset(h_pivot_to_col, 0xFF, sizeof(h_pivot_to_col));
        // All pivots unclaimed (already -1 from memset)

        int h_col_pivots[num_cols] = {2, 1};

        uint64_t *d_cols = nullptr;
        int *d_pivot_to_col = nullptr;
        int *d_col_pivots = nullptr;
        int *d_new_pivots = nullptr;
        int h_new_pivots[num_cols] = {};

        bool ok = check_cuda(cudaMalloc(&d_cols, sizeof(h_cols)), "async2 malloc cols") &&
                  check_cuda(cudaMalloc(&d_pivot_to_col, sizeof(h_pivot_to_col)),
                             "async2 malloc pivot_to_col") &&
                  check_cuda(cudaMalloc(&d_col_pivots, sizeof(h_col_pivots)),
                             "async2 malloc col_pivots") &&
                  check_cuda(cudaMalloc(&d_new_pivots, sizeof(int) * num_cols),
                             "async2 malloc new_pivots") &&
                  check_cuda(cudaMemcpy(d_cols, h_cols, sizeof(h_cols), cudaMemcpyHostToDevice),
                             "async2 memcpy cols") &&
                  check_cuda(cudaMemcpy(d_pivot_to_col, h_pivot_to_col, sizeof(h_pivot_to_col),
                                        cudaMemcpyHostToDevice),
                             "async2 memcpy pivot_to_col") &&
                  check_cuda(cudaMemcpy(d_col_pivots, h_col_pivots, sizeof(h_col_pivots),
                                        cudaMemcpyHostToDevice),
                             "async2 memcpy col_pivots") &&
                  check_cuda(cudaMemset(d_new_pivots, 0xFF, sizeof(int) * num_cols),
                             "async2 memset new_pivots");

        if (!ok)
        {
            cudaFree(d_cols);
            cudaFree(d_pivot_to_col);
            cudaFree(d_col_pivots);
            cudaFree(d_new_pivots);
            std::cerr << "FAIL: async_unclaimed setup\n";
            return 1;
        }

        nerve::persistence::gpu::launchPipelinedReduction(
            d_cols, d_col_pivots, d_pivot_to_col, num_words, num_cols, d_new_pivots, 0, true);
        if (!check_cuda(cudaDeviceSynchronize(), "async2 sync"))
        {
            cudaFree(d_cols);
            cudaFree(d_pivot_to_col);
            cudaFree(d_col_pivots);
            cudaFree(d_new_pivots);
            return 1;
        }

        check_cuda(
            cudaMemcpy(h_new_pivots, d_new_pivots, sizeof(int) * num_cols, cudaMemcpyDeviceToHost),
            "async2 result");

        // Both columns retain original pivots (unclaimed -> break)
        const int expected_pivots[num_cols] = {2, 1};
        for (int i = 0; i < num_cols; ++i)
        {
            if (h_new_pivots[i] != expected_pivots[i])
            {
                std::cerr << "FAIL: async_unclaimed col " << i << " pivot=" << h_new_pivots[i]
                          << " expected " << expected_pivots[i] << "\n";
                cudaFree(d_cols);
                cudaFree(d_pivot_to_col);
                cudaFree(d_col_pivots);
                cudaFree(d_new_pivots);
                return 1;
            }
        }

        // Verify data unchanged
        uint64_t h_result_cols[num_cols * num_words];
        check_cuda(cudaMemcpy(h_result_cols, d_cols, sizeof(h_result_cols), cudaMemcpyDeviceToHost),
                   "async2 cols result");
        for (int i = 0; i < num_cols; ++i)
        {
            if (h_result_cols[i] != h_cols[i])
            {
                std::cerr << "FAIL: async_unclaimed col " << i << " data changed\n";
                cudaFree(d_cols);
                cudaFree(d_pivot_to_col);
                cudaFree(d_col_pivots);
                cudaFree(d_new_pivots);
                return 1;
            }
        }

        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
    }

    // Async-copy pipelined reduction -- zero columns (no-op)
    {
        nerve::persistence::gpu::launchPipelinedReduction(nullptr, nullptr, nullptr, 1, 0, nullptr,
                                                          0, true);
    }

    // Async-copy pipelined reduction -- multi-word (2 words, use_async_copy=true)
    {
        constexpr int num_words = 2;
        constexpr int num_cols = 2;

        // Each column claims its own pivot across 2 words.
        // Column 0: word[0]=0b100, word[1]=0b000 -> pivot = 2 (word 0), self-clear
        // Column 1: word[0]=0b000, word[1]=0b001 -> pivot = 64 (word 1, bit 0), self-clear
        uint64_t h_cols[num_cols * num_words] = {0b100ULL, 0b000ULL, 0b000ULL, 0b001ULL};
        int h_pivot_to_col[128]; // num_words * 64
        std::memset(h_pivot_to_col, 0xFF, sizeof(h_pivot_to_col));
        h_pivot_to_col[2] = 0;  // col 0 claims pivot 2
        h_pivot_to_col[64] = 1; // col 1 claims pivot 64

        int h_col_pivots[num_cols] = {2, 64};

        uint64_t *d_cols = nullptr;
        int *d_pivot_to_col = nullptr;
        int *d_col_pivots = nullptr;
        int *d_new_pivots = nullptr;
        int h_new_pivots[num_cols] = {};

        bool ok = check_cuda(cudaMalloc(&d_cols, sizeof(h_cols)), "async3 malloc cols") &&
                  check_cuda(cudaMalloc(&d_pivot_to_col, sizeof(h_pivot_to_col)),
                             "async3 malloc pivot_to_col") &&
                  check_cuda(cudaMalloc(&d_col_pivots, sizeof(h_col_pivots)),
                             "async3 malloc col_pivots") &&
                  check_cuda(cudaMalloc(&d_new_pivots, sizeof(int) * num_cols),
                             "async3 malloc new_pivots") &&
                  check_cuda(cudaMemcpy(d_cols, h_cols, sizeof(h_cols), cudaMemcpyHostToDevice),
                             "async3 memcpy cols") &&
                  check_cuda(cudaMemcpy(d_pivot_to_col, h_pivot_to_col, sizeof(h_pivot_to_col),
                                        cudaMemcpyHostToDevice),
                             "async3 memcpy pivot_to_col") &&
                  check_cuda(cudaMemcpy(d_col_pivots, h_col_pivots, sizeof(h_col_pivots),
                                        cudaMemcpyHostToDevice),
                             "async3 memcpy col_pivots") &&
                  check_cuda(cudaMemset(d_new_pivots, 0xFF, sizeof(int) * num_cols),
                             "async3 memset new_pivots");

        if (!ok)
        {
            cudaFree(d_cols);
            cudaFree(d_pivot_to_col);
            cudaFree(d_col_pivots);
            cudaFree(d_new_pivots);
            std::cerr << "FAIL: async_multiword setup\n";
            return 1;
        }

        nerve::persistence::gpu::launchPipelinedReduction(
            d_cols, d_col_pivots, d_pivot_to_col, num_words, num_cols, d_new_pivots, 0, true);
        if (!check_cuda(cudaDeviceSynchronize(), "async3 sync"))
        {
            cudaFree(d_cols);
            cudaFree(d_pivot_to_col);
            cudaFree(d_col_pivots);
            cudaFree(d_new_pivots);
            return 1;
        }

        check_cuda(
            cudaMemcpy(h_new_pivots, d_new_pivots, sizeof(int) * num_cols, cudaMemcpyDeviceToHost),
            "async3 result");

        // Both columns should self-clear
        for (int i = 0; i < num_cols; ++i)
        {
            if (h_new_pivots[i] != -1)
            {
                std::cerr << "FAIL: async_multiword col " << i << " pivot=" << h_new_pivots[i]
                          << " expected -1\n";
                cudaFree(d_cols);
                cudaFree(d_pivot_to_col);
                cudaFree(d_col_pivots);
                cudaFree(d_new_pivots);
                return 1;
            }
        }

        // Verify both words of each column are zeroed
        uint64_t h_result_cols[num_cols * num_words];
        check_cuda(cudaMemcpy(h_result_cols, d_cols, sizeof(h_result_cols), cudaMemcpyDeviceToHost),
                   "async3 cols result");
        for (int i = 0; i < num_cols * num_words; ++i)
        {
            if (h_result_cols[i] != 0)
            {
                std::cerr << "FAIL: async_multiword element " << i << " data=0x" << std::hex
                          << h_result_cols[i] << std::dec << " expected 0\n";
                cudaFree(d_cols);
                cudaFree(d_pivot_to_col);
                cudaFree(d_col_pivots);
                cudaFree(d_new_pivots);
                return 1;
            }
        }

        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
    }

    // Async-copy shared memory overflow fallback (large num_words > 1024)
    // When WARPS_PER_BLOCK (8) * num_words * sizeof(uint64_t) > 65536,
    // the host function falls back to the non-async pipelinedReductionKernel.
    // We verify this case produces correct results (same as test 8 but via
    // the fallback path).
    {
        // 8 warps * 2048 words * 8 bytes = 131072 bytes > 65536 -> fallback
        // But allocating 2048 words * 8 columns * 8 bytes = 128KB, which is fine
        constexpr int num_words = 2048;
        constexpr int num_cols = 3;

        std::vector<uint64_t> h_cols(static_cast<std::size_t>(num_cols * num_words), 0);
        std::vector<int> h_pivot_to_col(static_cast<std::size_t>(num_words * 64), -1);
        std::vector<int> h_col_pivots(static_cast<std::size_t>(num_cols), -1);
        std::vector<int> h_new_pivots(static_cast<std::size_t>(num_cols), 0);

        // Set up self-claiming columns at different word offsets
        // Column 0: bit 63 of word 0 -> pivot = 63
        h_cols[0 * num_words + 0] = (1ULL << 63);
        h_pivot_to_col[63] = 0;
        h_col_pivots[0] = 63;

        // Column 1: bit 0 of word 200 -> pivot = 200 * 64 = 12800
        h_cols[1 * num_words + 200] = 1ULL;
        h_pivot_to_col[200 * 64] = 1;
        h_col_pivots[1] = 200 * 64;

        // Column 2: bit 31 of word 1000 -> pivot = 1000 * 64 + 31 = 64031
        h_cols[2 * num_words + 1000] = (1ULL << 31);
        h_pivot_to_col[1000 * 64 + 31] = 2;
        h_col_pivots[2] = 1000 * 64 + 31;

        std::size_t col_bytes = static_cast<std::size_t>(num_cols * num_words) * sizeof(uint64_t);
        std::size_t ptc_bytes = static_cast<std::size_t>(num_words * 64) * sizeof(int);
        std::size_t idx_bytes = static_cast<std::size_t>(num_cols) * sizeof(int);

        uint64_t *d_cols = nullptr;
        int *d_pivot_to_col = nullptr;
        int *d_col_pivots = nullptr;
        int *d_new_pivots = nullptr;

        bool ok = check_cuda(cudaMalloc(&d_cols, col_bytes), "fallback malloc cols") &&
                  check_cuda(cudaMalloc(&d_pivot_to_col, ptc_bytes), "fallback malloc ptc") &&
                  check_cuda(cudaMalloc(&d_col_pivots, idx_bytes), "fallback malloc pivots") &&
                  check_cuda(cudaMalloc(&d_new_pivots, idx_bytes), "fallback malloc new") &&
                  check_cuda(cudaMemcpy(d_cols, h_cols.data(), col_bytes, cudaMemcpyHostToDevice),
                             "fallback memcpy cols") &&
                  check_cuda(cudaMemcpy(d_pivot_to_col, h_pivot_to_col.data(), ptc_bytes,
                                        cudaMemcpyHostToDevice),
                             "fallback memcpy ptc") &&
                  check_cuda(cudaMemcpy(d_col_pivots, h_col_pivots.data(), idx_bytes,
                                        cudaMemcpyHostToDevice),
                             "fallback memcpy pivots") &&
                  check_cuda(cudaMemset(d_new_pivots, 0xFF, idx_bytes), "fallback memset new");

        if (!ok)
        {
            cudaFree(d_cols);
            cudaFree(d_pivot_to_col);
            cudaFree(d_col_pivots);
            cudaFree(d_new_pivots);
            std::cerr << "FAIL: async_fallback setup\n";
            return 1;
        }

        // use_async_copy=true but will fall back because shared mem exceeds 64KB
        nerve::persistence::gpu::launchPipelinedReduction(
            d_cols, d_col_pivots, d_pivot_to_col, num_words, num_cols, d_new_pivots, 0, true);
        if (!check_cuda(cudaDeviceSynchronize(), "fallback sync"))
        {
            cudaFree(d_cols);
            cudaFree(d_pivot_to_col);
            cudaFree(d_col_pivots);
            cudaFree(d_new_pivots);
            return 1;
        }

        check_cuda(cudaMemcpy(h_new_pivots.data(), d_new_pivots, idx_bytes, cudaMemcpyDeviceToHost),
                   "fallback result");

        // All columns should self-clear (pivot = -1)
        for (int i = 0; i < num_cols; ++i)
        {
            if (h_new_pivots[i] != -1)
            {
                std::cerr << "FAIL: async_fallback col " << i << " pivot=" << h_new_pivots[i]
                          << " expected -1\n";
                cudaFree(d_cols);
                cudaFree(d_pivot_to_col);
                cudaFree(d_col_pivots);
                cudaFree(d_new_pivots);
                return 1;
            }
        }

        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
    }

    // Async vs Regular comparison: same reduction with use_async_copy=false
    // vs true on the same data, verifying identical new_pivots and column data.
    //
    // We use a race-free cross-column scenario where NO column self-clears AND
    // no column that is read by another is also modified. This guarantees that
    // both the async and regular kernels produce identical results.
    //
    // Chain design (race-free):
    // Col 0: 0b110 (pivot=2). pivot_to_col[2]=1 -> XOR with col 1 (0b001).
    // 0b110 ^ 0b001 = 0b111. New pivot=2. Same -> break.
    // Result: data=0b111, new_pivot=2.
    // Col 1: 0b001 (pivot=0). pivot_to_col[0]=-1 -> break.
    // Result: data=0b001, new_pivot=0.
    // Col 2: 0b100 (pivot=2). pivot_to_col[2]=1 -> XOR with col 1.
    // 0b100 ^ 0b001 = 0b101. New pivot=2. Same -> break.
    // Result: data=0b101, new_pivot=2.
    {
        constexpr int num_words = 1;
        constexpr int num_cols = 3;

        // Column data
        uint64_t h_cols_initial[num_cols * num_words] = {0b110ULL, 0b001ULL, 0b100ULL};

        // pivot_to_col: col 1 claims pivot 2; pivots 0,1 unclaimed
        int h_pivot_to_col[64];
        std::memset(h_pivot_to_col, 0xFF, sizeof(h_pivot_to_col));
        h_pivot_to_col[2] = 1; // col 1 claims pivot 2
        // pivot_to_col[0] stays -1 (unclaimed)
        // pivot_to_col[1] stays -1 (unclaimed)

        // col_pivots: starting pivots for each column
        int h_col_pivots[num_cols] = {2, 0, 2};

        // Expected results (derived manually, see design comment above)
        const uint64_t expected_data[num_cols * num_words] = {0b111ULL, 0b001ULL, 0b101ULL};
        const int expected_pivots[num_cols] = {2, 0, 2};

        // Allocate device memory (shared between both runs)
        uint64_t *d_cols = nullptr;
        int *d_pivot_to_col = nullptr;
        int *d_col_pivots = nullptr;
        int *d_new_pivots = nullptr;

        bool ok =
            check_cuda(cudaMalloc(&d_cols, sizeof(h_cols_initial)), "cmp malloc cols") &&
            check_cuda(cudaMalloc(&d_pivot_to_col, sizeof(h_pivot_to_col)),
                       "cmp malloc pivot_to_col") &&
            check_cuda(cudaMalloc(&d_col_pivots, sizeof(h_col_pivots)), "cmp malloc col_pivots") &&
            check_cuda(cudaMalloc(&d_new_pivots, sizeof(int) * num_cols), "cmp malloc new_pivots");

        if (!ok)
        {
            cudaFree(d_cols);
            cudaFree(d_pivot_to_col);
            cudaFree(d_col_pivots);
            cudaFree(d_new_pivots);
            std::cerr << "FAIL: async_vs_regular setup\n";
            return 1;
        }

        // Run with use_async_copy=false
        {
            check_cuda(
                cudaMemcpy(d_cols, h_cols_initial, sizeof(h_cols_initial), cudaMemcpyHostToDevice),
                "cmp memcpy cols (reg)");
            check_cuda(cudaMemcpy(d_pivot_to_col, h_pivot_to_col, sizeof(h_pivot_to_col),
                                  cudaMemcpyHostToDevice),
                       "cmp memcpy pivot_to_col (reg)");
            check_cuda(cudaMemcpy(d_col_pivots, h_col_pivots, sizeof(h_col_pivots),
                                  cudaMemcpyHostToDevice),
                       "cmp memcpy col_pivots (reg)");
            check_cuda(cudaMemset(d_new_pivots, 0xFF, sizeof(int) * num_cols),
                       "cmp memset new_pivots (reg)");

            nerve::persistence::gpu::launchPipelinedReduction(
                d_cols, d_col_pivots, d_pivot_to_col, num_words, num_cols, d_new_pivots, 0, false);
            if (!check_cuda(cudaDeviceSynchronize(), "cmp sync (reg)"))
            {
                cudaFree(d_cols);
                cudaFree(d_pivot_to_col);
                cudaFree(d_col_pivots);
                cudaFree(d_new_pivots);
                return 1;
            }

            // Read back results into staging buffers
            uint64_t h_reg_data[num_cols * num_words];
            int h_reg_pivots[num_cols];
            check_cuda(cudaMemcpy(h_reg_data, d_cols, sizeof(h_reg_data), cudaMemcpyDeviceToHost),
                       "cmp d2h reg cols");
            check_cuda(cudaMemcpy(h_reg_pivots, d_new_pivots, sizeof(h_reg_pivots),
                                  cudaMemcpyDeviceToHost),
                       "cmp d2h reg pivots");

            // Verify regular kernel matches expected
            for (int i = 0; i < num_cols; ++i)
            {
                if (h_reg_data[i] != expected_data[i])
                {
                    std::cerr << "FAIL: async_vs_regular regular col " << i << " data=0x"
                              << std::hex << h_reg_data[i] << std::dec << " expected 0x" << std::hex
                              << expected_data[i] << std::dec << "\n";
                    cudaFree(d_cols);
                    cudaFree(d_pivot_to_col);
                    cudaFree(d_col_pivots);
                    cudaFree(d_new_pivots);
                    return 1;
                }
                if (h_reg_pivots[i] != expected_pivots[i])
                {
                    std::cerr << "FAIL: async_vs_regular regular col " << i
                              << " pivot=" << h_reg_pivots[i] << " expected " << expected_pivots[i]
                              << "\n";
                    cudaFree(d_cols);
                    cudaFree(d_pivot_to_col);
                    cudaFree(d_col_pivots);
                    cudaFree(d_new_pivots);
                    return 1;
                }
            }
        }

        // Run with use_async_copy=true on the same data
        {
            check_cuda(
                cudaMemcpy(d_cols, h_cols_initial, sizeof(h_cols_initial), cudaMemcpyHostToDevice),
                "cmp memcpy cols (async)");
            check_cuda(cudaMemcpy(d_pivot_to_col, h_pivot_to_col, sizeof(h_pivot_to_col),
                                  cudaMemcpyHostToDevice),
                       "cmp memcpy pivot_to_col (async)");
            check_cuda(cudaMemcpy(d_col_pivots, h_col_pivots, sizeof(h_col_pivots),
                                  cudaMemcpyHostToDevice),
                       "cmp memcpy col_pivots (async)");
            check_cuda(cudaMemset(d_new_pivots, 0xFF, sizeof(int) * num_cols),
                       "cmp memset new_pivots (async)");

            nerve::persistence::gpu::launchPipelinedReduction(
                d_cols, d_col_pivots, d_pivot_to_col, num_words, num_cols, d_new_pivots, 0, true);
            if (!check_cuda(cudaDeviceSynchronize(), "cmp sync (async)"))
            {
                cudaFree(d_cols);
                cudaFree(d_pivot_to_col);
                cudaFree(d_col_pivots);
                cudaFree(d_new_pivots);
                return 1;
            }

            // Read back results
            uint64_t h_async_data[num_cols * num_words];
            int h_async_pivots[num_cols];
            check_cuda(
                cudaMemcpy(h_async_data, d_cols, sizeof(h_async_data), cudaMemcpyDeviceToHost),
                "cmp d2h async cols");
            check_cuda(cudaMemcpy(h_async_pivots, d_new_pivots, sizeof(h_async_pivots),
                                  cudaMemcpyDeviceToHost),
                       "cmp d2h async pivots");

            // Compare against regular kernel results (already verified above)
            // and against expected values
            for (int i = 0; i < num_cols; ++i)
            {
                if (h_async_data[i] != expected_data[i])
                {
                    std::cerr << "FAIL: async_vs_regular async col " << i << " data=0x" << std::hex
                              << h_async_data[i] << std::dec << " expected 0x" << std::hex
                              << expected_data[i] << std::dec << "\n";
                    cudaFree(d_cols);
                    cudaFree(d_pivot_to_col);
                    cudaFree(d_col_pivots);
                    cudaFree(d_new_pivots);
                    return 1;
                }
                if (h_async_pivots[i] != expected_pivots[i])
                {
                    std::cerr << "FAIL: async_vs_regular async col " << i
                              << " pivot=" << h_async_pivots[i] << " expected "
                              << expected_pivots[i] << "\n";
                    cudaFree(d_cols);
                    cudaFree(d_pivot_to_col);
                    cudaFree(d_col_pivots);
                    cudaFree(d_new_pivots);
                    return 1;
                }
            }
        }

        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
    }

    // Cross-column reduction: single-warp reference vs production kernels
    //
    // Uses a file-scope singleWarpReductionRef kernel that processes all
    // columns SEQUENTIALLY in one warp (race-free by construction since
    // only one warp is active, so no inter-warp read-write conflict).
    //
    // The test:
    // Runs the single-warp reference kernel
    // Runs the regular kernel (use_async_copy=false) 5 times
    // Runs the async kernel (use_async_copy=true) 5 times
    // Verifies all new_pivots match the reference
    //
    // Cross-column chain (Col 2 is leaf, never writes to itself):
    // Col 0: 0b110 (pivot=2). pivot_to_col[2]=1. XOR with Col 1.
    // Col 1: 0b010 (pivot=1). pivot_to_col[1]=2. XOR with Col 2.
    // Col 2: 0b001 (pivot=0). pivot_to_col[0]=-1. Break (unclaimed).
    {
        constexpr int num_words = 1;
        constexpr int num_cols = 3;

        const uint64_t h_cols[num_cols * num_words] = {0b110ULL, 0b010ULL, 0b001ULL};
        const int h_col_pivots[num_cols] = {2, 1, 0};

        int h_pivot_to_col[64];
        std::memset(h_pivot_to_col, 0xFF, sizeof(h_pivot_to_col));
        h_pivot_to_col[2] = 1; // col 1 claims pivot 2
        h_pivot_to_col[1] = 2; // col 2 claims pivot 1
        // pivot_to_col[0] stays -1 (unclaimed)

        uint64_t *d_cols = nullptr;
        int *d_pivot_to_col = nullptr;
        int *d_col_pivots = nullptr;
        int *d_new_pivots = nullptr;

        bool ok =
            check_cuda(cudaMalloc(&d_cols, sizeof(h_cols)), "xref malloc cols") &&
            check_cuda(cudaMalloc(&d_pivot_to_col, sizeof(h_pivot_to_col)), "xref malloc ptc") &&
            check_cuda(cudaMalloc(&d_col_pivots, sizeof(h_col_pivots)), "xref malloc cp") &&
            check_cuda(cudaMalloc(&d_new_pivots, sizeof(int) * num_cols), "xref malloc np");

        if (!ok)
        {
            cudaFree(d_cols);
            cudaFree(d_pivot_to_col);
            cudaFree(d_col_pivots);
            cudaFree(d_new_pivots);
            std::cerr << "FAIL: xref_cross_col setup\n";
            return 1;
        }

        // Helper to upload initial data
        auto upload_data = [&]() -> bool {
            return check_cuda(cudaMemcpy(d_cols, h_cols, sizeof(h_cols), cudaMemcpyHostToDevice),
                              "xref memcpy cols") &&
                   check_cuda(cudaMemcpy(d_pivot_to_col, h_pivot_to_col, sizeof(h_pivot_to_col),
                                         cudaMemcpyHostToDevice),
                              "xref memcpy ptc") &&
                   check_cuda(cudaMemcpy(d_col_pivots, h_col_pivots, sizeof(h_col_pivots),
                                         cudaMemcpyHostToDevice),
                              "xref memcpy cp");
        };

        // single-warp reference (sequential, race-free)
        int ref_pivots[num_cols] = {};
        {
            if (!upload_data())
            {
                goto xref_cleanup;
            }
            check_cuda(cudaMemset(d_new_pivots, 0xFF, sizeof(int) * num_cols),
                       "xref memset np (ref)");

            singleWarpReductionRef<<<1, 32>>>(d_cols, d_pivot_to_col, num_words, num_cols,
                                              d_new_pivots);
            if (cudaDeviceSynchronize() != cudaSuccess)
            {
                std::cerr << "FAIL: xref_cross_col ref kernel launch failed\n";
                goto xref_cleanup;
            }

            check_cuda(
                cudaMemcpy(ref_pivots, d_new_pivots, sizeof(ref_pivots), cudaMemcpyDeviceToHost),
                "xref d2h ref pivots");

            for (int i = 0; i < num_cols; ++i)
            {
                if (ref_pivots[i] < 0)
                {
                    std::cerr << "FAIL: xref_cross_col ref pivot[" << i << "]=" << ref_pivots[i]
                              << " expected >=0\n";
                    goto xref_cleanup;
                }
            }
        }

        // regular kernel (use_async_copy=false), 5 runs
        {
            for (int run = 0; run < 5; ++run)
            {
                if (!upload_data())
                {
                    goto xref_cleanup;
                }
                check_cuda(cudaMemset(d_new_pivots, 0xFF, sizeof(int) * num_cols),
                           "xref memset np (reg)");

                nerve::persistence::gpu::launchPipelinedReduction(d_cols, d_col_pivots,
                                                                  d_pivot_to_col, num_words,
                                                                  num_cols, d_new_pivots, 0, false);
                if (cudaDeviceSynchronize() != cudaSuccess)
                {
                    std::cerr << "FAIL: xref_cross_col reg run " << run << " launch failed\n";
                    goto xref_cleanup;
                }

                int run_pivots[num_cols] = {};
                check_cuda(cudaMemcpy(run_pivots, d_new_pivots, sizeof(run_pivots),
                                      cudaMemcpyDeviceToHost),
                           "xref d2h reg pivots");

                for (int i = 0; i < num_cols; ++i)
                {
                    if (run_pivots[i] != ref_pivots[i])
                    {
                        std::cerr << "FAIL: xref_cross_col reg run " << run << " col " << i
                                  << " pivot=" << run_pivots[i] << " != ref " << ref_pivots[i]
                                  << "\n";
                        goto xref_cleanup;
                    }
                }
            }
        }

        // async kernel (use_async_copy=true), 5 runs
        {
            for (int run = 0; run < 5; ++run)
            {
                if (!upload_data())
                {
                    goto xref_cleanup;
                }
                check_cuda(cudaMemset(d_new_pivots, 0xFF, sizeof(int) * num_cols),
                           "xref memset np (async)");

                nerve::persistence::gpu::launchPipelinedReduction(d_cols, d_col_pivots,
                                                                  d_pivot_to_col, num_words,
                                                                  num_cols, d_new_pivots, 0, true);
                if (cudaDeviceSynchronize() != cudaSuccess)
                {
                    std::cerr << "FAIL: xref_cross_col async run " << run << " launch failed\n";
                    goto xref_cleanup;
                }

                int run_pivots[num_cols] = {};
                check_cuda(cudaMemcpy(run_pivots, d_new_pivots, sizeof(run_pivots),
                                      cudaMemcpyDeviceToHost),
                           "xref d2h async pivots");

                for (int i = 0; i < num_cols; ++i)
                {
                    if (run_pivots[i] != ref_pivots[i])
                    {
                        std::cerr << "FAIL: xref_cross_col async run " << run << " col " << i
                                  << " pivot=" << run_pivots[i] << " != ref " << ref_pivots[i]
                                  << "\n";
                        goto xref_cleanup;
                    }
                }
            }
        }

        // All runs succeeded: free and continue to return 0
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        goto xref_done;

xref_cleanup:
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        return 1;

xref_done:;
    }

    // Multi-word (2-word) cross-column reduction: single-warp reference
    // vs production kernels (regular + async).
    //
    // Cross-column chain where XOR with the leaf column CHANGES the pivot's
    // word (from word 0 to word 1), avoiding oscillation:
    //
    // Col 0: [0x0000, 0b100] (pivot=66). pivot_to_col[66]=1. XOR with Col 1.
    // Col 1: [0b010, 0x0000] (pivot=1).  pivot_to_col[1]=2.  XOR with Col 2.
    // Col 2: [0x0000, 0b001] (pivot=64). pivot_to_col[64]=-1. Break (unclaimed).
    //
    // After XOR with Col 2, Col 1's MSB moves from word 0 (pivot=1) to word 1
    // (pivot=64), which is unclaimed -> kernel breaks immediately. No oscillation.
    {
        constexpr int num_words = 2;
        constexpr int num_cols = 3;

        // Column data: stored as [col0_w0, col0_w1, col1_w0, col1_w1, col2_w0, col2_w1]
        const uint64_t h_cols[num_cols * num_words] = {
            0x0000ULL, 0b100ULL,  // col 0: pivot=66 (word 1, bit 2)
            0b010ULL,  0x0000ULL, // col 1: pivot=1 (word 0, bit 1)
            0x0000ULL, 0b001ULL   // col 2: pivot=64 (word 1, bit 0), unclaimed
        };

        const int h_col_pivots[num_cols] = {66, 1, 64};

        // pivot_to_col sized for 2 words * 64 = 128 possible pivots
        int h_pivot_to_col[128];
        std::memset(h_pivot_to_col, 0xFF, sizeof(h_pivot_to_col));
        h_pivot_to_col[66] = 1; // col 1 claims pivot 66 (tells col 0 to XOR with col 1)
        h_pivot_to_col[1] = 2;  // col 2 claims pivot 1 (tells col 1 to XOR with col 2)
        // pivot_to_col[64] stays -1 (unclaimed)

        // Expected: col 1's pivot changes to 64 after XOR, then breaks.
        // The reference processes sequentially and produces deterministic results.

        uint64_t *d_cols = nullptr;
        int *d_pivot_to_col = nullptr;
        int *d_col_pivots = nullptr;
        int *d_new_pivots = nullptr;

        std::size_t col_bytes = sizeof(h_cols);
        std::size_t ptc_bytes = sizeof(h_pivot_to_col);
        std::size_t idx_bytes = static_cast<std::size_t>(num_cols) * sizeof(int);

        bool ok = check_cuda(cudaMalloc(&d_cols, col_bytes), "mw2 malloc cols") &&
                  check_cuda(cudaMalloc(&d_pivot_to_col, ptc_bytes), "mw2 malloc ptc") &&
                  check_cuda(cudaMalloc(&d_col_pivots, idx_bytes), "mw2 malloc cp") &&
                  check_cuda(cudaMalloc(&d_new_pivots, idx_bytes), "mw2 malloc np");

        if (!ok)
        {
            cudaFree(d_cols);
            cudaFree(d_pivot_to_col);
            cudaFree(d_col_pivots);
            cudaFree(d_new_pivots);
            std::cerr << "FAIL: mw2_cross_col setup\n";
            return 1;
        }

        // Upload helper
        auto upload_mw2 = [&]() -> bool {
            return check_cuda(cudaMemcpy(d_cols, h_cols, col_bytes, cudaMemcpyHostToDevice),
                              "mw2 memcpy cols") &&
                   check_cuda(cudaMemcpy(d_pivot_to_col, h_pivot_to_col, ptc_bytes,
                                         cudaMemcpyHostToDevice),
                              "mw2 memcpy ptc") &&
                   check_cuda(
                       cudaMemcpy(d_col_pivots, h_col_pivots, idx_bytes, cudaMemcpyHostToDevice),
                       "mw2 memcpy cp");
        };

        // single-warp reference
        int ref_pivots_mw2[num_cols] = {};
        {
            if (!upload_mw2())
            {
                goto mw2_cleanup;
            }
            check_cuda(cudaMemset(d_new_pivots, 0xFF, idx_bytes), "mw2 memset np (ref)");

            singleWarpReductionRef<<<1, 32>>>(d_cols, d_pivot_to_col, num_words, num_cols,
                                              d_new_pivots);
            if (cudaDeviceSynchronize() != cudaSuccess)
            {
                std::cerr << "FAIL: mw2_cross_col ref kernel launch failed\n";
                goto mw2_cleanup;
            }

            check_cuda(cudaMemcpy(ref_pivots_mw2, d_new_pivots, sizeof(ref_pivots_mw2),
                                  cudaMemcpyDeviceToHost),
                       "mw2 d2h ref pivots");

            for (int i = 0; i < num_cols; ++i)
            {
                if (ref_pivots_mw2[i] < 0)
                {
                    std::cerr << "FAIL: mw2_cross_col ref pivot[" << i << "]=" << ref_pivots_mw2[i]
                              << " expected >=0\n";
                    goto mw2_cleanup;
                }
            }
        }

        // regular kernel, 5 runs
        {
            for (int run = 0; run < 5; ++run)
            {
                if (!upload_mw2())
                {
                    goto mw2_cleanup;
                }
                check_cuda(cudaMemset(d_new_pivots, 0xFF, idx_bytes), "mw2 memset np (reg)");

                nerve::persistence::gpu::launchPipelinedReduction(d_cols, d_col_pivots,
                                                                  d_pivot_to_col, num_words,
                                                                  num_cols, d_new_pivots, 0, false);
                if (cudaDeviceSynchronize() != cudaSuccess)
                {
                    std::cerr << "FAIL: mw2_cross_col reg run " << run << " launch failed\n";
                    goto mw2_cleanup;
                }

                int run_pivots_mw2[num_cols] = {};
                check_cuda(cudaMemcpy(run_pivots_mw2, d_new_pivots, sizeof(run_pivots_mw2),
                                      cudaMemcpyDeviceToHost),
                           "mw2 d2h reg pivots");

                for (int i = 0; i < num_cols; ++i)
                {
                    if (run_pivots_mw2[i] != ref_pivots_mw2[i])
                    {
                        std::cerr << "FAIL: mw2_cross_col reg run " << run << " col " << i
                                  << " pivot=" << run_pivots_mw2[i] << " != ref "
                                  << ref_pivots_mw2[i] << "\n";
                        goto mw2_cleanup;
                    }
                }
            }
        }

        // async kernel, 5 runs
        {
            for (int run = 0; run < 5; ++run)
            {
                if (!upload_mw2())
                {
                    goto mw2_cleanup;
                }
                check_cuda(cudaMemset(d_new_pivots, 0xFF, idx_bytes), "mw2 memset np (async)");

                nerve::persistence::gpu::launchPipelinedReduction(d_cols, d_col_pivots,
                                                                  d_pivot_to_col, num_words,
                                                                  num_cols, d_new_pivots, 0, true);
                if (cudaDeviceSynchronize() != cudaSuccess)
                {
                    std::cerr << "FAIL: mw2_cross_col async run " << run << " launch failed\n";
                    goto mw2_cleanup;
                }

                int run_pivots_mw2[num_cols] = {};
                check_cuda(cudaMemcpy(run_pivots_mw2, d_new_pivots, sizeof(run_pivots_mw2),
                                      cudaMemcpyDeviceToHost),
                           "mw2 d2h async pivots");

                for (int i = 0; i < num_cols; ++i)
                {
                    if (run_pivots_mw2[i] != ref_pivots_mw2[i])
                    {
                        std::cerr << "FAIL: mw2_cross_col async run " << run << " col " << i
                                  << " pivot=" << run_pivots_mw2[i] << " != ref "
                                  << ref_pivots_mw2[i] << "\n";
                        goto mw2_cleanup;
                    }
                }
            }
        }

        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        goto mw2_done;

mw2_cleanup:
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        return 1;

mw2_done:;
    }

    // __syncthreads-barrier reduction vs single-warp ref vs production kernels
    //
    // Uses the block-synchronized kernel syncthreadsReductionRef which processes
    // all columns concurrently (one warp per column) with __syncthreads()
    // barriers between iterations. This provides a fully synchronized reference
    // that is race-free by construction.
    //
    // The test:
    // Runs single-warp reference (sequential, race-free)
    // Runs syncthreads reference (parallel, barrier-synchronized)
    // Runs regular kernel (use_async_copy=false) 5 times
    // Runs async kernel (use_async_copy=true) 5 times
    // Verifies all new_pivots match the single-warp reference
    //
    // Cross-column chain (same as test 22):
    // Col 0: 0b110 (pivot=2). pivot_to_col[2]=1. XOR with Col 1.
    // Col 1: 0b010 (pivot=1). pivot_to_col[1]=2. XOR with Col 2.
    // Col 2: 0b001 (pivot=0). pivot_to_col[0]=-1. Break (unclaimed).
    //
    // On this race-free chain, ALL variants produce {2, 1, 0} because col 2
    // (the leaf) is never modified and both col 0/1 oscillate to consistent
    // values after 4 iterations.
    {
        constexpr int num_words = 1;
        constexpr int num_cols = 3;

        const uint64_t h_cols[num_cols * num_words] = {0b110ULL, 0b010ULL, 0b001ULL};
        const int h_col_pivots[num_cols] = {2, 1, 0};

        int h_pivot_to_col[64];
        std::memset(h_pivot_to_col, 0xFF, sizeof(h_pivot_to_col));
        h_pivot_to_col[2] = 1; // col 1 claims pivot 2
        h_pivot_to_col[1] = 2; // col 2 claims pivot 1
        // pivot_to_col[0] stays -1 (unclaimed)

        uint64_t *d_cols = nullptr;
        int *d_pivot_to_col = nullptr;
        int *d_col_pivots = nullptr;
        int *d_new_pivots = nullptr;
        std::size_t idx_bytes = static_cast<std::size_t>(num_cols) * sizeof(int);

        bool ok =
            check_cuda(cudaMalloc(&d_cols, sizeof(h_cols)), "sync malloc cols") &&
            check_cuda(cudaMalloc(&d_pivot_to_col, sizeof(h_pivot_to_col)), "sync malloc ptc") &&
            check_cuda(cudaMalloc(&d_col_pivots, sizeof(h_col_pivots)), "sync malloc cp") &&
            check_cuda(cudaMalloc(&d_new_pivots, idx_bytes), "sync malloc np");

        if (!ok)
        {
            cudaFree(d_cols);
            cudaFree(d_pivot_to_col);
            cudaFree(d_col_pivots);
            cudaFree(d_new_pivots);
            std::cerr << "FAIL: syncthreads_cross_col setup\n";
            return 1;
        }

        // Upload helper
        auto upload_sync = [&]() -> bool {
            return check_cuda(cudaMemcpy(d_cols, h_cols, sizeof(h_cols), cudaMemcpyHostToDevice),
                              "sync memcpy cols") &&
                   check_cuda(cudaMemcpy(d_pivot_to_col, h_pivot_to_col, sizeof(h_pivot_to_col),
                                         cudaMemcpyHostToDevice),
                              "sync memcpy ptc") &&
                   check_cuda(cudaMemcpy(d_col_pivots, h_col_pivots, sizeof(h_col_pivots),
                                         cudaMemcpyHostToDevice),
                              "sync memcpy cp");
        };

        // single-warp reference (sequential, race-free)
        int ref_pivots[num_cols] = {};
        {
            if (!upload_sync())
            {
                goto sync_cleanup;
            }
            check_cuda(cudaMemset(d_new_pivots, 0xFF, idx_bytes), "sync memset np (sw ref)");

            singleWarpReductionRef<<<1, 32>>>(d_cols, d_pivot_to_col, num_words, num_cols,
                                              d_new_pivots);
            if (cudaDeviceSynchronize() != cudaSuccess)
            {
                std::cerr << "FAIL: syncthreads_cross_col sw ref launch failed\n";
                goto sync_cleanup;
            }

            check_cuda(
                cudaMemcpy(ref_pivots, d_new_pivots, sizeof(ref_pivots), cudaMemcpyDeviceToHost),
                "sync d2h sw ref");
        }

        // syncthreads-barrier reference (block-synchronized, race-free)
        {
            int st_pivots[num_cols] = {};
            if (!upload_sync())
            {
                goto sync_cleanup;
            }
            check_cuda(cudaMemset(d_new_pivots, 0xFF, idx_bytes), "sync memset np (st ref)");

            const int block_threads = max(32, num_cols * 32);
            syncthreadsReductionRef<<<1, block_threads>>>(d_cols, d_col_pivots, d_pivot_to_col,
                                                          num_words, num_cols, d_new_pivots);
            if (cudaDeviceSynchronize() != cudaSuccess)
            {
                std::cerr << "FAIL: syncthreads_cross_col st ref launch failed\n";
                goto sync_cleanup;
            }

            check_cuda(
                cudaMemcpy(st_pivots, d_new_pivots, sizeof(st_pivots), cudaMemcpyDeviceToHost),
                "sync d2h st ref");

            // Verify syncthreads ref matches single-warp ref
            for (int i = 0; i < num_cols; ++i)
            {
                if (st_pivots[i] != ref_pivots[i])
                {
                    std::cerr << "FAIL: syncthreads_cross_col st pivot[" << i
                              << "]=" << st_pivots[i] << " != sw ref " << ref_pivots[i] << "\n";
                    goto sync_cleanup;
                }
            }
        }

        // regular kernel (use_async_copy=false), 5 runs
        {
            for (int run = 0; run < 5; ++run)
            {
                if (!upload_sync())
                {
                    goto sync_cleanup;
                }
                check_cuda(cudaMemset(d_new_pivots, 0xFF, idx_bytes), "sync memset np (reg)");

                nerve::persistence::gpu::launchPipelinedReduction(d_cols, d_col_pivots,
                                                                  d_pivot_to_col, num_words,
                                                                  num_cols, d_new_pivots, 0, false);
                if (cudaDeviceSynchronize() != cudaSuccess)
                {
                    std::cerr << "FAIL: syncthreads_cross_col reg run " << run
                              << " launch failed\n";
                    goto sync_cleanup;
                }

                int run_pivots[num_cols] = {};
                check_cuda(cudaMemcpy(run_pivots, d_new_pivots, sizeof(run_pivots),
                                      cudaMemcpyDeviceToHost),
                           "sync d2h reg");

                for (int i = 0; i < num_cols; ++i)
                {
                    if (run_pivots[i] != ref_pivots[i])
                    {
                        std::cerr << "FAIL: syncthreads_cross_col reg run " << run << " col " << i
                                  << " pivot=" << run_pivots[i] << " != ref " << ref_pivots[i]
                                  << "\n";
                        goto sync_cleanup;
                    }
                }
            }
        }

        // async kernel (use_async_copy=true), 5 runs
        {
            for (int run = 0; run < 5; ++run)
            {
                if (!upload_sync())
                {
                    goto sync_cleanup;
                }
                check_cuda(cudaMemset(d_new_pivots, 0xFF, idx_bytes), "sync memset np (async)");

                nerve::persistence::gpu::launchPipelinedReduction(d_cols, d_col_pivots,
                                                                  d_pivot_to_col, num_words,
                                                                  num_cols, d_new_pivots, 0, true);
                if (cudaDeviceSynchronize() != cudaSuccess)
                {
                    std::cerr << "FAIL: syncthreads_cross_col async run " << run
                              << " launch failed\n";
                    goto sync_cleanup;
                }

                int run_pivots[num_cols] = {};
                check_cuda(cudaMemcpy(run_pivots, d_new_pivots, sizeof(run_pivots),
                                      cudaMemcpyDeviceToHost),
                           "sync d2h async");

                for (int i = 0; i < num_cols; ++i)
                {
                    if (run_pivots[i] != ref_pivots[i])
                    {
                        std::cerr << "FAIL: syncthreads_cross_col async run " << run << " col " << i
                                  << " pivot=" << run_pivots[i] << " != ref " << ref_pivots[i]
                                  << "\n";
                        goto sync_cleanup;
                    }
                }
            }
        }

        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        goto sync_done;

sync_cleanup:
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        return 1;

sync_done:;
    }

    // Async-copy cross-column determinism: 10 runs, identical results each time
    //
    // Runs the async kernel (use_async_copy=true) 10 times on the same cross-column
    // chain, verifying that every run produces identical pivots AND column data.
    //
    // The async kernel reads col_pivots as starting pivots and uses shared memory
    // staging via async_pipeline_stage_and_xor. This test validates that the shared
    // memory path is deterministic across multiple launches.
    //
    // Chain (same as tests 22/24, race-free):
    // Col 0: 0b110 (pivot=2). pivot_to_col[2]=1. XOR with Col 1.
    // Col 1: 0b010 (pivot=1). pivot_to_col[1]=2. XOR with Col 2.
    // Col 2: 0b001 (pivot=0). pivot_to_col[0]=-1. Break (unclaimed).
    //
    // Expected: all 10 runs produce identical new_pivots AND column data.
    {
        constexpr int num_words = 1;
        constexpr int num_cols = 3;
        constexpr int num_runs = 10;

        const uint64_t h_cols[num_cols * num_words] = {0b110ULL, 0b010ULL, 0b001ULL};
        const int h_col_pivots[num_cols] = {2, 1, 0};

        int h_pivot_to_col[64];
        std::memset(h_pivot_to_col, 0xFF, sizeof(h_pivot_to_col));
        h_pivot_to_col[2] = 1; // col 1 claims pivot 2
        h_pivot_to_col[1] = 2; // col 2 claims pivot 1
        // pivot_to_col[0] stays -1 (unclaimed)

        uint64_t *d_cols = nullptr;
        int *d_pivot_to_col = nullptr;
        int *d_col_pivots = nullptr;
        int *d_new_pivots = nullptr;
        std::size_t col_bytes = sizeof(h_cols);
        std::size_t ptc_bytes = sizeof(h_pivot_to_col);
        std::size_t idx_bytes = static_cast<std::size_t>(num_cols) * sizeof(int);

        bool ok = check_cuda(cudaMalloc(&d_cols, col_bytes), "det malloc cols") &&
                  check_cuda(cudaMalloc(&d_pivot_to_col, ptc_bytes), "det malloc ptc") &&
                  check_cuda(cudaMalloc(&d_col_pivots, idx_bytes), "det malloc cp") &&
                  check_cuda(cudaMalloc(&d_new_pivots, idx_bytes), "det malloc np");

        if (!ok)
        {
            cudaFree(d_cols);
            cudaFree(d_pivot_to_col);
            cudaFree(d_col_pivots);
            cudaFree(d_new_pivots);
            std::cerr << "FAIL: async_cross_col_determinism setup\n";
            return 1;
        }

        // Upload initial data (shared across all runs)
        if (!check_cuda(
                cudaMemcpy(d_pivot_to_col, h_pivot_to_col, ptc_bytes, cudaMemcpyHostToDevice),
                "det memcpy ptc") ||
            !check_cuda(cudaMemcpy(d_col_pivots, h_col_pivots, idx_bytes, cudaMemcpyHostToDevice),
                        "det memcpy cp"))
        {
            cudaFree(d_cols);
            cudaFree(d_pivot_to_col);
            cudaFree(d_col_pivots);
            cudaFree(d_new_pivots);
            std::cerr << "FAIL: async_cross_col_determinism upload ptc/cp\n";
            return 1;
        }

        // Run 0: store reference results
        int ref_pivots[num_cols] = {};
        uint64_t ref_data[num_cols * num_words] = {};
        {
            if (!check_cuda(cudaMemcpy(d_cols, h_cols, col_bytes, cudaMemcpyHostToDevice),
                            "det memcpy cols (run 0)"))
            {
                goto det_cleanup;
            }
            check_cuda(cudaMemset(d_new_pivots, 0xFF, idx_bytes), "det memset np (run 0)");

            nerve::persistence::gpu::launchPipelinedReduction(
                d_cols, d_col_pivots, d_pivot_to_col, num_words, num_cols, d_new_pivots, 0, true);
            if (cudaDeviceSynchronize() != cudaSuccess)
            {
                std::cerr << "FAIL: async_cross_col_determinism run 0 launch failed\n";
                goto det_cleanup;
            }

            check_cuda(
                cudaMemcpy(ref_pivots, d_new_pivots, sizeof(ref_pivots), cudaMemcpyDeviceToHost),
                "det d2h pivots (run 0)");
            check_cuda(cudaMemcpy(ref_data, d_cols, sizeof(ref_data), cudaMemcpyDeviceToHost),
                       "det d2h cols (run 0)");
        }

        // Runs 1..num_runs-1: compare against reference
        for (int run = 1; run < num_runs; ++run)
        {
            if (!check_cuda(cudaMemcpy(d_cols, h_cols, col_bytes, cudaMemcpyHostToDevice),
                            "det memcpy cols"))
            {
                goto det_cleanup;
            }
            check_cuda(cudaMemset(d_new_pivots, 0xFF, idx_bytes), "det memset np");

            nerve::persistence::gpu::launchPipelinedReduction(
                d_cols, d_col_pivots, d_pivot_to_col, num_words, num_cols, d_new_pivots, 0, true);
            if (cudaDeviceSynchronize() != cudaSuccess)
            {
                std::cerr << "FAIL: async_cross_col_determinism run " << run << " launch failed\n";
                goto det_cleanup;
            }

            int run_pivots[num_cols] = {};
            uint64_t run_data[num_cols * num_words] = {};
            check_cuda(
                cudaMemcpy(run_pivots, d_new_pivots, sizeof(run_pivots), cudaMemcpyDeviceToHost),
                "det d2h pivots");
            check_cuda(cudaMemcpy(run_data, d_cols, sizeof(run_data), cudaMemcpyDeviceToHost),
                       "det d2h cols");

            // Compare pivots
            for (int i = 0; i < num_cols; ++i)
            {
                if (run_pivots[i] != ref_pivots[i])
                {
                    std::cerr << "FAIL: async_cross_col_determinism run " << run << " col " << i
                              << " pivot=" << run_pivots[i] << " != ref " << ref_pivots[i] << "\n";
                    goto det_cleanup;
                }
            }

            // Compare column data
            for (int i = 0; i < num_cols * num_words; ++i)
            {
                if (run_data[i] != ref_data[i])
                {
                    std::cerr << "FAIL: async_cross_col_determinism run " << run << " element " << i
                              << " data=0x" << std::hex << run_data[i] << std::dec << " != ref 0x"
                              << std::hex << ref_data[i] << std::dec << "\n";
                    goto det_cleanup;
                }
            }
        }

        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        goto det_done;

det_cleanup:
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        return 1;

det_done:;
    }

    // 3-word cross-column reduction: single-warp ref vs syncthreads vs
    // regular (5x) vs async (5x)
    //
    // Cross-column chain with pivots spanning all 3 words.
    // Col 1 oscillates between [0b001, 0b100, 0] and [0, 0b100, 0] by XOR-ing
    // with Col 2 (immutable leaf). Col 0 XORs with Col 1 across all 3 words.
    // After 12 iterations (even), ALL variants converge to identical results
    // because Col 0 starts with word[0]=0 and the XOR-cancelation property
    // holds over a full period of Col 1's oscillation.
    //
    // Chain:
    // Col 0: [0, 0, 0b100] (pivot=130=2*64+2).
    // pivot_to_col[130]=1 -> XOR with Col 1.
    // Col 1: [0, 0b100, 0]  (pivot=66=1*64+2).
    // pivot_to_col[66]=2 -> XOR with Col 2.
    // Col 2: [0b001, 0, 0]  (pivot=0).
    // pivot_to_col[0]=-1 -> Break (unclaimed).
    //
    // Expected (all models): pivots={130, 66, 0}
    // Col 0=[0, 0, 0b100], Col 1=[0, 0b100, 0], Col 2=[0b001, 0, 0]
    {
        constexpr int num_words = 3;
        constexpr int num_cols = 3;

        // Column data: stored as [col0_w0, col0_w1, col0_w2, col1_w0, ...]
        const uint64_t h_cols[num_cols * num_words] = {
            0x0000ULL, 0x0000ULL, 0b100ULL,  // col 0: pivot=130 (word 2, bit 2)
            0x0000ULL, 0b100ULL,  0x0000ULL, // col 1: pivot=66 (word 1, bit 2)
            0b001ULL,  0x0000ULL, 0x0000ULL  // col 2: pivot=0 (word 0, bit 0), unclaimed
        };

        const int h_col_pivots[num_cols] = {130, 66, 0};

        // pivot_to_col sized for 3 words * 64 = 192 possible pivots
        int h_pivot_to_col[192];
        std::memset(h_pivot_to_col, 0xFF, sizeof(h_pivot_to_col));
        h_pivot_to_col[130] = 1; // col 1 claims pivot 130 (tells col 0 to XOR with col 1)
        h_pivot_to_col[66] = 2;  // col 2 claims pivot 66 (tells col 1 to XOR with col 2)
        // pivot_to_col[0] stays -1 (unclaimed leaf)

        uint64_t *d_cols = nullptr;
        int *d_pivot_to_col = nullptr;
        int *d_col_pivots = nullptr;
        int *d_new_pivots = nullptr;

        std::size_t col_bytes = sizeof(h_cols);
        std::size_t ptc_bytes = sizeof(h_pivot_to_col);
        std::size_t idx_bytes = static_cast<std::size_t>(num_cols) * sizeof(int);

        bool ok = check_cuda(cudaMalloc(&d_cols, col_bytes), "mw3 malloc cols") &&
                  check_cuda(cudaMalloc(&d_pivot_to_col, ptc_bytes), "mw3 malloc ptc") &&
                  check_cuda(cudaMalloc(&d_col_pivots, idx_bytes), "mw3 malloc cp") &&
                  check_cuda(cudaMalloc(&d_new_pivots, idx_bytes), "mw3 malloc np");

        if (!ok)
        {
            cudaFree(d_cols);
            cudaFree(d_pivot_to_col);
            cudaFree(d_col_pivots);
            cudaFree(d_new_pivots);
            std::cerr << "FAIL: mw3_cross_col setup\n";
            return 1;
        }

        // Upload helper
        auto upload_mw3 = [&]() -> bool {
            return check_cuda(cudaMemcpy(d_cols, h_cols, col_bytes, cudaMemcpyHostToDevice),
                              "mw3 memcpy cols") &&
                   check_cuda(cudaMemcpy(d_pivot_to_col, h_pivot_to_col, ptc_bytes,
                                         cudaMemcpyHostToDevice),
                              "mw3 memcpy ptc") &&
                   check_cuda(
                       cudaMemcpy(d_col_pivots, h_col_pivots, idx_bytes, cudaMemcpyHostToDevice),
                       "mw3 memcpy cp");
        };

        // Expected pivots and column data (verified manually for all models)
        const int expected_pivots[num_cols] = {130, 66, 0};
        const uint64_t expected_data[num_cols * num_words] = {
            0x0000ULL, 0x0000ULL, 0b100ULL,  // col 0
            0x0000ULL, 0b100ULL,  0x0000ULL, // col 1
            0b001ULL,  0x0000ULL, 0x0000ULL  // col 2
        };

        // single-warp reference (sequential, race-free)
        int ref_pivots[num_cols] = {};
        uint64_t ref_data[num_cols * num_words] = {};
        {
            if (!upload_mw3())
            {
                goto mw3_cleanup;
            }
            check_cuda(cudaMemset(d_new_pivots, 0xFF, idx_bytes), "mw3 memset np (ref)");

            singleWarpReductionRef<<<1, 32>>>(d_cols, d_pivot_to_col, num_words, num_cols,
                                              d_new_pivots);
            if (cudaDeviceSynchronize() != cudaSuccess)
            {
                std::cerr << "FAIL: mw3_cross_col ref kernel launch failed\n";
                goto mw3_cleanup;
            }

            check_cuda(
                cudaMemcpy(ref_pivots, d_new_pivots, sizeof(ref_pivots), cudaMemcpyDeviceToHost),
                "mw3 d2h ref pivots");
            check_cuda(cudaMemcpy(ref_data, d_cols, sizeof(ref_data), cudaMemcpyDeviceToHost),
                       "mw3 d2h ref cols");

            // Verify reference matches expected
            for (int i = 0; i < num_cols; ++i)
            {
                if (ref_pivots[i] != expected_pivots[i])
                {
                    std::cerr << "FAIL: mw3_cross_col ref pivot[" << i << "]=" << ref_pivots[i]
                              << " != expected " << expected_pivots[i] << "\n";
                    goto mw3_cleanup;
                }
            }
            for (int i = 0; i < num_cols * num_words; ++i)
            {
                if (ref_data[i] != expected_data[i])
                {
                    std::cerr << "FAIL: mw3_cross_col ref elem " << i << "=0x" << std::hex
                              << ref_data[i] << std::dec << " != expected 0x" << std::hex
                              << expected_data[i] << std::dec << "\n";
                    goto mw3_cleanup;
                }
            }
        }

        // syncthreads-barrier reference
        {
            if (!upload_mw3())
            {
                goto mw3_cleanup;
            }
            check_cuda(cudaMemset(d_new_pivots, 0xFF, idx_bytes), "mw3 memset np (st)");

            const int block_threads = max(32, num_cols * 32);
            syncthreadsReductionRef<<<1, block_threads>>>(d_cols, d_col_pivots, d_pivot_to_col,
                                                          num_words, num_cols, d_new_pivots);
            if (cudaDeviceSynchronize() != cudaSuccess)
            {
                std::cerr << "FAIL: mw3_cross_col st ref launch failed\n";
                goto mw3_cleanup;
            }

            int st_pivots[num_cols] = {};
            check_cuda(
                cudaMemcpy(st_pivots, d_new_pivots, sizeof(st_pivots), cudaMemcpyDeviceToHost),
                "mw3 d2h st pivots");

            for (int i = 0; i < num_cols; ++i)
            {
                if (st_pivots[i] != ref_pivots[i])
                {
                    std::cerr << "FAIL: mw3_cross_col st pivot[" << i << "]=" << st_pivots[i]
                              << " != ref " << ref_pivots[i] << "\n";
                    goto mw3_cleanup;
                }
            }
        }

        // regular kernel (use_async_copy=false), 5 runs
        {
            for (int run = 0; run < 5; ++run)
            {
                if (!upload_mw3())
                {
                    goto mw3_cleanup;
                }
                check_cuda(cudaMemset(d_new_pivots, 0xFF, idx_bytes), "mw3 memset np (reg)");

                nerve::persistence::gpu::launchPipelinedReduction(d_cols, d_col_pivots,
                                                                  d_pivot_to_col, num_words,
                                                                  num_cols, d_new_pivots, 0, false);
                if (cudaDeviceSynchronize() != cudaSuccess)
                {
                    std::cerr << "FAIL: mw3_cross_col reg run " << run << " launch failed\n";
                    goto mw3_cleanup;
                }

                int run_pivots[num_cols] = {};
                check_cuda(cudaMemcpy(run_pivots, d_new_pivots, sizeof(run_pivots),
                                      cudaMemcpyDeviceToHost),
                           "mw3 d2h reg pivots");

                for (int i = 0; i < num_cols; ++i)
                {
                    if (run_pivots[i] != ref_pivots[i])
                    {
                        std::cerr << "FAIL: mw3_cross_col reg run " << run << " col " << i
                                  << " pivot=" << run_pivots[i] << " != ref " << ref_pivots[i]
                                  << "\n";
                        goto mw3_cleanup;
                    }
                }
            }
        }

        // async kernel (use_async_copy=true), 5 runs
        {
            for (int run = 0; run < 5; ++run)
            {
                if (!upload_mw3())
                {
                    goto mw3_cleanup;
                }
                check_cuda(cudaMemset(d_new_pivots, 0xFF, idx_bytes), "mw3 memset np (async)");

                nerve::persistence::gpu::launchPipelinedReduction(d_cols, d_col_pivots,
                                                                  d_pivot_to_col, num_words,
                                                                  num_cols, d_new_pivots, 0, true);
                if (cudaDeviceSynchronize() != cudaSuccess)
                {
                    std::cerr << "FAIL: mw3_cross_col async run " << run << " launch failed\n";
                    goto mw3_cleanup;
                }

                int run_pivots[num_cols] = {};
                check_cuda(cudaMemcpy(run_pivots, d_new_pivots, sizeof(run_pivots),
                                      cudaMemcpyDeviceToHost),
                           "mw3 d2h async pivots");

                for (int i = 0; i < num_cols; ++i)
                {
                    if (run_pivots[i] != ref_pivots[i])
                    {
                        std::cerr << "FAIL: mw3_cross_col async run " << run << " col " << i
                                  << " pivot=" << run_pivots[i] << " != ref " << ref_pivots[i]
                                  << "\n";
                        goto mw3_cleanup;
                    }
                }
            }
        }

        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        goto mw3_done;

mw3_cleanup:
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        return 1;

mw3_done:;
    }

    // Multi-word (2-word) self-clearing: single-warp ref vs syncthreads
    // vs regular (5x) vs async (5x)
    //
    // Each column claims its own pivot (self-clear). No cross-column reads,
    // so trivially race-free -- all variants produce identical results.
    //
    // Col 0: [0b100, 0x0000] (pivot=2).  pivot_to_col[2]=0  -> self-clear.
    // Col 1: [0x0000, 0b001] (pivot=64). pivot_to_col[64]=1 -> self-clear.
    // Col 2: [0b010, 0x0000] (pivot=1).  pivot_to_col[1]=2  -> self-clear.
    //
    // Expected: all columns zeroed, all pivots = -1.
    {
        constexpr int num_words = 2;
        constexpr int num_cols = 3;

        // Column data: [col0_w0, col0_w1, col1_w0, col1_w1, col2_w0, col2_w1]
        const uint64_t h_cols[num_cols * num_words] = {
            0b100ULL,  0x0000ULL, // col 0: pivot=2 (word 0, bit 2)
            0x0000ULL, 0b001ULL,  // col 1: pivot=64 (word 1, bit 0)
            0b010ULL,  0x0000ULL  // col 2: pivot=1 (word 0, bit 1)
        };

        const int h_col_pivots[num_cols] = {2, 64, 1};

        // Each column claims its own pivot
        int h_pivot_to_col[128]; // num_words * 64 = 2 * 64 = 128
        std::memset(h_pivot_to_col, 0xFF, sizeof(h_pivot_to_col));
        h_pivot_to_col[2] = 0;  // col 0 claims pivot 2
        h_pivot_to_col[64] = 1; // col 1 claims pivot 64
        h_pivot_to_col[1] = 2;  // col 2 claims pivot 1

        uint64_t *d_cols = nullptr;
        int *d_pivot_to_col = nullptr;
        int *d_col_pivots = nullptr;
        int *d_new_pivots = nullptr;

        std::size_t col_bytes = sizeof(h_cols);
        std::size_t ptc_bytes = sizeof(h_pivot_to_col);
        std::size_t idx_bytes = static_cast<std::size_t>(num_cols) * sizeof(int);

        bool ok = check_cuda(cudaMalloc(&d_cols, col_bytes), "sc2 malloc cols") &&
                  check_cuda(cudaMalloc(&d_pivot_to_col, ptc_bytes), "sc2 malloc ptc") &&
                  check_cuda(cudaMalloc(&d_col_pivots, idx_bytes), "sc2 malloc cp") &&
                  check_cuda(cudaMalloc(&d_new_pivots, idx_bytes), "sc2 malloc np");

        if (!ok)
        {
            cudaFree(d_cols);
            cudaFree(d_pivot_to_col);
            cudaFree(d_col_pivots);
            cudaFree(d_new_pivots);
            std::cerr << "FAIL: self_clear_2word setup\n";
            return 1;
        }

        // Upload helper
        auto upload_sc2 = [&]() -> bool {
            return check_cuda(cudaMemcpy(d_cols, h_cols, col_bytes, cudaMemcpyHostToDevice),
                              "sc2 memcpy cols") &&
                   check_cuda(cudaMemcpy(d_pivot_to_col, h_pivot_to_col, ptc_bytes,
                                         cudaMemcpyHostToDevice),
                              "sc2 memcpy ptc") &&
                   check_cuda(
                       cudaMemcpy(d_col_pivots, h_col_pivots, idx_bytes, cudaMemcpyHostToDevice),
                       "sc2 memcpy cp");
        };

        // Expected: all zeroed, all pivots = -1
        const int expected_pivots[num_cols] = {-1, -1, -1};

        // single-warp reference (sequential, race-free)
        int ref_pivots[num_cols] = {};
        uint64_t ref_data[num_cols * num_words] = {};
        {
            if (!upload_sc2())
            {
                goto sc2_cleanup;
            }
            check_cuda(cudaMemset(d_new_pivots, 0xFF, idx_bytes), "sc2 memset np (ref)");

            singleWarpReductionRef<<<1, 32>>>(d_cols, d_pivot_to_col, num_words, num_cols,
                                              d_new_pivots);
            if (cudaDeviceSynchronize() != cudaSuccess)
            {
                std::cerr << "FAIL: self_clear_2word ref kernel launch failed\n";
                goto sc2_cleanup;
            }

            check_cuda(
                cudaMemcpy(ref_pivots, d_new_pivots, sizeof(ref_pivots), cudaMemcpyDeviceToHost),
                "sc2 d2h ref pivots");
            check_cuda(cudaMemcpy(ref_data, d_cols, sizeof(ref_data), cudaMemcpyDeviceToHost),
                       "sc2 d2h ref cols");

            // Verify reference matches expected
            for (int i = 0; i < num_cols; ++i)
            {
                if (ref_pivots[i] != expected_pivots[i])
                {
                    std::cerr << "FAIL: self_clear_2word ref pivot[" << i << "]=" << ref_pivots[i]
                              << " != expected -1\n";
                    goto sc2_cleanup;
                }
            }
            for (int i = 0; i < num_cols * num_words; ++i)
            {
                if (ref_data[i] != 0)
                {
                    std::cerr << "FAIL: self_clear_2word ref elem " << i << "=0x" << std::hex
                              << ref_data[i] << std::dec << " != 0\n";
                    goto sc2_cleanup;
                }
            }
        }

        // syncthreads-barrier reference
        {
            if (!upload_sc2())
            {
                goto sc2_cleanup;
            }
            check_cuda(cudaMemset(d_new_pivots, 0xFF, idx_bytes), "sc2 memset np (st)");

            const int block_threads = max(32, num_cols * 32);
            syncthreadsReductionRef<<<1, block_threads>>>(d_cols, d_col_pivots, d_pivot_to_col,
                                                          num_words, num_cols, d_new_pivots);
            if (cudaDeviceSynchronize() != cudaSuccess)
            {
                std::cerr << "FAIL: self_clear_2word st ref launch failed\n";
                goto sc2_cleanup;
            }

            int st_pivots[num_cols] = {};
            check_cuda(
                cudaMemcpy(st_pivots, d_new_pivots, sizeof(st_pivots), cudaMemcpyDeviceToHost),
                "sc2 d2h st pivots");

            for (int i = 0; i < num_cols; ++i)
            {
                if (st_pivots[i] != ref_pivots[i])
                {
                    std::cerr << "FAIL: self_clear_2word st pivot[" << i << "]=" << st_pivots[i]
                              << " != ref " << ref_pivots[i] << "\n";
                    goto sc2_cleanup;
                }
            }
        }

        // regular kernel (use_async_copy=false), 5 runs
        {
            for (int run = 0; run < 5; ++run)
            {
                if (!upload_sc2())
                {
                    goto sc2_cleanup;
                }
                check_cuda(cudaMemset(d_new_pivots, 0xFF, idx_bytes), "sc2 memset np (reg)");

                nerve::persistence::gpu::launchPipelinedReduction(d_cols, d_col_pivots,
                                                                  d_pivot_to_col, num_words,
                                                                  num_cols, d_new_pivots, 0, false);
                if (cudaDeviceSynchronize() != cudaSuccess)
                {
                    std::cerr << "FAIL: self_clear_2word reg run " << run << " launch failed\n";
                    goto sc2_cleanup;
                }

                int run_pivots[num_cols] = {};
                check_cuda(cudaMemcpy(run_pivots, d_new_pivots, sizeof(run_pivots),
                                      cudaMemcpyDeviceToHost),
                           "sc2 d2h reg pivots");

                for (int i = 0; i < num_cols; ++i)
                {
                    if (run_pivots[i] != ref_pivots[i])
                    {
                        std::cerr << "FAIL: self_clear_2word reg run " << run << " col " << i
                                  << " pivot=" << run_pivots[i] << " != ref " << ref_pivots[i]
                                  << "\n";
                        goto sc2_cleanup;
                    }
                }
            }
        }

        // async kernel (use_async_copy=true), 5 runs
        {
            for (int run = 0; run < 5; ++run)
            {
                if (!upload_sc2())
                {
                    goto sc2_cleanup;
                }
                check_cuda(cudaMemset(d_new_pivots, 0xFF, idx_bytes), "sc2 memset np (async)");

                nerve::persistence::gpu::launchPipelinedReduction(d_cols, d_col_pivots,
                                                                  d_pivot_to_col, num_words,
                                                                  num_cols, d_new_pivots, 0, true);
                if (cudaDeviceSynchronize() != cudaSuccess)
                {
                    std::cerr << "FAIL: self_clear_2word async run " << run << " launch failed\n";
                    goto sc2_cleanup;
                }

                int run_pivots[num_cols] = {};
                check_cuda(cudaMemcpy(run_pivots, d_new_pivots, sizeof(run_pivots),
                                      cudaMemcpyDeviceToHost),
                           "sc2 d2h async pivots");

                for (int i = 0; i < num_cols; ++i)
                {
                    if (run_pivots[i] != ref_pivots[i])
                    {
                        std::cerr << "FAIL: self_clear_2word async run " << run << " col " << i
                                  << " pivot=" << run_pivots[i] << " != ref " << ref_pivots[i]
                                  << "\n";
                        goto sc2_cleanup;
                    }
                }
            }
        }

        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        goto sc2_done;

sc2_cleanup:
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        return 1;

sc2_done:;
    }

    // Regular kernel cross-column determinism: 10 runs, identical results
    //
    // Runs the regular kernel (use_async_copy=false) 10 times on the same
    // cross-column chain, verifying that every run produces identical pivots
    // AND column data.
    //
    // Chain (same as tests 22/24/25, race-free):
    // Col 0: 0b110 (pivot=2). pivot_to_col[2]=1. XOR with Col 1.
    // Col 1: 0b010 (pivot=1). pivot_to_col[1]=2. XOR with Col 2.
    // Col 2: 0b001 (pivot=0). pivot_to_col[0]=-1. Break (unclaimed).
    //
    // Expected: all 10 runs produce identical new_pivots AND column data.
    {
        constexpr int num_words = 1;
        constexpr int num_cols = 3;
        constexpr int num_runs = 10;

        const uint64_t h_cols[num_cols * num_words] = {0b110ULL, 0b010ULL, 0b001ULL};
        const int h_col_pivots[num_cols] = {2, 1, 0};

        int h_pivot_to_col[64];
        std::memset(h_pivot_to_col, 0xFF, sizeof(h_pivot_to_col));
        h_pivot_to_col[2] = 1; // col 1 claims pivot 2
        h_pivot_to_col[1] = 2; // col 2 claims pivot 1
        // pivot_to_col[0] stays -1 (unclaimed)

        uint64_t *d_cols = nullptr;
        int *d_pivot_to_col = nullptr;
        int *d_col_pivots = nullptr;
        int *d_new_pivots = nullptr;
        std::size_t col_bytes = sizeof(h_cols);
        std::size_t ptc_bytes = sizeof(h_pivot_to_col);
        std::size_t idx_bytes = static_cast<std::size_t>(num_cols) * sizeof(int);

        bool ok = check_cuda(cudaMalloc(&d_cols, col_bytes), "reg_det malloc cols") &&
                  check_cuda(cudaMalloc(&d_pivot_to_col, ptc_bytes), "reg_det malloc ptc") &&
                  check_cuda(cudaMalloc(&d_col_pivots, idx_bytes), "reg_det malloc cp") &&
                  check_cuda(cudaMalloc(&d_new_pivots, idx_bytes), "reg_det malloc np");

        if (!ok)
        {
            cudaFree(d_cols);
            cudaFree(d_pivot_to_col);
            cudaFree(d_col_pivots);
            cudaFree(d_new_pivots);
            std::cerr << "FAIL: reg_cross_col_determinism setup\n";
            return 1;
        }

        // Upload initial data (shared across all runs)
        if (!check_cuda(
                cudaMemcpy(d_pivot_to_col, h_pivot_to_col, ptc_bytes, cudaMemcpyHostToDevice),
                "reg_det memcpy ptc") ||
            !check_cuda(cudaMemcpy(d_col_pivots, h_col_pivots, idx_bytes, cudaMemcpyHostToDevice),
                        "reg_det memcpy cp"))
        {
            cudaFree(d_cols);
            cudaFree(d_pivot_to_col);
            cudaFree(d_col_pivots);
            cudaFree(d_new_pivots);
            std::cerr << "FAIL: reg_cross_col_determinism upload ptc/cp\n";
            return 1;
        }

        // Run 0: store reference results
        int ref_pivots[num_cols] = {};
        uint64_t ref_data[num_cols * num_words] = {};
        {
            if (!check_cuda(cudaMemcpy(d_cols, h_cols, col_bytes, cudaMemcpyHostToDevice),
                            "reg_det memcpy cols (run 0)"))
            {
                goto reg_det_cleanup;
            }
            check_cuda(cudaMemset(d_new_pivots, 0xFF, idx_bytes), "reg_det memset np (run 0)");

            nerve::persistence::gpu::launchPipelinedReduction(
                d_cols, d_col_pivots, d_pivot_to_col, num_words, num_cols, d_new_pivots, 0, false);
            if (cudaDeviceSynchronize() != cudaSuccess)
            {
                std::cerr << "FAIL: reg_cross_col_determinism run 0 launch failed\n";
                goto reg_det_cleanup;
            }

            check_cuda(
                cudaMemcpy(ref_pivots, d_new_pivots, sizeof(ref_pivots), cudaMemcpyDeviceToHost),
                "reg_det d2h pivots (run 0)");
            check_cuda(cudaMemcpy(ref_data, d_cols, sizeof(ref_data), cudaMemcpyDeviceToHost),
                       "reg_det d2h cols (run 0)");
        }

        // Runs 1..num_runs-1: compare against reference
        for (int run = 1; run < num_runs; ++run)
        {
            if (!check_cuda(cudaMemcpy(d_cols, h_cols, col_bytes, cudaMemcpyHostToDevice),
                            "reg_det memcpy cols"))
            {
                goto reg_det_cleanup;
            }
            check_cuda(cudaMemset(d_new_pivots, 0xFF, idx_bytes), "reg_det memset np");

            nerve::persistence::gpu::launchPipelinedReduction(
                d_cols, d_col_pivots, d_pivot_to_col, num_words, num_cols, d_new_pivots, 0, false);
            if (cudaDeviceSynchronize() != cudaSuccess)
            {
                std::cerr << "FAIL: reg_cross_col_determinism run " << run << " launch failed\n";
                goto reg_det_cleanup;
            }

            int run_pivots[num_cols] = {};
            uint64_t run_data[num_cols * num_words] = {};
            check_cuda(
                cudaMemcpy(run_pivots, d_new_pivots, sizeof(run_pivots), cudaMemcpyDeviceToHost),
                "reg_det d2h pivots");
            check_cuda(cudaMemcpy(run_data, d_cols, sizeof(run_data), cudaMemcpyDeviceToHost),
                       "reg_det d2h cols");

            // Compare pivots
            for (int i = 0; i < num_cols; ++i)
            {
                if (run_pivots[i] != ref_pivots[i])
                {
                    std::cerr << "FAIL: reg_cross_col_determinism run " << run << " col " << i
                              << " pivot=" << run_pivots[i] << " != ref " << ref_pivots[i] << "\n";
                    goto reg_det_cleanup;
                }
            }

            // Compare column data
            for (int i = 0; i < num_cols * num_words; ++i)
            {
                if (run_data[i] != ref_data[i])
                {
                    std::cerr << "FAIL: reg_cross_col_determinism run " << run << " element " << i
                              << " data=0x" << std::hex << run_data[i] << std::dec << " != ref 0x"
                              << std::hex << ref_data[i] << std::dec << "\n";
                    goto reg_det_cleanup;
                }
            }
        }

        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        goto reg_det_done;

reg_det_cleanup:
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        return 1;

reg_det_done:;
    }

    // Multi-word (2-word) async-copy determinism: 10 runs, identical results
    //
    // Runs the async kernel (use_async_copy=true) 10 times on a 2-word cross-column
    // chain, verifying that every run produces identical pivots AND column data.
    //
    // Chain (same as test 23, race-free):
    // Col 0: [0x0000, 0b100] (pivot=66). pivot_to_col[66]=1. XOR with Col 1.
    // Col 1: [0b010, 0x0000] (pivot=1).  pivot_to_col[1]=2.  XOR with Col 2.
    // Col 2: [0x0000, 0b001] (pivot=64). pivot_to_col[64]=-1. Break (unclaimed).
    //
    // After XOR with Col 2, Col 1's MSB moves from word 0 (pivot=1) to word 1
    // (pivot=64), which is unclaimed -> kernel breaks immediately.
    //
    // Expected: all 10 runs produce identical new_pivots AND column data.
    {
        constexpr int num_words = 2;
        constexpr int num_cols = 3;
        constexpr int num_runs = 10;

        const uint64_t h_cols[num_cols * num_words] = {
            0x0000ULL, 0b100ULL,  // col 0: pivot=66 (word 1, bit 2)
            0b010ULL,  0x0000ULL, // col 1: pivot=1 (word 0, bit 1)
            0x0000ULL, 0b001ULL   // col 2: pivot=64 (word 1, bit 0), unclaimed
        };

        const int h_col_pivots[num_cols] = {66, 1, 64};

        int h_pivot_to_col[128]; // num_words * 64 = 2 * 64 = 128
        std::memset(h_pivot_to_col, 0xFF, sizeof(h_pivot_to_col));
        h_pivot_to_col[66] = 1; // col 1 claims pivot 66
        h_pivot_to_col[1] = 2;  // col 2 claims pivot 1
        // pivot_to_col[64] stays -1 (unclaimed leaf)

        uint64_t *d_cols = nullptr;
        int *d_pivot_to_col = nullptr;
        int *d_col_pivots = nullptr;
        int *d_new_pivots = nullptr;
        std::size_t col_bytes = sizeof(h_cols);
        std::size_t ptc_bytes = sizeof(h_pivot_to_col);
        std::size_t idx_bytes = static_cast<std::size_t>(num_cols) * sizeof(int);

        bool ok = check_cuda(cudaMalloc(&d_cols, col_bytes), "mw2_det malloc cols") &&
                  check_cuda(cudaMalloc(&d_pivot_to_col, ptc_bytes), "mw2_det malloc ptc") &&
                  check_cuda(cudaMalloc(&d_col_pivots, idx_bytes), "mw2_det malloc cp") &&
                  check_cuda(cudaMalloc(&d_new_pivots, idx_bytes), "mw2_det malloc np");

        if (!ok)
        {
            cudaFree(d_cols);
            cudaFree(d_pivot_to_col);
            cudaFree(d_col_pivots);
            cudaFree(d_new_pivots);
            std::cerr << "FAIL: mw2_async_determinism setup\n";
            return 1;
        }

        // Upload initial data (shared across all runs)
        if (!check_cuda(
                cudaMemcpy(d_pivot_to_col, h_pivot_to_col, ptc_bytes, cudaMemcpyHostToDevice),
                "mw2_det memcpy ptc") ||
            !check_cuda(cudaMemcpy(d_col_pivots, h_col_pivots, idx_bytes, cudaMemcpyHostToDevice),
                        "mw2_det memcpy cp"))
        {
            cudaFree(d_cols);
            cudaFree(d_pivot_to_col);
            cudaFree(d_col_pivots);
            cudaFree(d_new_pivots);
            std::cerr << "FAIL: mw2_async_determinism upload ptc/cp\n";
            return 1;
        }

        // Run 0: store reference results
        int ref_pivots[num_cols] = {};
        uint64_t ref_data[num_cols * num_words] = {};
        {
            if (!check_cuda(cudaMemcpy(d_cols, h_cols, col_bytes, cudaMemcpyHostToDevice),
                            "mw2_det memcpy cols (run 0)"))
            {
                goto mw2_det_cleanup;
            }
            check_cuda(cudaMemset(d_new_pivots, 0xFF, idx_bytes), "mw2_det memset np (run 0)");

            nerve::persistence::gpu::launchPipelinedReduction(
                d_cols, d_col_pivots, d_pivot_to_col, num_words, num_cols, d_new_pivots, 0, true);
            if (cudaDeviceSynchronize() != cudaSuccess)
            {
                std::cerr << "FAIL: mw2_async_determinism run 0 launch failed\n";
                goto mw2_det_cleanup;
            }

            check_cuda(
                cudaMemcpy(ref_pivots, d_new_pivots, sizeof(ref_pivots), cudaMemcpyDeviceToHost),
                "mw2_det d2h pivots (run 0)");
            check_cuda(cudaMemcpy(ref_data, d_cols, sizeof(ref_data), cudaMemcpyDeviceToHost),
                       "mw2_det d2h cols (run 0)");
        }

        // Runs 1..num_runs-1: compare against reference
        for (int run = 1; run < num_runs; ++run)
        {
            if (!check_cuda(cudaMemcpy(d_cols, h_cols, col_bytes, cudaMemcpyHostToDevice),
                            "mw2_det memcpy cols"))
            {
                goto mw2_det_cleanup;
            }
            check_cuda(cudaMemset(d_new_pivots, 0xFF, idx_bytes), "mw2_det memset np");

            nerve::persistence::gpu::launchPipelinedReduction(
                d_cols, d_col_pivots, d_pivot_to_col, num_words, num_cols, d_new_pivots, 0, true);
            if (cudaDeviceSynchronize() != cudaSuccess)
            {
                std::cerr << "FAIL: mw2_async_determinism run " << run << " launch failed\n";
                goto mw2_det_cleanup;
            }

            int run_pivots[num_cols] = {};
            uint64_t run_data[num_cols * num_words] = {};
            check_cuda(
                cudaMemcpy(run_pivots, d_new_pivots, sizeof(run_pivots), cudaMemcpyDeviceToHost),
                "mw2_det d2h pivots");
            check_cuda(cudaMemcpy(run_data, d_cols, sizeof(run_data), cudaMemcpyDeviceToHost),
                       "mw2_det d2h cols");

            // Compare pivots
            for (int i = 0; i < num_cols; ++i)
            {
                if (run_pivots[i] != ref_pivots[i])
                {
                    std::cerr << "FAIL: mw2_async_determinism run " << run << " col " << i
                              << " pivot=" << run_pivots[i] << " != ref " << ref_pivots[i] << "\n";
                    goto mw2_det_cleanup;
                }
            }

            // Compare column data
            for (int i = 0; i < num_cols * num_words; ++i)
            {
                if (run_data[i] != ref_data[i])
                {
                    std::cerr << "FAIL: mw2_async_determinism run " << run << " element " << i
                              << " data=0x" << std::hex << run_data[i] << std::dec << " != ref 0x"
                              << std::hex << ref_data[i] << std::dec << "\n";
                    goto mw2_det_cleanup;
                }
            }
        }

        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        goto mw2_det_done;

mw2_det_cleanup:
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        return 1;

mw2_det_done:;
    }
    return 0;

    // Multi-word (2-word) unclaimed pivot test: no self-claims, race-free
    //
    // All pivots map outside the column's own index (no self-clearing). Two of
    // the three columns are unclaimed leaves (never written), making this
    // trivially race-free.
    //
    // Col 0: [0b100,   0x0000] (pivot=2).  pivot_to_col[2]=1  -> XOR with Col 1
    // Col 1: [0x0000,  0b001 ] (pivot=64). pivot_to_col[64]=-1 -> Break (unclaimed)
    // Col 2: [0b001,   0x0000] (pivot=0).  pivot_to_col[0]=-1  -> Break (unclaimed)
    //
    // Col 0 XOR with Col 1: [0b100,0] ^ [0,0b001] = [0b100,0b001]. MSB moves from
    // word 0 (pivot=2) to word 1 (pivot=64), which is unclaimed -> break.
    //
    // Expected pivots: {64, 64, 0}
    // Expected data:  [0b100, 0b001 | 0, 0b001 | 0b001, 0]
    {
        constexpr int num_words = 2;
        constexpr int num_cols = 3;

        // Column data: [col0_w0, col0_w1, col1_w0, col1_w1, col2_w0, col2_w1]
        const uint64_t h_cols[num_cols * num_words] = {
            0b100ULL, 0x0000,   // Col 0: pivot=2
            0x0000,   0b001ULL, // Col 1: pivot=64 (unclaimed)
            0b001ULL, 0x0000    // Col 2: pivot=0  (unclaimed)
        };
        const int h_pivot_to_col[128] = {};
        // Set only the entries used by this test
        const_cast<int *>(h_pivot_to_col)[2] = 1;   // Col 0's pivot -> Col 1
        const_cast<int *>(h_pivot_to_col)[64] = -1; // Col 1's pivot -> unclaimed
        const_cast<int *>(h_pivot_to_col)[0] = -1;  // Col 2's pivot -> unclaimed
        const int h_col_pivots[num_cols] = {2, 64, 0};

        const int expected_pivots[num_cols] = {64, 64, 0};
        const uint64_t expected_data[num_cols * num_words] = {
            0b100ULL, 0b001ULL, // Col 0 after XOR
            0x0000,   0b001ULL, // Col 1 unchanged
            0b001ULL, 0x0000    // Col 2 unchanged
        };

        const size_t col_bytes = sizeof(h_cols);
        const size_t ptc_bytes = sizeof(h_pivot_to_col);
        const size_t idx_bytes = num_cols * sizeof(int);
        const int block_threads = max(32, num_cols * 32);

        // Device allocations
        uint64_t *d_cols;
        int *d_pivot_to_col, *d_col_pivots, *d_new_pivots;
        check_cuda(cudaMalloc(&d_cols, col_bytes), "uc2 d_cols");
        check_cuda(cudaMalloc(&d_pivot_to_col, ptc_bytes), "uc2 d_ptc");
        check_cuda(cudaMalloc(&d_col_pivots, idx_bytes), "uc2 d_cp");
        check_cuda(cudaMalloc(&d_new_pivots, idx_bytes), "uc2 d_np");
        check_cuda(cudaMemcpy(d_pivot_to_col, h_pivot_to_col, ptc_bytes, cudaMemcpyHostToDevice),
                   "uc2 h2d ptc");
        check_cuda(cudaMemcpy(d_col_pivots, h_col_pivots, idx_bytes, cudaMemcpyHostToDevice),
                   "uc2 h2d cp");

        auto upload_data = [&]() -> bool {
            bool ok = true;
            ok &= check_cuda(cudaMemcpy(d_cols, h_cols, col_bytes, cudaMemcpyHostToDevice),
                             "uc2 h2d cols");
            ok &= check_cuda(cudaMemset(d_new_pivots, 0xFF, idx_bytes), "uc2 memset new_pivots");
            return ok;
        };

        // single-warp reference
        int ref_pivots[num_cols];
        uint64_t ref_data[num_cols * num_words];
        {
            if (!upload_data())
                goto uc2_cleanup;
            singleWarpReductionRef<<<1, 32>>>(d_cols, d_pivot_to_col, num_words, num_cols,
                                              d_new_pivots);
            check_cuda(cudaDeviceSynchronize(), "uc2 ref sync");
            check_cuda(
                cudaMemcpy(ref_pivots, d_new_pivots, sizeof(ref_pivots), cudaMemcpyDeviceToHost),
                "uc2 ref d2h pivots");
            check_cuda(cudaMemcpy(ref_data, d_cols, sizeof(ref_data), cudaMemcpyDeviceToHost),
                       "uc2 ref d2h data");

            // Validate reference against expected
            bool ref_ok = true;
            for (int i = 0; i < num_cols; ++i)
                if (ref_pivots[i] != expected_pivots[i])
                {
                    fprintf(stderr, "uc2 ref pivot[%d] = %d, expected %d\n", i, ref_pivots[i],
                            expected_pivots[i]);
                    ref_ok = false;
                }
            for (int i = 0; i < num_cols * num_words; ++i)
                if (ref_data[i] != expected_data[i])
                {
                    fprintf(stderr, "uc2 ref data[%d] = 0x%lx, expected 0x%lx\n", i, ref_data[i],
                            expected_data[i]);
                    ref_ok = false;
                }
            if (!ref_ok)
                goto uc2_cleanup;
        }

        // syncthreads reference
        {
            if (!upload_data())
                goto uc2_cleanup;
            syncthreadsReductionRef<<<1, block_threads>>>(d_cols, d_col_pivots, d_pivot_to_col,
                                                          num_words, num_cols, d_new_pivots);
            check_cuda(cudaDeviceSynchronize(), "uc2 sync sync");
            int sync_pivots[num_cols];
            check_cuda(
                cudaMemcpy(sync_pivots, d_new_pivots, sizeof(sync_pivots), cudaMemcpyDeviceToHost),
                "uc2 sync d2h");
            bool sync_ok = true;
            for (int i = 0; i < num_cols; ++i)
                if (sync_pivots[i] != ref_pivots[i])
                {
                    fprintf(stderr, "uc2 sync pivot[%d] = %d, ref %d\n", i, sync_pivots[i],
                            ref_pivots[i]);
                    sync_ok = false;
                }
            if (!sync_ok)
                goto uc2_cleanup;
        }

        // regular kernel (use_async_copy=false) 5x
        for (int run = 0; run < 5; ++run)
        {
            if (!upload_data())
                goto uc2_cleanup;
            nerve::persistence::gpu::launchPipelinedReduction(
                d_cols, d_col_pivots, d_pivot_to_col, num_words, num_cols, d_new_pivots, 0, false);
            int run_pivots[num_cols];
            check_cuda(
                cudaMemcpy(run_pivots, d_new_pivots, sizeof(run_pivots), cudaMemcpyDeviceToHost),
                "uc2 reg d2h");
            bool run_ok = true;
            for (int i = 0; i < num_cols; ++i)
                if (run_pivots[i] != ref_pivots[i])
                {
                    fprintf(stderr, "uc2 reg[run=%d] pivot[%d] = %d, ref %d\n", run, i,
                            run_pivots[i], ref_pivots[i]);
                    run_ok = false;
                }
            if (!run_ok)
                goto uc2_cleanup;
        }

        // async kernel (use_async_copy=true) 5x
        for (int run = 0; run < 5; ++run)
        {
            if (!upload_data())
                goto uc2_cleanup;
            nerve::persistence::gpu::launchPipelinedReduction(
                d_cols, d_col_pivots, d_pivot_to_col, num_words, num_cols, d_new_pivots, 0, true);
            int run_pivots[num_cols];
            check_cuda(
                cudaMemcpy(run_pivots, d_new_pivots, sizeof(run_pivots), cudaMemcpyDeviceToHost),
                "uc2 async d2h");
            bool run_ok = true;
            for (int i = 0; i < num_cols; ++i)
                if (run_pivots[i] != ref_pivots[i])
                {
                    fprintf(stderr, "uc2 async[run=%d] pivot[%d] = %d, ref %d\n", run, i,
                            run_pivots[i], ref_pivots[i]);
                    run_ok = false;
                }
            if (!run_ok)
                goto uc2_cleanup;
        }

        // All passed
        printf("  PASSED: (multi-word unclaimed pivot)\n");
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        goto uc2_done;

uc2_cleanup:
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        return 1;

uc2_done:;
    }

    // Multi-word (3-word) self-clearing: single-warp ref vs syncthreads
    // vs regular (5x) vs async (5x)
    //
    // Each column claims its own pivot (self-clear). No cross-column reads,
    // so trivially race-free -- all variants produce identical results.
    //
    // Col 0: [0b100, 0x0000, 0x0000] (pivot=2).   pivot_to_col[2]=0   -> self-clear.
    // Col 1: [0x0000, 0b001,  0x0000] (pivot=64).  pivot_to_col[64]=1  -> self-clear.
    // Col 2: [0x0000, 0x0000, 0b010]  (pivot=128). pivot_to_col[128]=2 -> self-clear.
    //
    // Expected: all columns zeroed, all pivots = -1.
    {
        constexpr int num_words = 3;
        constexpr int num_cols = 3;

        // Column data: [col0_w0..w2, col1_w0..w2, col2_w0..w2]
        const uint64_t h_cols[num_cols * num_words] = {
            0b100ULL, 0x0000,   0x0000,  // Col 0: pivot=2
            0x0000,   0b001ULL, 0x0000,  // Col 1: pivot=64
            0x0000,   0x0000,   0b010ULL // Col 2: pivot=128
        };
        // pivot_to_col: 192 entries (3 words * 64)
        const int h_pivot_to_col[192] = {};
        const_cast<int *>(h_pivot_to_col)[2] = 0;   // Col 0 self
        const_cast<int *>(h_pivot_to_col)[64] = 1;  // Col 1 self
        const_cast<int *>(h_pivot_to_col)[128] = 2; // Col 2 self
        const int h_col_pivots[num_cols] = {2, 64, 128};

        const int expected_pivots[num_cols] = {-1, -1, -1};
        const uint64_t expected_data[num_cols * num_words] = {0};

        const size_t col_bytes = sizeof(h_cols);
        const size_t ptc_bytes = sizeof(h_pivot_to_col);
        const size_t idx_bytes = num_cols * sizeof(int);
        const int block_threads = max(32, num_cols * 32);

        // Device allocations
        uint64_t *d_cols;
        int *d_pivot_to_col, *d_col_pivots, *d_new_pivots;
        check_cuda(cudaMalloc(&d_cols, col_bytes), "sc3 d_cols");
        check_cuda(cudaMalloc(&d_pivot_to_col, ptc_bytes), "sc3 d_ptc");
        check_cuda(cudaMalloc(&d_col_pivots, idx_bytes), "sc3 d_cp");
        check_cuda(cudaMalloc(&d_new_pivots, idx_bytes), "sc3 d_np");
        check_cuda(cudaMemcpy(d_pivot_to_col, h_pivot_to_col, ptc_bytes, cudaMemcpyHostToDevice),
                   "sc3 h2d ptc");
        check_cuda(cudaMemcpy(d_col_pivots, h_col_pivots, idx_bytes, cudaMemcpyHostToDevice),
                   "sc3 h2d cp");

        auto upload_data = [&]() -> bool {
            bool ok = true;
            ok &= check_cuda(cudaMemcpy(d_cols, h_cols, col_bytes, cudaMemcpyHostToDevice),
                             "sc3 h2d cols");
            ok &= check_cuda(cudaMemset(d_new_pivots, 0xFF, idx_bytes), "sc3 memset new_pivots");
            return ok;
        };

        // single-warp reference
        int ref_pivots[num_cols];
        uint64_t ref_data[num_cols * num_words];
        {
            if (!upload_data())
                goto sc3_cleanup;
            singleWarpReductionRef<<<1, 32>>>(d_cols, d_pivot_to_col, num_words, num_cols,
                                              d_new_pivots);
            check_cuda(cudaDeviceSynchronize(), "sc3 ref sync");
            check_cuda(
                cudaMemcpy(ref_pivots, d_new_pivots, sizeof(ref_pivots), cudaMemcpyDeviceToHost),
                "sc3 ref d2h pivots");
            check_cuda(cudaMemcpy(ref_data, d_cols, sizeof(ref_data), cudaMemcpyDeviceToHost),
                       "sc3 ref d2h data");

            // Validate reference against expected
            bool ref_ok = true;
            for (int i = 0; i < num_cols; ++i)
                if (ref_pivots[i] != expected_pivots[i])
                {
                    fprintf(stderr, "sc3 ref pivot[%d] = %d, expected %d\n", i, ref_pivots[i],
                            expected_pivots[i]);
                    ref_ok = false;
                }
            for (int i = 0; i < num_cols * num_words; ++i)
                if (ref_data[i] != expected_data[i])
                {
                    fprintf(stderr, "sc3 ref data[%d] = 0x%lx, expected 0x%lx\n", i, ref_data[i],
                            expected_data[i]);
                    ref_ok = false;
                }
            if (!ref_ok)
                goto sc3_cleanup;
        }

        // syncthreads reference
        {
            if (!upload_data())
                goto sc3_cleanup;
            syncthreadsReductionRef<<<1, block_threads>>>(d_cols, d_col_pivots, d_pivot_to_col,
                                                          num_words, num_cols, d_new_pivots);
            check_cuda(cudaDeviceSynchronize(), "sc3 sync sync");
            int sync_pivots[num_cols];
            check_cuda(
                cudaMemcpy(sync_pivots, d_new_pivots, sizeof(sync_pivots), cudaMemcpyDeviceToHost),
                "sc3 sync d2h");
            bool sync_ok = true;
            for (int i = 0; i < num_cols; ++i)
                if (sync_pivots[i] != ref_pivots[i])
                {
                    fprintf(stderr, "sc3 sync pivot[%d] = %d, ref %d\n", i, sync_pivots[i],
                            ref_pivots[i]);
                    sync_ok = false;
                }
            if (!sync_ok)
                goto sc3_cleanup;
        }

        // regular kernel (use_async_copy=false) 5x
        for (int run = 0; run < 5; ++run)
        {
            if (!upload_data())
                goto sc3_cleanup;
            nerve::persistence::gpu::launchPipelinedReduction(
                d_cols, d_col_pivots, d_pivot_to_col, num_words, num_cols, d_new_pivots, 0, false);
            int run_pivots[num_cols];
            check_cuda(
                cudaMemcpy(run_pivots, d_new_pivots, sizeof(run_pivots), cudaMemcpyDeviceToHost),
                "sc3 reg d2h");
            bool run_ok = true;
            for (int i = 0; i < num_cols; ++i)
                if (run_pivots[i] != ref_pivots[i])
                {
                    fprintf(stderr, "sc3 reg[run=%d] pivot[%d] = %d, ref %d\n", run, i,
                            run_pivots[i], ref_pivots[i]);
                    run_ok = false;
                }
            if (!run_ok)
                goto sc3_cleanup;
        }

        // async kernel (use_async_copy=true) 5x
        for (int run = 0; run < 5; ++run)
        {
            if (!upload_data())
                goto sc3_cleanup;
            nerve::persistence::gpu::launchPipelinedReduction(
                d_cols, d_col_pivots, d_pivot_to_col, num_words, num_cols, d_new_pivots, 0, true);
            int run_pivots[num_cols];
            check_cuda(
                cudaMemcpy(run_pivots, d_new_pivots, sizeof(run_pivots), cudaMemcpyDeviceToHost),
                "sc3 async d2h");
            bool run_ok = true;
            for (int i = 0; i < num_cols; ++i)
                if (run_pivots[i] != ref_pivots[i])
                {
                    fprintf(stderr, "sc3 async[run=%d] pivot[%d] = %d, ref %d\n", run, i,
                            run_pivots[i], ref_pivots[i]);
                    run_ok = false;
                }
            if (!run_ok)
                goto sc3_cleanup;
        }

        // All passed
        printf("  PASSED: (3-word self-clearing)\n");
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        goto sc3_done;

sc3_cleanup:
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        return 1;

sc3_done:;
    }

    return 0;

    // Syncthreads kernel determinism: 10 runs, identical results
    //
    // Runs the syncthreadsReductionRef kernel 10 times on a race-free cross-column
    // chain, verifying that every run produces identical pivots AND column data.
    //
    // Chain (same as tests 22/24/25, race-free):
    // Col 0: 0b110 (pivot=2). pivot_to_col[2]=1. XOR with Col 1.
    // Col 1: 0b010 (pivot=1). pivot_to_col[1]=2. XOR with Col 2.
    // Col 2: 0b001 (pivot=0). pivot_to_col[0]=-1. Break (unclaimed).
    //
    // Expected: all 10 runs produce identical new_pivots AND column data.
    {
        constexpr int num_words = 1;
        constexpr int num_cols = 3;
        constexpr int num_runs = 10;

        const uint64_t h_cols[num_cols] = {0b110ULL, 0b010ULL, 0b001ULL};
        const int h_pivot_to_col[64] = {};
        const_cast<int *>(h_pivot_to_col)[0] = -1;
        const_cast<int *>(h_pivot_to_col)[1] = 2;
        const_cast<int *>(h_pivot_to_col)[2] = 1;
        const int h_col_pivots[num_cols] = {2, 1, 0};

        const size_t col_bytes = sizeof(h_cols);
        const size_t ptc_bytes = sizeof(h_pivot_to_col);
        const size_t idx_bytes = num_cols * sizeof(int);
        const int block_threads = max(32, num_cols * 32);

        // Device allocations
        uint64_t *d_cols;
        int *d_pivot_to_col, *d_col_pivots, *d_new_pivots;
        check_cuda(cudaMalloc(&d_cols, col_bytes), "sdet d_cols");
        check_cuda(cudaMalloc(&d_pivot_to_col, ptc_bytes), "sdet d_ptc");
        check_cuda(cudaMalloc(&d_col_pivots, idx_bytes), "sdet d_cp");
        check_cuda(cudaMalloc(&d_new_pivots, idx_bytes), "sdet d_np");
        check_cuda(cudaMemcpy(d_pivot_to_col, h_pivot_to_col, ptc_bytes, cudaMemcpyHostToDevice),
                   "sdet h2d ptc");
        check_cuda(cudaMemcpy(d_col_pivots, h_col_pivots, idx_bytes, cudaMemcpyHostToDevice),
                   "sdet h2d cp");

        // Reference storage (from run 0)
        int ref_pivots[num_cols];
        uint64_t ref_data[num_cols];

        for (int run = 0; run < num_runs; ++run)
        {
            // Upload fresh data before each run
            check_cuda(cudaMemcpy(d_cols, h_cols, col_bytes, cudaMemcpyHostToDevice),
                       "sdet h2d cols");
            check_cuda(cudaMemset(d_new_pivots, 0xFF, idx_bytes), "sdet memset new_pivots");

            syncthreadsReductionRef<<<1, block_threads>>>(d_cols, d_col_pivots, d_pivot_to_col,
                                                          num_words, num_cols, d_new_pivots);
            check_cuda(cudaDeviceSynchronize(), "sdet sync");

            int run_pivots[num_cols];
            uint64_t run_data[num_cols];
            check_cuda(
                cudaMemcpy(run_pivots, d_new_pivots, sizeof(run_pivots), cudaMemcpyDeviceToHost),
                "sdet d2h pivots");
            check_cuda(cudaMemcpy(run_data, d_cols, sizeof(run_data), cudaMemcpyDeviceToHost),
                       "sdet d2h cols");

            if (run == 0)
            {
                // Store reference
                for (int i = 0; i < num_cols; ++i)
                    ref_pivots[i] = run_pivots[i];
                for (int i = 0; i < num_cols; ++i)
                    ref_data[i] = run_data[i];
            }
            else
            {
                // Compare against reference
                bool ok = true;
                for (int i = 0; i < num_cols; ++i)
                    if (run_pivots[i] != ref_pivots[i])
                    {
                        fprintf(stderr, "sdet[run=%d] pivot[%d] = %d, ref %d\n", run, i,
                                run_pivots[i], ref_pivots[i]);
                        ok = false;
                    }
                for (int i = 0; i < num_cols; ++i)
                    if (run_data[i] != ref_data[i])
                    {
                        fprintf(stderr, "sdet[run=%d] data[%d] = 0x%lx, ref 0x%lx\n", run, i,
                                run_data[i], ref_data[i]);
                        ok = false;
                    }
                if (!ok)
                    goto sdet_cleanup;
            }
        }

        // All passed
        printf("  PASSED: (syncthreads determinism)\n");
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        goto sdet_done;

sdet_cleanup:
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        return 1;

sdet_done:;
    }

    // Multi-word (4-word) cross-column reduction: single-warp ref vs
    // syncthreads vs regular (5x) vs async (5x)
    //
    // Chain (race-free): Col 0 -> Col 1 -> Col 2 (unclaimed leaf). Pivots span
    // all four words to exercise multi-word pivot-find across the full range.
    //
    // Col 0: [0x0000, 0x0000, 0x0000, 0b100 ] (pivot=194, word 3).  p2c[194]=1
    // Col 1: [0x0000, 0b010,  0x0000, 0x0000] (pivot=65,  word 1).  p2c[65]=2
    // Col 2: [0b001,  0x0000, 0x0000, 0x0000] (pivot=0,   word 0).  p2c[0]=-1
    //
    // Convergence: Col 1 oscillates period-2 against unclaimed Col 2, returning
    // to original after even iterations. Col 0's word[3] is isolated from Col 1's
    // modifications (words 0-1 only), so Col 0 also converges deterministically.
    //
    // Expected pivots: {194, 65, 0}
    // Expected data:  same as initial (after 16 even iterations, all return to
    // original state)
    {
        constexpr int num_words = 4;
        constexpr int num_cols = 3;

        // Column data: [col0_w0..w3, col1_w0..w3, col2_w0..w3]
        const uint64_t h_cols[num_cols * num_words] = {
            // Col 0: pivot=194 (word 3, bit 2)
            0x0000, 0x0000, 0x0000, 0b100ULL,
            // Col 1: pivot=65  (word 1, bit 1)
            0x0000, 0b010ULL, 0x0000, 0x0000,
            // Col 2: pivot=0   (word 0, bit 0) -- unclaimed leaf
            0b001ULL, 0x0000, 0x0000, 0x0000};
        // pivot_to_col: 256 entries (4 words * 64)
        const int h_pivot_to_col[256] = {};
        const_cast<int *>(h_pivot_to_col)[0] = -1;  // Col 2 unclaimed
        const_cast<int *>(h_pivot_to_col)[65] = 2;  // Col 1 -> Col 2
        const_cast<int *>(h_pivot_to_col)[194] = 1; // Col 0 -> Col 1
        const int h_col_pivots[num_cols] = {194, 65, 0};

        const int expected_pivots[num_cols] = {194, 65, 0};
        // Expected data equals initial (converges back after even iterations)

        const size_t col_bytes = sizeof(h_cols);
        const size_t ptc_bytes = sizeof(h_pivot_to_col);
        const size_t idx_bytes = num_cols * sizeof(int);
        const int block_threads = max(32, num_cols * 32);

        // Device allocations
        uint64_t *d_cols;
        int *d_pivot_to_col, *d_col_pivots, *d_new_pivots;
        check_cuda(cudaMalloc(&d_cols, col_bytes), "mw4 d_cols");
        check_cuda(cudaMalloc(&d_pivot_to_col, ptc_bytes), "mw4 d_ptc");
        check_cuda(cudaMalloc(&d_col_pivots, idx_bytes), "mw4 d_cp");
        check_cuda(cudaMalloc(&d_new_pivots, idx_bytes), "mw4 d_np");
        check_cuda(cudaMemcpy(d_pivot_to_col, h_pivot_to_col, ptc_bytes, cudaMemcpyHostToDevice),
                   "mw4 h2d ptc");
        check_cuda(cudaMemcpy(d_col_pivots, h_col_pivots, idx_bytes, cudaMemcpyHostToDevice),
                   "mw4 h2d cp");

        auto upload_data = [&]() -> bool {
            bool ok = true;
            ok &= check_cuda(cudaMemcpy(d_cols, h_cols, col_bytes, cudaMemcpyHostToDevice),
                             "mw4 h2d cols");
            ok &= check_cuda(cudaMemset(d_new_pivots, 0xFF, idx_bytes), "mw4 memset new_pivots");
            return ok;
        };

        // single-warp reference
        int ref_pivots[num_cols];
        uint64_t ref_data[num_cols * num_words];
        {
            if (!upload_data())
                goto mw4_cleanup;
            singleWarpReductionRef<<<1, 32>>>(d_cols, d_pivot_to_col, num_words, num_cols,
                                              d_new_pivots);
            check_cuda(cudaDeviceSynchronize(), "mw4 ref sync");
            check_cuda(
                cudaMemcpy(ref_pivots, d_new_pivots, sizeof(ref_pivots), cudaMemcpyDeviceToHost),
                "mw4 ref d2h pivots");
            check_cuda(cudaMemcpy(ref_data, d_cols, sizeof(ref_data), cudaMemcpyDeviceToHost),
                       "mw4 ref d2h data");

            // Validate reference against expected
            bool ref_ok = true;
            for (int i = 0; i < num_cols; ++i)
                if (ref_pivots[i] != expected_pivots[i])
                {
                    fprintf(stderr, "mw4 ref pivot[%d] = %d, expected %d\n", i, ref_pivots[i],
                            expected_pivots[i]);
                    ref_ok = false;
                }
            for (int i = 0; i < num_cols * num_words; ++i)
                if (ref_data[i] != h_cols[i])
                {
                    fprintf(stderr, "mw4 ref data[%d] = 0x%lx, expected 0x%lx\n", i, ref_data[i],
                            h_cols[i]);
                    ref_ok = false;
                }
            if (!ref_ok)
                goto mw4_cleanup;
        }

        // syncthreads reference
        {
            if (!upload_data())
                goto mw4_cleanup;
            syncthreadsReductionRef<<<1, block_threads>>>(d_cols, d_col_pivots, d_pivot_to_col,
                                                          num_words, num_cols, d_new_pivots);
            check_cuda(cudaDeviceSynchronize(), "mw4 sync sync");
            int sync_pivots[num_cols];
            check_cuda(
                cudaMemcpy(sync_pivots, d_new_pivots, sizeof(sync_pivots), cudaMemcpyDeviceToHost),
                "mw4 sync d2h");
            bool sync_ok = true;
            for (int i = 0; i < num_cols; ++i)
                if (sync_pivots[i] != ref_pivots[i])
                {
                    fprintf(stderr, "mw4 sync pivot[%d] = %d, ref %d\n", i, sync_pivots[i],
                            ref_pivots[i]);
                    sync_ok = false;
                }
            if (!sync_ok)
                goto mw4_cleanup;
        }

        // regular kernel (use_async_copy=false) 5x
        for (int run = 0; run < 5; ++run)
        {
            if (!upload_data())
                goto mw4_cleanup;
            nerve::persistence::gpu::launchPipelinedReduction(
                d_cols, d_col_pivots, d_pivot_to_col, num_words, num_cols, d_new_pivots, 0, false);
            int run_pivots[num_cols];
            check_cuda(
                cudaMemcpy(run_pivots, d_new_pivots, sizeof(run_pivots), cudaMemcpyDeviceToHost),
                "mw4 reg d2h");
            bool run_ok = true;
            for (int i = 0; i < num_cols; ++i)
                if (run_pivots[i] != ref_pivots[i])
                {
                    fprintf(stderr, "mw4 reg[run=%d] pivot[%d] = %d, ref %d\n", run, i,
                            run_pivots[i], ref_pivots[i]);
                    run_ok = false;
                }
            if (!run_ok)
                goto mw4_cleanup;
        }

        // async kernel (use_async_copy=true) 5x
        for (int run = 0; run < 5; ++run)
        {
            if (!upload_data())
                goto mw4_cleanup;
            nerve::persistence::gpu::launchPipelinedReduction(
                d_cols, d_col_pivots, d_pivot_to_col, num_words, num_cols, d_new_pivots, 0, true);
            int run_pivots[num_cols];
            check_cuda(
                cudaMemcpy(run_pivots, d_new_pivots, sizeof(run_pivots), cudaMemcpyDeviceToHost),
                "mw4 async d2h");
            bool run_ok = true;
            for (int i = 0; i < num_cols; ++i)
                if (run_pivots[i] != ref_pivots[i])
                {
                    fprintf(stderr, "mw4 async[run=%d] pivot[%d] = %d, ref %d\n", run, i,
                            run_pivots[i], ref_pivots[i]);
                    run_ok = false;
                }
            if (!run_ok)
                goto mw4_cleanup;
        }

        // All passed
        printf("  PASSED: (4-word cross-column reduction)\n");
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        goto mw4_done;

mw4_cleanup:
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        return 1;

mw4_done:;
    }

    // 3-word syncthreads determinism: 10 runs, identical results
    //
    // Runs syncthreadsReductionRef 10 times on the 3-word cross-column chain
    // from test 26, verifying all runs produce identical pivots AND column data.
    //
    // Chain (race-free):
    // Col 0: [0, 0, 0b100] (pivot=130, word 2).  pivot_to_col[130]=1.
    // Col 1: [0, 0b100, 0]  (pivot=66,  word 1).  pivot_to_col[66]=2.
    // Col 2: [0b001, 0, 0]  (pivot=0,   word 0).  pivot_to_col[0]=-1.
    //
    // Expected pivots: {130, 66, 0}
    // Expected data:  same as initial (converges after even iterations)
    {
        constexpr int num_words = 3;
        constexpr int num_cols = 3;
        constexpr int num_runs = 10;

        const uint64_t h_cols[num_cols * num_words] = {
            0x0000,   0x0000,   0b100ULL, // Col 0: pivot=130
            0x0000,   0b100ULL, 0x0000,   // Col 1: pivot=66
            0b001ULL, 0x0000,   0x0000    // Col 2: pivot=0  (unclaimed)
        };
        const int h_pivot_to_col[192] = {};
        const_cast<int *>(h_pivot_to_col)[0] = -1;
        const_cast<int *>(h_pivot_to_col)[66] = 2;
        const_cast<int *>(h_pivot_to_col)[130] = 1;
        const int h_col_pivots[num_cols] = {130, 66, 0};

        const size_t col_bytes = sizeof(h_cols);
        const size_t ptc_bytes = sizeof(h_pivot_to_col);
        const size_t idx_bytes = num_cols * sizeof(int);
        const int block_threads = max(32, num_cols * 32);

        // Device allocations
        uint64_t *d_cols;
        int *d_pivot_to_col, *d_col_pivots, *d_new_pivots;
        check_cuda(cudaMalloc(&d_cols, col_bytes), "mw3s d_cols");
        check_cuda(cudaMalloc(&d_pivot_to_col, ptc_bytes), "mw3s d_ptc");
        check_cuda(cudaMalloc(&d_col_pivots, idx_bytes), "mw3s d_cp");
        check_cuda(cudaMalloc(&d_new_pivots, idx_bytes), "mw3s d_np");
        check_cuda(cudaMemcpy(d_pivot_to_col, h_pivot_to_col, ptc_bytes, cudaMemcpyHostToDevice),
                   "mw3s h2d ptc");
        check_cuda(cudaMemcpy(d_col_pivots, h_col_pivots, idx_bytes, cudaMemcpyHostToDevice),
                   "mw3s h2d cp");

        // Reference storage (from run 0)
        int ref_pivots[num_cols];
        uint64_t ref_data[num_cols * num_words];

        for (int run = 0; run < num_runs; ++run)
        {
            // Upload fresh data before each run
            check_cuda(cudaMemcpy(d_cols, h_cols, col_bytes, cudaMemcpyHostToDevice),
                       "mw3s h2d cols");
            check_cuda(cudaMemset(d_new_pivots, 0xFF, idx_bytes), "mw3s memset new_pivots");

            syncthreadsReductionRef<<<1, block_threads>>>(d_cols, d_col_pivots, d_pivot_to_col,
                                                          num_words, num_cols, d_new_pivots);
            check_cuda(cudaDeviceSynchronize(), "mw3s sync");

            int run_pivots[num_cols];
            uint64_t run_data[num_cols * num_words];
            check_cuda(
                cudaMemcpy(run_pivots, d_new_pivots, sizeof(run_pivots), cudaMemcpyDeviceToHost),
                "mw3s d2h pivots");
            check_cuda(cudaMemcpy(run_data, d_cols, sizeof(run_data), cudaMemcpyDeviceToHost),
                       "mw3s d2h cols");

            if (run == 0)
            {
                // Store reference
                for (int i = 0; i < num_cols; ++i)
                    ref_pivots[i] = run_pivots[i];
                for (int i = 0; i < num_cols * num_words; ++i)
                    ref_data[i] = run_data[i];
            }
            else
            {
                // Compare against reference
                bool ok = true;
                for (int i = 0; i < num_cols; ++i)
                    if (run_pivots[i] != ref_pivots[i])
                    {
                        fprintf(stderr, "mw3s[run=%d] pivot[%d] = %d, ref %d\n", run, i,
                                run_pivots[i], ref_pivots[i]);
                        ok = false;
                    }
                for (int i = 0; i < num_cols * num_words; ++i)
                    if (run_data[i] != ref_data[i])
                    {
                        fprintf(stderr, "mw3s[run=%d] data[%d] = 0x%lx, ref 0x%lx\n", run, i,
                                run_data[i], ref_data[i]);
                        ok = false;
                    }
                if (!ok)
                    goto mw3s_cleanup;
            }
        }

        // All passed
        printf("  PASSED: (3-word syncthreads determinism)\n");
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        goto mw3s_done;

mw3s_cleanup:
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        return 1;

mw3s_done:;
    }

    // Multi-word (2-word) unclaimed pivot cross-column chain
    //
    // Each column's MSB moves to unclaimed (word 1) after XOR with the next
    // column in the chain. All pivots converge to unclaimed territory,
    // demonstrating "chain-to-unclaimed" behavior.
    //
    // Chain: Col 0 -> Col 1 -> Col 2 (unclaimed leaf)
    //
    // Col 0: [0b100, 0x0000] (pivot=2,   word 0). p2c[2]=1   -> XOR with Col 1
    // Col 1: [0b010, 0x0000] (pivot=1,   word 0). p2c[1]=2   -> XOR with Col 2
    // Col 2: [0x0000, 0b001 ] (pivot=64,  word 1). p2c[64]=-1 -> Break (unclaimed)
    //
    // Col 1 XOR with Col 2 -> [0b010, 0b001], MSB=64 (unclaimed). Break in 1 iter.
    // Col 0 XOR with Col 1 -> [0b110, 0b001], MSB=64 (unclaimed). Break in 1 iter.
    //
    // Race-free: Col 2 unmodified; Col 1 converges in 1 iteration;
    // Col 0 reads stable Col 1 data.
    //
    // Expected pivots: {64, 64, 64}
    {
        constexpr int num_words = 2;
        constexpr int num_cols = 3;

        // Column data: [col0_w0, col0_w1, col1_w0, col1_w1, col2_w0, col2_w1]
        const uint64_t h_cols[num_cols * num_words] = {
            0b100ULL, 0x0000,  // Col 0: pivot=2
            0b010ULL, 0x0000,  // Col 1: pivot=1
            0x0000,   0b001ULL // Col 2: pivot=64 (unclaimed)
        };
        const int h_pivot_to_col[128] = {};
        const_cast<int *>(h_pivot_to_col)[2] = 1;   // Col 0 -> Col 1
        const_cast<int *>(h_pivot_to_col)[1] = 2;   // Col 1 -> Col 2
        const_cast<int *>(h_pivot_to_col)[64] = -1; // Col 2 unclaimed
        const int h_col_pivots[num_cols] = {2, 1, 64};

        const int expected_pivots[num_cols] = {64, 64, 64};
        const uint64_t expected_data[num_cols * num_words] = {
            0b110ULL, 0b001ULL, // Col 0 after XOR
            0b010ULL, 0b001ULL, // Col 1 after XOR
            0x0000,   0b001ULL  // Col 2 unchanged
        };

        const size_t col_bytes = sizeof(h_cols);
        const size_t ptc_bytes = sizeof(h_pivot_to_col);
        const size_t idx_bytes = num_cols * sizeof(int);
        const int block_threads = max(32, num_cols * 32);

        // Device allocations
        uint64_t *d_cols;
        int *d_pivot_to_col, *d_col_pivots, *d_new_pivots;
        check_cuda(cudaMalloc(&d_cols, col_bytes), "ucx d_cols");
        check_cuda(cudaMalloc(&d_pivot_to_col, ptc_bytes), "ucx d_ptc");
        check_cuda(cudaMalloc(&d_col_pivots, idx_bytes), "ucx d_cp");
        check_cuda(cudaMalloc(&d_new_pivots, idx_bytes), "ucx d_np");
        check_cuda(cudaMemcpy(d_pivot_to_col, h_pivot_to_col, ptc_bytes, cudaMemcpyHostToDevice),
                   "ucx h2d ptc");
        check_cuda(cudaMemcpy(d_col_pivots, h_col_pivots, idx_bytes, cudaMemcpyHostToDevice),
                   "ucx h2d cp");

        auto upload_data = [&]() -> bool {
            bool ok = true;
            ok &= check_cuda(cudaMemcpy(d_cols, h_cols, col_bytes, cudaMemcpyHostToDevice),
                             "ucx h2d cols");
            ok &= check_cuda(cudaMemset(d_new_pivots, 0xFF, idx_bytes), "ucx memset new_pivots");
            return ok;
        };

        // single-warp reference
        int ref_pivots[num_cols];
        uint64_t ref_data[num_cols * num_words];
        {
            if (!upload_data())
                goto ucx_cleanup;
            singleWarpReductionRef<<<1, 32>>>(d_cols, d_pivot_to_col, num_words, num_cols,
                                              d_new_pivots);
            check_cuda(cudaDeviceSynchronize(), "ucx ref sync");
            check_cuda(
                cudaMemcpy(ref_pivots, d_new_pivots, sizeof(ref_pivots), cudaMemcpyDeviceToHost),
                "ucx ref d2h pivots");
            check_cuda(cudaMemcpy(ref_data, d_cols, sizeof(ref_data), cudaMemcpyDeviceToHost),
                       "ucx ref d2h data");

            // Validate reference against expected
            bool ref_ok = true;
            for (int i = 0; i < num_cols; ++i)
                if (ref_pivots[i] != expected_pivots[i])
                {
                    fprintf(stderr, "ucx ref pivot[%d] = %d, expected %d\n", i, ref_pivots[i],
                            expected_pivots[i]);
                    ref_ok = false;
                }
            for (int i = 0; i < num_cols * num_words; ++i)
                if (ref_data[i] != expected_data[i])
                {
                    fprintf(stderr, "ucx ref data[%d] = 0x%lx, expected 0x%lx\n", i, ref_data[i],
                            expected_data[i]);
                    ref_ok = false;
                }
            if (!ref_ok)
                goto ucx_cleanup;
        }

        // syncthreads reference
        {
            if (!upload_data())
                goto ucx_cleanup;
            syncthreadsReductionRef<<<1, block_threads>>>(d_cols, d_col_pivots, d_pivot_to_col,
                                                          num_words, num_cols, d_new_pivots);
            check_cuda(cudaDeviceSynchronize(), "ucx sync sync");
            int sync_pivots[num_cols];
            check_cuda(
                cudaMemcpy(sync_pivots, d_new_pivots, sizeof(sync_pivots), cudaMemcpyDeviceToHost),
                "ucx sync d2h");
            bool sync_ok = true;
            for (int i = 0; i < num_cols; ++i)
                if (sync_pivots[i] != ref_pivots[i])
                {
                    fprintf(stderr, "ucx sync pivot[%d] = %d, ref %d\n", i, sync_pivots[i],
                            ref_pivots[i]);
                    sync_ok = false;
                }
            if (!sync_ok)
                goto ucx_cleanup;
        }

        // regular kernel (use_async_copy=false) 5x
        for (int run = 0; run < 5; ++run)
        {
            if (!upload_data())
                goto ucx_cleanup;
            nerve::persistence::gpu::launchPipelinedReduction(
                d_cols, d_col_pivots, d_pivot_to_col, num_words, num_cols, d_new_pivots, 0, false);
            int run_pivots[num_cols];
            check_cuda(
                cudaMemcpy(run_pivots, d_new_pivots, sizeof(run_pivots), cudaMemcpyDeviceToHost),
                "ucx reg d2h");
            bool run_ok = true;
            for (int i = 0; i < num_cols; ++i)
                if (run_pivots[i] != ref_pivots[i])
                {
                    fprintf(stderr, "ucx reg[run=%d] pivot[%d] = %d, ref %d\n", run, i,
                            run_pivots[i], ref_pivots[i]);
                    run_ok = false;
                }
            if (!run_ok)
                goto ucx_cleanup;
        }

        // async kernel (use_async_copy=true) 5x
        for (int run = 0; run < 5; ++run)
        {
            if (!upload_data())
                goto ucx_cleanup;
            nerve::persistence::gpu::launchPipelinedReduction(
                d_cols, d_col_pivots, d_pivot_to_col, num_words, num_cols, d_new_pivots, 0, true);
            int run_pivots[num_cols];
            check_cuda(
                cudaMemcpy(run_pivots, d_new_pivots, sizeof(run_pivots), cudaMemcpyDeviceToHost),
                "ucx async d2h");
            bool run_ok = true;
            for (int i = 0; i < num_cols; ++i)
                if (run_pivots[i] != ref_pivots[i])
                {
                    fprintf(stderr, "ucx async[run=%d] pivot[%d] = %d, ref %d\n", run, i,
                            run_pivots[i], ref_pivots[i]);
                    run_ok = false;
                }
            if (!run_ok)
                goto ucx_cleanup;
        }

        // All passed
        printf("  PASSED: (2-word unclaimed pivot cross-column chain)\n");
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        goto ucx_done;

ucx_cleanup:
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        return 1;

ucx_done:;
    }

    // 4-word async-copy determinism: 10 runs, identical results
    //
    // Runs launchPipelinedReduction with use_async_copy=true 10 times on the
    // 4-word cross-column chain from test 33, verifying all runs produce
    // identical pivots AND column data.
    //
    // Chain (same as test 33, race-free):
    // Col 0: [0, 0, 0, 0b100] (pivot=194, word 3).  pivot_to_col[194]=1.
    // Col 1: [0, 0b010, 0, 0]  (pivot=65,  word 1).  pivot_to_col[65]=2.
    // Col 2: [0b001, 0, 0, 0]  (pivot=0,   word 0).  pivot_to_col[0]=-1.
    //
    // Expected: all 10 runs produce identical new_pivots AND column data.
    // Expected pivots: {194, 65, 0} (data returns to initial after 16 even iter.)
    {
        constexpr int num_words = 4;
        constexpr int num_cols = 3;
        constexpr int num_runs = 10;

        const uint64_t h_cols[num_cols * num_words] = {
            0x0000,   0x0000,   0x0000, 0b100ULL, // Col 0: pivot=194
            0x0000,   0b010ULL, 0x0000, 0x0000,   // Col 1: pivot=65
            0b001ULL, 0x0000,   0x0000, 0x0000    // Col 2: pivot=0  (unclaimed)
        };
        const int h_pivot_to_col[256] = {};
        const_cast<int *>(h_pivot_to_col)[0] = -1;
        const_cast<int *>(h_pivot_to_col)[65] = 2;
        const_cast<int *>(h_pivot_to_col)[194] = 1;
        const int h_col_pivots[num_cols] = {194, 65, 0};

        const size_t col_bytes = sizeof(h_cols);
        const size_t ptc_bytes = sizeof(h_pivot_to_col);
        const size_t idx_bytes = num_cols * sizeof(int);

        // Device allocations
        uint64_t *d_cols;
        int *d_pivot_to_col, *d_col_pivots, *d_new_pivots;
        check_cuda(cudaMalloc(&d_cols, col_bytes), "mw4d d_cols");
        check_cuda(cudaMalloc(&d_pivot_to_col, ptc_bytes), "mw4d d_ptc");
        check_cuda(cudaMalloc(&d_col_pivots, idx_bytes), "mw4d d_cp");
        check_cuda(cudaMalloc(&d_new_pivots, idx_bytes), "mw4d d_np");
        check_cuda(cudaMemcpy(d_pivot_to_col, h_pivot_to_col, ptc_bytes, cudaMemcpyHostToDevice),
                   "mw4d h2d ptc");
        check_cuda(cudaMemcpy(d_col_pivots, h_col_pivots, idx_bytes, cudaMemcpyHostToDevice),
                   "mw4d h2d cp");

        // Reference storage (from run 0)
        int ref_pivots[num_cols];
        uint64_t ref_data[num_cols * num_words];

        for (int run = 0; run < num_runs; ++run)
        {
            // Upload fresh data before each run
            check_cuda(cudaMemcpy(d_cols, h_cols, col_bytes, cudaMemcpyHostToDevice),
                       "mw4d h2d cols");
            check_cuda(cudaMemset(d_new_pivots, 0xFF, idx_bytes), "mw4d memset new_pivots");

            nerve::persistence::gpu::launchPipelinedReduction(
                d_cols, d_col_pivots, d_pivot_to_col, num_words, num_cols, d_new_pivots, 0, true);

            int run_pivots[num_cols];
            uint64_t run_data[num_cols * num_words];
            check_cuda(
                cudaMemcpy(run_pivots, d_new_pivots, sizeof(run_pivots), cudaMemcpyDeviceToHost),
                "mw4d d2h pivots");
            check_cuda(cudaMemcpy(run_data, d_cols, sizeof(run_data), cudaMemcpyDeviceToHost),
                       "mw4d d2h cols");

            if (run == 0)
            {
                // Store reference
                for (int i = 0; i < num_cols; ++i)
                    ref_pivots[i] = run_pivots[i];
                for (int i = 0; i < num_cols * num_words; ++i)
                    ref_data[i] = run_data[i];
            }
            else
            {
                // Compare against reference
                bool ok = true;
                for (int i = 0; i < num_cols; ++i)
                    if (run_pivots[i] != ref_pivots[i])
                    {
                        fprintf(stderr, "mw4d[run=%d] pivot[%d] = %d, ref %d\n", run, i,
                                run_pivots[i], ref_pivots[i]);
                        ok = false;
                    }
                for (int i = 0; i < num_cols * num_words; ++i)
                    if (run_data[i] != ref_data[i])
                    {
                        fprintf(stderr, "mw4d[run=%d] data[%d] = 0x%lx, ref 0x%lx\n", run, i,
                                run_data[i], ref_data[i]);
                        ok = false;
                    }
                if (!ok)
                    goto mw4d_cleanup;
            }
        }

        // All passed
        printf("  PASSED: (4-word async determinism)\n");
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        goto mw4d_done;

mw4d_cleanup:
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        return 1;

mw4d_done:;
    }

    // Multi-word (4-word) self-clearing: single-warp ref vs syncthreads
    // vs regular (5x) vs async (5x)
    //
    // Each column claims its own pivot (self-clear). No cross-column reads,
    // so trivially race-free -- all variants produce identical results.
    //
    // Col 0: [0b100, 0x0000, 0x0000, 0x0000] (pivot=2,   word 0). p2c[2]=0
    // Col 1: [0x0000, 0x0000, 0x0000, 0b001 ] (pivot=192, word 3). p2c[192]=1
    // Col 2: [0x0000, 0x0000, 0b010,  0x0000] (pivot=128, word 2). p2c[128]=2
    //
    // Expected: all columns zeroed, all pivots = -1.
    {
        constexpr int num_words = 4;
        constexpr int num_cols = 3;

        // Column data: [col0_w0..w3, col1_w0..w3, col2_w0..w3]
        const uint64_t h_cols[num_cols * num_words] = {
            0b100ULL, 0x0000, 0x0000,   0x0000,   // Col 0: pivot=2   (word 0)
            0x0000,   0x0000, 0x0000,   0b001ULL, // Col 1: pivot=192 (word 3)
            0x0000,   0x0000, 0b010ULL, 0x0000    // Col 2: pivot=128 (word 2)
        };
        // pivot_to_col: 256 entries (4 words * 64)
        const int h_pivot_to_col[256] = {};
        const_cast<int *>(h_pivot_to_col)[2] = 0;   // Col 0 self
        const_cast<int *>(h_pivot_to_col)[192] = 1; // Col 1 self
        const_cast<int *>(h_pivot_to_col)[128] = 2; // Col 2 self
        const int h_col_pivots[num_cols] = {2, 192, 128};

        const int expected_pivots[num_cols] = {-1, -1, -1};
        const uint64_t expected_data[num_cols * num_words] = {0};

        const size_t col_bytes = sizeof(h_cols);
        const size_t ptc_bytes = sizeof(h_pivot_to_col);
        const size_t idx_bytes = num_cols * sizeof(int);
        const int block_threads = max(32, num_cols * 32);

        // Device allocations
        uint64_t *d_cols;
        int *d_pivot_to_col, *d_col_pivots, *d_new_pivots;
        check_cuda(cudaMalloc(&d_cols, col_bytes), "sc4 d_cols");
        check_cuda(cudaMalloc(&d_pivot_to_col, ptc_bytes), "sc4 d_ptc");
        check_cuda(cudaMalloc(&d_col_pivots, idx_bytes), "sc4 d_cp");
        check_cuda(cudaMalloc(&d_new_pivots, idx_bytes), "sc4 d_np");
        check_cuda(cudaMemcpy(d_pivot_to_col, h_pivot_to_col, ptc_bytes, cudaMemcpyHostToDevice),
                   "sc4 h2d ptc");
        check_cuda(cudaMemcpy(d_col_pivots, h_col_pivots, idx_bytes, cudaMemcpyHostToDevice),
                   "sc4 h2d cp");

        auto upload_data = [&]() -> bool {
            bool ok = true;
            ok &= check_cuda(cudaMemcpy(d_cols, h_cols, col_bytes, cudaMemcpyHostToDevice),
                             "sc4 h2d cols");
            ok &= check_cuda(cudaMemset(d_new_pivots, 0xFF, idx_bytes), "sc4 memset new_pivots");
            return ok;
        };

        // single-warp reference
        int ref_pivots[num_cols];
        uint64_t ref_data[num_cols * num_words];
        {
            if (!upload_data())
                goto sc4_cleanup;
            singleWarpReductionRef<<<1, 32>>>(d_cols, d_pivot_to_col, num_words, num_cols,
                                              d_new_pivots);
            check_cuda(cudaDeviceSynchronize(), "sc4 ref sync");
            check_cuda(
                cudaMemcpy(ref_pivots, d_new_pivots, sizeof(ref_pivots), cudaMemcpyDeviceToHost),
                "sc4 ref d2h pivots");
            check_cuda(cudaMemcpy(ref_data, d_cols, sizeof(ref_data), cudaMemcpyDeviceToHost),
                       "sc4 ref d2h data");

            // Validate reference against expected
            bool ref_ok = true;
            for (int i = 0; i < num_cols; ++i)
                if (ref_pivots[i] != expected_pivots[i])
                {
                    fprintf(stderr, "sc4 ref pivot[%d] = %d, expected %d\n", i, ref_pivots[i],
                            expected_pivots[i]);
                    ref_ok = false;
                }
            for (int i = 0; i < num_cols * num_words; ++i)
                if (ref_data[i] != expected_data[i])
                {
                    fprintf(stderr, "sc4 ref data[%d] = 0x%lx, expected 0x%lx\n", i, ref_data[i],
                            expected_data[i]);
                    ref_ok = false;
                }
            if (!ref_ok)
                goto sc4_cleanup;
        }

        // syncthreads reference
        {
            if (!upload_data())
                goto sc4_cleanup;
            syncthreadsReductionRef<<<1, block_threads>>>(d_cols, d_col_pivots, d_pivot_to_col,
                                                          num_words, num_cols, d_new_pivots);
            check_cuda(cudaDeviceSynchronize(), "sc4 sync sync");
            int sync_pivots[num_cols];
            check_cuda(
                cudaMemcpy(sync_pivots, d_new_pivots, sizeof(sync_pivots), cudaMemcpyDeviceToHost),
                "sc4 sync d2h");
            bool sync_ok = true;
            for (int i = 0; i < num_cols; ++i)
                if (sync_pivots[i] != ref_pivots[i])
                {
                    fprintf(stderr, "sc4 sync pivot[%d] = %d, ref %d\n", i, sync_pivots[i],
                            ref_pivots[i]);
                    sync_ok = false;
                }
            if (!sync_ok)
                goto sc4_cleanup;
        }

        // regular kernel (use_async_copy=false) 5x
        for (int run = 0; run < 5; ++run)
        {
            if (!upload_data())
                goto sc4_cleanup;
            nerve::persistence::gpu::launchPipelinedReduction(
                d_cols, d_col_pivots, d_pivot_to_col, num_words, num_cols, d_new_pivots, 0, false);
            int run_pivots[num_cols];
            check_cuda(
                cudaMemcpy(run_pivots, d_new_pivots, sizeof(run_pivots), cudaMemcpyDeviceToHost),
                "sc4 reg d2h");
            bool run_ok = true;
            for (int i = 0; i < num_cols; ++i)
                if (run_pivots[i] != ref_pivots[i])
                {
                    fprintf(stderr, "sc4 reg[run=%d] pivot[%d] = %d, ref %d\n", run, i,
                            run_pivots[i], ref_pivots[i]);
                    run_ok = false;
                }
            if (!run_ok)
                goto sc4_cleanup;
        }

        // async kernel (use_async_copy=true) 5x
        for (int run = 0; run < 5; ++run)
        {
            if (!upload_data())
                goto sc4_cleanup;
            nerve::persistence::gpu::launchPipelinedReduction(
                d_cols, d_col_pivots, d_pivot_to_col, num_words, num_cols, d_new_pivots, 0, true);
            int run_pivots[num_cols];
            check_cuda(
                cudaMemcpy(run_pivots, d_new_pivots, sizeof(run_pivots), cudaMemcpyDeviceToHost),
                "sc4 async d2h");
            bool run_ok = true;
            for (int i = 0; i < num_cols; ++i)
                if (run_pivots[i] != ref_pivots[i])
                {
                    fprintf(stderr, "sc4 async[run=%d] pivot[%d] = %d, ref %d\n", run, i,
                            run_pivots[i], ref_pivots[i]);
                    run_ok = false;
                }
            if (!run_ok)
                goto sc4_cleanup;
        }

        // All passed
        printf("  PASSED: (4-word self-clearing)\n");
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        goto sc4_done;

sc4_cleanup:
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        return 1;

sc4_done:;
    }

    // 4-word async-copy determinism (test 25's pattern): 10 runs, identical
    //
    // Runs launchPipelinedReduction with use_async_copy=true 10 times on the
    // 1-word cross-column chain from test 25, extended to 4-word columns.
    //
    // Chain (same as test 25, race-free):
    // Col 0: [0b110, 0, 0, 0] (pivot=2,   word 0).  pivot_to_col[2]=1.
    // Col 1: [0b010, 0, 0, 0] (pivot=1,   word 0).  pivot_to_col[1]=2.
    // Col 2: [0b001, 0, 0, 0] (pivot=0,   word 0).  pivot_to_col[0]=-1.
    //
    // Expected: all 10 runs produce identical new_pivots AND column data.
    // Expected pivots: {2, 1, 0} (data returns to initial after 16 even iter.)
    {
        constexpr int num_words = 4;
        constexpr int num_cols = 3;
        constexpr int num_runs = 10;

        const uint64_t h_cols[num_cols * num_words] = {
            0b110ULL, 0x0000, 0x0000, 0x0000, // Col 0: pivot=2
            0b010ULL, 0x0000, 0x0000, 0x0000, // Col 1: pivot=1
            0b001ULL, 0x0000, 0x0000, 0x0000  // Col 2: pivot=0 (unclaimed)
        };
        const int h_pivot_to_col[256] = {};
        const_cast<int *>(h_pivot_to_col)[0] = -1;
        const_cast<int *>(h_pivot_to_col)[1] = 2;
        const_cast<int *>(h_pivot_to_col)[2] = 1;
        const int h_col_pivots[num_cols] = {2, 1, 0};

        const size_t col_bytes = sizeof(h_cols);
        const size_t ptc_bytes = sizeof(h_pivot_to_col);
        const size_t idx_bytes = num_cols * sizeof(int);

        // Device allocations
        uint64_t *d_cols;
        int *d_pivot_to_col, *d_col_pivots, *d_new_pivots;
        check_cuda(cudaMalloc(&d_cols, col_bytes), "mw4t d_cols");
        check_cuda(cudaMalloc(&d_pivot_to_col, ptc_bytes), "mw4t d_ptc");
        check_cuda(cudaMalloc(&d_col_pivots, idx_bytes), "mw4t d_cp");
        check_cuda(cudaMalloc(&d_new_pivots, idx_bytes), "mw4t d_np");
        check_cuda(cudaMemcpy(d_pivot_to_col, h_pivot_to_col, ptc_bytes, cudaMemcpyHostToDevice),
                   "mw4t h2d ptc");
        check_cuda(cudaMemcpy(d_col_pivots, h_col_pivots, idx_bytes, cudaMemcpyHostToDevice),
                   "mw4t h2d cp");

        // Reference storage (from run 0)
        int ref_pivots[num_cols];
        uint64_t ref_data[num_cols * num_words];

        for (int run = 0; run < num_runs; ++run)
        {
            // Upload fresh data before each run
            check_cuda(cudaMemcpy(d_cols, h_cols, col_bytes, cudaMemcpyHostToDevice),
                       "mw4t h2d cols");
            check_cuda(cudaMemset(d_new_pivots, 0xFF, idx_bytes), "mw4t memset new_pivots");

            nerve::persistence::gpu::launchPipelinedReduction(
                d_cols, d_col_pivots, d_pivot_to_col, num_words, num_cols, d_new_pivots, 0, true);

            int run_pivots[num_cols];
            uint64_t run_data[num_cols * num_words];
            check_cuda(
                cudaMemcpy(run_pivots, d_new_pivots, sizeof(run_pivots), cudaMemcpyDeviceToHost),
                "mw4t d2h pivots");
            check_cuda(cudaMemcpy(run_data, d_cols, sizeof(run_data), cudaMemcpyDeviceToHost),
                       "mw4t d2h cols");

            if (run == 0)
            {
                // Store reference
                for (int i = 0; i < num_cols; ++i)
                    ref_pivots[i] = run_pivots[i];
                for (int i = 0; i < num_cols * num_words; ++i)
                    ref_data[i] = run_data[i];
            }
            else
            {
                // Compare against reference
                bool ok = true;
                for (int i = 0; i < num_cols; ++i)
                    if (run_pivots[i] != ref_pivots[i])
                    {
                        fprintf(stderr, "mw4t[run=%d] pivot[%d] = %d, ref %d\n", run, i,
                                run_pivots[i], ref_pivots[i]);
                        ok = false;
                    }
                for (int i = 0; i < num_cols * num_words; ++i)
                    if (run_data[i] != ref_data[i])
                    {
                        fprintf(stderr, "mw4t[run=%d] data[%d] = 0x%lx, ref 0x%lx\n", run, i,
                                run_data[i], ref_data[i]);
                        ok = false;
                    }
                if (!ok)
                    goto mw4t_cleanup;
            }
        }

        // All passed
        printf("  PASSED: (4-word async determinism, test 25 pattern)\n");
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        goto mw4t_done;

mw4t_cleanup:
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        return 1;

mw4t_done:;
    }

    // 4-word syncthreads determinism: 10 runs, identical results
    //
    // Runs syncthreadsReductionRef 10 times on the 4-word cross-column chain
    // from test 33, verifying all runs produce identical pivots AND column data.
    //
    // Chain (same as test 33, race-free):
    // Col 0: [0, 0, 0, 0b100] (pivot=194, word 3).  pivot_to_col[194]=1.
    // Col 1: [0, 0b010, 0, 0]  (pivot=65,  word 1).  pivot_to_col[65]=2.
    // Col 2: [0b001, 0, 0, 0]  (pivot=0,   word 0).  pivot_to_col[0]=-1.
    //
    // Expected: all 10 runs produce identical pivots and column data.
    // Expected pivots: {194, 65, 0} (data returns to initial after 16 even iter.)
    {
        constexpr int num_words = 4;
        constexpr int num_cols = 3;
        constexpr int num_runs = 10;

        const uint64_t h_cols[num_cols * num_words] = {
            0x0000,   0x0000,   0x0000, 0b100ULL, // Col 0
            0x0000,   0b010ULL, 0x0000, 0x0000,   // Col 1
            0b001ULL, 0x0000,   0x0000, 0x0000    // Col 2
        };
        const int h_pivot_to_col[256] = {};
        const_cast<int *>(h_pivot_to_col)[0] = -1;
        const_cast<int *>(h_pivot_to_col)[65] = 2;
        const_cast<int *>(h_pivot_to_col)[194] = 1;
        const int h_col_pivots[num_cols] = {194, 65, 0};

        const size_t col_bytes = sizeof(h_cols);
        const size_t ptc_bytes = sizeof(h_pivot_to_col);
        const size_t idx_bytes = num_cols * sizeof(int);
        const int block_threads = max(32, num_cols * 32);

        uint64_t *d_cols;
        int *d_pivot_to_col, *d_col_pivots, *d_new_pivots;
        check_cuda(cudaMalloc(&d_cols, col_bytes), "mw4s d_cols");
        check_cuda(cudaMalloc(&d_pivot_to_col, ptc_bytes), "mw4s d_ptc");
        check_cuda(cudaMalloc(&d_col_pivots, idx_bytes), "mw4s d_cp");
        check_cuda(cudaMalloc(&d_new_pivots, idx_bytes), "mw4s d_np");
        check_cuda(cudaMemcpy(d_pivot_to_col, h_pivot_to_col, ptc_bytes, cudaMemcpyHostToDevice),
                   "mw4s h2d ptc");
        check_cuda(cudaMemcpy(d_col_pivots, h_col_pivots, idx_bytes, cudaMemcpyHostToDevice),
                   "mw4s h2d cp");

        int ref_pivots[num_cols];
        uint64_t ref_data[num_cols * num_words];

        for (int run = 0; run < num_runs; ++run)
        {
            check_cuda(cudaMemcpy(d_cols, h_cols, col_bytes, cudaMemcpyHostToDevice),
                       "mw4s h2d cols");
            check_cuda(cudaMemset(d_new_pivots, 0xFF, idx_bytes), "mw4s memset new_pivots");

            syncthreadsReductionRef<<<1, block_threads>>>(d_cols, d_col_pivots, d_pivot_to_col,
                                                          num_words, num_cols, d_new_pivots);
            check_cuda(cudaDeviceSynchronize(), "mw4s sync");

            int run_pivots[num_cols];
            uint64_t run_data[num_cols * num_words];
            check_cuda(
                cudaMemcpy(run_pivots, d_new_pivots, sizeof(run_pivots), cudaMemcpyDeviceToHost),
                "mw4s d2h pivots");
            check_cuda(cudaMemcpy(run_data, d_cols, sizeof(run_data), cudaMemcpyDeviceToHost),
                       "mw4s d2h cols");

            if (run == 0)
            {
                for (int i = 0; i < num_cols; ++i)
                    ref_pivots[i] = run_pivots[i];
                for (int i = 0; i < num_cols * num_words; ++i)
                    ref_data[i] = run_data[i];
            }
            else
            {
                bool ok = true;
                for (int i = 0; i < num_cols; ++i)
                    if (run_pivots[i] != ref_pivots[i])
                    {
                        fprintf(stderr, "mw4s[run=%d] pivot[%d] = %d, ref %d\n", run, i,
                                run_pivots[i], ref_pivots[i]);
                        ok = false;
                    }
                for (int i = 0; i < num_cols * num_words; ++i)
                    if (run_data[i] != ref_data[i])
                    {
                        fprintf(stderr, "mw4s[run=%d] data[%d] = 0x%lx, ref 0x%lx\n", run, i,
                                run_data[i], ref_data[i]);
                        ok = false;
                    }
                if (!ok)
                    goto mw4s_cleanup;
            }
        }

        printf("  PASSED: (4-word syncthreads determinism)\n");
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        goto mw4s_done;

mw4s_cleanup:
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        return 1;

mw4s_done:;
    }

    // Multi-word (4-word) unclaimed pivot cross-column chain
    //
    // Each column's MSB moves to unclaimed (word 3) after XOR with the next
    // column in the chain. All pivots converge to unclaimed territory,
    // demonstrating "chain-to-unclaimed" behavior with 4-word columns.
    //
    // Chain: Col 0 -> Col 1 -> Col 2 (unclaimed leaf)
    //
    // Col 0: [0b100, 0, 0, 0x0000] (pivot=2,   word 0). p2c[2]=1   -> XOR with Col 1
    // Col 1: [0, 0, 0b010, 0x0000] (pivot=128, word 2). p2c[128]=2 -> XOR with Col 2
    // Col 2: [0, 0, 0, 0b001]      (pivot=192, word 3). p2c[192]=-1 -> Break (unclaimed)
    //
    // Col 1 XOR with Col 2 -> [0,0,0b010,0b001], MSB=192 (unclaimed). Break in 1 iter.
    // Col 0 XOR with Col 1 -> [0b100,0,0b010,0b001], MSB=192 (unclaimed). Break in 1 iter.
    //
    // Race-free: Col 2 unmodified; Col 1 converges in 1 iteration;
    // Col 0 reads stable Col 1 data.
    //
    // Expected pivots: {192, 192, 192}
    {
        constexpr int num_words = 4;
        constexpr int num_cols = 3;

        // Column data: [col0_w0..w3, col1_w0..w3, col2_w0..w3]
        const uint64_t h_cols[num_cols * num_words] = {
            0b100ULL, 0x0000, 0x0000,   0x0000,  // Col 0: pivot=2
            0x0000,   0x0000, 0b010ULL, 0x0000,  // Col 1: pivot=128
            0x0000,   0x0000, 0x0000,   0b001ULL // Col 2: pivot=192 (unclaimed)
        };
        const int h_pivot_to_col[256] = {};
        const_cast<int *>(h_pivot_to_col)[2] = 1;    // Col 0 -> Col 1
        const_cast<int *>(h_pivot_to_col)[128] = 2;  // Col 1 -> Col 2
        const_cast<int *>(h_pivot_to_col)[192] = -1; // Col 2 unclaimed
        const int h_col_pivots[num_cols] = {2, 128, 192};

        const int expected_pivots[num_cols] = {192, 192, 192};
        const uint64_t expected_data[num_cols * num_words] = {
            0b100ULL, 0x0000, 0b010ULL, 0b001ULL, // Col 0 after XOR
            0x0000,   0x0000, 0b010ULL, 0b001ULL, // Col 1 after XOR
            0x0000,   0x0000, 0x0000,   0b001ULL  // Col 2 unchanged
        };

        const size_t col_bytes = sizeof(h_cols);
        const size_t ptc_bytes = sizeof(h_pivot_to_col);
        const size_t idx_bytes = num_cols * sizeof(int);
        const int block_threads = max(32, num_cols * 32);

        // Device allocations
        uint64_t *d_cols;
        int *d_pivot_to_col, *d_col_pivots, *d_new_pivots;
        check_cuda(cudaMalloc(&d_cols, col_bytes), "uc4 d_cols");
        check_cuda(cudaMalloc(&d_pivot_to_col, ptc_bytes), "uc4 d_ptc");
        check_cuda(cudaMalloc(&d_col_pivots, idx_bytes), "uc4 d_cp");
        check_cuda(cudaMalloc(&d_new_pivots, idx_bytes), "uc4 d_np");
        check_cuda(cudaMemcpy(d_pivot_to_col, h_pivot_to_col, ptc_bytes, cudaMemcpyHostToDevice),
                   "uc4 h2d ptc");
        check_cuda(cudaMemcpy(d_col_pivots, h_col_pivots, idx_bytes, cudaMemcpyHostToDevice),
                   "uc4 h2d cp");

        auto upload_data = [&]() -> bool {
            bool ok = true;
            ok &= check_cuda(cudaMemcpy(d_cols, h_cols, col_bytes, cudaMemcpyHostToDevice),
                             "uc4 h2d cols");
            ok &= check_cuda(cudaMemset(d_new_pivots, 0xFF, idx_bytes), "uc4 memset new_pivots");
            return ok;
        };

        // single-warp reference
        int ref_pivots[num_cols];
        uint64_t ref_data[num_cols * num_words];
        {
            if (!upload_data())
                goto uc4_cleanup;
            singleWarpReductionRef<<<1, 32>>>(d_cols, d_pivot_to_col, num_words, num_cols,
                                              d_new_pivots);
            check_cuda(cudaDeviceSynchronize(), "uc4 ref sync");
            check_cuda(
                cudaMemcpy(ref_pivots, d_new_pivots, sizeof(ref_pivots), cudaMemcpyDeviceToHost),
                "uc4 ref d2h pivots");
            check_cuda(cudaMemcpy(ref_data, d_cols, sizeof(ref_data), cudaMemcpyDeviceToHost),
                       "uc4 ref d2h data");

            // Validate reference against expected
            bool ref_ok = true;
            for (int i = 0; i < num_cols; ++i)
                if (ref_pivots[i] != expected_pivots[i])
                {
                    fprintf(stderr, "uc4 ref pivot[%d] = %d, expected %d\n", i, ref_pivots[i],
                            expected_pivots[i]);
                    ref_ok = false;
                }
            for (int i = 0; i < num_cols * num_words; ++i)
                if (ref_data[i] != expected_data[i])
                {
                    fprintf(stderr, "uc4 ref data[%d] = 0x%lx, expected 0x%lx\n", i, ref_data[i],
                            expected_data[i]);
                    ref_ok = false;
                }
            if (!ref_ok)
                goto uc4_cleanup;
        }

        // syncthreads reference
        {
            if (!upload_data())
                goto uc4_cleanup;
            syncthreadsReductionRef<<<1, block_threads>>>(d_cols, d_col_pivots, d_pivot_to_col,
                                                          num_words, num_cols, d_new_pivots);
            check_cuda(cudaDeviceSynchronize(), "uc4 sync sync");
            int sync_pivots[num_cols];
            check_cuda(
                cudaMemcpy(sync_pivots, d_new_pivots, sizeof(sync_pivots), cudaMemcpyDeviceToHost),
                "uc4 sync d2h");
            bool sync_ok = true;
            for (int i = 0; i < num_cols; ++i)
                if (sync_pivots[i] != ref_pivots[i])
                {
                    fprintf(stderr, "uc4 sync pivot[%d] = %d, ref %d\n", i, sync_pivots[i],
                            ref_pivots[i]);
                    sync_ok = false;
                }
            if (!sync_ok)
                goto uc4_cleanup;
        }

        // regular kernel (use_async_copy=false) 5x
        for (int run = 0; run < 5; ++run)
        {
            if (!upload_data())
                goto uc4_cleanup;
            nerve::persistence::gpu::launchPipelinedReduction(
                d_cols, d_col_pivots, d_pivot_to_col, num_words, num_cols, d_new_pivots, 0, false);
            int run_pivots[num_cols];
            check_cuda(
                cudaMemcpy(run_pivots, d_new_pivots, sizeof(run_pivots), cudaMemcpyDeviceToHost),
                "uc4 reg d2h");
            bool run_ok = true;
            for (int i = 0; i < num_cols; ++i)
                if (run_pivots[i] != ref_pivots[i])
                {
                    fprintf(stderr, "uc4 reg[run=%d] pivot[%d] = %d, ref %d\n", run, i,
                            run_pivots[i], ref_pivots[i]);
                    run_ok = false;
                }
            if (!run_ok)
                goto uc4_cleanup;
        }

        // async kernel (use_async_copy=true) 5x
        for (int run = 0; run < 5; ++run)
        {
            if (!upload_data())
                goto uc4_cleanup;
            nerve::persistence::gpu::launchPipelinedReduction(
                d_cols, d_col_pivots, d_pivot_to_col, num_words, num_cols, d_new_pivots, 0, true);
            int run_pivots[num_cols];
            check_cuda(
                cudaMemcpy(run_pivots, d_new_pivots, sizeof(run_pivots), cudaMemcpyDeviceToHost),
                "uc4 async d2h");
            bool run_ok = true;
            for (int i = 0; i < num_cols; ++i)
                if (run_pivots[i] != ref_pivots[i])
                {
                    fprintf(stderr, "uc4 async[run=%d] pivot[%d] = %d, ref %d\n", run, i,
                            run_pivots[i], ref_pivots[i]);
                    run_ok = false;
                }
            if (!run_ok)
                goto uc4_cleanup;
        }

        // All passed
        printf("  PASSED: (4-word unclaimed pivot cross-column chain)\n");
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        goto uc4_done;

uc4_cleanup:
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        return 1;

uc4_done:;
    }

    // 2-word unclaimed pivot cross-column determinism: 10 async runs
    //
    // Runs launchPipelinedReduction with use_async_copy=true 10 times on the
    // 2-word unclaimed pivot chain from test 35, verifying all runs produce
    // identical pivots AND column data.
    //
    // Chain (same as test 35, race-free):
    // Col 0: [0b100, 0x0000] (pivot=2).   pivot_to_col[2]=1  -> XOR with Col 1
    // Col 1: [0b010, 0x0000] (pivot=1).   pivot_to_col[1]=2  -> XOR with Col 2
    // Col 2: [0x0000, 0b001 ] (pivot=64).  pivot_to_col[64]=-1 -> Break (unclaimed)
    //
    // Both Col 1 and Col 0 converge to unclaimed pivot=64 after 1 iteration.
    // Expected pivots: {64, 64, 64}
    {
        constexpr int num_words = 2;
        constexpr int num_cols = 3;
        constexpr int num_runs = 10;

        const uint64_t h_cols[num_cols * num_words] = {
            0b100ULL, 0x0000,  // Col 0: pivot=2
            0b010ULL, 0x0000,  // Col 1: pivot=1
            0x0000,   0b001ULL // Col 2: pivot=64 (unclaimed)
        };
        const int h_pivot_to_col[128] = {};
        const_cast<int *>(h_pivot_to_col)[2] = 1;
        const_cast<int *>(h_pivot_to_col)[1] = 2;
        const_cast<int *>(h_pivot_to_col)[64] = -1;
        const int h_col_pivots[num_cols] = {2, 1, 64};

        const size_t col_bytes = sizeof(h_cols);
        const size_t ptc_bytes = sizeof(h_pivot_to_col);
        const size_t idx_bytes = num_cols * sizeof(int);

        // Device allocations
        uint64_t *d_cols;
        int *d_pivot_to_col, *d_col_pivots, *d_new_pivots;
        check_cuda(cudaMalloc(&d_cols, col_bytes), "ucd d_cols");
        check_cuda(cudaMalloc(&d_pivot_to_col, ptc_bytes), "ucd d_ptc");
        check_cuda(cudaMalloc(&d_col_pivots, idx_bytes), "ucd d_cp");
        check_cuda(cudaMalloc(&d_new_pivots, idx_bytes), "ucd d_np");
        check_cuda(cudaMemcpy(d_pivot_to_col, h_pivot_to_col, ptc_bytes, cudaMemcpyHostToDevice),
                   "ucd h2d ptc");
        check_cuda(cudaMemcpy(d_col_pivots, h_col_pivots, idx_bytes, cudaMemcpyHostToDevice),
                   "ucd h2d cp");

        // Reference storage (from run 0)
        int ref_pivots[num_cols];
        uint64_t ref_data[num_cols * num_words];

        for (int run = 0; run < num_runs; ++run)
        {
            check_cuda(cudaMemcpy(d_cols, h_cols, col_bytes, cudaMemcpyHostToDevice),
                       "ucd h2d cols");
            check_cuda(cudaMemset(d_new_pivots, 0xFF, idx_bytes), "ucd memset new_pivots");

            nerve::persistence::gpu::launchPipelinedReduction(
                d_cols, d_col_pivots, d_pivot_to_col, num_words, num_cols, d_new_pivots, 0, true);

            int run_pivots[num_cols];
            uint64_t run_data[num_cols * num_words];
            check_cuda(
                cudaMemcpy(run_pivots, d_new_pivots, sizeof(run_pivots), cudaMemcpyDeviceToHost),
                "ucd d2h pivots");
            check_cuda(cudaMemcpy(run_data, d_cols, sizeof(run_data), cudaMemcpyDeviceToHost),
                       "ucd d2h cols");

            if (run == 0)
            {
                for (int i = 0; i < num_cols; ++i)
                    ref_pivots[i] = run_pivots[i];
                for (int i = 0; i < num_cols * num_words; ++i)
                    ref_data[i] = run_data[i];
            }
            else
            {
                bool ok = true;
                for (int i = 0; i < num_cols; ++i)
                    if (run_pivots[i] != ref_pivots[i])
                    {
                        fprintf(stderr, "ucd[run=%d] pivot[%d] = %d, ref %d\n", run, i,
                                run_pivots[i], ref_pivots[i]);
                        ok = false;
                    }
                for (int i = 0; i < num_cols * num_words; ++i)
                    if (run_data[i] != ref_data[i])
                    {
                        fprintf(stderr, "ucd[run=%d] data[%d] = 0x%lx, ref 0x%lx\n", run, i,
                                run_data[i], ref_data[i]);
                        ok = false;
                    }
                if (!ok)
                    goto ucd_cleanup;
            }
        }

        // All passed
        printf("  PASSED: (2-word unclaimed pivot async determinism)\n");
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        goto ucd_done;

ucd_cleanup:
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        return 1;

ucd_done:;
    }

    // 2-word self-clearing async determinism: 10 runs, identical results
    //
    // Runs launchPipelinedReduction with use_async_copy=true 10 times on test 27's
    // 2-word self-clearing chain, verifying all runs produce identical pivots
    // AND column data.
    //
    // Chain (same as test 27, trivially race-free):
    // Col 0: [0b100, 0] (pivot=2).   pivot_to_col[2]=0   -> self-clear.
    // Col 1: [0, 0b001] (pivot=64).  pivot_to_col[64]=1  -> self-clear.
    // Col 2: [0b010, 0] (pivot=1).   pivot_to_col[1]=2   -> self-clear.
    //
    // Expected: all columns zeroed, all new_pivots = -1.
    {
        constexpr int num_words = 2;
        constexpr int num_cols = 3;
        constexpr int num_runs = 10;

        const uint64_t h_cols[num_cols * num_words] = {
            0b100ULL, 0x0000,   // Col 0: pivot=2
            0x0000,   0b001ULL, // Col 1: pivot=64
            0b010ULL, 0x0000    // Col 2: pivot=1
        };
        const int h_pivot_to_col[128] = {};
        const_cast<int *>(h_pivot_to_col)[2] = 0;
        const_cast<int *>(h_pivot_to_col)[64] = 1;
        const_cast<int *>(h_pivot_to_col)[1] = 2;
        const int h_col_pivots[num_cols] = {2, 64, 1};

        const size_t col_bytes = sizeof(h_cols);
        const size_t ptc_bytes = sizeof(h_pivot_to_col);
        const size_t idx_bytes = num_cols * sizeof(int);

        // Device allocations
        uint64_t *d_cols;
        int *d_pivot_to_col, *d_col_pivots, *d_new_pivots;
        check_cuda(cudaMalloc(&d_cols, col_bytes), "scd d_cols");
        check_cuda(cudaMalloc(&d_pivot_to_col, ptc_bytes), "scd d_ptc");
        check_cuda(cudaMalloc(&d_col_pivots, idx_bytes), "scd d_cp");
        check_cuda(cudaMalloc(&d_new_pivots, idx_bytes), "scd d_np");
        check_cuda(cudaMemcpy(d_pivot_to_col, h_pivot_to_col, ptc_bytes, cudaMemcpyHostToDevice),
                   "scd h2d ptc");
        check_cuda(cudaMemcpy(d_col_pivots, h_col_pivots, idx_bytes, cudaMemcpyHostToDevice),
                   "scd h2d cp");

        // Reference storage (from run 0)
        int ref_pivots[num_cols];
        uint64_t ref_data[num_cols * num_words];

        for (int run = 0; run < num_runs; ++run)
        {
            check_cuda(cudaMemcpy(d_cols, h_cols, col_bytes, cudaMemcpyHostToDevice),
                       "scd h2d cols");
            check_cuda(cudaMemset(d_new_pivots, 0xFF, idx_bytes), "scd memset new_pivots");

            nerve::persistence::gpu::launchPipelinedReduction(
                d_cols, d_col_pivots, d_pivot_to_col, num_words, num_cols, d_new_pivots, 0, true);

            int run_pivots[num_cols];
            uint64_t run_data[num_cols * num_words];
            check_cuda(
                cudaMemcpy(run_pivots, d_new_pivots, sizeof(run_pivots), cudaMemcpyDeviceToHost),
                "scd d2h pivots");
            check_cuda(cudaMemcpy(run_data, d_cols, sizeof(run_data), cudaMemcpyDeviceToHost),
                       "scd d2h cols");

            if (run == 0)
            {
                for (int i = 0; i < num_cols; ++i)
                    ref_pivots[i] = run_pivots[i];
                for (int i = 0; i < num_cols * num_words; ++i)
                    ref_data[i] = run_data[i];
            }
            else
            {
                bool ok = true;
                for (int i = 0; i < num_cols; ++i)
                    if (run_pivots[i] != ref_pivots[i])
                    {
                        fprintf(stderr, "scd[run=%d] pivot[%d] = %d, ref %d\n", run, i,
                                run_pivots[i], ref_pivots[i]);
                        ok = false;
                    }
                for (int i = 0; i < num_cols * num_words; ++i)
                    if (run_data[i] != ref_data[i])
                    {
                        fprintf(stderr, "scd[run=%d] data[%d] = 0x%lx, ref 0x%lx\n", run, i,
                                run_data[i], ref_data[i]);
                        ok = false;
                    }
                if (!ok)
                    goto scd_cleanup;
            }
        }

        // All passed
        printf("  PASSED: (2-word self-clearing async determinism)\n");
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        goto scd_done;

scd_cleanup:
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        return 1;

scd_done:;
    }

    // 4-word regular kernel determinism (test 33's chain): 10 runs
    //
    // Runs launchPipelinedReduction with use_async_copy=false 10 times on the
    // 4-word cross-column chain from test 33, verifying all runs produce
    // identical pivots AND column data.
    //
    // Chain (same as test 33, race-free):
    // Col 0: [0, 0, 0, 0b100] (pivot=194, word 3).  pivot_to_col[194]=1.
    // Col 1: [0, 0b010, 0, 0]  (pivot=65,  word 1).  pivot_to_col[65]=2.
    // Col 2: [0b001, 0, 0, 0]  (pivot=0,   word 0).  pivot_to_col[0]=-1.
    //
    // Expected: all 10 runs produce identical new_pivots AND column data.
    // Expected pivots: {194, 65, 0} (data returns to initial after 16 even iter.)
    {
        constexpr int num_words = 4;
        constexpr int num_cols = 3;
        constexpr int num_runs = 10;

        const uint64_t h_cols[num_cols * num_words] = {
            0x0000,   0x0000,   0x0000, 0b100ULL, // Col 0: pivot=194
            0x0000,   0b010ULL, 0x0000, 0x0000,   // Col 1: pivot=65
            0b001ULL, 0x0000,   0x0000, 0x0000    // Col 2: pivot=0  (unclaimed)
        };
        const int h_pivot_to_col[256] = {};
        const_cast<int *>(h_pivot_to_col)[0] = -1;
        const_cast<int *>(h_pivot_to_col)[65] = 2;
        const_cast<int *>(h_pivot_to_col)[194] = 1;
        const int h_col_pivots[num_cols] = {194, 65, 0};

        const size_t col_bytes = sizeof(h_cols);
        const size_t ptc_bytes = sizeof(h_pivot_to_col);
        const size_t idx_bytes = num_cols * sizeof(int);

        // Device allocations
        uint64_t *d_cols;
        int *d_pivot_to_col, *d_col_pivots, *d_new_pivots;
        check_cuda(cudaMalloc(&d_cols, col_bytes), "mw4r d_cols");
        check_cuda(cudaMalloc(&d_pivot_to_col, ptc_bytes), "mw4r d_ptc");
        check_cuda(cudaMalloc(&d_col_pivots, idx_bytes), "mw4r d_cp");
        check_cuda(cudaMalloc(&d_new_pivots, idx_bytes), "mw4r d_np");
        check_cuda(cudaMemcpy(d_pivot_to_col, h_pivot_to_col, ptc_bytes, cudaMemcpyHostToDevice),
                   "mw4r h2d ptc");
        check_cuda(cudaMemcpy(d_col_pivots, h_col_pivots, idx_bytes, cudaMemcpyHostToDevice),
                   "mw4r h2d cp");

        // Reference storage (from run 0)
        int ref_pivots[num_cols];
        uint64_t ref_data[num_cols * num_words];

        for (int run = 0; run < num_runs; ++run)
        {
            check_cuda(cudaMemcpy(d_cols, h_cols, col_bytes, cudaMemcpyHostToDevice),
                       "mw4r h2d cols");
            check_cuda(cudaMemset(d_new_pivots, 0xFF, idx_bytes), "mw4r memset new_pivots");

            nerve::persistence::gpu::launchPipelinedReduction(
                d_cols, d_col_pivots, d_pivot_to_col, num_words, num_cols, d_new_pivots, 0, false);

            int run_pivots[num_cols];
            uint64_t run_data[num_cols * num_words];
            check_cuda(
                cudaMemcpy(run_pivots, d_new_pivots, sizeof(run_pivots), cudaMemcpyDeviceToHost),
                "mw4r d2h pivots");
            check_cuda(cudaMemcpy(run_data, d_cols, sizeof(run_data), cudaMemcpyDeviceToHost),
                       "mw4r d2h cols");

            if (run == 0)
            {
                for (int i = 0; i < num_cols; ++i)
                    ref_pivots[i] = run_pivots[i];
                for (int i = 0; i < num_cols * num_words; ++i)
                    ref_data[i] = run_data[i];
            }
            else
            {
                bool ok = true;
                for (int i = 0; i < num_cols; ++i)
                    if (run_pivots[i] != ref_pivots[i])
                    {
                        fprintf(stderr, "mw4r[run=%d] pivot[%d] = %d, ref %d\n", run, i,
                                run_pivots[i], ref_pivots[i]);
                        ok = false;
                    }
                for (int i = 0; i < num_cols * num_words; ++i)
                    if (run_data[i] != ref_data[i])
                    {
                        fprintf(stderr, "mw4r[run=%d] data[%d] = 0x%lx, ref 0x%lx\n", run, i,
                                run_data[i], ref_data[i]);
                        ok = false;
                    }
                if (!ok)
                    goto mw4r_cleanup;
            }
        }

        // All passed
        printf("  PASSED: (4-word regular kernel determinism)\n");
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        goto mw4r_done;

mw4r_cleanup:
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        return 1;

mw4r_done:;
    }

    // 2-word unclaimed pivot regular kernel determinism: 10 runs
    //
    // Runs launchPipelinedReduction with use_async_copy=false 10 times on the
    // 2-word unclaimed pivot chain from test 35, verifying all runs produce
    // identical pivots AND column data.
    //
    // Chain (same as test 35, race-free):
    // Col 0: [0b100, 0x0000] (pivot=2).   pivot_to_col[2]=1  -> XOR with Col 1
    // Col 1: [0b010, 0x0000] (pivot=1).   pivot_to_col[1]=2  -> XOR with Col 2
    // Col 2: [0x0000, 0b001 ] (pivot=64).  pivot_to_col[64]=-1 -> Break (unclaimed)
    //
    // Both Col 1 and Col 0 converge to unclaimed pivot=64 after 1 iteration.
    // Expected pivots: {64, 64, 64}
    {
        constexpr int num_words = 2;
        constexpr int num_cols = 3;
        constexpr int num_runs = 10;

        const uint64_t h_cols[num_cols * num_words] = {
            0b100ULL, 0x0000,  // Col 0: pivot=2
            0b010ULL, 0x0000,  // Col 1: pivot=1
            0x0000,   0b001ULL // Col 2: pivot=64 (unclaimed)
        };
        const int h_pivot_to_col[128] = {};
        const_cast<int *>(h_pivot_to_col)[2] = 1;
        const_cast<int *>(h_pivot_to_col)[1] = 2;
        const_cast<int *>(h_pivot_to_col)[64] = -1;
        const int h_col_pivots[num_cols] = {2, 1, 64};

        const size_t col_bytes = sizeof(h_cols);
        const size_t ptc_bytes = sizeof(h_pivot_to_col);
        const size_t idx_bytes = num_cols * sizeof(int);

        // Device allocations
        uint64_t *d_cols;
        int *d_pivot_to_col, *d_col_pivots, *d_new_pivots;
        check_cuda(cudaMalloc(&d_cols, col_bytes), "ucr d_cols");
        check_cuda(cudaMalloc(&d_pivot_to_col, ptc_bytes), "ucr d_ptc");
        check_cuda(cudaMalloc(&d_col_pivots, idx_bytes), "ucr d_cp");
        check_cuda(cudaMalloc(&d_new_pivots, idx_bytes), "ucr d_np");
        check_cuda(cudaMemcpy(d_pivot_to_col, h_pivot_to_col, ptc_bytes, cudaMemcpyHostToDevice),
                   "ucr h2d ptc");
        check_cuda(cudaMemcpy(d_col_pivots, h_col_pivots, idx_bytes, cudaMemcpyHostToDevice),
                   "ucr h2d cp");

        // Reference storage (from run 0)
        int ref_pivots[num_cols];
        uint64_t ref_data[num_cols * num_words];

        for (int run = 0; run < num_runs; ++run)
        {
            check_cuda(cudaMemcpy(d_cols, h_cols, col_bytes, cudaMemcpyHostToDevice),
                       "ucr h2d cols");
            check_cuda(cudaMemset(d_new_pivots, 0xFF, idx_bytes), "ucr memset new_pivots");

            nerve::persistence::gpu::launchPipelinedReduction(
                d_cols, d_col_pivots, d_pivot_to_col, num_words, num_cols, d_new_pivots, 0, false);

            int run_pivots[num_cols];
            uint64_t run_data[num_cols * num_words];
            check_cuda(
                cudaMemcpy(run_pivots, d_new_pivots, sizeof(run_pivots), cudaMemcpyDeviceToHost),
                "ucr d2h pivots");
            check_cuda(cudaMemcpy(run_data, d_cols, sizeof(run_data), cudaMemcpyDeviceToHost),
                       "ucr d2h cols");

            if (run == 0)
            {
                for (int i = 0; i < num_cols; ++i)
                    ref_pivots[i] = run_pivots[i];
                for (int i = 0; i < num_cols * num_words; ++i)
                    ref_data[i] = run_data[i];
            }
            else
            {
                bool ok = true;
                for (int i = 0; i < num_cols; ++i)
                    if (run_pivots[i] != ref_pivots[i])
                    {
                        fprintf(stderr, "ucr[run=%d] pivot[%d] = %d, ref %d\n", run, i,
                                run_pivots[i], ref_pivots[i]);
                        ok = false;
                    }
                for (int i = 0; i < num_cols * num_words; ++i)
                    if (run_data[i] != ref_data[i])
                    {
                        fprintf(stderr, "ucr[run=%d] data[%d] = 0x%lx, ref 0x%lx\n", run, i,
                                run_data[i], ref_data[i]);
                        ok = false;
                    }
                if (!ok)
                    goto ucr_cleanup;
            }
        }

        // All passed
        printf("  PASSED: (2-word unclaimed pivot regular kernel determinism)\n");
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        goto ucr_done;

ucr_cleanup:
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        return 1;

ucr_done:;
    }

    // 3-word async determinism (test 26's chain): 10 runs, identical results
    //
    // Runs launchPipelinedReduction with use_async_copy=true 10 times on the
    // 3-word cross-column chain from test 26, verifying all runs produce
    // identical pivots AND column data.
    //
    // Chain (same as test 26, race-free):
    // Col 0: [0, 0, 0b100] (pivot=130, word 2).  pivot_to_col[130]=1.
    // Col 1: [0, 0b100, 0]  (pivot=66,  word 1).  pivot_to_col[66]=2.
    // Col 2: [0b001, 0, 0]  (pivot=0,   word 0).  pivot_to_col[0]=-1.
    //
    // Expected: all 10 runs produce identical new_pivots AND column data.
    // Expected pivots: {130, 66, 0} (data returns to initial after 12 even iter.)
    {
        constexpr int num_words = 3;
        constexpr int num_cols = 3;
        constexpr int num_runs = 10;

        const uint64_t h_cols[num_cols * num_words] = {
            0x0000ULL, 0x0000ULL, 0b100ULL,  // Col 0: pivot=130
            0x0000ULL, 0b100ULL,  0x0000ULL, // Col 1: pivot=66
            0b001ULL,  0x0000ULL, 0x0000ULL  // Col 2: pivot=0 (unclaimed)
        };
        const int h_pivot_to_col[192] = {};
        const_cast<int *>(h_pivot_to_col)[0] = -1;
        const_cast<int *>(h_pivot_to_col)[66] = 2;
        const_cast<int *>(h_pivot_to_col)[130] = 1;
        const int h_col_pivots[num_cols] = {130, 66, 0};

        const size_t col_bytes = sizeof(h_cols);
        const size_t ptc_bytes = sizeof(h_pivot_to_col);
        const size_t idx_bytes = num_cols * sizeof(int);

        // Device allocations
        uint64_t *d_cols;
        int *d_pivot_to_col, *d_col_pivots, *d_new_pivots;
        check_cuda(cudaMalloc(&d_cols, col_bytes), "mw3a d_cols");
        check_cuda(cudaMalloc(&d_pivot_to_col, ptc_bytes), "mw3a d_ptc");
        check_cuda(cudaMalloc(&d_col_pivots, idx_bytes), "mw3a d_cp");
        check_cuda(cudaMalloc(&d_new_pivots, idx_bytes), "mw3a d_np");
        check_cuda(cudaMemcpy(d_pivot_to_col, h_pivot_to_col, ptc_bytes, cudaMemcpyHostToDevice),
                   "mw3a h2d ptc");
        check_cuda(cudaMemcpy(d_col_pivots, h_col_pivots, idx_bytes, cudaMemcpyHostToDevice),
                   "mw3a h2d cp");

        int ref_pivots[num_cols];
        uint64_t ref_data[num_cols * num_words];

        for (int run = 0; run < num_runs; ++run)
        {
            check_cuda(cudaMemcpy(d_cols, h_cols, col_bytes, cudaMemcpyHostToDevice),
                       "mw3a h2d cols");
            check_cuda(cudaMemset(d_new_pivots, 0xFF, idx_bytes), "mw3a memset new_pivots");

            nerve::persistence::gpu::launchPipelinedReduction(
                d_cols, d_col_pivots, d_pivot_to_col, num_words, num_cols, d_new_pivots, 0, true);

            int run_pivots[num_cols];
            uint64_t run_data[num_cols * num_words];
            check_cuda(
                cudaMemcpy(run_pivots, d_new_pivots, sizeof(run_pivots), cudaMemcpyDeviceToHost),
                "mw3a d2h pivots");
            check_cuda(cudaMemcpy(run_data, d_cols, sizeof(run_data), cudaMemcpyDeviceToHost),
                       "mw3a d2h cols");

            if (run == 0)
            {
                for (int i = 0; i < num_cols; ++i)
                    ref_pivots[i] = run_pivots[i];
                for (int i = 0; i < num_cols * num_words; ++i)
                    ref_data[i] = run_data[i];
            }
            else
            {
                bool ok = true;
                for (int i = 0; i < num_cols; ++i)
                    if (run_pivots[i] != ref_pivots[i])
                    {
                        fprintf(stderr, "mw3a[run=%d] pivot[%d] = %d, ref %d\n", run, i,
                                run_pivots[i], ref_pivots[i]);
                        ok = false;
                    }
                for (int i = 0; i < num_cols * num_words; ++i)
                    if (run_data[i] != ref_data[i])
                    {
                        fprintf(stderr, "mw3a[run=%d] data[%d] = 0x%lx, ref 0x%lx\n", run, i,
                                run_data[i], ref_data[i]);
                        ok = false;
                    }
                if (!ok)
                    goto mw3a_cleanup;
            }
        }

        printf("  PASSED: (3-word async determinism)\n");
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        goto mw3a_done;

mw3a_cleanup:
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        return 1;

mw3a_done:;
    }

    // 3-word regular kernel determinism (test 26's chain): 10 runs
    //
    // Runs launchPipelinedReduction with use_async_copy=false 10 times on the
    // 3-word cross-column chain from test 26, verifying all runs produce
    // identical pivots AND column data.
    //
    // Chain (same as test 26, race-free):
    // Col 0: [0, 0, 0b100] (pivot=130, word 2).  pivot_to_col[130]=1.
    // Col 1: [0, 0b100, 0]  (pivot=66,  word 1).  pivot_to_col[66]=2.
    // Col 2: [0b001, 0, 0]  (pivot=0,   word 0).  pivot_to_col[0]=-1.
    //
    // Expected: all 10 runs produce identical new_pivots AND column data.
    // Expected pivots: {130, 66, 0} (data returns to initial after 12 even iter.)
    {
        constexpr int num_words = 3;
        constexpr int num_cols = 3;
        constexpr int num_runs = 10;

        const uint64_t h_cols[num_cols * num_words] = {
            0x0000,   0x0000,   0b100ULL, // Col 0: pivot=130
            0x0000,   0b100ULL, 0x0000,   // Col 1: pivot=66
            0b001ULL, 0x0000,   0x0000    // Col 2: pivot=0  (unclaimed)
        };
        const int h_pivot_to_col[192] = {};
        const_cast<int *>(h_pivot_to_col)[0] = -1;  // unclaimed leaf (already -1 by {})
        const_cast<int *>(h_pivot_to_col)[66] = 2;  // col 2 claims pivot 66
        const_cast<int *>(h_pivot_to_col)[130] = 1; // col 1 claims pivot 130
        const int h_col_pivots[num_cols] = {130, 66, 0};

        const size_t col_bytes = sizeof(h_cols);
        const size_t ptc_bytes = sizeof(h_pivot_to_col);
        const size_t idx_bytes = num_cols * sizeof(int);

        // Device allocations
        uint64_t *d_cols;
        int *d_pivot_to_col, *d_col_pivots, *d_new_pivots;
        check_cuda(cudaMalloc(&d_cols, col_bytes), "mw3r d_cols");
        check_cuda(cudaMalloc(&d_pivot_to_col, ptc_bytes), "mw3r d_ptc");
        check_cuda(cudaMalloc(&d_col_pivots, idx_bytes), "mw3r d_cp");
        check_cuda(cudaMalloc(&d_new_pivots, idx_bytes), "mw3r d_np");
        check_cuda(cudaMemcpy(d_pivot_to_col, h_pivot_to_col, ptc_bytes, cudaMemcpyHostToDevice),
                   "mw3r h2d ptc");
        check_cuda(cudaMemcpy(d_col_pivots, h_col_pivots, idx_bytes, cudaMemcpyHostToDevice),
                   "mw3r h2d cp");

        // Reference storage (from run 0)
        int ref_pivots[num_cols];
        uint64_t ref_data[num_cols * num_words];

        for (int run = 0; run < num_runs; ++run)
        {
            check_cuda(cudaMemcpy(d_cols, h_cols, col_bytes, cudaMemcpyHostToDevice),
                       "mw3r h2d cols");
            check_cuda(cudaMemset(d_new_pivots, 0xFF, idx_bytes), "mw3r memset new_pivots");

            nerve::persistence::gpu::launchPipelinedReduction(
                d_cols, d_col_pivots, d_pivot_to_col, num_words, num_cols, d_new_pivots, 0, false);

            int run_pivots[num_cols];
            uint64_t run_data[num_cols * num_words];
            check_cuda(
                cudaMemcpy(run_pivots, d_new_pivots, sizeof(run_pivots), cudaMemcpyDeviceToHost),
                "mw3r d2h pivots");
            check_cuda(cudaMemcpy(run_data, d_cols, sizeof(run_data), cudaMemcpyDeviceToHost),
                       "mw3r d2h cols");

            if (run == 0)
            {
                for (int i = 0; i < num_cols; ++i)
                    ref_pivots[i] = run_pivots[i];
                for (int i = 0; i < num_cols * num_words; ++i)
                    ref_data[i] = run_data[i];
            }
            else
            {
                bool ok = true;
                for (int i = 0; i < num_cols; ++i)
                    if (run_pivots[i] != ref_pivots[i])
                    {
                        fprintf(stderr, "mw3r[run=%d] pivot[%d] = %d, ref %d\n", run, i,
                                run_pivots[i], ref_pivots[i]);
                        ok = false;
                    }
                for (int i = 0; i < num_cols * num_words; ++i)
                    if (run_data[i] != ref_data[i])
                    {
                        fprintf(stderr, "mw3r[run=%d] data[%d] = 0x%lx, ref 0x%lx\n", run, i,
                                run_data[i], ref_data[i]);
                        ok = false;
                    }
                if (!ok)
                    goto mw3r_cleanup;
            }
        }

        // All passed
        printf("  PASSED: (3-word regular kernel determinism)\n");
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        goto mw3r_done;

mw3r_cleanup:
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        return 1;

mw3r_done:;
    }
    // 2-word cross-column syncthreads determinism (test 23's chain): 10 runs
    //
    // Runs syncthreadsReductionRef 10 times on the 2-word cross-column chain
    // from test 23, verifying all runs produce identical pivots AND column data.
    //
    // Chain (same as test 23, race-free):
    //   Col 0: [0x0000, 0b100] (pivot=66,  word 1).  pivot_to_col[66]=1.
    //   Col 1: [0b010,  0x0000] (pivot=1,   word 0).  pivot_to_col[1]=2.
    //   Col 2: [0x0000, 0b001] (pivot=64,  word 1).  pivot_to_col[64]=-1.
    //
    // After XOR with Col 2, Col 1's MSB moves to word 1 (pivot=64, unclaimed).
    // All variants converge to {66, 64, 64} after 1 iteration.
    // Expected: all 10 runs produce identical new_pivots AND column data.
    {
        constexpr int num_words = 2;
        constexpr int num_cols = 3;
        constexpr int num_runs = 10;

        const uint64_t h_cols[num_cols * num_words] = {
            0x0000ULL, 0b100ULL,  // Col 0: pivot=66
            0b010ULL,  0x0000ULL, // Col 1: pivot=1
            0x0000ULL, 0b001ULL   // Col 2: pivot=64 (unclaimed)
        };
        const int h_col_pivots[num_cols] = {66, 1, 64};
        const int h_pivot_to_col[128] = {};
        const_cast<int *>(h_pivot_to_col)[66] = 1; // col 1 claims pivot 66
        const_cast<int *>(h_pivot_to_col)[1] = 2;  // col 2 claims pivot 1
        // pivot_to_col[64] stays -1 (unclaimed)

        const size_t col_bytes = sizeof(h_cols);
        const size_t ptc_bytes = sizeof(h_pivot_to_col);
        const size_t idx_bytes = num_cols * sizeof(int);

        // Device allocations
        uint64_t *d_cols;
        int *d_pivot_to_col, *d_col_pivots, *d_new_pivots;
        check_cuda(cudaMalloc(&d_cols, col_bytes), "sd2 d_cols");
        check_cuda(cudaMalloc(&d_pivot_to_col, ptc_bytes), "sd2 d_ptc");
        check_cuda(cudaMalloc(&d_col_pivots, idx_bytes), "sd2 d_cp");
        check_cuda(cudaMalloc(&d_new_pivots, idx_bytes), "sd2 d_np");
        check_cuda(cudaMemcpy(d_pivot_to_col, h_pivot_to_col, ptc_bytes, cudaMemcpyHostToDevice),
                   "sd2 h2d ptc");
        check_cuda(cudaMemcpy(d_col_pivots, h_col_pivots, idx_bytes, cudaMemcpyHostToDevice),
                   "sd2 h2d cp");

        // Reference storage (from run 0)
        int ref_pivots[num_cols];
        uint64_t ref_data[num_cols * num_words];

        const int block_threads = max(32, num_cols * 32);

        for (int run = 0; run < num_runs; ++run)
        {
            check_cuda(cudaMemcpy(d_cols, h_cols, col_bytes, cudaMemcpyHostToDevice),
                       "sd2 h2d cols");
            check_cuda(cudaMemset(d_new_pivots, 0xFF, idx_bytes), "sd2 memset new_pivots");

            syncthreadsReductionRef<<<1, block_threads>>>(d_cols, d_col_pivots, d_pivot_to_col,
                                                          num_words, num_cols, d_new_pivots);
            check_cuda(cudaDeviceSynchronize(), "sd2 sync");

            int run_pivots[num_cols];
            uint64_t run_data[num_cols * num_words];
            check_cuda(
                cudaMemcpy(run_pivots, d_new_pivots, sizeof(run_pivots), cudaMemcpyDeviceToHost),
                "sd2 d2h pivots");
            check_cuda(cudaMemcpy(run_data, d_cols, sizeof(run_data), cudaMemcpyDeviceToHost),
                       "sd2 d2h cols");

            if (run == 0)
            {
                for (int i = 0; i < num_cols; ++i)
                    ref_pivots[i] = run_pivots[i];
                for (int i = 0; i < num_cols * num_words; ++i)
                    ref_data[i] = run_data[i];
            }
            else
            {
                bool ok = true;
                for (int i = 0; i < num_cols; ++i)
                    if (run_pivots[i] != ref_pivots[i])
                    {
                        fprintf(stderr, "sd2[run=%d] pivot[%d] = %d, ref %d\n", run, i,
                                run_pivots[i], ref_pivots[i]);
                        ok = false;
                    }
                for (int i = 0; i < num_cols * num_words; ++i)
                    if (run_data[i] != ref_data[i])
                    {
                        fprintf(stderr, "sd2[run=%d] data[%d] = 0x%lx, ref 0x%lx\n", run, i,
                                run_data[i], ref_data[i]);
                        ok = false;
                    }
                if (!ok)
                    goto sd2_cleanup;
            }
        }

        printf("  PASSED: (2-word cross-column syncthreads determinism)\n");
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        goto sd2_done;

sd2_cleanup:
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        return 1;

sd2_done:;
    }
    // 4-word unclaimed pivot regular kernel determinism (test 40's chain): 10 runs
    //
    // Runs launchPipelinedReduction with use_async_copy=false 10 times on the
    // 4-word unclaimed pivot chain from test 40, verifying all runs produce
    // identical pivots AND column data.
    //
    // Chain (same as test 40, race-free):
    //   Col 0: [0b100, 0, 0, 0] (pivot=2,   word 0).  pivot_to_col[2]=1.
    //   Col 1: [0, 0, 0b010, 0] (pivot=128, word 2).  pivot_to_col[128]=2.
    //   Col 2: [0, 0, 0, 0b001] (pivot=192, word 3).  pivot_to_col[192]=-1.
    //
    // Both Col 1 and Col 0 converge to unclaimed pivot=192 after 1 iteration.
    // Expected: all 10 runs produce identical new_pivots AND column data.
    {
        constexpr int num_words = 4;
        constexpr int num_cols = 3;
        constexpr int num_runs = 10;

        const uint64_t h_cols[num_cols * num_words] = {
            0b100ULL, 0x0000, 0x0000,   0x0000,  // Col 0: pivot=2
            0x0000,   0x0000, 0b010ULL, 0x0000,  // Col 1: pivot=128
            0x0000,   0x0000, 0x0000,   0b001ULL // Col 2: pivot=192 (unclaimed)
        };
        const int h_pivot_to_col[256] = {};
        const_cast<int *>(h_pivot_to_col)[2] = 1;
        const_cast<int *>(h_pivot_to_col)[128] = 2;
        const_cast<int *>(h_pivot_to_col)[192] = -1;
        const int h_col_pivots[num_cols] = {2, 128, 192};

        const size_t col_bytes = sizeof(h_cols);
        const size_t ptc_bytes = sizeof(h_pivot_to_col);
        const size_t idx_bytes = num_cols * sizeof(int);

        // Device allocations
        uint64_t *d_cols;
        int *d_pivot_to_col, *d_col_pivots, *d_new_pivots;
        check_cuda(cudaMalloc(&d_cols, col_bytes), "uc4r d_cols");
        check_cuda(cudaMalloc(&d_pivot_to_col, ptc_bytes), "uc4r d_ptc");
        check_cuda(cudaMalloc(&d_col_pivots, idx_bytes), "uc4r d_cp");
        check_cuda(cudaMalloc(&d_new_pivots, idx_bytes), "uc4r d_np");
        check_cuda(cudaMemcpy(d_pivot_to_col, h_pivot_to_col, ptc_bytes, cudaMemcpyHostToDevice),
                   "uc4r h2d ptc");
        check_cuda(cudaMemcpy(d_col_pivots, h_col_pivots, idx_bytes, cudaMemcpyHostToDevice),
                   "uc4r h2d cp");

        // Reference storage (from run 0)
        int ref_pivots[num_cols];
        uint64_t ref_data[num_cols * num_words];

        for (int run = 0; run < num_runs; ++run)
        {
            check_cuda(cudaMemcpy(d_cols, h_cols, col_bytes, cudaMemcpyHostToDevice),
                       "uc4r h2d cols");
            check_cuda(cudaMemset(d_new_pivots, 0xFF, idx_bytes), "uc4r memset new_pivots");

            nerve::persistence::gpu::launchPipelinedReduction(
                d_cols, d_col_pivots, d_pivot_to_col, num_words, num_cols, d_new_pivots, 0, false);

            int run_pivots[num_cols];
            uint64_t run_data[num_cols * num_words];
            check_cuda(
                cudaMemcpy(run_pivots, d_new_pivots, sizeof(run_pivots), cudaMemcpyDeviceToHost),
                "uc4r d2h pivots");
            check_cuda(cudaMemcpy(run_data, d_cols, sizeof(run_data), cudaMemcpyDeviceToHost),
                       "uc4r d2h cols");

            if (run == 0)
            {
                for (int i = 0; i < num_cols; ++i)
                    ref_pivots[i] = run_pivots[i];
                for (int i = 0; i < num_cols * num_words; ++i)
                    ref_data[i] = run_data[i];
            }
            else
            {
                bool ok = true;
                for (int i = 0; i < num_cols; ++i)
                    if (run_pivots[i] != ref_pivots[i])
                    {
                        fprintf(stderr, "uc4r[run=%d] pivot[%d] = %d, ref %d\n", run, i,
                                run_pivots[i], ref_pivots[i]);
                        ok = false;
                    }
                for (int i = 0; i < num_cols * num_words; ++i)
                    if (run_data[i] != ref_data[i])
                    {
                        fprintf(stderr, "uc4r[run=%d] data[%d] = 0x%lx, ref 0x%lx\n", run, i,
                                run_data[i], ref_data[i]);
                        ok = false;
                    }
                if (!ok)
                    goto uc4r_cleanup;
            }
        }

        printf("  PASSED: (4-word unclaimed pivot regular kernel determinism)\n");
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        goto uc4r_done;

uc4r_cleanup:
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        return 1;

uc4r_done:;
    }
    // 2-word unclaimed pivot syncthreads determinism (test 35's chain): 10 runs
    //
    // Runs syncthreadsReductionRef 10 times on the 2-word unclaimed pivot chain
    // from test 35, verifying all runs produce identical pivots AND column data.
    //
    // Chain (same as test 35, race-free):
    //   Col 0: [0b100, 0x0000] (pivot=2,   word 0).  pivot_to_col[2]=1.
    //   Col 1: [0b010, 0x0000] (pivot=1,   word 0).  pivot_to_col[1]=2.
    //   Col 2: [0x0000, 0b001 ] (pivot=64,  word 1).  pivot_to_col[64]=-1.
    //
    // Col 1 XOR with Col 2 -> [0b010, 0b001], MSB=64 (unclaimed). Break in 1 iter.
    // Col 0 XOR with Col 1 -> [0b110, 0b001], MSB=64 (unclaimed). Break in 1 iter.
    // Expected: all 10 runs produce identical new_pivots AND column data.
    // Expected pivots: {64, 64, 64}
    {
        constexpr int num_words = 2;
        constexpr int num_cols = 3;
        constexpr int num_runs = 10;

        const uint64_t h_cols[num_cols * num_words] = {
            0b100ULL, 0x0000,  // Col 0: pivot=2
            0b010ULL, 0x0000,  // Col 1: pivot=1
            0x0000,   0b001ULL // Col 2: pivot=64 (unclaimed)
        };
        const int h_pivot_to_col[128] = {};
        const_cast<int *>(h_pivot_to_col)[2] = 1;
        const_cast<int *>(h_pivot_to_col)[1] = 2;
        const_cast<int *>(h_pivot_to_col)[64] = -1;
        const int h_col_pivots[num_cols] = {2, 1, 64};

        const int block_threads = (32 > num_cols * 32) ? 32 : num_cols * 32;
        const size_t col_bytes = sizeof(h_cols);
        const size_t ptc_bytes = sizeof(h_pivot_to_col);
        const size_t idx_bytes = sizeof(int) * num_cols;

        uint64_t *d_cols;
        int *d_pivot_to_col, *d_col_pivots, *d_new_pivots;
        check_cuda(cudaMalloc(&d_cols, col_bytes), "uc2s d_cols");
        check_cuda(cudaMalloc(&d_pivot_to_col, ptc_bytes), "uc2s d_ptc");
        check_cuda(cudaMalloc(&d_col_pivots, idx_bytes), "uc2s d_cp");
        check_cuda(cudaMalloc(&d_new_pivots, idx_bytes), "uc2s d_np");
        check_cuda(cudaMemcpy(d_pivot_to_col, h_pivot_to_col, ptc_bytes, cudaMemcpyHostToDevice),
                   "uc2s h2d ptc");
        check_cuda(cudaMemcpy(d_col_pivots, h_col_pivots, idx_bytes, cudaMemcpyHostToDevice),
                   "uc2s h2d cp");

        int ref_pivots[num_cols] = {};
        uint64_t ref_data[num_cols * num_words] = {};

        for (int run = 0; run < num_runs; ++run)
        {
            check_cuda(cudaMemcpy(d_cols, h_cols, col_bytes, cudaMemcpyHostToDevice),
                       "uc2s h2d cols");
            check_cuda(cudaMemset(d_new_pivots, 0xFF, idx_bytes), "uc2s memset new_pivots");

            syncthreadsReductionRef<<<1, block_threads>>>(d_cols, d_col_pivots, d_pivot_to_col,
                                                          num_words, num_cols, d_new_pivots);
            check_cuda(cudaDeviceSynchronize(), "uc2s sync");

            int run_pivots[num_cols] = {};
            check_cuda(cudaMemcpy(run_pivots, d_new_pivots, idx_bytes, cudaMemcpyDeviceToHost),
                       "uc2s d2h pivots");

            uint64_t run_data[num_cols * num_words] = {};
            check_cuda(cudaMemcpy(run_data, d_cols, col_bytes, cudaMemcpyDeviceToHost),
                       "uc2s d2h data");

            if (run == 0)
            {
                for (int i = 0; i < num_cols; ++i)
                    ref_pivots[i] = run_pivots[i];
                for (int i = 0; i < num_cols * num_words; ++i)
                    ref_data[i] = run_data[i];
            }
            else
            {
                bool ok = true;
                for (int i = 0; i < num_cols; ++i)
                    if (run_pivots[i] != ref_pivots[i])
                    {
                        fprintf(stderr, "uc2s[run=%d] pivot[%d] = %d, ref %d\n", run, i,
                                run_pivots[i], ref_pivots[i]);
                        ok = false;
                    }
                for (int i = 0; i < num_cols * num_words; ++i)
                    if (run_data[i] != ref_data[i])
                    {
                        fprintf(stderr, "uc2s[run=%d] data[%d] = 0x%lx, ref 0x%lx\n", run, i,
                                run_data[i], ref_data[i]);
                        ok = false;
                    }
                if (!ok)
                    goto uc2s_cleanup;
            }
        }

        printf("  PASSED: (2-word unclaimed pivot syncthreads determinism)\n");
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        goto uc2s_done;

uc2s_cleanup:
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        return 1;

uc2s_done:;
    }
    // 3-word cross-column syncthreads determinism (test 26's chain): 10 runs
    //
    // Runs syncthreadsReductionRef 10 times on the 3-word cross-column chain
    // from test 26, verifying all runs produce identical pivots AND column data.
    //
    // Chain (same as test 26):
    //   Col 0: [0, 0, 0b100] (pivot=130, word 2).  pivot_to_col[130]=1.
    //   Col 1: [0, 0b100, 0] (pivot=66,   word 1).  pivot_to_col[66]=2.
    //   Col 2: [0b001, 0, 0]  (pivot=0,    word 0).  pivot_to_col[0]=-1.
    //
    // Returns to initial after 12 even iterations.
    // Expected: all 10 runs produce identical new_pivots AND column data.
    // Expected pivots: {130, 66, 0}
    {
        constexpr int num_words = 3;
        constexpr int num_cols = 3;
        constexpr int num_runs = 10;

        const uint64_t h_cols[num_cols * num_words] = {
            0x0000ULL, 0x0000ULL, 0b100ULL,  // col 0: pivot=130
            0x0000ULL, 0b100ULL,  0x0000ULL, // col 1: pivot=66
            0b001ULL,  0x0000ULL, 0x0000ULL  // col 2: pivot=0
        };
        const int h_pivot_to_col[192] = {};
        const_cast<int *>(h_pivot_to_col)[130] = 1;
        const_cast<int *>(h_pivot_to_col)[66] = 2;
        const int h_col_pivots[num_cols] = {130, 66, 0};

        const int block_threads = (32 > num_cols * 32) ? 32 : num_cols * 32;
        const size_t col_bytes = sizeof(h_cols);
        const size_t ptc_bytes = sizeof(h_pivot_to_col);
        const size_t idx_bytes = sizeof(int) * num_cols;

        uint64_t *d_cols;
        int *d_pivot_to_col, *d_col_pivots, *d_new_pivots;
        check_cuda(cudaMalloc(&d_cols, col_bytes), "xc3 d_cols");
        check_cuda(cudaMalloc(&d_pivot_to_col, ptc_bytes), "xc3 d_ptc");
        check_cuda(cudaMalloc(&d_col_pivots, idx_bytes), "xc3 d_cp");
        check_cuda(cudaMalloc(&d_new_pivots, idx_bytes), "xc3 d_np");
        check_cuda(cudaMemcpy(d_pivot_to_col, h_pivot_to_col, ptc_bytes, cudaMemcpyHostToDevice),
                   "xc3 h2d ptc");
        check_cuda(cudaMemcpy(d_col_pivots, h_col_pivots, idx_bytes, cudaMemcpyHostToDevice),
                   "xc3 h2d cp");

        int ref_pivots[num_cols] = {};
        uint64_t ref_data[num_cols * num_words] = {};

        for (int run = 0; run < num_runs; ++run)
        {
            check_cuda(cudaMemcpy(d_cols, h_cols, col_bytes, cudaMemcpyHostToDevice),
                       "xc3 h2d cols");
            check_cuda(cudaMemset(d_new_pivots, 0xFF, idx_bytes), "xc3 memset new_pivots");

            syncthreadsReductionRef<<<1, block_threads>>>(d_cols, d_col_pivots, d_pivot_to_col,
                                                          num_words, num_cols, d_new_pivots);
            check_cuda(cudaDeviceSynchronize(), "xc3 sync");

            int run_pivots[num_cols] = {};
            check_cuda(cudaMemcpy(run_pivots, d_new_pivots, idx_bytes, cudaMemcpyDeviceToHost),
                       "xc3 d2h pivots");

            uint64_t run_data[num_cols * num_words] = {};
            check_cuda(cudaMemcpy(run_data, d_cols, col_bytes, cudaMemcpyDeviceToHost),
                       "xc3 d2h data");

            if (run == 0)
            {
                for (int i = 0; i < num_cols; ++i)
                    ref_pivots[i] = run_pivots[i];
                for (int i = 0; i < num_cols * num_words; ++i)
                    ref_data[i] = run_data[i];
            }
            else
            {
                bool ok = true;
                for (int i = 0; i < num_cols; ++i)
                    if (run_pivots[i] != ref_pivots[i])
                    {
                        fprintf(stderr, "xc3[run=%d] pivot[%d] = %d, ref %d\n", run, i,
                                run_pivots[i], ref_pivots[i]);
                        ok = false;
                    }
                for (int i = 0; i < num_cols * num_words; ++i)
                    if (run_data[i] != ref_data[i])
                    {
                        fprintf(stderr, "xc3[run=%d] data[%d] = 0x%lx, ref 0x%lx\n", run, i,
                                run_data[i], ref_data[i]);
                        ok = false;
                    }
                if (!ok)
                    goto xc3_cleanup;
            }
        }

        printf("  PASSED: (3-word cross-column syncthreads determinism)\n");
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        goto xc3_done;

xc3_cleanup:
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        return 1;

xc3_done:;
    }
    // 2-word self-clearing syncthreads determinism (test 27's chain): 10 runs
    //
    // Runs syncthreadsReductionRef 10 times on the 2-word self-clearing chain
    // from test 27, verifying all runs produce identical pivots AND column data.
    //
    // Chain (same as test 27, race-free):
    //   Col 0: [0b100, 0x0000] (pivot=2,  word 0).  pivot_to_col[2]=0.
    //   Col 1: [0x0000, 0b001] (pivot=64, word 1).  pivot_to_col[64]=1.
    //   Col 2: [0b010, 0x0000] (pivot=1,  word 0).  pivot_to_col[1]=2.
    //
    // Each column claims its own pivot. All columns are zeroed.
    // New pivots are all -1.
    // Expected: all 10 runs produce identical new_pivots AND column data.
    {
        constexpr int num_words = 2;
        constexpr int num_cols = 3;
        constexpr int num_runs = 10;

        const uint64_t h_cols[num_cols * num_words] = {
            0b100ULL,  0x0000ULL, // col 0: pivot=2
            0x0000ULL, 0b001ULL,  // col 1: pivot=64
            0b010ULL,  0x0000ULL  // col 2: pivot=1
        };
        int h_pivot_to_col[128];
        std::memset(h_pivot_to_col, 0xFF, sizeof(h_pivot_to_col));
        h_pivot_to_col[2] = 0;
        h_pivot_to_col[64] = 1;
        h_pivot_to_col[1] = 2;
        const int h_col_pivots[num_cols] = {2, 64, 1};

        const int block_threads = (32 > num_cols * 32) ? 32 : num_cols * 32;
        const size_t col_bytes = sizeof(h_cols);
        const size_t ptc_bytes = sizeof(h_pivot_to_col);
        const size_t idx_bytes = sizeof(int) * num_cols;

        uint64_t *d_cols;
        int *d_pivot_to_col, *d_col_pivots, *d_new_pivots;
        check_cuda(cudaMalloc(&d_cols, col_bytes), "sc2s d_cols");
        check_cuda(cudaMalloc(&d_pivot_to_col, ptc_bytes), "sc2s d_ptc");
        check_cuda(cudaMalloc(&d_col_pivots, idx_bytes), "sc2s d_cp");
        check_cuda(cudaMalloc(&d_new_pivots, idx_bytes), "sc2s d_np");
        check_cuda(cudaMemcpy(d_pivot_to_col, h_pivot_to_col, ptc_bytes, cudaMemcpyHostToDevice),
                   "sc2s h2d ptc");
        check_cuda(cudaMemcpy(d_col_pivots, h_col_pivots, idx_bytes, cudaMemcpyHostToDevice),
                   "sc2s h2d cp");

        int ref_pivots[num_cols] = {};
        uint64_t ref_data[num_cols * num_words] = {};

        for (int run = 0; run < num_runs; ++run)
        {
            check_cuda(cudaMemcpy(d_cols, h_cols, col_bytes, cudaMemcpyHostToDevice),
                       "sc2s h2d cols");
            check_cuda(cudaMemset(d_new_pivots, 0xFF, idx_bytes), "sc2s memset new_pivots");

            syncthreadsReductionRef<<<1, block_threads>>>(d_cols, d_col_pivots, d_pivot_to_col,
                                                          num_words, num_cols, d_new_pivots);
            check_cuda(cudaDeviceSynchronize(), "sc2s sync");

            int run_pivots[num_cols] = {};
            check_cuda(cudaMemcpy(run_pivots, d_new_pivots, idx_bytes, cudaMemcpyDeviceToHost),
                       "sc2s d2h pivots");

            uint64_t run_data[num_cols * num_words] = {};
            check_cuda(cudaMemcpy(run_data, d_cols, col_bytes, cudaMemcpyDeviceToHost),
                       "sc2s d2h data");

            if (run == 0)
            {
                for (int i = 0; i < num_cols; ++i)
                    ref_pivots[i] = run_pivots[i];
                for (int i = 0; i < num_cols * num_words; ++i)
                    ref_data[i] = run_data[i];
            }
            else
            {
                bool ok = true;
                for (int i = 0; i < num_cols; ++i)
                    if (run_pivots[i] != ref_pivots[i])
                    {
                        fprintf(stderr, "sc2s[run=%d] pivot[%d] = %d, ref %d\n", run, i,
                                run_pivots[i], ref_pivots[i]);
                        ok = false;
                    }
                for (int i = 0; i < num_cols * num_words; ++i)
                    if (run_data[i] != ref_data[i])
                    {
                        fprintf(stderr, "sc2s[run=%d] data[%d] = 0x%lx, ref 0x%lx\n", run, i,
                                run_data[i], ref_data[i]);
                        ok = false;
                    }
                if (!ok)
                    goto sc2s_cleanup;
            }
        }

        printf("  PASSED: (2-word self-clearing syncthreads determinism)\n");
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        goto sc2s_done;

sc2s_cleanup:
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        return 1;

sc2s_done:;
    }
    // 2-word self-clearing regular kernel determinism (test 27's chain): 10 runs
    //
    // Runs launchPipelinedReduction with use_async_copy=false 10 times on
    // the 2-word self-clearing chain from test 27, verifying all runs
    // produce identical pivots AND column data.
    //
    // Chain (same as test 27, race-free):
    //   Col 0: [0b100, 0x0000] (pivot=2,  word 0).  pivot_to_col[2]=0.
    //   Col 1: [0x0000, 0b001] (pivot=64, word 1).  pivot_to_col[64]=1.
    //   Col 2: [0b010, 0x0000] (pivot=1,  word 0).  pivot_to_col[1]=2.
    //
    // Each column claims its own pivot. All columns are zeroed.
    // New pivots are all -1.
    // Expected: all 10 runs produce identical new_pivots AND column data.
    {
        constexpr int num_words = 2;
        constexpr int num_cols = 3;
        constexpr int num_runs = 10;

        const uint64_t h_cols[num_cols * num_words] = {
            0b100ULL,  0x0000ULL, // col 0: pivot=2
            0x0000ULL, 0b001ULL,  // col 1: pivot=64
            0b010ULL,  0x0000ULL  // col 2: pivot=1
        };
        int h_pivot_to_col[128];
        std::memset(h_pivot_to_col, 0xFF, sizeof(h_pivot_to_col));
        h_pivot_to_col[2] = 0;
        h_pivot_to_col[64] = 1;
        h_pivot_to_col[1] = 2;
        const int h_col_pivots[num_cols] = {2, 64, 1};

        const size_t col_bytes = sizeof(h_cols);
        const size_t ptc_bytes = sizeof(h_pivot_to_col);
        const size_t idx_bytes = sizeof(int) * num_cols;

        uint64_t *d_cols;
        int *d_pivot_to_col, *d_col_pivots, *d_new_pivots;
        check_cuda(cudaMalloc(&d_cols, col_bytes), "sc2r d_cols");
        check_cuda(cudaMalloc(&d_pivot_to_col, ptc_bytes), "sc2r d_ptc");
        check_cuda(cudaMalloc(&d_col_pivots, idx_bytes), "sc2r d_cp");
        check_cuda(cudaMalloc(&d_new_pivots, idx_bytes), "sc2r d_np");
        check_cuda(cudaMemcpy(d_pivot_to_col, h_pivot_to_col, ptc_bytes, cudaMemcpyHostToDevice),
                   "sc2r h2d ptc");
        check_cuda(cudaMemcpy(d_col_pivots, h_col_pivots, idx_bytes, cudaMemcpyHostToDevice),
                   "sc2r h2d cp");

        int ref_pivots[num_cols] = {};
        uint64_t ref_data[num_cols * num_words] = {};

        for (int run = 0; run < num_runs; ++run)
        {
            check_cuda(cudaMemcpy(d_cols, h_cols, col_bytes, cudaMemcpyHostToDevice),
                       "sc2r h2d cols");
            check_cuda(cudaMemset(d_new_pivots, 0xFF, idx_bytes), "sc2r memset new_pivots");

            nerve::persistence::gpu::launchPipelinedReduction(
                d_cols, d_col_pivots, d_pivot_to_col, num_words, num_cols, d_new_pivots, 0, false);
            check_cuda(cudaDeviceSynchronize(), "sc2r sync");

            int run_pivots[num_cols] = {};
            check_cuda(cudaMemcpy(run_pivots, d_new_pivots, idx_bytes, cudaMemcpyDeviceToHost),
                       "sc2r d2h pivots");

            uint64_t run_data[num_cols * num_words] = {};
            check_cuda(cudaMemcpy(run_data, d_cols, col_bytes, cudaMemcpyDeviceToHost),
                       "sc2r d2h data");

            if (run == 0)
            {
                for (int i = 0; i < num_cols; ++i)
                    ref_pivots[i] = run_pivots[i];
                for (int i = 0; i < num_cols * num_words; ++i)
                    ref_data[i] = run_data[i];
            }
            else
            {
                bool ok = true;
                for (int i = 0; i < num_cols; ++i)
                    if (run_pivots[i] != ref_pivots[i])
                    {
                        fprintf(stderr, "sc2r[run=%d] pivot[%d] = %d, ref %d\n", run, i,
                                run_pivots[i], ref_pivots[i]);
                        ok = false;
                    }
                for (int i = 0; i < num_cols * num_words; ++i)
                    if (run_data[i] != ref_data[i])
                    {
                        fprintf(stderr, "sc2r[run=%d] data[%d] = 0x%lx, ref 0x%lx\n", run, i,
                                run_data[i], ref_data[i]);
                        ok = false;
                    }
                if (!ok)
                    goto sc2r_cleanup;
            }
        }

        printf("  PASSED: (2-word self-clearing regular kernel determinism)\n");
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        goto sc2r_done;

sc2r_cleanup:
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        return 1;

sc2r_done:;
    }
    // 3-word self-clearing syncthreads determinism (test 31's chain): 10 runs
    //
    // Runs syncthreadsReductionRef 10 times on the 3-word self-clearing chain
    // from test 31, verifying all runs produce identical pivots AND column data.
    //
    // Chain (same as test 31, race-free):
    //   Col 0: [0b100, 0x0000, 0x0000] (pivot=2,   word 0).  pivot_to_col[2]=0.
    //   Col 1: [0x0000, 0b001, 0x0000] (pivot=64,  word 1).  pivot_to_col[64]=1.
    //   Col 2: [0x0000, 0x0000, 0b010] (pivot=128, word 2).  pivot_to_col[128]=2.
    //
    // Each column claims its own pivot. All columns are zeroed.
    // New pivots are all -1.
    // Expected: all 10 runs produce identical new_pivots AND column data.
    {
        constexpr int num_words = 3;
        constexpr int num_cols = 3;
        constexpr int num_runs = 10;

        const uint64_t h_cols[num_cols * num_words] = {
            0b100ULL, 0x0000,   0x0000,  // col 0: pivot=2
            0x0000,   0b001ULL, 0x0000,  // col 1: pivot=64
            0x0000,   0x0000,   0b010ULL // col 2: pivot=128
        };
        const int h_pivot_to_col[192] = {};
        const_cast<int *>(h_pivot_to_col)[2] = 0;
        const_cast<int *>(h_pivot_to_col)[64] = 1;
        const_cast<int *>(h_pivot_to_col)[128] = 2;
        const int h_col_pivots[num_cols] = {2, 64, 128};

        const int block_threads = (32 > num_cols * 32) ? 32 : num_cols * 32;
        const size_t col_bytes = sizeof(h_cols);
        const size_t ptc_bytes = sizeof(h_pivot_to_col);
        const size_t idx_bytes = sizeof(int) * num_cols;

        uint64_t *d_cols;
        int *d_pivot_to_col, *d_col_pivots, *d_new_pivots;
        check_cuda(cudaMalloc(&d_cols, col_bytes), "sc3s d_cols");
        check_cuda(cudaMalloc(&d_pivot_to_col, ptc_bytes), "sc3s d_ptc");
        check_cuda(cudaMalloc(&d_col_pivots, idx_bytes), "sc3s d_cp");
        check_cuda(cudaMalloc(&d_new_pivots, idx_bytes), "sc3s d_np");
        check_cuda(cudaMemcpy(d_pivot_to_col, h_pivot_to_col, ptc_bytes, cudaMemcpyHostToDevice),
                   "sc3s h2d ptc");
        check_cuda(cudaMemcpy(d_col_pivots, h_col_pivots, idx_bytes, cudaMemcpyHostToDevice),
                   "sc3s h2d cp");

        int ref_pivots[num_cols] = {};
        uint64_t ref_data[num_cols * num_words] = {};

        for (int run = 0; run < num_runs; ++run)
        {
            check_cuda(cudaMemcpy(d_cols, h_cols, col_bytes, cudaMemcpyHostToDevice),
                       "sc3s h2d cols");
            check_cuda(cudaMemset(d_new_pivots, 0xFF, idx_bytes), "sc3s memset new_pivots");

            syncthreadsReductionRef<<<1, block_threads>>>(d_cols, d_col_pivots, d_pivot_to_col,
                                                          num_words, num_cols, d_new_pivots);
            check_cuda(cudaDeviceSynchronize(), "sc3s sync");

            int run_pivots[num_cols] = {};
            check_cuda(cudaMemcpy(run_pivots, d_new_pivots, idx_bytes, cudaMemcpyDeviceToHost),
                       "sc3s d2h pivots");

            uint64_t run_data[num_cols * num_words] = {};
            check_cuda(cudaMemcpy(run_data, d_cols, col_bytes, cudaMemcpyDeviceToHost),
                       "sc3s d2h data");

            if (run == 0)
            {
                for (int i = 0; i < num_cols; ++i)
                    ref_pivots[i] = run_pivots[i];
                for (int i = 0; i < num_cols * num_words; ++i)
                    ref_data[i] = run_data[i];
            }
            else
            {
                bool ok = true;
                for (int i = 0; i < num_cols; ++i)
                    if (run_pivots[i] != ref_pivots[i])
                    {
                        fprintf(stderr, "sc3s[run=%d] pivot[%d] = %d, ref %d\n", run, i,
                                run_pivots[i], ref_pivots[i]);
                        ok = false;
                    }
                for (int i = 0; i < num_cols * num_words; ++i)
                    if (run_data[i] != ref_data[i])
                    {
                        fprintf(stderr, "sc3s[run=%d] data[%d] = 0x%lx, ref 0x%lx\n", run, i,
                                run_data[i], ref_data[i]);
                        ok = false;
                    }
                if (!ok)
                    goto sc3s_cleanup;
            }
        }

        printf("  PASSED: (3-word self-clearing syncthreads determinism)\n");
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        goto sc3s_done;

sc3s_cleanup:
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        return 1;

sc3s_done:;
    }
    // 2-word cross-column regular kernel determinism (test 23's chain): 10 runs
    //
    // Runs launchPipelinedReduction with use_async_copy=false 10 times on
    // the 2-word cross-column chain from test 23, verifying all runs
    // produce identical pivots AND column data.
    //
    // Chain (same as test 23, race-free):
    //   Col 0: [0x0000, 0b100] (pivot=66, word 1).  pivot_to_col[66]=1.
    //   Col 1: [0b010, 0x0000] (pivot=1,  word 0).  pivot_to_col[1]=2.
    //   Col 2: [0x0000, 0b001] (pivot=64, word 1).  pivot_to_col[64]=-1.
    //
    // Col 1 XOR with Col 2: MSB moves to w1 (pivot=64, unclaimed). Break.
    // Col 0 XOR with Col 1: MSB moves to w1 (pivot=64, unclaimed). Break.
    // Expected: all 10 runs produce identical new_pivots AND column data.
    {
        constexpr int num_words = 2;
        constexpr int num_cols = 3;
        constexpr int num_runs = 10;

        const uint64_t h_cols[num_cols * num_words] = {
            0x0000ULL, 0b100ULL,  // col 0: pivot=66
            0b010ULL,  0x0000ULL, // col 1: pivot=1
            0x0000ULL, 0b001ULL   // col 2: pivot=64 (unclaimed)
        };
        const int h_pivot_to_col[128] = {};
        const_cast<int *>(h_pivot_to_col)[66] = 1;
        const_cast<int *>(h_pivot_to_col)[1] = 2;
        const int h_col_pivots[num_cols] = {66, 1, 64};

        const size_t col_bytes = sizeof(h_cols);
        const size_t ptc_bytes = sizeof(h_pivot_to_col);
        const size_t idx_bytes = sizeof(int) * num_cols;

        uint64_t *d_cols;
        int *d_pivot_to_col, *d_col_pivots, *d_new_pivots;
        check_cuda(cudaMalloc(&d_cols, col_bytes), "xc2 d_cols");
        check_cuda(cudaMalloc(&d_pivot_to_col, ptc_bytes), "xc2 d_ptc");
        check_cuda(cudaMalloc(&d_col_pivots, idx_bytes), "xc2 d_cp");
        check_cuda(cudaMalloc(&d_new_pivots, idx_bytes), "xc2 d_np");
        check_cuda(cudaMemcpy(d_pivot_to_col, h_pivot_to_col, ptc_bytes, cudaMemcpyHostToDevice),
                   "xc2 h2d ptc");
        check_cuda(cudaMemcpy(d_col_pivots, h_col_pivots, idx_bytes, cudaMemcpyHostToDevice),
                   "xc2 h2d cp");

        int ref_pivots[num_cols] = {};
        uint64_t ref_data[num_cols * num_words] = {};

        for (int run = 0; run < num_runs; ++run)
        {
            check_cuda(cudaMemcpy(d_cols, h_cols, col_bytes, cudaMemcpyHostToDevice),
                       "xc2 h2d cols");
            check_cuda(cudaMemset(d_new_pivots, 0xFF, idx_bytes), "xc2 memset new_pivots");

            nerve::persistence::gpu::launchPipelinedReduction(
                d_cols, d_col_pivots, d_pivot_to_col, num_words, num_cols, d_new_pivots, 0, false);
            check_cuda(cudaDeviceSynchronize(), "xc2 sync");

            int run_pivots[num_cols] = {};
            check_cuda(cudaMemcpy(run_pivots, d_new_pivots, idx_bytes, cudaMemcpyDeviceToHost),
                       "xc2 d2h pivots");

            uint64_t run_data[num_cols * num_words] = {};
            check_cuda(cudaMemcpy(run_data, d_cols, col_bytes, cudaMemcpyDeviceToHost),
                       "xc2 d2h data");

            if (run == 0)
            {
                for (int i = 0; i < num_cols; ++i)
                    ref_pivots[i] = run_pivots[i];
                for (int i = 0; i < num_cols * num_words; ++i)
                    ref_data[i] = run_data[i];
            }
            else
            {
                bool ok = true;
                for (int i = 0; i < num_cols; ++i)
                    if (run_pivots[i] != ref_pivots[i])
                    {
                        fprintf(stderr, "xc2[run=%d] pivot[%d] = %d, ref %d\n", run, i,
                                run_pivots[i], ref_pivots[i]);
                        ok = false;
                    }
                for (int i = 0; i < num_cols * num_words; ++i)
                    if (run_data[i] != ref_data[i])
                    {
                        fprintf(stderr, "xc2[run=%d] data[%d] = 0x%lx, ref 0x%lx\n", run, i,
                                run_data[i], ref_data[i]);
                        ok = false;
                    }
                if (!ok)
                    goto xc2_cleanup;
            }
        }

        printf("  PASSED: (2-word cross-column regular kernel determinism)\n");
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        goto xc2_done;

xc2_cleanup:
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        return 1;

xc2_done:;
    }
    // 4-word self-clearing syncthreads determinism (test 37's chain): 10 runs
    //
    // Runs syncthreadsReductionRef 10 times on the 4-word self-clearing chain
    // from test 37, verifying all runs produce identical pivots AND column data.
    //
    // Chain (same as test 37, race-free):
    //   Col 0: [0b100, 0x0000, 0x0000, 0x0000] (pivot=2,   word 0).  pivot_to_col[2]=0.
    //   Col 1: [0x0000, 0x0000, 0x0000, 0b001] (pivot=192, word 3).  pivot_to_col[192]=1.
    //   Col 2: [0x0000, 0x0000, 0b010, 0x0000] (pivot=128, word 2).  pivot_to_col[128]=2.
    //
    // Each column claims its own pivot. All columns are zeroed.
    // New pivots are all -1.
    // Expected: all 10 runs produce identical new_pivots AND column data.
    {
        constexpr int num_words = 4;
        constexpr int num_cols = 3;
        constexpr int num_runs = 10;

        const uint64_t h_cols[num_cols * num_words] = {
            0b100ULL, 0x0000, 0x0000,   0x0000,   // col 0: pivot=2
            0x0000,   0x0000, 0x0000,   0b001ULL, // col 1: pivot=192
            0x0000,   0x0000, 0b010ULL, 0x0000    // col 2: pivot=128
        };
        const int h_pivot_to_col[256] = {};
        const_cast<int *>(h_pivot_to_col)[2] = 0;
        const_cast<int *>(h_pivot_to_col)[192] = 1;
        const_cast<int *>(h_pivot_to_col)[128] = 2;
        const int h_col_pivots[num_cols] = {2, 192, 128};

        const int block_threads = (32 > num_cols * 32) ? 32 : num_cols * 32;
        const size_t col_bytes = sizeof(h_cols);
        const size_t ptc_bytes = sizeof(h_pivot_to_col);
        const size_t idx_bytes = sizeof(int) * num_cols;

        uint64_t *d_cols;
        int *d_pivot_to_col, *d_col_pivots, *d_new_pivots;
        check_cuda(cudaMalloc(&d_cols, col_bytes), "sc4s d_cols");
        check_cuda(cudaMalloc(&d_pivot_to_col, ptc_bytes), "sc4s d_ptc");
        check_cuda(cudaMalloc(&d_col_pivots, idx_bytes), "sc4s d_cp");
        check_cuda(cudaMalloc(&d_new_pivots, idx_bytes), "sc4s d_np");
        check_cuda(cudaMemcpy(d_pivot_to_col, h_pivot_to_col, ptc_bytes, cudaMemcpyHostToDevice),
                   "sc4s h2d ptc");
        check_cuda(cudaMemcpy(d_col_pivots, h_col_pivots, idx_bytes, cudaMemcpyHostToDevice),
                   "sc4s h2d cp");

        int ref_pivots[num_cols] = {};
        uint64_t ref_data[num_cols * num_words] = {};

        for (int run = 0; run < num_runs; ++run)
        {
            check_cuda(cudaMemcpy(d_cols, h_cols, col_bytes, cudaMemcpyHostToDevice),
                       "sc4s h2d cols");
            check_cuda(cudaMemset(d_new_pivots, 0xFF, idx_bytes), "sc4s memset new_pivots");

            syncthreadsReductionRef<<<1, block_threads>>>(d_cols, d_col_pivots, d_pivot_to_col,
                                                          num_words, num_cols, d_new_pivots);
            check_cuda(cudaDeviceSynchronize(), "sc4s sync");

            int run_pivots[num_cols] = {};
            check_cuda(cudaMemcpy(run_pivots, d_new_pivots, idx_bytes, cudaMemcpyDeviceToHost),
                       "sc4s d2h pivots");

            uint64_t run_data[num_cols * num_words] = {};
            check_cuda(cudaMemcpy(run_data, d_cols, col_bytes, cudaMemcpyDeviceToHost),
                       "sc4s d2h data");

            if (run == 0)
            {
                for (int i = 0; i < num_cols; ++i)
                    ref_pivots[i] = run_pivots[i];
                for (int i = 0; i < num_cols * num_words; ++i)
                    ref_data[i] = run_data[i];
            }
            else
            {
                bool ok = true;
                for (int i = 0; i < num_cols; ++i)
                    if (run_pivots[i] != ref_pivots[i])
                    {
                        fprintf(stderr, "sc4s[run=%d] pivot[%d] = %d, ref %d\n", run, i,
                                run_pivots[i], ref_pivots[i]);
                        ok = false;
                    }
                for (int i = 0; i < num_cols * num_words; ++i)
                    if (run_data[i] != ref_data[i])
                    {
                        fprintf(stderr, "sc4s[run=%d] data[%d] = 0x%lx, ref 0x%lx\n", run, i,
                                run_data[i], ref_data[i]);
                        ok = false;
                    }
                if (!ok)
                    goto sc4s_cleanup;
            }
        }

        printf("  PASSED: (4-word self-clearing syncthreads determinism)\n");
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        goto sc4s_done;

sc4s_cleanup:
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        return 1;

sc4s_done:;
    }
    // 4-word self-clearing regular kernel determinism (test 37's chain): 10 runs
    //
    // Runs launchPipelinedReduction with use_async_copy=false 10 times on
    // the 4-word self-clearing chain from test 37, verifying all runs
    // produce identical pivots AND column data.
    //
    // Chain (same as test 37, race-free):
    //   Col 0: [0b100, 0x0000, 0x0000, 0x0000] (pivot=2,   word 0).  pivot_to_col[2]=0.
    //   Col 1: [0x0000, 0x0000, 0x0000, 0b001] (pivot=192, word 3).  pivot_to_col[192]=1.
    //   Col 2: [0x0000, 0x0000, 0b010, 0x0000] (pivot=128, word 2).  pivot_to_col[128]=2.
    //
    // Each column claims its own pivot. All columns are zeroed.
    // New pivots are all -1.
    // Expected: all 10 runs produce identical new_pivots AND column data.
    {
        constexpr int num_words = 4;
        constexpr int num_cols = 3;
        constexpr int num_runs = 10;

        const uint64_t h_cols[num_cols * num_words] = {
            0b100ULL, 0x0000, 0x0000,   0x0000,   // col 0: pivot=2
            0x0000,   0x0000, 0x0000,   0b001ULL, // col 1: pivot=192
            0x0000,   0x0000, 0b010ULL, 0x0000    // col 2: pivot=128
        };
        const int h_pivot_to_col[256] = {};
        const_cast<int *>(h_pivot_to_col)[2] = 0;
        const_cast<int *>(h_pivot_to_col)[192] = 1;
        const_cast<int *>(h_pivot_to_col)[128] = 2;
        const int h_col_pivots[num_cols] = {2, 192, 128};

        const size_t col_bytes = sizeof(h_cols);
        const size_t ptc_bytes = sizeof(h_pivot_to_col);
        const size_t idx_bytes = sizeof(int) * num_cols;

        uint64_t *d_cols;
        int *d_pivot_to_col, *d_col_pivots, *d_new_pivots;
        check_cuda(cudaMalloc(&d_cols, col_bytes), "sc4r d_cols");
        check_cuda(cudaMalloc(&d_pivot_to_col, ptc_bytes), "sc4r d_ptc");
        check_cuda(cudaMalloc(&d_col_pivots, idx_bytes), "sc4r d_cp");
        check_cuda(cudaMalloc(&d_new_pivots, idx_bytes), "sc4r d_np");
        check_cuda(cudaMemcpy(d_pivot_to_col, h_pivot_to_col, ptc_bytes, cudaMemcpyHostToDevice),
                   "sc4r h2d ptc");
        check_cuda(cudaMemcpy(d_col_pivots, h_col_pivots, idx_bytes, cudaMemcpyHostToDevice),
                   "sc4r h2d cp");

        int ref_pivots[num_cols] = {};
        uint64_t ref_data[num_cols * num_words] = {};

        for (int run = 0; run < num_runs; ++run)
        {
            check_cuda(cudaMemcpy(d_cols, h_cols, col_bytes, cudaMemcpyHostToDevice),
                       "sc4r h2d cols");
            check_cuda(cudaMemset(d_new_pivots, 0xFF, idx_bytes), "sc4r memset new_pivots");

            nerve::persistence::gpu::launchPipelinedReduction(
                d_cols, d_col_pivots, d_pivot_to_col, num_words, num_cols, d_new_pivots, 0, false);
            check_cuda(cudaDeviceSynchronize(), "sc4r sync");

            int run_pivots[num_cols] = {};
            check_cuda(cudaMemcpy(run_pivots, d_new_pivots, idx_bytes, cudaMemcpyDeviceToHost),
                       "sc4r d2h pivots");

            uint64_t run_data[num_cols * num_words] = {};
            check_cuda(cudaMemcpy(run_data, d_cols, col_bytes, cudaMemcpyDeviceToHost),
                       "sc4r d2h data");

            if (run == 0)
            {
                for (int i = 0; i < num_cols; ++i)
                    ref_pivots[i] = run_pivots[i];
                for (int i = 0; i < num_cols * num_words; ++i)
                    ref_data[i] = run_data[i];
            }
            else
            {
                bool ok = true;
                for (int i = 0; i < num_cols; ++i)
                    if (run_pivots[i] != ref_pivots[i])
                    {
                        fprintf(stderr, "sc4r[run=%d] pivot[%d] = %d, ref %d\n", run, i,
                                run_pivots[i], ref_pivots[i]);
                        ok = false;
                    }
                for (int i = 0; i < num_cols * num_words; ++i)
                    if (run_data[i] != ref_data[i])
                    {
                        fprintf(stderr, "sc4r[run=%d] data[%d] = 0x%lx, ref 0x%lx\n", run, i,
                                run_data[i], ref_data[i]);
                        ok = false;
                    }
                if (!ok)
                    goto sc4r_cleanup;
            }
        }

        printf("  PASSED: (4-word self-clearing regular kernel determinism)\n");
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        goto sc4r_done;

sc4r_cleanup:
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        return 1;

sc4r_done:;
    }
    // 4-word cross-column regular kernel determinism (test 33's chain): 10 runs
    //
    // Runs launchPipelinedReduction with use_async_copy=false 10 times on
    // the 4-word cross-column chain from test 33, verifying all runs
    // produce identical pivots AND column data.
    //
    // Chain (same as test 33, race-free):
    //   Col 0: [0, 0, 0, 0b100] (pivot=194, word 3).  pivot_to_col[194]=1.
    //   Col 1: [0, 0b010, 0, 0] (pivot=65,  word 1).  pivot_to_col[65]=2.
    //   Col 2: [0b001, 0, 0, 0]  (pivot=0,   word 0).  pivot_to_col[0]=-1.
    //
    // Expected: all 10 runs produce identical new_pivots AND column data.
    // Expected pivots: {194, 65, 0}
    {
        constexpr int num_words = 4;
        constexpr int num_cols = 3;
        constexpr int num_runs = 10;

        const uint64_t h_cols[num_cols * num_words] = {
            0x0000,   0x0000,   0x0000, 0b100ULL, // col 0: pivot=194
            0x0000,   0b010ULL, 0x0000, 0x0000,   // col 1: pivot=65
            0b001ULL, 0x0000,   0x0000, 0x0000    // col 2: pivot=0
        };
        int h_pivot_to_col[256];
        std::memset(h_pivot_to_col, 0xFF, sizeof(h_pivot_to_col));
        h_pivot_to_col[194] = 1;
        h_pivot_to_col[65] = 2;
        const int h_col_pivots[num_cols] = {194, 65, 0};

        const size_t col_bytes = sizeof(h_cols);
        const size_t ptc_bytes = sizeof(h_pivot_to_col);
        const size_t idx_bytes = sizeof(int) * num_cols;

        uint64_t *d_cols;
        int *d_pivot_to_col, *d_col_pivots, *d_new_pivots;
        check_cuda(cudaMalloc(&d_cols, col_bytes), "xc4 d_cols");
        check_cuda(cudaMalloc(&d_pivot_to_col, ptc_bytes), "xc4 d_ptc");
        check_cuda(cudaMalloc(&d_col_pivots, idx_bytes), "xc4 d_cp");
        check_cuda(cudaMalloc(&d_new_pivots, idx_bytes), "xc4 d_np");
        check_cuda(cudaMemcpy(d_pivot_to_col, h_pivot_to_col, ptc_bytes, cudaMemcpyHostToDevice),
                   "xc4 h2d ptc");
        check_cuda(cudaMemcpy(d_col_pivots, h_col_pivots, idx_bytes, cudaMemcpyHostToDevice),
                   "xc4 h2d cp");

        int ref_pivots[num_cols] = {};
        uint64_t ref_data[num_cols * num_words] = {};

        for (int run = 0; run < num_runs; ++run)
        {
            check_cuda(cudaMemcpy(d_cols, h_cols, col_bytes, cudaMemcpyHostToDevice),
                       "xc4 h2d cols");
            check_cuda(cudaMemset(d_new_pivots, 0xFF, idx_bytes), "xc4 memset new_pivots");

            nerve::persistence::gpu::launchPipelinedReduction(
                d_cols, d_col_pivots, d_pivot_to_col, num_words, num_cols, d_new_pivots, 0, false);
            check_cuda(cudaDeviceSynchronize(), "xc4 sync");

            int run_pivots[num_cols] = {};
            check_cuda(cudaMemcpy(run_pivots, d_new_pivots, idx_bytes, cudaMemcpyDeviceToHost),
                       "xc4 d2h pivots");

            uint64_t run_data[num_cols * num_words] = {};
            check_cuda(cudaMemcpy(run_data, d_cols, col_bytes, cudaMemcpyDeviceToHost),
                       "xc4 d2h data");

            if (run == 0)
            {
                for (int i = 0; i < num_cols; ++i)
                    ref_pivots[i] = run_pivots[i];
                for (int i = 0; i < num_cols * num_words; ++i)
                    ref_data[i] = run_data[i];
            }
            else
            {
                bool ok = true;
                for (int i = 0; i < num_cols; ++i)
                    if (run_pivots[i] != ref_pivots[i])
                    {
                        fprintf(stderr, "xc4[run=%d] pivot[%d] = %d, ref %d\n", run, i,
                                run_pivots[i], ref_pivots[i]);
                        ok = false;
                    }
                for (int i = 0; i < num_cols * num_words; ++i)
                    if (run_data[i] != ref_data[i])
                    {
                        fprintf(stderr, "xc4[run=%d] data[%d] = 0x%lx, ref 0x%lx\n", run, i,
                                run_data[i], ref_data[i]);
                        ok = false;
                    }
                if (!ok)
                    goto xc4_cleanup;
            }
        }

        printf("  PASSED: (4-word cross-column regular kernel determinism)\n");
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        goto xc4_done;

xc4_cleanup:
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        return 1;

xc4_done:;
    }
    // 3-word cross-column regular kernel determinism (test 26's chain): 10 runs
    //
    // Runs launchPipelinedReduction with use_async_copy=false 10 times on
    // the 3-word cross-column chain from test 26, verifying all runs
    // produce identical pivots AND column data.
    //
    // Chain (same as test 26):
    //   Col 0: [0, 0, 0b100] (pivot=130, word 2).  pivot_to_col[130]=1.
    //   Col 1: [0, 0b100, 0] (pivot=66,  word 1).  pivot_to_col[66]=2.
    //   Col 2: [0b001, 0, 0] (pivot=0,   word 0).  pivot_to_col[0]=-1.
    //
    // Returns to initial after 12 even iterations.
    // Expected: all 10 runs produce identical new_pivots AND column data.
    // Expected pivots: {130, 66, 0}
    {
        constexpr int num_words = 3;
        constexpr int num_cols = 3;
        constexpr int num_runs = 10;

        const uint64_t h_cols[num_cols * num_words] = {
            0x0000ULL, 0x0000ULL, 0b100ULL,  // col 0: pivot=130
            0x0000ULL, 0b100ULL,  0x0000ULL, // col 1: pivot=66
            0b001ULL,  0x0000ULL, 0x0000ULL  // col 2: pivot=0
        };
        const int h_pivot_to_col[192] = {};
        const_cast<int *>(h_pivot_to_col)[130] = 1;
        const_cast<int *>(h_pivot_to_col)[66] = 2;
        const int h_col_pivots[num_cols] = {130, 66, 0};

        const size_t col_bytes = sizeof(h_cols);
        const size_t ptc_bytes = sizeof(h_pivot_to_col);
        const size_t idx_bytes = sizeof(int) * num_cols;

        uint64_t *d_cols;
        int *d_pivot_to_col, *d_col_pivots, *d_new_pivots;
        check_cuda(cudaMalloc(&d_cols, col_bytes), "xc3r d_cols");
        check_cuda(cudaMalloc(&d_pivot_to_col, ptc_bytes), "xc3r d_ptc");
        check_cuda(cudaMalloc(&d_col_pivots, idx_bytes), "xc3r d_cp");
        check_cuda(cudaMalloc(&d_new_pivots, idx_bytes), "xc3r d_np");
        check_cuda(cudaMemcpy(d_pivot_to_col, h_pivot_to_col, ptc_bytes, cudaMemcpyHostToDevice),
                   "xc3r h2d ptc");
        check_cuda(cudaMemcpy(d_col_pivots, h_col_pivots, idx_bytes, cudaMemcpyHostToDevice),
                   "xc3r h2d cp");

        int ref_pivots[num_cols] = {};
        uint64_t ref_data[num_cols * num_words] = {};

        for (int run = 0; run < num_runs; ++run)
        {
            check_cuda(cudaMemcpy(d_cols, h_cols, col_bytes, cudaMemcpyHostToDevice),
                       "xc3r h2d cols");
            check_cuda(cudaMemset(d_new_pivots, 0xFF, idx_bytes), "xc3r memset new_pivots");

            nerve::persistence::gpu::launchPipelinedReduction(
                d_cols, d_col_pivots, d_pivot_to_col, num_words, num_cols, d_new_pivots, 0, false);
            check_cuda(cudaDeviceSynchronize(), "xc3r sync");

            int run_pivots[num_cols] = {};
            check_cuda(cudaMemcpy(run_pivots, d_new_pivots, idx_bytes, cudaMemcpyDeviceToHost),
                       "xc3r d2h pivots");

            uint64_t run_data[num_cols * num_words] = {};
            check_cuda(cudaMemcpy(run_data, d_cols, col_bytes, cudaMemcpyDeviceToHost),
                       "xc3r d2h data");

            if (run == 0)
            {
                for (int i = 0; i < num_cols; ++i)
                    ref_pivots[i] = run_pivots[i];
                for (int i = 0; i < num_cols * num_words; ++i)
                    ref_data[i] = run_data[i];
            }
            else
            {
                bool ok = true;
                for (int i = 0; i < num_cols; ++i)
                    if (run_pivots[i] != ref_pivots[i])
                    {
                        fprintf(stderr, "xc3r[run=%d] pivot[%d] = %d, ref %d\n", run, i,
                                run_pivots[i], ref_pivots[i]);
                        ok = false;
                    }
                for (int i = 0; i < num_cols * num_words; ++i)
                    if (run_data[i] != ref_data[i])
                    {
                        fprintf(stderr, "xc3r[run=%d] data[%d] = 0x%lx, ref 0x%lx\n", run, i,
                                run_data[i], ref_data[i]);
                        ok = false;
                    }
                if (!ok)
                    goto xc3r_cleanup;
            }
        }

        printf("  PASSED: (3-word cross-column regular kernel determinism)\n");
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        goto xc3r_done;

xc3r_cleanup:
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        return 1;

xc3r_done:;
    }
    // 4-word unclaimed pivot syncthreads determinism (test 40's chain): 10 runs
    //
    // Runs syncthreadsReductionRef 10 times on the 4-word unclaimed pivot
    // chain from test 40, verifying all runs produce identical pivots AND
    // column data.
    //
    // Chain (same as test 40, race-free):
    //   Col 0: [0b100, 0, 0, 0] (pivot=2,   word 0).  pivot_to_col[2]=1.
    //   Col 1: [0, 0, 0b010, 0] (pivot=128, word 2).  pivot_to_col[128]=2.
    //   Col 2: [0, 0, 0, 0b001] (pivot=192, word 3).  pivot_to_col[192]=-1.
    //
    // Both Col 1 and Col 0 converge to unclaimed pivot=192 after 1 iter.
    // Expected: all 10 runs produce identical new_pivots AND column data.
    {
        constexpr int num_words = 4;
        constexpr int num_cols = 3;
        constexpr int num_runs = 10;

        const uint64_t h_cols[num_cols * num_words] = {
            0b100ULL, 0x0000, 0x0000,   0x0000,  // col 0: pivot=2
            0x0000,   0x0000, 0b010ULL, 0x0000,  // col 1: pivot=128
            0x0000,   0x0000, 0x0000,   0b001ULL // col 2: pivot=192 (unclaimed)
        };
        const int h_pivot_to_col[256] = {};
        const_cast<int *>(h_pivot_to_col)[2] = 1;
        const_cast<int *>(h_pivot_to_col)[128] = 2;
        const_cast<int *>(h_pivot_to_col)[192] = -1;
        const int h_col_pivots[num_cols] = {2, 128, 192};

        const int block_threads = (32 > num_cols * 32) ? 32 : num_cols * 32;
        const size_t col_bytes = sizeof(h_cols);
        const size_t ptc_bytes = sizeof(h_pivot_to_col);
        const size_t idx_bytes = sizeof(int) * num_cols;

        uint64_t *d_cols;
        int *d_pivot_to_col, *d_col_pivots, *d_new_pivots;
        check_cuda(cudaMalloc(&d_cols, col_bytes), "uc4s d_cols");
        check_cuda(cudaMalloc(&d_pivot_to_col, ptc_bytes), "uc4s d_ptc");
        check_cuda(cudaMalloc(&d_col_pivots, idx_bytes), "uc4s d_cp");
        check_cuda(cudaMalloc(&d_new_pivots, idx_bytes), "uc4s d_np");
        check_cuda(cudaMemcpy(d_pivot_to_col, h_pivot_to_col, ptc_bytes, cudaMemcpyHostToDevice),
                   "uc4s h2d ptc");
        check_cuda(cudaMemcpy(d_col_pivots, h_col_pivots, idx_bytes, cudaMemcpyHostToDevice),
                   "uc4s h2d cp");

        int ref_pivots[num_cols] = {};
        uint64_t ref_data[num_cols * num_words] = {};

        for (int run = 0; run < num_runs; ++run)
        {
            check_cuda(cudaMemcpy(d_cols, h_cols, col_bytes, cudaMemcpyHostToDevice),
                       "uc4s h2d cols");
            check_cuda(cudaMemset(d_new_pivots, 0xFF, idx_bytes), "uc4s memset new_pivots");

            syncthreadsReductionRef<<<1, block_threads>>>(d_cols, d_col_pivots, d_pivot_to_col,
                                                          num_words, num_cols, d_new_pivots);
            check_cuda(cudaDeviceSynchronize(), "uc4s sync");

            int run_pivots[num_cols] = {};
            check_cuda(cudaMemcpy(run_pivots, d_new_pivots, idx_bytes, cudaMemcpyDeviceToHost),
                       "uc4s d2h pivots");

            uint64_t run_data[num_cols * num_words] = {};
            check_cuda(cudaMemcpy(run_data, d_cols, col_bytes, cudaMemcpyDeviceToHost),
                       "uc4s d2h data");

            if (run == 0)
            {
                for (int i = 0; i < num_cols; ++i)
                    ref_pivots[i] = run_pivots[i];
                for (int i = 0; i < num_cols * num_words; ++i)
                    ref_data[i] = run_data[i];
            }
            else
            {
                bool ok = true;
                for (int i = 0; i < num_cols; ++i)
                    if (run_pivots[i] != ref_pivots[i])
                    {
                        fprintf(stderr, "uc4s[run=%d] pivot[%d] = %d, ref %d\n", run, i,
                                run_pivots[i], ref_pivots[i]);
                        ok = false;
                    }
                for (int i = 0; i < num_cols * num_words; ++i)
                    if (run_data[i] != ref_data[i])
                    {
                        fprintf(stderr, "uc4s[run=%d] data[%d] = 0x%lx, ref 0x%lx\n", run, i,
                                run_data[i], ref_data[i]);
                        ok = false;
                    }
                if (!ok)
                    goto uc4s_cleanup;
            }
        }

        printf("  PASSED: (4-word unclaimed pivot syncthreads determinism)\n");
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        goto uc4s_done;

uc4s_cleanup:
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        return 1;

uc4s_done:;
    }
    // 3-word self-clearing regular kernel determinism (test 31's chain): 10 runs
    //
    // Runs launchPipelinedReduction with use_async_copy=false 10 times on
    // the 3-word self-clearing chain from test 31, verifying all runs
    // produce identical pivots AND column data.
    //
    // Chain (same as test 31, race-free):
    //   Col 0: [0b100, 0x0000, 0x0000] (pivot=2,   word 0).  pivot_to_col[2]=0.
    //   Col 1: [0x0000, 0b001, 0x0000] (pivot=64,  word 1).  pivot_to_col[64]=1.
    //   Col 2: [0x0000, 0x0000, 0b010] (pivot=128, word 2).  pivot_to_col[128]=2.
    //
    // Each column claims its own pivot. All columns are zeroed.
    // New pivots are all -1.
    // Expected: all 10 runs produce identical new_pivots AND column data.
    {
        constexpr int num_words = 3;
        constexpr int num_cols = 3;
        constexpr int num_runs = 10;

        const uint64_t h_cols[num_cols * num_words] = {
            0b100ULL, 0x0000,   0x0000,  // col 0: pivot=2
            0x0000,   0b001ULL, 0x0000,  // col 1: pivot=64
            0x0000,   0x0000,   0b010ULL // col 2: pivot=128
        };
        const int h_pivot_to_col[192] = {};
        const_cast<int *>(h_pivot_to_col)[2] = 0;
        const_cast<int *>(h_pivot_to_col)[64] = 1;
        const_cast<int *>(h_pivot_to_col)[128] = 2;
        const int h_col_pivots[num_cols] = {2, 64, 128};

        const size_t col_bytes = sizeof(h_cols);
        const size_t ptc_bytes = sizeof(h_pivot_to_col);
        const size_t idx_bytes = sizeof(int) * num_cols;

        uint64_t *d_cols;
        int *d_pivot_to_col, *d_col_pivots, *d_new_pivots;
        check_cuda(cudaMalloc(&d_cols, col_bytes), "sc3r d_cols");
        check_cuda(cudaMalloc(&d_pivot_to_col, ptc_bytes), "sc3r d_ptc");
        check_cuda(cudaMalloc(&d_col_pivots, idx_bytes), "sc3r d_cp");
        check_cuda(cudaMalloc(&d_new_pivots, idx_bytes), "sc3r d_np");
        check_cuda(cudaMemcpy(d_pivot_to_col, h_pivot_to_col, ptc_bytes, cudaMemcpyHostToDevice),
                   "sc3r h2d ptc");
        check_cuda(cudaMemcpy(d_col_pivots, h_col_pivots, idx_bytes, cudaMemcpyHostToDevice),
                   "sc3r h2d cp");

        int ref_pivots[num_cols] = {};
        uint64_t ref_data[num_cols * num_words] = {};

        for (int run = 0; run < num_runs; ++run)
        {
            check_cuda(cudaMemcpy(d_cols, h_cols, col_bytes, cudaMemcpyHostToDevice),
                       "sc3r h2d cols");
            check_cuda(cudaMemset(d_new_pivots, 0xFF, idx_bytes), "sc3r memset new_pivots");

            nerve::persistence::gpu::launchPipelinedReduction(
                d_cols, d_col_pivots, d_pivot_to_col, num_words, num_cols, d_new_pivots, 0, false);
            check_cuda(cudaDeviceSynchronize(), "sc3r sync");

            int run_pivots[num_cols] = {};
            check_cuda(cudaMemcpy(run_pivots, d_new_pivots, idx_bytes, cudaMemcpyDeviceToHost),
                       "sc3r d2h pivots");

            uint64_t run_data[num_cols * num_words] = {};
            check_cuda(cudaMemcpy(run_data, d_cols, col_bytes, cudaMemcpyDeviceToHost),
                       "sc3r d2h data");

            if (run == 0)
            {
                for (int i = 0; i < num_cols; ++i)
                    ref_pivots[i] = run_pivots[i];
                for (int i = 0; i < num_cols * num_words; ++i)
                    ref_data[i] = run_data[i];
            }
            else
            {
                bool ok = true;
                for (int i = 0; i < num_cols; ++i)
                    if (run_pivots[i] != ref_pivots[i])
                    {
                        fprintf(stderr, "sc3r[run=%d] pivot[%d] = %d, ref %d\n", run, i,
                                run_pivots[i], ref_pivots[i]);
                        ok = false;
                    }
                for (int i = 0; i < num_cols * num_words; ++i)
                    if (run_data[i] != ref_data[i])
                    {
                        fprintf(stderr, "sc3r[run=%d] data[%d] = 0x%lx, ref 0x%lx\n", run, i,
                                run_data[i], ref_data[i]);
                        ok = false;
                    }
                if (!ok)
                    goto sc3r_cleanup;
            }
        }

        printf("  PASSED: (3-word self-clearing regular kernel determinism)\n");
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        goto sc3r_done;

sc3r_cleanup:
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        return 1;

sc3r_done:;
    }
    // 4-word cross-column syncthreads determinism (test 33's chain): 10 runs
    //
    // Runs syncthreadsReductionRef 10 times on the 4-word cross-column chain
    // from test 33, verifying all runs produce identical pivots AND column data.
    //
    // Chain (same as test 33):
    //   Col 0: [0, 0, 0, 0b100] (pivot=194, word 3).  pivot_to_col[194]=1.
    //   Col 1: [0, 0b010, 0, 0] (pivot=65,  word 1).  pivot_to_col[65]=2.
    //   Col 2: [0b001, 0, 0, 0]  (pivot=0,   word 0).  pivot_to_col[0]=-1.
    //
    // Expected: all 10 runs produce identical new_pivots AND column data.
    // Expected pivots: {194, 65, 0}
    {
        constexpr int num_words = 4;
        constexpr int num_cols = 3;
        constexpr int num_runs = 10;

        const uint64_t h_cols[num_cols * num_words] = {
            0x0000,   0x0000,   0x0000, 0b100ULL, // col 0: pivot=194
            0x0000,   0b010ULL, 0x0000, 0x0000,   // col 1: pivot=65
            0b001ULL, 0x0000,   0x0000, 0x0000    // col 2: pivot=0
        };
        int h_pivot_to_col[256];
        std::memset(h_pivot_to_col, 0xFF, sizeof(h_pivot_to_col));
        h_pivot_to_col[194] = 1;
        h_pivot_to_col[65] = 2;
        const int h_col_pivots[num_cols] = {194, 65, 0};

        const int block_threads = (32 > num_cols * 32) ? 32 : num_cols * 32;
        const size_t col_bytes = sizeof(h_cols);
        const size_t ptc_bytes = sizeof(h_pivot_to_col);
        const size_t idx_bytes = sizeof(int) * num_cols;

        uint64_t *d_cols;
        int *d_pivot_to_col, *d_col_pivots, *d_new_pivots;
        check_cuda(cudaMalloc(&d_cols, col_bytes), "xc4s d_cols");
        check_cuda(cudaMalloc(&d_pivot_to_col, ptc_bytes), "xc4s d_ptc");
        check_cuda(cudaMalloc(&d_col_pivots, idx_bytes), "xc4s d_cp");
        check_cuda(cudaMalloc(&d_new_pivots, idx_bytes), "xc4s d_np");
        check_cuda(cudaMemcpy(d_pivot_to_col, h_pivot_to_col, ptc_bytes, cudaMemcpyHostToDevice),
                   "xc4s h2d ptc");
        check_cuda(cudaMemcpy(d_col_pivots, h_col_pivots, idx_bytes, cudaMemcpyHostToDevice),
                   "xc4s h2d cp");

        int ref_pivots[num_cols] = {};
        uint64_t ref_data[num_cols * num_words] = {};

        for (int run = 0; run < num_runs; ++run)
        {
            check_cuda(cudaMemcpy(d_cols, h_cols, col_bytes, cudaMemcpyHostToDevice),
                       "xc4s h2d cols");
            check_cuda(cudaMemset(d_new_pivots, 0xFF, idx_bytes), "xc4s memset new_pivots");

            syncthreadsReductionRef<<<1, block_threads>>>(d_cols, d_col_pivots, d_pivot_to_col,
                                                          num_words, num_cols, d_new_pivots);
            check_cuda(cudaDeviceSynchronize(), "xc4s sync");

            int run_pivots[num_cols] = {};
            check_cuda(cudaMemcpy(run_pivots, d_new_pivots, idx_bytes, cudaMemcpyDeviceToHost),
                       "xc4s d2h pivots");

            uint64_t run_data[num_cols * num_words] = {};
            check_cuda(cudaMemcpy(run_data, d_cols, col_bytes, cudaMemcpyDeviceToHost),
                       "xc4s d2h data");

            if (run == 0)
            {
                for (int i = 0; i < num_cols; ++i)
                    ref_pivots[i] = run_pivots[i];
                for (int i = 0; i < num_cols * num_words; ++i)
                    ref_data[i] = run_data[i];
            }
            else
            {
                bool ok = true;
                for (int i = 0; i < num_cols; ++i)
                    if (run_pivots[i] != ref_pivots[i])
                    {
                        fprintf(stderr, "xc4s[run=%d] pivot[%d] = %d, ref %d\n", run, i,
                                run_pivots[i], ref_pivots[i]);
                        ok = false;
                    }
                for (int i = 0; i < num_cols * num_words; ++i)
                    if (run_data[i] != ref_data[i])
                    {
                        fprintf(stderr, "xc4s[run=%d] data[%d] = 0x%lx, ref 0x%lx\n", run, i,
                                run_data[i], ref_data[i]);
                        ok = false;
                    }
                if (!ok)
                    goto xc4s_cleanup;
            }
        }

        printf("  PASSED: (4-word cross-column syncthreads determinism)\n");
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        goto xc4s_done;

xc4s_cleanup:
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        return 1;

xc4s_done:;
    }
    // 3-word unclaimed pivot syncthreads determinism: 10 runs
    //
    // Runs syncthreadsReductionRef 10 times on a 3-word unclaimed pivot
    // chain, verifying all runs produce identical pivots AND column data.
    //
    // Chain (race-free):
    //   Col 0: [0b100, 0, 0] (pivot=2,   word 0).  pivot_to_col[2]=1.
    //   Col 1: [0, 0b010, 0] (pivot=65,  word 1).  pivot_to_col[65]=2.
    //   Col 2: [0, 0, 0b001] (pivot=128, word 2).  pivot_to_col[128]=-1.
    //
    // Both Col 1 and Col 0 converge to unclaimed pivot=128 after 1 iter.
    // Expected: all 10 runs produce identical new_pivots AND column data.
    {
        constexpr int num_words = 3;
        constexpr int num_cols = 3;
        constexpr int num_runs = 10;

        const uint64_t h_cols[num_cols * num_words] = {
            0b100ULL, 0x0000,   0x0000,  // col 0: pivot=2
            0x0000,   0b010ULL, 0x0000,  // col 1: pivot=65
            0x0000,   0x0000,   0b001ULL // col 2: pivot=128 (unclaimed)
        };
        const int h_pivot_to_col[192] = {};
        const_cast<int *>(h_pivot_to_col)[2] = 1;
        const_cast<int *>(h_pivot_to_col)[65] = 2;
        const_cast<int *>(h_pivot_to_col)[128] = -1;
        const int h_col_pivots[num_cols] = {2, 65, 128};

        const int block_threads = (32 > num_cols * 32) ? 32 : num_cols * 32;
        const size_t col_bytes = sizeof(h_cols);
        const size_t ptc_bytes = sizeof(h_pivot_to_col);
        const size_t idx_bytes = sizeof(int) * num_cols;

        uint64_t *d_cols;
        int *d_pivot_to_col, *d_col_pivots, *d_new_pivots;
        check_cuda(cudaMalloc(&d_cols, col_bytes), "uc3s d_cols");
        check_cuda(cudaMalloc(&d_pivot_to_col, ptc_bytes), "uc3s d_ptc");
        check_cuda(cudaMalloc(&d_col_pivots, idx_bytes), "uc3s d_cp");
        check_cuda(cudaMalloc(&d_new_pivots, idx_bytes), "uc3s d_np");
        check_cuda(cudaMemcpy(d_pivot_to_col, h_pivot_to_col, ptc_bytes, cudaMemcpyHostToDevice),
                   "uc3s h2d ptc");
        check_cuda(cudaMemcpy(d_col_pivots, h_col_pivots, idx_bytes, cudaMemcpyHostToDevice),
                   "uc3s h2d cp");

        int ref_pivots[num_cols] = {};
        uint64_t ref_data[num_cols * num_words] = {};

        for (int run = 0; run < num_runs; ++run)
        {
            check_cuda(cudaMemcpy(d_cols, h_cols, col_bytes, cudaMemcpyHostToDevice),
                       "uc3s h2d cols");
            check_cuda(cudaMemset(d_new_pivots, 0xFF, idx_bytes), "uc3s memset new_pivots");

            syncthreadsReductionRef<<<1, block_threads>>>(d_cols, d_col_pivots, d_pivot_to_col,
                                                          num_words, num_cols, d_new_pivots);
            check_cuda(cudaDeviceSynchronize(), "uc3s sync");

            int run_pivots[num_cols] = {};
            check_cuda(cudaMemcpy(run_pivots, d_new_pivots, idx_bytes, cudaMemcpyDeviceToHost),
                       "uc3s d2h pivots");

            uint64_t run_data[num_cols * num_words] = {};
            check_cuda(cudaMemcpy(run_data, d_cols, col_bytes, cudaMemcpyDeviceToHost),
                       "uc3s d2h data");

            if (run == 0)
            {
                for (int i = 0; i < num_cols; ++i)
                    ref_pivots[i] = run_pivots[i];
                for (int i = 0; i < num_cols * num_words; ++i)
                    ref_data[i] = run_data[i];
            }
            else
            {
                bool ok = true;
                for (int i = 0; i < num_cols; ++i)
                    if (run_pivots[i] != ref_pivots[i])
                    {
                        fprintf(stderr, "uc3s[run=%d] pivot[%d] = %d, ref %d\n", run, i,
                                run_pivots[i], ref_pivots[i]);
                        ok = false;
                    }
                for (int i = 0; i < num_cols * num_words; ++i)
                    if (run_data[i] != ref_data[i])
                    {
                        fprintf(stderr, "uc3s[run=%d] data[%d] = 0x%lx, ref 0x%lx\n", run, i,
                                run_data[i], ref_data[i]);
                        ok = false;
                    }
                if (!ok)
                    goto uc3s_cleanup;
            }
        }

        printf("  PASSED: (3-word unclaimed pivot syncthreads determinism)\n");
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        goto uc3s_done;

uc3s_cleanup:
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        return 1;

uc3s_done:;
    }
    // 3-word unclaimed pivot regular kernel determinism: 10 runs
    //
    // Runs launchPipelinedReduction with use_async_copy=false 10 times on
    // the 3-word unclaimed pivot chain, verifying all runs produce
    // identical pivots AND column data.
    //
    // Chain (same as test 63, race-free):
    //   Col 0: [0b100, 0, 0] (pivot=2,   word 0).  pivot_to_col[2]=1.
    //   Col 1: [0, 0b010, 0] (pivot=65,  word 1).  pivot_to_col[65]=2.
    //   Col 2: [0, 0, 0b001] (pivot=128, word 2).  pivot_to_col[128]=-1.
    //
    // Both Col 1 and Col 0 converge to unclaimed pivot=128 after 1 iter.
    // Expected: all 10 runs produce identical new_pivots AND column data.
    {
        constexpr int num_words = 3;
        constexpr int num_cols = 3;
        constexpr int num_runs = 10;

        const uint64_t h_cols[num_cols * num_words] = {
            0b100ULL, 0x0000,   0x0000,  // col 0: pivot=2
            0x0000,   0b010ULL, 0x0000,  // col 1: pivot=65
            0x0000,   0x0000,   0b001ULL // col 2: pivot=128 (unclaimed)
        };
        const int h_pivot_to_col[192] = {};
        const_cast<int *>(h_pivot_to_col)[2] = 1;
        const_cast<int *>(h_pivot_to_col)[65] = 2;
        const_cast<int *>(h_pivot_to_col)[128] = -1;
        const int h_col_pivots[num_cols] = {2, 65, 128};

        const size_t col_bytes = sizeof(h_cols);
        const size_t ptc_bytes = sizeof(h_pivot_to_col);
        const size_t idx_bytes = sizeof(int) * num_cols;

        uint64_t *d_cols;
        int *d_pivot_to_col, *d_col_pivots, *d_new_pivots;
        check_cuda(cudaMalloc(&d_cols, col_bytes), "uc3r d_cols");
        check_cuda(cudaMalloc(&d_pivot_to_col, ptc_bytes), "uc3r d_ptc");
        check_cuda(cudaMalloc(&d_col_pivots, idx_bytes), "uc3r d_cp");
        check_cuda(cudaMalloc(&d_new_pivots, idx_bytes), "uc3r d_np");
        check_cuda(cudaMemcpy(d_pivot_to_col, h_pivot_to_col, ptc_bytes, cudaMemcpyHostToDevice),
                   "uc3r h2d ptc");
        check_cuda(cudaMemcpy(d_col_pivots, h_col_pivots, idx_bytes, cudaMemcpyHostToDevice),
                   "uc3r h2d cp");

        int ref_pivots[num_cols] = {};
        uint64_t ref_data[num_cols * num_words] = {};

        for (int run = 0; run < num_runs; ++run)
        {
            check_cuda(cudaMemcpy(d_cols, h_cols, col_bytes, cudaMemcpyHostToDevice),
                       "uc3r h2d cols");
            check_cuda(cudaMemset(d_new_pivots, 0xFF, idx_bytes), "uc3r memset new_pivots");

            nerve::persistence::gpu::launchPipelinedReduction(
                d_cols, d_col_pivots, d_pivot_to_col, num_words, num_cols, d_new_pivots, 0, false);
            check_cuda(cudaDeviceSynchronize(), "uc3r sync");

            int run_pivots[num_cols] = {};
            check_cuda(cudaMemcpy(run_pivots, d_new_pivots, idx_bytes, cudaMemcpyDeviceToHost),
                       "uc3r d2h pivots");

            uint64_t run_data[num_cols * num_words] = {};
            check_cuda(cudaMemcpy(run_data, d_cols, col_bytes, cudaMemcpyDeviceToHost),
                       "uc3r d2h data");

            if (run == 0)
            {
                for (int i = 0; i < num_cols; ++i)
                    ref_pivots[i] = run_pivots[i];
                for (int i = 0; i < num_cols * num_words; ++i)
                    ref_data[i] = run_data[i];
            }
            else
            {
                bool ok = true;
                for (int i = 0; i < num_cols; ++i)
                    if (run_pivots[i] != ref_pivots[i])
                    {
                        fprintf(stderr, "uc3r[run=%d] pivot[%d] = %d, ref %d\n", run, i,
                                run_pivots[i], ref_pivots[i]);
                        ok = false;
                    }
                for (int i = 0; i < num_cols * num_words; ++i)
                    if (run_data[i] != ref_data[i])
                    {
                        fprintf(stderr, "uc3r[run=%d] data[%d] = 0x%lx, ref 0x%lx\n", run, i,
                                run_data[i], ref_data[i]);
                        ok = false;
                    }
                if (!ok)
                    goto uc3r_cleanup;
            }
        }

        printf("  PASSED: (3-word unclaimed pivot regular kernel determinism)\n");
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        goto uc3r_done;

uc3r_cleanup:
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        return 1;

uc3r_done:;
    }
    // 2-word cross-column async determinism (test 23's chain): 10 runs
    //
    // Runs launchPipelinedReduction with use_async_copy=true 10 times on
    // the 2-word cross-column chain from test 23, verifying all runs
    // produce identical pivots AND column data.
    //
    // Chain (same as test 23, race-free):
    //   Col 0: [0x0000, 0b100] (pivot=66, word 1).  pivot_to_col[66]=1.
    //   Col 1: [0b010, 0x0000] (pivot=1,  word 0).  pivot_to_col[1]=2.
    //   Col 2: [0x0000, 0b001] (pivot=64, word 1).  pivot_to_col[64]=-1.
    //
    // Expected: all 10 runs produce identical new_pivots AND column data.
    {
        constexpr int num_words = 2;
        constexpr int num_cols = 3;
        constexpr int num_runs = 10;

        const uint64_t h_cols[num_cols * num_words] = {
            0x0000ULL, 0b100ULL,  // col 0: pivot=66
            0b010ULL,  0x0000ULL, // col 1: pivot=1
            0x0000ULL, 0b001ULL   // col 2: pivot=64 (unclaimed)
        };
        const int h_pivot_to_col[128] = {};
        const_cast<int *>(h_pivot_to_col)[66] = 1;
        const_cast<int *>(h_pivot_to_col)[1] = 2;
        const int h_col_pivots[num_cols] = {66, 1, 64};

        const size_t col_bytes = sizeof(h_cols);
        const size_t ptc_bytes = sizeof(h_pivot_to_col);
        const size_t idx_bytes = sizeof(int) * num_cols;

        uint64_t *d_cols;
        int *d_pivot_to_col, *d_col_pivots, *d_new_pivots;
        check_cuda(cudaMalloc(&d_cols, col_bytes), "xca d_cols");
        check_cuda(cudaMalloc(&d_pivot_to_col, ptc_bytes), "xca d_ptc");
        check_cuda(cudaMalloc(&d_col_pivots, idx_bytes), "xca d_cp");
        check_cuda(cudaMalloc(&d_new_pivots, idx_bytes), "xca d_np");
        check_cuda(cudaMemcpy(d_pivot_to_col, h_pivot_to_col, ptc_bytes, cudaMemcpyHostToDevice),
                   "xca h2d ptc");
        check_cuda(cudaMemcpy(d_col_pivots, h_col_pivots, idx_bytes, cudaMemcpyHostToDevice),
                   "xca h2d cp");

        int ref_pivots[num_cols] = {};
        uint64_t ref_data[num_cols * num_words] = {};

        for (int run = 0; run < num_runs; ++run)
        {
            check_cuda(cudaMemcpy(d_cols, h_cols, col_bytes, cudaMemcpyHostToDevice),
                       "xca h2d cols");
            check_cuda(cudaMemset(d_new_pivots, 0xFF, idx_bytes), "xca memset new_pivots");

            nerve::persistence::gpu::launchPipelinedReduction(
                d_cols, d_col_pivots, d_pivot_to_col, num_words, num_cols, d_new_pivots, 0, true);
            check_cuda(cudaDeviceSynchronize(), "xca sync");

            int run_pivots[num_cols] = {};
            check_cuda(cudaMemcpy(run_pivots, d_new_pivots, idx_bytes, cudaMemcpyDeviceToHost),
                       "xca d2h pivots");

            uint64_t run_data[num_cols * num_words] = {};
            check_cuda(cudaMemcpy(run_data, d_cols, col_bytes, cudaMemcpyDeviceToHost),
                       "xca d2h data");

            if (run == 0)
            {
                for (int i = 0; i < num_cols; ++i)
                    ref_pivots[i] = run_pivots[i];
                for (int i = 0; i < num_cols * num_words; ++i)
                    ref_data[i] = run_data[i];
            }
            else
            {
                bool ok = true;
                for (int i = 0; i < num_cols; ++i)
                    if (run_pivots[i] != ref_pivots[i])
                    {
                        fprintf(stderr, "xca[run=%d] pivot[%d] = %d, ref %d\n", run, i,
                                run_pivots[i], ref_pivots[i]);
                        ok = false;
                    }
                for (int i = 0; i < num_cols * num_words; ++i)
                    if (run_data[i] != ref_data[i])
                    {
                        fprintf(stderr, "xca[run=%d] data[%d] = 0x%lx, ref 0x%lx\n", run, i,
                                run_data[i], ref_data[i]);
                        ok = false;
                    }
                if (!ok)
                    goto xca_cleanup;
            }
        }

        printf("  PASSED: (2-word cross-column async determinism)\n");
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        goto xca_done;

xca_cleanup:
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        return 1;

xca_done:;
    }
    // 2-word self-clearing async determinism (test 27's chain): 10 runs
    //
    // Runs launchPipelinedReduction with use_async_copy=true 10 times on
    // the 2-word self-clearing chain from test 27, verifying all runs
    // produce identical pivots AND column data.
    //
    // Chain (same as test 27, race-free):
    //   Col 0: [0b100, 0x0000] (pivot=2,  word 0).  pivot_to_col[2]=0.
    //   Col 1: [0x0000, 0b001] (pivot=64, word 1).  pivot_to_col[64]=1.
    //   Col 2: [0b010, 0x0000] (pivot=1,  word 0).  pivot_to_col[1]=2.
    //
    // Each column claims its own pivot. All columns are zeroed.
    // Expected: all 10 runs produce identical new_pivots AND column data.
    {
        constexpr int num_words = 2;
        constexpr int num_cols = 3;
        constexpr int num_runs = 10;

        const uint64_t h_cols[num_cols * num_words] = {
            0b100ULL,  0x0000ULL, // col 0: pivot=2
            0x0000ULL, 0b001ULL,  // col 1: pivot=64
            0b010ULL,  0x0000ULL  // col 2: pivot=1
        };
        int h_pivot_to_col[128];
        std::memset(h_pivot_to_col, 0xFF, sizeof(h_pivot_to_col));
        h_pivot_to_col[2] = 0;
        h_pivot_to_col[64] = 1;
        h_pivot_to_col[1] = 2;
        const int h_col_pivots[num_cols] = {2, 64, 1};

        const size_t col_bytes = sizeof(h_cols);
        const size_t ptc_bytes = sizeof(h_pivot_to_col);
        const size_t idx_bytes = sizeof(int) * num_cols;

        uint64_t *d_cols;
        int *d_pivot_to_col, *d_col_pivots, *d_new_pivots;
        check_cuda(cudaMalloc(&d_cols, col_bytes), "sca d_cols");
        check_cuda(cudaMalloc(&d_pivot_to_col, ptc_bytes), "sca d_ptc");
        check_cuda(cudaMalloc(&d_col_pivots, idx_bytes), "sca d_cp");
        check_cuda(cudaMalloc(&d_new_pivots, idx_bytes), "sca d_np");
        check_cuda(cudaMemcpy(d_pivot_to_col, h_pivot_to_col, ptc_bytes, cudaMemcpyHostToDevice),
                   "sca h2d ptc");
        check_cuda(cudaMemcpy(d_col_pivots, h_col_pivots, idx_bytes, cudaMemcpyHostToDevice),
                   "sca h2d cp");

        int ref_pivots[num_cols] = {};
        uint64_t ref_data[num_cols * num_words] = {};

        for (int run = 0; run < num_runs; ++run)
        {
            check_cuda(cudaMemcpy(d_cols, h_cols, col_bytes, cudaMemcpyHostToDevice),
                       "sca h2d cols");
            check_cuda(cudaMemset(d_new_pivots, 0xFF, idx_bytes), "sca memset new_pivots");

            nerve::persistence::gpu::launchPipelinedReduction(
                d_cols, d_col_pivots, d_pivot_to_col, num_words, num_cols, d_new_pivots, 0, true);
            check_cuda(cudaDeviceSynchronize(), "sca sync");

            int run_pivots[num_cols] = {};
            check_cuda(cudaMemcpy(run_pivots, d_new_pivots, idx_bytes, cudaMemcpyDeviceToHost),
                       "sca d2h pivots");

            uint64_t run_data[num_cols * num_words] = {};
            check_cuda(cudaMemcpy(run_data, d_cols, col_bytes, cudaMemcpyDeviceToHost),
                       "sca d2h data");

            if (run == 0)
            {
                for (int i = 0; i < num_cols; ++i)
                    ref_pivots[i] = run_pivots[i];
                for (int i = 0; i < num_cols * num_words; ++i)
                    ref_data[i] = run_data[i];
            }
            else
            {
                bool ok = true;
                for (int i = 0; i < num_cols; ++i)
                    if (run_pivots[i] != ref_pivots[i])
                    {
                        fprintf(stderr, "sca[run=%d] pivot[%d] = %d, ref %d\n", run, i,
                                run_pivots[i], ref_pivots[i]);
                        ok = false;
                    }
                for (int i = 0; i < num_cols * num_words; ++i)
                    if (run_data[i] != ref_data[i])
                    {
                        fprintf(stderr, "sca[run=%d] data[%d] = 0x%lx, ref 0x%lx\n", run, i,
                                run_data[i], ref_data[i]);
                        ok = false;
                    }
                if (!ok)
                    goto sca_cleanup;
            }
        }

        printf("  PASSED: (2-word self-clearing async determinism)\n");
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        goto sca_done;

sca_cleanup:
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        return 1;

sca_done:;
    }
    // 2-word unclaimed pivot async determinism (test 35's chain): 10 runs
    //
    // Runs launchPipelinedReduction with use_async_copy=true 10 times on
    // the 2-word unclaimed pivot chain from test 35, verifying all runs
    // produce identical pivots AND column data.
    //
    // Chain (same as test 35, race-free):
    //   Col 0: [0b100, 0x0000] (pivot=2,  word 0).  pivot_to_col[2]=1.
    //   Col 1: [0b010, 0x0000] (pivot=1,  word 0).  pivot_to_col[1]=2.
    //   Col 2: [0x0000, 0b001 ] (pivot=64, word 1).  pivot_to_col[64]=-1.
    //
    // Both Col 1 and Col 0 converge to unclaimed pivot=64 after 1 iter.
    // Expected: all 10 runs produce identical new_pivots AND column data.
    {
        constexpr int num_words = 2;
        constexpr int num_cols = 3;
        constexpr int num_runs = 10;

        const uint64_t h_cols[num_cols * num_words] = {
            0b100ULL, 0x0000,  // col 0: pivot=2
            0b010ULL, 0x0000,  // col 1: pivot=1
            0x0000,   0b001ULL // col 2: pivot=64 (unclaimed)
        };
        const int h_pivot_to_col[128] = {};
        const_cast<int *>(h_pivot_to_col)[2] = 1;
        const_cast<int *>(h_pivot_to_col)[1] = 2;
        const_cast<int *>(h_pivot_to_col)[64] = -1;
        const int h_col_pivots[num_cols] = {2, 1, 64};

        const size_t col_bytes = sizeof(h_cols);
        const size_t ptc_bytes = sizeof(h_pivot_to_col);
        const size_t idx_bytes = sizeof(int) * num_cols;

        uint64_t *d_cols;
        int *d_pivot_to_col, *d_col_pivots, *d_new_pivots;
        check_cuda(cudaMalloc(&d_cols, col_bytes), "uca d_cols");
        check_cuda(cudaMalloc(&d_pivot_to_col, ptc_bytes), "uca d_ptc");
        check_cuda(cudaMalloc(&d_col_pivots, idx_bytes), "uca d_cp");
        check_cuda(cudaMalloc(&d_new_pivots, idx_bytes), "uca d_np");
        check_cuda(cudaMemcpy(d_pivot_to_col, h_pivot_to_col, ptc_bytes, cudaMemcpyHostToDevice),
                   "uca h2d ptc");
        check_cuda(cudaMemcpy(d_col_pivots, h_col_pivots, idx_bytes, cudaMemcpyHostToDevice),
                   "uca h2d cp");

        int ref_pivots[num_cols] = {};
        uint64_t ref_data[num_cols * num_words] = {};

        for (int run = 0; run < num_runs; ++run)
        {
            check_cuda(cudaMemcpy(d_cols, h_cols, col_bytes, cudaMemcpyHostToDevice),
                       "uca h2d cols");
            check_cuda(cudaMemset(d_new_pivots, 0xFF, idx_bytes), "uca memset new_pivots");

            nerve::persistence::gpu::launchPipelinedReduction(
                d_cols, d_col_pivots, d_pivot_to_col, num_words, num_cols, d_new_pivots, 0, true);
            check_cuda(cudaDeviceSynchronize(), "uca sync");

            int run_pivots[num_cols] = {};
            check_cuda(cudaMemcpy(run_pivots, d_new_pivots, idx_bytes, cudaMemcpyDeviceToHost),
                       "uca d2h pivots");

            uint64_t run_data[num_cols * num_words] = {};
            check_cuda(cudaMemcpy(run_data, d_cols, col_bytes, cudaMemcpyDeviceToHost),
                       "uca d2h data");

            if (run == 0)
            {
                for (int i = 0; i < num_cols; ++i)
                    ref_pivots[i] = run_pivots[i];
                for (int i = 0; i < num_cols * num_words; ++i)
                    ref_data[i] = run_data[i];
            }
            else
            {
                bool ok = true;
                for (int i = 0; i < num_cols; ++i)
                    if (run_pivots[i] != ref_pivots[i])
                    {
                        fprintf(stderr, "uca[run=%d] pivot[%d] = %d, ref %d\n", run, i,
                                run_pivots[i], ref_pivots[i]);
                        ok = false;
                    }
                for (int i = 0; i < num_cols * num_words; ++i)
                    if (run_data[i] != ref_data[i])
                    {
                        fprintf(stderr, "uca[run=%d] data[%d] = 0x%lx, ref 0x%lx\n", run, i,
                                run_data[i], ref_data[i]);
                        ok = false;
                    }
                if (!ok)
                    goto uca_cleanup;
            }
        }

        printf("  PASSED: (2-word unclaimed pivot async determinism)\n");
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        goto uca_done;

uca_cleanup:
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        return 1;

uca_done:;
    }
    // Column Add determinism (2 words): 10 runs
    //
    // Runs launchWarpSpecializedColumnAdd 10 times with 2-word columns,
    // verifying all runs produce identical destination column data.
    {
        constexpr int num_words = 2;
        constexpr int num_cols = 3;
        constexpr int num_runs = 10;

        const uint64_t h_dest[num_cols * num_words] = {
            0b100ULL, 0x0000, // col 0
            0b010ULL, 0x0000, // col 1
            0b001ULL, 0x0000  // col 2
        };
        const uint64_t h_src[num_cols * num_words] = {
            0b010ULL, 0x0000, // col 0 src
            0b001ULL, 0x0000, // col 1 src
            0b100ULL, 0x0000  // col 2 src
        };
        const int h_col_sizes[num_cols] = {num_words, num_words, num_words};

        const size_t col_bytes = sizeof(h_dest);

        uint64_t *d_dest, *d_src;
        int *d_col_sizes;
        check_cuda(cudaMalloc(&d_dest, col_bytes), "ca2 d_dest");
        check_cuda(cudaMalloc(&d_src, col_bytes), "ca2 d_src");
        check_cuda(cudaMalloc(&d_col_sizes, sizeof(h_col_sizes)), "ca2 d_sizes");
        check_cuda(cudaMemcpy(d_src, h_src, col_bytes, cudaMemcpyHostToDevice), "ca2 h2d src");
        check_cuda(
            cudaMemcpy(d_col_sizes, h_col_sizes, sizeof(h_col_sizes), cudaMemcpyHostToDevice),
            "ca2 h2d sizes");

        uint64_t ref_data[num_cols * num_words] = {};

        for (int run = 0; run < num_runs; ++run)
        {
            check_cuda(cudaMemcpy(d_dest, h_dest, col_bytes, cudaMemcpyHostToDevice),
                       "ca2 h2d dest");

            nerve::persistence::gpu::launchWarpSpecializedColumnAdd(d_dest, d_src, d_col_sizes,
                                                                    num_words, num_cols);
            check_cuda(cudaDeviceSynchronize(), "ca2 sync");

            uint64_t run_data[num_cols * num_words] = {};
            check_cuda(cudaMemcpy(run_data, d_dest, col_bytes, cudaMemcpyDeviceToHost),
                       "ca2 d2h data");

            if (run == 0)
            {
                for (int i = 0; i < num_cols * num_words; ++i)
                    ref_data[i] = run_data[i];
            }
            else
            {
                bool ok = true;
                for (int i = 0; i < num_cols * num_words; ++i)
                    if (run_data[i] != ref_data[i])
                    {
                        fprintf(stderr, "ca2[run=%d] data[%d] = 0x%lx, ref 0x%lx\n", run, i,
                                run_data[i], ref_data[i]);
                        ok = false;
                    }
                if (!ok)
                    goto ca2_cleanup;
            }
        }

        printf("  PASSED: (Column Add 2w determinism)\n");
        cudaFree(d_dest);
        cudaFree(d_src);
        cudaFree(d_col_sizes);
        goto ca2_done;

ca2_cleanup:
        cudaFree(d_dest);
        cudaFree(d_src);
        cudaFree(d_col_sizes);
        return 1;

ca2_done:;
    }
    // Column Add determinism (3 words): 10 runs
    //
    // Runs launchWarpSpecializedColumnAdd 10 times with 3-word columns.
    {
        constexpr int num_words = 3;
        constexpr int num_cols = 3;
        constexpr int num_runs = 10;

        const uint64_t h_dest[num_cols * num_words] = {0b100ULL, 0x0000, 0x0000, 0x0000,  0b010ULL,
                                                       0x0000,   0x0000, 0x0000, 0b001ULL};
        const uint64_t h_src[num_cols * num_words] = {0b010ULL, 0x0000, 0x0000, 0x0000,  0b001ULL,
                                                      0x0000,   0x0000, 0x0000, 0b100ULL};
        const int h_col_sizes[num_cols] = {num_words, num_words, num_words};

        const size_t col_bytes = sizeof(h_dest);

        uint64_t *d_dest, *d_src;
        int *d_col_sizes;
        check_cuda(cudaMalloc(&d_dest, col_bytes), "ca3 d_dest");
        check_cuda(cudaMalloc(&d_src, col_bytes), "ca3 d_src");
        check_cuda(cudaMalloc(&d_col_sizes, sizeof(h_col_sizes)), "ca3 d_sizes");
        check_cuda(cudaMemcpy(d_src, h_src, col_bytes, cudaMemcpyHostToDevice), "ca3 h2d src");
        check_cuda(
            cudaMemcpy(d_col_sizes, h_col_sizes, sizeof(h_col_sizes), cudaMemcpyHostToDevice),
            "ca3 h2d sizes");

        uint64_t ref_data[num_cols * num_words] = {};

        for (int run = 0; run < num_runs; ++run)
        {
            check_cuda(cudaMemcpy(d_dest, h_dest, col_bytes, cudaMemcpyHostToDevice),
                       "ca3 h2d dest");

            nerve::persistence::gpu::launchWarpSpecializedColumnAdd(d_dest, d_src, d_col_sizes,
                                                                    num_words, num_cols);
            check_cuda(cudaDeviceSynchronize(), "ca3 sync");

            uint64_t run_data[num_cols * num_words] = {};
            check_cuda(cudaMemcpy(run_data, d_dest, col_bytes, cudaMemcpyDeviceToHost),
                       "ca3 d2h data");

            if (run == 0)
            {
                for (int i = 0; i < num_cols * num_words; ++i)
                    ref_data[i] = run_data[i];
            }
            else
            {
                bool ok = true;
                for (int i = 0; i < num_cols * num_words; ++i)
                    if (run_data[i] != ref_data[i])
                    {
                        fprintf(stderr, "ca3[run=%d] data[%d] = 0x%lx, ref 0x%lx\n", run, i,
                                run_data[i], ref_data[i]);
                        ok = false;
                    }
                if (!ok)
                    goto ca3_cleanup;
            }
        }

        printf("  PASSED: (Column Add 3w determinism)\n");
        cudaFree(d_dest);
        cudaFree(d_src);
        cudaFree(d_col_sizes);
        goto ca3_done;

ca3_cleanup:
        cudaFree(d_dest);
        cudaFree(d_src);
        cudaFree(d_col_sizes);
        return 1;

ca3_done:;
    }
    // Pivot Find determinism (2 words): 10 runs
    //
    // Runs launchWarpSpecializedPivotFind 10 times with 2-word columns.
    {
        constexpr int num_words = 2;
        constexpr int num_cols = 3;
        constexpr int num_runs = 10;

        const uint64_t h_cols[num_cols * num_words] = {
            0b100ULL, 0x0000,   // col 0: pivot=2
            0x0000,   0b001ULL, // col 1: pivot=64
            0b010ULL, 0x0000    // col 2: pivot=1
        };
        const int h_col_sizes[num_cols] = {num_words, num_words, num_words};

        const size_t col_bytes = sizeof(h_cols);

        uint64_t *d_cols;
        int *d_col_sizes, *d_pivots;
        check_cuda(cudaMalloc(&d_cols, col_bytes), "pf2 d_cols");
        check_cuda(cudaMalloc(&d_col_sizes, sizeof(h_col_sizes)), "pf2 d_sizes");
        check_cuda(cudaMalloc(&d_pivots, sizeof(int) * num_cols), "pf2 d_pivots");
        check_cuda(
            cudaMemcpy(d_col_sizes, h_col_sizes, sizeof(h_col_sizes), cudaMemcpyHostToDevice),
            "pf2 h2d sizes");

        int ref_pivots[num_cols] = {};

        for (int run = 0; run < num_runs; ++run)
        {
            check_cuda(cudaMemcpy(d_cols, h_cols, col_bytes, cudaMemcpyHostToDevice),
                       "pf2 h2d cols");
            check_cuda(cudaMemset(d_pivots, 0xFF, sizeof(int) * num_cols), "pf2 memset");

            nerve::persistence::gpu::launchWarpSpecializedPivotFind(d_cols, d_col_sizes, num_words,
                                                                    num_cols, d_pivots);
            check_cuda(cudaDeviceSynchronize(), "pf2 sync");

            int run_pivots[num_cols] = {};
            check_cuda(
                cudaMemcpy(run_pivots, d_pivots, sizeof(int) * num_cols, cudaMemcpyDeviceToHost),
                "pf2 d2h pivots");

            if (run == 0)
            {
                for (int i = 0; i < num_cols; ++i)
                    ref_pivots[i] = run_pivots[i];
            }
            else
            {
                bool ok = true;
                for (int i = 0; i < num_cols; ++i)
                    if (run_pivots[i] != ref_pivots[i])
                    {
                        fprintf(stderr, "pf2[run=%d] pivot[%d] = %d, ref %d\n", run, i,
                                run_pivots[i], ref_pivots[i]);
                        ok = false;
                    }
                if (!ok)
                    goto pf2_cleanup;
            }
        }

        printf("  PASSED: (Pivot Find 2w determinism)\n");
        cudaFree(d_cols);
        cudaFree(d_col_sizes);
        cudaFree(d_pivots);
        goto pf2_done;

pf2_cleanup:
        cudaFree(d_cols);
        cudaFree(d_col_sizes);
        cudaFree(d_pivots);
        return 1;

pf2_done:;
    }
    // Pivot Find determinism (3 words): 10 runs
    {
        constexpr int num_words = 3;
        constexpr int num_cols = 3;
        constexpr int num_runs = 10;

        const uint64_t h_cols[num_cols * num_words] = {
            0b100ULL, 0x0000,   0x0000,  // col 0: pivot=2
            0x0000,   0b001ULL, 0x0000,  // col 1: pivot=64
            0x0000,   0x0000,   0b010ULL // col 2: pivot=128
        };
        const int h_col_sizes[num_cols] = {num_words, num_words, num_words};

        const size_t col_bytes = sizeof(h_cols);

        uint64_t *d_cols;
        int *d_col_sizes, *d_pivots;
        check_cuda(cudaMalloc(&d_cols, col_bytes), "pf3 d_cols");
        check_cuda(cudaMalloc(&d_col_sizes, sizeof(h_col_sizes)), "pf3 d_sizes");
        check_cuda(cudaMalloc(&d_pivots, sizeof(int) * num_cols), "pf3 d_pivots");
        check_cuda(
            cudaMemcpy(d_col_sizes, h_col_sizes, sizeof(h_col_sizes), cudaMemcpyHostToDevice),
            "pf3 h2d sizes");

        int ref_pivots[num_cols] = {};

        for (int run = 0; run < num_runs; ++run)
        {
            check_cuda(cudaMemcpy(d_cols, h_cols, col_bytes, cudaMemcpyHostToDevice),
                       "pf3 h2d cols");
            check_cuda(cudaMemset(d_pivots, 0xFF, sizeof(int) * num_cols), "pf3 memset");

            nerve::persistence::gpu::launchWarpSpecializedPivotFind(d_cols, d_col_sizes, num_words,
                                                                    num_cols, d_pivots);
            check_cuda(cudaDeviceSynchronize(), "pf3 sync");

            int run_pivots[num_cols] = {};
            check_cuda(
                cudaMemcpy(run_pivots, d_pivots, sizeof(int) * num_cols, cudaMemcpyDeviceToHost),
                "pf3 d2h pivots");

            if (run == 0)
            {
                for (int i = 0; i < num_cols; ++i)
                    ref_pivots[i] = run_pivots[i];
            }
            else
            {
                bool ok = true;
                for (int i = 0; i < num_cols; ++i)
                    if (run_pivots[i] != ref_pivots[i])
                    {
                        fprintf(stderr, "pf3[run=%d] pivot[%d] = %d, ref %d\n", run, i,
                                run_pivots[i], ref_pivots[i]);
                        ok = false;
                    }
                if (!ok)
                    goto pf3_cleanup;
            }
        }

        printf("  PASSED: (Pivot Find 3w determinism)\n");
        cudaFree(d_cols);
        cudaFree(d_col_sizes);
        cudaFree(d_pivots);
        goto pf3_done;

pf3_cleanup:
        cudaFree(d_cols);
        cudaFree(d_col_sizes);
        cudaFree(d_pivots);
        return 1;

pf3_done:;
    }
    // 8-word self-clearing syncthreads determinism: 10 runs
    //
    // 8 words per column exercises the loop-based MSB scanning path
    // in the pivot find code (pivots span across 512 bits).
    {
        constexpr int num_words = 8;
        constexpr int num_cols = 3;
        constexpr int num_runs = 10;

        uint64_t h_cols[num_cols * num_words] = {};
        h_cols[0] = 0b100ULL;                 // col 0: pivot=2   (word 0, bit 2)
        h_cols[num_words + 4] = 0b001ULL;     // col 1: pivot=256 (word 4, bit 0)
        h_cols[2 * num_words + 7] = 0b010ULL; // col 2: pivot=449 (word 7, bit 1)

        const int h_pivot_to_col[512] = {};
        const_cast<int *>(h_pivot_to_col)[2] = 0;
        const_cast<int *>(h_pivot_to_col)[256] = 1;
        const_cast<int *>(h_pivot_to_col)[449] = 2;
        const int h_col_pivots[num_cols] = {2, 256, 449};

        const int block_threads = (32 > num_cols * 32) ? 32 : num_cols * 32;
        const size_t col_bytes = sizeof(h_cols);
        const size_t ptc_bytes = sizeof(h_pivot_to_col);
        const size_t idx_bytes = sizeof(int) * num_cols;

        uint64_t *d_cols;
        int *d_pivot_to_col, *d_col_pivots, *d_new_pivots;
        check_cuda(cudaMalloc(&d_cols, col_bytes), "lw8 d_cols");
        check_cuda(cudaMalloc(&d_pivot_to_col, ptc_bytes), "lw8 d_ptc");
        check_cuda(cudaMalloc(&d_col_pivots, idx_bytes), "lw8 d_cp");
        check_cuda(cudaMalloc(&d_new_pivots, idx_bytes), "lw8 d_np");
        check_cuda(cudaMemcpy(d_pivot_to_col, h_pivot_to_col, ptc_bytes, cudaMemcpyHostToDevice),
                   "lw8 h2d ptc");
        check_cuda(cudaMemcpy(d_col_pivots, h_col_pivots, idx_bytes, cudaMemcpyHostToDevice),
                   "lw8 h2d cp");

        int ref_pivots[num_cols] = {};
        uint64_t ref_data[num_cols * num_words] = {};

        for (int run = 0; run < num_runs; ++run)
        {
            check_cuda(cudaMemcpy(d_cols, h_cols, col_bytes, cudaMemcpyHostToDevice),
                       "lw8 h2d cols");
            check_cuda(cudaMemset(d_new_pivots, 0xFF, idx_bytes), "lw8 memset new_pivots");

            syncthreadsReductionRef<<<1, block_threads>>>(d_cols, d_col_pivots, d_pivot_to_col,
                                                          num_words, num_cols, d_new_pivots);
            check_cuda(cudaDeviceSynchronize(), "lw8 sync");

            int run_pivots[num_cols] = {};
            check_cuda(cudaMemcpy(run_pivots, d_new_pivots, idx_bytes, cudaMemcpyDeviceToHost),
                       "lw8 d2h pivots");

            uint64_t run_data[num_cols * num_words] = {};
            check_cuda(cudaMemcpy(run_data, d_cols, col_bytes, cudaMemcpyDeviceToHost),
                       "lw8 d2h data");

            if (run == 0)
            {
                for (int i = 0; i < num_cols; ++i)
                    ref_pivots[i] = run_pivots[i];
                for (int i = 0; i < num_cols * num_words; ++i)
                    ref_data[i] = run_data[i];
            }
            else
            {
                bool ok = true;
                for (int i = 0; i < num_cols; ++i)
                    if (run_pivots[i] != ref_pivots[i])
                    {
                        fprintf(stderr, "lw8[run=%d] pivot[%d] = %d, ref %d\n", run, i,
                                run_pivots[i], ref_pivots[i]);
                        ok = false;
                    }
                for (int i = 0; i < num_cols * num_words; ++i)
                    if (run_data[i] != ref_data[i])
                    {
                        fprintf(stderr, "lw8[run=%d] data[%d] = 0x%lx, ref 0x%lx\n", run, i,
                                run_data[i], ref_data[i]);
                        ok = false;
                    }
                if (!ok)
                    goto lw8_cleanup;
            }
        }

        printf("  PASSED: (8-word self-clearing syncthreads determinism)\n");
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        goto lw8_done;

lw8_cleanup:
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        return 1;

lw8_done:;
    }
    // 8-word unclaimed pivot syncthreads determinism: 10 runs
    {
        constexpr int num_words = 8;
        constexpr int num_cols = 3;
        constexpr int num_runs = 10;

        uint64_t h_cols[num_cols * num_words] = {};
        h_cols[0] = 0b100ULL;                 // col 0: pivot=2   -> col 1
        h_cols[num_words + 4] = 0b001ULL;     // col 1: pivot=256 -> col 2
        h_cols[2 * num_words + 7] = 0b010ULL; // col 2: pivot=449 (unclaimed)

        const int h_pivot_to_col[512] = {};
        const_cast<int *>(h_pivot_to_col)[2] = 1;
        const_cast<int *>(h_pivot_to_col)[256] = 2;
        const_cast<int *>(h_pivot_to_col)[449] = -1;
        const int h_col_pivots[num_cols] = {2, 256, 449};

        const int block_threads = (32 > num_cols * 32) ? 32 : num_cols * 32;
        const size_t col_bytes = sizeof(h_cols);
        const size_t ptc_bytes = sizeof(h_pivot_to_col);
        const size_t idx_bytes = sizeof(int) * num_cols;

        uint64_t *d_cols;
        int *d_pivot_to_col, *d_col_pivots, *d_new_pivots;
        check_cuda(cudaMalloc(&d_cols, col_bytes), "lu8 d_cols");
        check_cuda(cudaMalloc(&d_pivot_to_col, ptc_bytes), "lu8 d_ptc");
        check_cuda(cudaMalloc(&d_col_pivots, idx_bytes), "lu8 d_cp");
        check_cuda(cudaMalloc(&d_new_pivots, idx_bytes), "lu8 d_np");
        check_cuda(cudaMemcpy(d_pivot_to_col, h_pivot_to_col, ptc_bytes, cudaMemcpyHostToDevice),
                   "lu8 h2d ptc");
        check_cuda(cudaMemcpy(d_col_pivots, h_col_pivots, idx_bytes, cudaMemcpyHostToDevice),
                   "lu8 h2d cp");

        int ref_pivots[num_cols] = {};
        uint64_t ref_data[num_cols * num_words] = {};

        for (int run = 0; run < num_runs; ++run)
        {
            check_cuda(cudaMemcpy(d_cols, h_cols, col_bytes, cudaMemcpyHostToDevice),
                       "lu8 h2d cols");
            check_cuda(cudaMemset(d_new_pivots, 0xFF, idx_bytes), "lu8 memset new_pivots");

            syncthreadsReductionRef<<<1, block_threads>>>(d_cols, d_col_pivots, d_pivot_to_col,
                                                          num_words, num_cols, d_new_pivots);
            check_cuda(cudaDeviceSynchronize(), "lu8 sync");

            int run_pivots[num_cols] = {};
            check_cuda(cudaMemcpy(run_pivots, d_new_pivots, idx_bytes, cudaMemcpyDeviceToHost),
                       "lu8 d2h pivots");

            uint64_t run_data[num_cols * num_words] = {};
            check_cuda(cudaMemcpy(run_data, d_cols, col_bytes, cudaMemcpyDeviceToHost),
                       "lu8 d2h data");

            if (run == 0)
            {
                for (int i = 0; i < num_cols; ++i)
                    ref_pivots[i] = run_pivots[i];
                for (int i = 0; i < num_cols * num_words; ++i)
                    ref_data[i] = run_data[i];
            }
            else
            {
                bool ok = true;
                for (int i = 0; i < num_cols; ++i)
                    if (run_pivots[i] != ref_pivots[i])
                    {
                        fprintf(stderr, "lu8[run=%d] pivot[%d] = %d, ref %d\n", run, i,
                                run_pivots[i], ref_pivots[i]);
                        ok = false;
                    }
                for (int i = 0; i < num_cols * num_words; ++i)
                    if (run_data[i] != ref_data[i])
                    {
                        fprintf(stderr, "lu8[run=%d] data[%d] = 0x%lx, ref 0x%lx\n", run, i,
                                run_data[i], ref_data[i]);
                        ok = false;
                    }
                if (!ok)
                    goto lu8_cleanup;
            }
        }

        printf("  PASSED: (8-word unclaimed pivot syncthreads determinism)\n");
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        goto lu8_done;

lu8_cleanup:
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        return 1;

lu8_done:;
    }
    // 16-column self-clearing syncthreads determinism: 10 runs
    //
    // 16 columns exercises __syncthreads barriers with more warps
    // (block_threads = 16 * 32 = 512).
    {
        constexpr int num_words = 2;
        constexpr int num_cols = 16;
        constexpr int num_runs = 10;

        uint64_t h_cols[num_cols * num_words] = {};
        int h_pivot_to_col[128] = {};
        int h_col_pivots[num_cols] = {};
        for (int c = 0; c < num_cols; ++c)
        {
            h_cols[c * num_words] = (1ULL << c); // each col has bit c
            h_pivot_to_col[c] = c;               // self-claim pivot c
            h_col_pivots[c] = c;
        }

        const int block_threads = (32 > num_cols * 32) ? 32 : num_cols * 32;
        const size_t col_bytes = sizeof(h_cols);
        const size_t ptc_bytes = sizeof(h_pivot_to_col);
        const size_t idx_bytes = sizeof(int) * num_cols;

        uint64_t *d_cols;
        int *d_pivot_to_col, *d_col_pivots, *d_new_pivots;
        check_cuda(cudaMalloc(&d_cols, col_bytes), "lc16 d_cols");
        check_cuda(cudaMalloc(&d_pivot_to_col, ptc_bytes), "lc16 d_ptc");
        check_cuda(cudaMalloc(&d_col_pivots, idx_bytes), "lc16 d_cp");
        check_cuda(cudaMalloc(&d_new_pivots, idx_bytes), "lc16 d_np");
        check_cuda(cudaMemcpy(d_pivot_to_col, h_pivot_to_col, ptc_bytes, cudaMemcpyHostToDevice),
                   "lc16 h2d ptc");
        check_cuda(cudaMemcpy(d_col_pivots, h_col_pivots, idx_bytes, cudaMemcpyHostToDevice),
                   "lc16 h2d cp");

        int ref_pivots[num_cols] = {};
        uint64_t ref_data[num_cols * num_words] = {};

        for (int run = 0; run < num_runs; ++run)
        {
            check_cuda(cudaMemcpy(d_cols, h_cols, col_bytes, cudaMemcpyHostToDevice),
                       "lc16 h2d cols");
            check_cuda(cudaMemset(d_new_pivots, 0xFF, idx_bytes), "lc16 memset new_pivots");

            syncthreadsReductionRef<<<1, block_threads>>>(d_cols, d_col_pivots, d_pivot_to_col,
                                                          num_words, num_cols, d_new_pivots);
            check_cuda(cudaDeviceSynchronize(), "lc16 sync");

            int run_pivots[num_cols] = {};
            check_cuda(cudaMemcpy(run_pivots, d_new_pivots, idx_bytes, cudaMemcpyDeviceToHost),
                       "lc16 d2h pivots");

            uint64_t run_data[num_cols * num_words] = {};
            check_cuda(cudaMemcpy(run_data, d_cols, col_bytes, cudaMemcpyDeviceToHost),
                       "lc16 d2h data");

            if (run == 0)
            {
                for (int i = 0; i < num_cols; ++i)
                    ref_pivots[i] = run_pivots[i];
                for (int i = 0; i < num_cols * num_words; ++i)
                    ref_data[i] = run_data[i];
            }
            else
            {
                bool ok = true;
                for (int i = 0; i < num_cols; ++i)
                    if (run_pivots[i] != ref_pivots[i])
                    {
                        fprintf(stderr, "lc16[run=%d] pivot[%d] = %d, ref %d\n", run, i,
                                run_pivots[i], ref_pivots[i]);
                        ok = false;
                    }
                for (int i = 0; i < num_cols * num_words; ++i)
                    if (run_data[i] != ref_data[i])
                    {
                        fprintf(stderr, "lc16[run=%d] data[%d] = 0x%lx, ref 0x%lx\n", run, i,
                                run_data[i], ref_data[i]);
                        ok = false;
                    }
                if (!ok)
                    goto lc16_cleanup;
            }
        }

        printf("  PASSED: (16-column self-clearing syncthreads determinism)\n");
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        goto lc16_done;

lc16_cleanup:
        cudaFree(d_cols);
        cudaFree(d_pivot_to_col);
        cudaFree(d_col_pivots);
        cudaFree(d_new_pivots);
        return 1;

lc16_done:;
    }
    return 0;
}

#include "nerve/persistence/cuda/cuda_matrix_reduction.hpp"
#include "nerve/gpu/gpu_ptx_ops.cuh"
#include "nerve/gpu/packed_column_primitives.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace nerve::persistence::accelerated
{
using namespace gpu_kernels;

static constexpr Size BITS_PER_WORD = 64;
static constexpr Size MAX_COLUMN_WORDS = (MAX_DIM + BITS_PER_WORD - 1) / BITS_PER_WORD;

__host__ __device__ __forceinline__ Size words_for_dim(Size dim)
{
    return (dim + BITS_PER_WORD - 1) / BITS_PER_WORD;
}

__global__ void __launch_bounds__(256)
    matrixReductionAcceleratedKernel(const int *__restrict__ columns,
                                     const Size *__restrict__ column_sizes,
                                     const double *__restrict__ weights,
                                     int *__restrict__ low_row_to_col, int *__restrict__ col_pivot,
                                     Size n_columns, Size max_dim, bool use_clearing)
{
    (void)weights;
    (void)use_clearing;
    Size global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_idx >= n_columns)
    {
        return;
    }

    Size column_size = column_sizes[global_idx];
    if (column_size == 0)
    {
        return;
    }

    int pivot = -1;
    for (Size i = 0; i < column_size && i < max_dim; ++i)
    {
        int row = columns[global_idx * MAX_DIM + i];
        if (row >= 0)
        {
            int existing_col = atomicCAS(&low_row_to_col[row], -1, static_cast<int>(global_idx));
            if (existing_col == -1)
            {
                pivot = row;
                break;
            }
        }
    }

    if (pivot != -1)
    {
        col_pivot[global_idx] = pivot;
    }
}

__global__ void __launch_bounds__(256)
    matrixReductionStreamingKernel(const int *__restrict__ columns,
                                   const Size *__restrict__ column_sizes,
                                   const double *__restrict__ weights,
                                   int *__restrict__ low_row_to_col, int *__restrict__ col_pivot,
                                   Size n_columns, Size max_dim, bool use_clearing, Size chunk_size,
                                   Size chunk_offset)
{
    (void)weights;
    (void)use_clearing;
    Size global_idx = chunk_offset + blockIdx.x * blockDim.x + threadIdx.x;

    if (global_idx >= chunk_offset + chunk_size || global_idx >= n_columns)
    {
        return;
    }

    Size column_size = column_sizes[global_idx];
    if (column_size == 0)
    {
        return;
    }

    int pivot = -1;
    for (Size i = 0; i < column_size && i < max_dim; ++i)
    {
        int row = columns[global_idx * MAX_DIM + i];
        if (row >= 0)
        {
            int existing_col = atomicCAS(&low_row_to_col[row], -1, static_cast<int>(global_idx));
            if (existing_col == -1)
            {
                pivot = row;
                break;
            }
        }
    }

    if (pivot != -1)
    {
        col_pivot[global_idx] = pivot;
    }
}

__global__ void __launch_bounds__(256)
    computeApparentPairsKernel(const int *__restrict__ low_row_to_col,
                               const int *__restrict__ col_pivot,
                               const double *__restrict__ weights, Size n_columns,
                               int *__restrict__ pair_count, int *__restrict__ pair_values,
                               Size max_pairs, bool use_optimization)
{
    (void)low_row_to_col;
    Size global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_idx >= n_columns)
    {
        return;
    }

    int pivot = col_pivot[global_idx];
    if (pivot < 0)
    {
        return;
    }

    if (use_optimization && weights[global_idx] >= weights[static_cast<Size>(pivot)])
    {
        return;
    }

    int idx = atomicAdd(pair_count, 1);
    if (idx >= 0 && static_cast<Size>(idx) < max_pairs)
    {
        pair_values[idx] = pivot;
    }
}

__global__ void __launch_bounds__(256)
    hybridMatrixReductionKernel(const int *__restrict__ columns,
                                const Size *__restrict__ column_sizes,
                                const double *__restrict__ weights,
                                int *__restrict__ low_row_to_col, int *__restrict__ col_pivot,
                                Size n_columns, Size max_dim, Size gpu_columns, bool use_clearing)
{
    (void)weights;
    (void)use_clearing;
    Size global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_idx >= gpu_columns)
    {
        return;
    }

    Size column_size = column_sizes[global_idx];
    if (column_size == 0)
    {
        return;
    }

    int pivot = -1;
    for (Size i = 0; i < column_size && i < max_dim; ++i)
    {
        int row = columns[global_idx * MAX_DIM + i];
        if (row >= 0)
        {
            int existing_col = atomicCAS(&low_row_to_col[row], -1, static_cast<int>(global_idx));
            if (existing_col == -1)
            {
                pivot = row;
                break;
            }
        }
    }

    if (pivot != -1)
    {
        col_pivot[global_idx] = pivot;
    }
}

// PTX-Optimized Kernels
// Columns stored as packed uint64_t words.
// Row r   word index r / 64, bit position r % 64.
// low_row_storage: uint64_t array, bit set means row is claimed.

__device__ __forceinline__ int find_pivot_row_packed(const uint64_t *col_words, Size nw)
{
    return packed::packed_column_find_pivot(col_words, static_cast<int>(nw));
}

__device__ __forceinline__ bool column_is_empty_packed(const uint64_t *col_words, Size nw)
{
    return packed::packed_column_is_empty(col_words, static_cast<int>(nw));
}

__device__ __forceinline__ void xor_column_inplace(uint64_t *dst, const uint64_t *src, Size nw)
{
    packed::packed_column_xor_hw(dst, src, static_cast<int>(nw), threadIdx.x & 31);
}

__device__ __forceinline__ bool atomically_claim_row(uint64_t *low_row_storage, int pivot_row,
                                                       Size num_storage_words)
{
    int word_idx = pivot_row / static_cast<int>(BITS_PER_WORD);
    int bit_idx = pivot_row % static_cast<int>(BITS_PER_WORD);
    uint64_t bit_mask = 1ULL << bit_idx;

    if (static_cast<Size>(word_idx) >= num_storage_words)
    {
        return false;
    }

    uint64_t old = low_row_storage[word_idx];
    uint64_t assumed;
    do
    {
        if (old & bit_mask)
        {
            return false;
        }
        assumed = old;
        old = atomicCAS(&low_row_storage[word_idx], assumed, assumed | bit_mask);
    } while (assumed != old);
    return true;
}

__device__ __forceinline__ void warp_column_xor_global(uint64_t *dest, const uint64_t *src,
                                                        Size nw)
{
    packed::packed_column_xor_atomic_hw(dest, src, static_cast<int>(nw), threadIdx.x & 31);
}

__global__ void __launch_bounds__(256)
    matrixReductionAcceleratedKernel_Ptx(const uint64_t *__restrict__ columns,
                                          const Size *__restrict__ column_sizes,
                                          uint64_t *__restrict__ low_row_storage,
                                          int *__restrict__ col_pivot, Size n_columns,
                                          Size num_words, Size max_dim)
{
    Size global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_idx >= n_columns)
    {
        return;
    }

    Size nw = column_sizes[global_idx];
    if (nw == 0)
    {
        return;
    }
    if (nw > MAX_COLUMN_WORDS)
    {
        nw = MAX_COLUMN_WORDS;
    }

    const uint64_t *col_base = columns + global_idx * num_words;

    uint64_t col_words[MAX_COLUMN_WORDS];
#pragma unroll
    for (Size w = 0; w < MAX_COLUMN_WORDS; ++w)
    {
        col_words[w] = (w < nw) ? col_base[w] : 0ULL;
    }

    if (global_idx + 1 < n_columns)
    {
        ptx::prefetch_l2(columns + (global_idx + 1) * num_words);
    }

    Size iter_limit = nw * 4;
    if (iter_limit < 1)
    {
        iter_limit = 1;
    }

    for (Size it = 0; it < iter_limit; ++it)
    {
        if (column_is_empty_packed(col_words, nw))
        {
            break;
        }

        int pivot_row = find_pivot_row_packed(col_words, nw);
        if (pivot_row < 0)
        {
            break;
        }

        bool claimed = atomically_claim_row(low_row_storage, pivot_row, num_words);
        int selected_pivot = ptx::slct_s32(claimed, pivot_row, -1);

        if (claimed)
        {
            col_pivot[global_idx] = selected_pivot;
            return;
        }

        int pw = pivot_row / static_cast<int>(BITS_PER_WORD);
        int pb = pivot_row % static_cast<int>(BITS_PER_WORD);
        if (pw < static_cast<int>(nw))
        {
            col_words[pw] &= ~(1ULL << pb);
        }

        warp_column_xor_global(low_row_storage, col_words, nw);
        __threadfence();
        for (Size w = 0; w < nw; ++w)
        {
            col_words[w] = low_row_storage[w];
        }
    }
}

__global__ void __launch_bounds__(256)
    matrixReductionStreamingKernel_Ptx(const uint64_t *__restrict__ columns,
                                        const Size *__restrict__ column_sizes,
                                        uint64_t *__restrict__ low_row_storage,
                                        int *__restrict__ col_pivot, Size n_columns,
                                        Size num_words, Size max_dim, Size chunk_size,
                                        Size chunk_offset)
{
    Size global_idx = chunk_offset + blockIdx.x * blockDim.x + threadIdx.x;

    if (global_idx >= chunk_offset + chunk_size || global_idx >= n_columns)
    {
        return;
    }

    Size nw = column_sizes[global_idx];
    if (nw == 0)
    {
        return;
    }
    if (nw > MAX_COLUMN_WORDS)
    {
        nw = MAX_COLUMN_WORDS;
    }

    __shared__ uint64_t s_columns[BLOCK_SIZE * MAX_COLUMN_WORDS];
    const uint64_t *col_base = columns + global_idx * num_words;
    Size shmem_offset = threadIdx.x * MAX_COLUMN_WORDS;

    packed::async_pipeline_stage_column(col_base, &s_columns[shmem_offset], static_cast<int>(nw),
                                         threadIdx.x & 31);

    uint64_t col_words[MAX_COLUMN_WORDS];
#pragma unroll
    for (Size w = 0; w < MAX_COLUMN_WORDS; ++w)
    {
        col_words[w] = (w < nw) ? s_columns[shmem_offset + w] : 0ULL;
    }

    __syncthreads();

    if (global_idx + blockDim.x < n_columns)
    {
        ptx::prefetch_l2(columns + (global_idx + blockDim.x) * num_words);
    }

    Size iter_limit = nw * 4;
    if (iter_limit < 1)
    {
        iter_limit = 1;
    }

    int final_pivot = -1;

    for (Size it = 0; it < iter_limit; ++it)
    {
        if (column_is_empty_packed(col_words, nw))
        {
            break;
        }

        int pivot_row = find_pivot_row_packed(col_words, nw);
        if (pivot_row < 0)
        {
            break;
        }

        bool claimed = atomically_claim_row(low_row_storage, pivot_row, num_words);
        int selected = ptx::slct_s32(claimed, pivot_row, -1);

        if (claimed)
        {
            final_pivot = selected;
            break;
        }

        int pw = pivot_row / static_cast<int>(BITS_PER_WORD);
        int pb = pivot_row % static_cast<int>(BITS_PER_WORD);
        if (pw < static_cast<int>(nw))
        {
            col_words[pw] ^= (1ULL << pb);
        }

        warp_column_xor_global(low_row_storage, col_words, nw);
        __syncthreads();
        for (Size w = 0; w < nw; ++w)
        {
            col_words[w] = low_row_storage[w];
        }
        __syncthreads();
    }

    if (final_pivot >= 0)
    {
        col_pivot[global_idx] = final_pivot;
    }
}

__global__ void __launch_bounds__(256)
    computeApparentPairsKernel_Ptx(const uint64_t *__restrict__ low_row_storage,
                                    const int *__restrict__ col_pivot,
                                    const double *__restrict__ weights, Size n_columns,
                                    Size num_storage_words,
                                    int *__restrict__ pair_count, int *__restrict__ pair_values,
                                    Size max_pairs, bool use_optimization,
                                    float *__restrict__ out_weight_f32,
                                    double *__restrict__ out_weight_f64)
{
    (void)low_row_storage;
    (void)num_storage_words;

    Size global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_idx >= n_columns)
    {
        return;
    }

    int pivot = col_pivot[global_idx];
    if (pivot < 0)
    {
        return;
    }

    double w_col = weights[global_idx];
    double w_piv = weights[static_cast<Size>(pivot)];

    if (use_optimization && w_col >= w_piv)
    {
        return;
    }

    int idx = atomicAdd(pair_count, 1);
    if (idx >= 0 && static_cast<Size>(idx) < max_pairs)
    {
        pair_values[idx] = pivot;

        if (out_weight_f32)
        {
            ptx::st_global_cs_f32(&out_weight_f32[idx], static_cast<float>(w_col));
        }
        if (out_weight_f64)
        {
            ptx::st_global_cs_f64(&out_weight_f64[idx], w_piv);
        }
    }
}

__global__ void __launch_bounds__(256)
    hybridMatrixReductionKernel_Ptx(const uint64_t *__restrict__ columns,
                                     const Size *__restrict__ column_sizes,
                                     uint64_t *__restrict__ low_row_storage,
                                     int *__restrict__ col_pivot, Size n_columns, Size num_words,
                                     Size max_dim, Size gpu_columns)
{
    Size global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_idx >= gpu_columns)
    {
        return;
    }

    Size nw = column_sizes[global_idx];
    if (nw == 0)
    {
        return;
    }
    if (nw > MAX_COLUMN_WORDS)
    {
        nw = MAX_COLUMN_WORDS;
    }

    const uint64_t *col_base = columns + global_idx * num_words;

    uint64_t col_words[MAX_COLUMN_WORDS];
#pragma unroll
    for (Size w = 0; w < MAX_COLUMN_WORDS; ++w)
    {
        col_words[w] = (w < nw) ? col_base[w] : 0ULL;
    }

    if (global_idx + 1 < gpu_columns)
    {
        ptx::prefetch_l2(columns + (global_idx + 1) * num_words);
    }

    unsigned int lane_id = threadIdx.x & 31;
    unsigned int warp_mask = 0xFFFFFFFFu;

    Size iter_limit = nw * 4;
    if (iter_limit < 1)
    {
        iter_limit = 1;
    }

    for (Size it = 0; it < iter_limit; ++it)
    {
        if (column_is_empty_packed(col_words, nw))
        {
            break;
        }

        int pivot_row = find_pivot_row_packed(col_words, nw);
        if (pivot_row < 0)
        {
            break;
        }

        unsigned int matching_lanes =
            ptx::match_any_sync_u64(warp_mask, static_cast<unsigned long long>(pivot_row));

        if ((matching_lanes & (1u << lane_id)) != 0)
        {
            bool claimed = atomically_claim_row(low_row_storage, pivot_row, num_words);
            int selected = ptx::slct_s32(claimed, pivot_row, -1);

            if (claimed)
            {
                col_pivot[global_idx] = selected;
                return;
            }
        }

        int pw = pivot_row / static_cast<int>(BITS_PER_WORD);
        int pb = pivot_row % static_cast<int>(BITS_PER_WORD);
        if (pw < static_cast<int>(nw))
        {
            col_words[pw] ^= (1ULL << pb);
        }

        warp_column_xor_global(low_row_storage, col_words, nw);
        __threadfence();
        for (Size w = 0; w < nw; ++w)
        {
            col_words[w] = low_row_storage[w];
        }
    }
}

// Dispatch Functions

__host__ bool should_use_ptx_variant()
{
    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess)
    {
        return false;
    }
    cudaDeviceProp props;
    err = cudaGetDeviceProperties(&props, device);
    if (err != cudaSuccess)
    {
        return false;
    }
    return props.major >= 8;
}

__host__ void launch_ptx_dispatched(const uint64_t *d_columns, const Size *d_column_sizes,
                                     const double *d_weights, uint64_t *d_low_row_storage,
                                     int *d_col_pivot, Size n_columns, Size num_words,
                                     Size max_dim, const MatrixReductionConfig &config,
                                     cudaStream_t stream)
{
    Size block_size = BLOCK_SIZE;
    Size grid_size = (n_columns + block_size - 1) / block_size;

    if (config.enable_streaming)
    {
        Size chunk_size = std::min(config.streaming_chunk_size, n_columns);
        Size num_chunks = (n_columns + chunk_size - 1) / chunk_size;
        for (Size c = 0; c < num_chunks; ++c)
        {
            Size offset = c * chunk_size;
            Size current_chunk = std::min(chunk_size, n_columns - offset);
            Size cur_grid = (current_chunk + block_size - 1) / block_size;

            matrixReductionStreamingKernel_Ptx<<<cur_grid, block_size, 0, stream>>>(
                d_columns, d_column_sizes, d_low_row_storage, d_col_pivot, n_columns, num_words,
                max_dim, current_chunk, offset);
        }
    }
    else if (config.enable_hybrid_processing)
    {
        (void)d_weights;
        Size gpu_cols = std::min(n_columns, config.gpu_columns_threshold);
        grid_size = (gpu_cols + block_size - 1) / block_size;

        hybridMatrixReductionKernel_Ptx<<<grid_size, block_size, 0, stream>>>(
            d_columns, d_column_sizes, d_low_row_storage, d_col_pivot, n_columns, num_words,
            max_dim, gpu_cols);
    }
    else
    {
        matrixReductionAcceleratedKernel_Ptx<<<grid_size, block_size, 0, stream>>>(
            d_columns, d_column_sizes, d_low_row_storage, d_col_pivot, n_columns, num_words,
            max_dim);
    }
}

__host__ void launch_reduction_dispatched(const int *d_columns, const Size *d_column_sizes,
                                           const double *d_weights, int *d_low_row_to_col,
                                           int *d_col_pivot, Size n_columns, Size max_dim,
                                           const MatrixReductionConfig &config, cudaStream_t stream)
{
    Size block_size = BLOCK_SIZE;
    Size grid_size = (n_columns + block_size - 1) / block_size;

    if (config.enable_streaming)
    {
        Size chunk_size = std::min(config.streaming_chunk_size, n_columns);
        Size num_chunks = (n_columns + chunk_size - 1) / chunk_size;
        for (Size c = 0; c < num_chunks; ++c)
        {
            Size offset = c * chunk_size;
            Size current_chunk = std::min(chunk_size, n_columns - offset);
            Size cur_grid = (current_chunk + block_size - 1) / block_size;

            matrixReductionStreamingKernel<<<cur_grid, block_size, 0, stream>>>(
                d_columns, d_column_sizes, d_weights, d_low_row_to_col, d_col_pivot, n_columns,
                max_dim, config.enable_clearing, current_chunk, offset);
        }
    }
    else if (config.enable_hybrid_processing)
    {
        Size gpu_cols = std::min(n_columns, config.gpu_columns_threshold);
        grid_size = (gpu_cols + block_size - 1) / block_size;

        hybridMatrixReductionKernel<<<grid_size, block_size, 0, stream>>>(
            d_columns, d_column_sizes, d_weights, d_low_row_to_col, d_col_pivot, n_columns,
            max_dim, gpu_cols, config.enable_clearing);
    }
    else
    {
        matrixReductionAcceleratedKernel<<<grid_size, block_size, 0, stream>>>(
            d_columns, d_column_sizes, d_weights, d_low_row_to_col, d_col_pivot, n_columns,
            max_dim, config.enable_clearing);
    }
}

} // namespace nerve::persistence::accelerated

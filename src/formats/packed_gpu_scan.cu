#include "nerve/formats/packed_boundary_matrix.hpp"
#include "nerve/formats/packed_gpu_scan.hpp"
#include "nerve/gpu/gpu_ptx_ops.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>

namespace nerve::formats::gpu
{

namespace
{

constexpr int kScanBlockSize = 256;
constexpr Size kMaxLocalWords = 16;

int gridFor(Size items)
{
    if (items == 0)
        return 0;
    const Size blocks = (items + static_cast<Size>(kScanBlockSize) - 1) / static_cast<Size>(kScanBlockSize);
    return static_cast<int>(blocks);
}

struct PackedGpuScanBuffers
{
    PackedWord *d_columns = nullptr;
    Size *d_column_sizes = nullptr;
    Index *d_pivot_candidates = nullptr;
    int *d_zero_addition = nullptr;
    int *d_stable_columns = nullptr;
    int *d_low_to_col = nullptr;
    Index *d_stable_list = nullptr;
    Index *d_unstable_list = nullptr;
    Size *d_stable_count = nullptr;
    Size *d_unstable_count = nullptr;
};

void freeGpuBuffers(PackedGpuScanBuffers &b)
{
    if (b.d_columns)
        cudaFree(b.d_columns);
    if (b.d_column_sizes)
        cudaFree(b.d_column_sizes);
    if (b.d_pivot_candidates)
        cudaFree(b.d_pivot_candidates);
    if (b.d_zero_addition)
        cudaFree(b.d_zero_addition);
    if (b.d_stable_columns)
        cudaFree(b.d_stable_columns);
    if (b.d_low_to_col)
        cudaFree(b.d_low_to_col);
    if (b.d_stable_list)
        cudaFree(b.d_stable_list);
    if (b.d_unstable_list)
        cudaFree(b.d_unstable_list);
    if (b.d_stable_count)
        cudaFree(b.d_stable_count);
    if (b.d_unstable_count)
        cudaFree(b.d_unstable_count);
    b = PackedGpuScanBuffers{};
}

__global__ __launch_bounds__(kScanBlockSize)
void packedScanKernel(const PackedWord *__restrict__ columns,
                      const Size *__restrict__ column_sizes,
                      Index *__restrict__ pivot_candidates,
                      int *__restrict__ zero_addition,
                      int *__restrict__ stable_columns,
                      Size n_columns, Size n_words)
{
    Size col = static_cast<Size>(blockIdx.x) * static_cast<Size>(blockDim.x) + static_cast<Size>(threadIdx.x);
    if (col >= n_columns)
        return;

    Size nw = column_sizes[col];
    if (nw == 0)
    {
        pivot_candidates[col] = -1;
        zero_addition[col] = 0;
        stable_columns[col] = 1;
        return;
    }
    if (nw > kMaxLocalWords)
        nw = kMaxLocalWords;

    const PackedWord *col_base = columns + col * n_words;

    Index highest_pivot = -1;
    int nz_count = 0;

    for (Size w = 0; w < nw; ++w)
    {
        PackedWord word = col_base[w];
        if (word == 0)
            continue;
        int msb = nerve::gpu::ptx::find_msb_u64(word);
        if (msb >= 0)
        {
            Index global_row = static_cast<Index>(w) * static_cast<Index>(kBitsPerPackedWord) +
                               static_cast<Index>(msb);
            if (global_row > highest_pivot)
            {
                highest_pivot = global_row;
            }
        }
        nz_count += static_cast<int>(nerve::gpu::ptx::popc_u64(word));
    }

    if (nz_count == 0)
    {
        pivot_candidates[col] = -1;
        zero_addition[col] = 0;
        stable_columns[col] = 1;
        return;
    }

    pivot_candidates[col] = highest_pivot;

    if (nz_count == 1)
    {
        zero_addition[col] = 1;
        stable_columns[col] = 0;
    }
    else
    {
        zero_addition[col] = 0;
        stable_columns[col] = 0;
    }
}

__global__ __launch_bounds__(kScanBlockSize)
void packedPivotClaimKernel(const Index *__restrict__ pivot_candidates,
                            int *__restrict__ low_to_col, Size n_columns,
                            Index *__restrict__ stable_list,
                            Index *__restrict__ unstable_list,
                            Size *__restrict__ stable_count,
                            Size *__restrict__ unstable_count)
{
    Size col = static_cast<Size>(blockIdx.x) * static_cast<Size>(blockDim.x) + static_cast<Size>(threadIdx.x);
    if (col >= n_columns)
        return;

    Index pivot = pivot_candidates[col];

    if (pivot < 0)
    {
        Index idx = atomicAdd(reinterpret_cast<unsigned long long *>(stable_count), 1ULL);
        stable_list[idx] = static_cast<Index>(col);
        return;
    }

    int expected = -1;
    int actual = atomicCAS(&low_to_col[pivot], -1, static_cast<int>(col));
    if (actual == -1 || actual == static_cast<int>(col))
    {
        Index idx = atomicAdd(reinterpret_cast<unsigned long long *>(stable_count), 1ULL);
        stable_list[idx] = static_cast<Index>(col);
    }
    else
    {
        Index idx = atomicAdd(reinterpret_cast<unsigned long long *>(unstable_count), 1ULL);
        unstable_list[idx] = static_cast<Index>(col);
    }
}

__global__ __launch_bounds__(kScanBlockSize)
void packedLeftmostOneKernel(const PackedWord *__restrict__ columns,
                             const Size *__restrict__ column_sizes,
                             const Index *__restrict__ stable_list,
                             Size stable_count,
                             Index *__restrict__ leftmost_column_by_row,
                             Size n_rows, Size n_words)
{
    Size idx = static_cast<Size>(blockIdx.x) * static_cast<Size>(blockDim.x) + static_cast<Size>(threadIdx.x);
    if (idx >= stable_count)
        return;

    Index col = stable_list[idx];
    Size nw = column_sizes[col];
    if (nw == 0 || nw > kMaxLocalWords)
        return;

    const PackedWord *col_base = columns + col * n_words;

    for (Size w = 0; w < nw; ++w)
    {
        PackedWord word = col_base[w];
        if (word == 0)
            continue;
        int msb = nerve::gpu::ptx::find_msb_u64(word);
        if (msb >= 0)
        {
            Index global_row = static_cast<Index>(w) * static_cast<Index>(kBitsPerPackedWord) + static_cast<Index>(msb);
            if (global_row >= 0 && static_cast<Size>(global_row) < n_rows)
            {
                Index expected = -1;
                atomicCAS(&leftmost_column_by_row[global_row], static_cast<Index>(-1), col);
            }
        }
    }
}

} // namespace

errors::ErrorResult<GpuScanResult>
launchPackedScan(const GpuPackedLayout &layout, void *stream_handle, int device_id)
{
    GpuScanResult result;
    if (layout.num_columns == 0)
        return errors::ErrorResult<GpuScanResult>::success(std::move(result));

    cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);

    PackedGpuScanBuffers buf{};

    const Size n_cols = layout.num_columns;
    const Size n_rows = layout.num_rows;
    const Size n_words = layout.max_words_per_column;
    const Size flat_size = layout.columns_flat.size();
    const Size col_data_bytes = flat_size * kPackedWordBytes;
    const Size size_array_bytes = n_cols * sizeof(Size);
    const Size index_array_bytes = n_cols * sizeof(Index);
    const Size int_array_bytes = n_cols * sizeof(int);

    int prev_device = 0;
    cudaGetDevice(&prev_device);
    cudaSetDevice(device_id);

    cudaError_t err;
    err = cudaMalloc(&buf.d_columns, col_data_bytes);
    if (err != cudaSuccess)
        goto cleanup;
    err = cudaMemcpyAsync(buf.d_columns, layout.columns_flat.data(), col_data_bytes,
                          cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
        goto cleanup;

    err = cudaMalloc(&buf.d_column_sizes, size_array_bytes);
    if (err != cudaSuccess)
        goto cleanup;
    {
        std::vector<Size> h_sizes(n_cols, n_words);
        for (Size i = 0; i < n_cols && i < layout.column_offsets.size() - 1; ++i)
        {
            Size off = layout.column_offsets[i];
            Size end = layout.column_offsets[i + 1];
            Size nw = (end > off) ? (end - off) : n_words;
            Size actual = 0;
            for (Size w = nw; w > 0; --w)
            {
                if (layout.columns_flat[off + (w - 1)] != 0)
                {
                    actual = w;
                    break;
                }
            }
            h_sizes[i] = actual;
        }
        err = cudaMemcpyAsync(buf.d_column_sizes, h_sizes.data(), size_array_bytes,
                              cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess)
            goto cleanup;
    }

    err = cudaMalloc(&buf.d_pivot_candidates, index_array_bytes);
    if (err != cudaSuccess)
        goto cleanup;
    err = cudaMalloc(&buf.d_zero_addition, int_array_bytes);
    if (err != cudaSuccess)
        goto cleanup;
    err = cudaMalloc(&buf.d_stable_columns, int_array_bytes);
    if (err != cudaSuccess)
        goto cleanup;
    err = cudaMalloc(&buf.d_low_to_col, int_array_bytes);
    if (err != cudaSuccess)
        goto cleanup;
    err = cudaMalloc(&buf.d_stable_list, index_array_bytes);
    if (err != cudaSuccess)
        goto cleanup;
    err = cudaMalloc(&buf.d_unstable_list, index_array_bytes);
    if (err != cudaSuccess)
        goto cleanup;
    err = cudaMalloc(&buf.d_stable_count, sizeof(Size));
    if (err != cudaSuccess)
        goto cleanup;
    err = cudaMalloc(&buf.d_unstable_count, sizeof(Size));
    if (err != cudaSuccess)
        goto cleanup;

    cudaMemsetAsync(buf.d_low_to_col, 0xFF, int_array_bytes, stream);
    cudaMemsetAsync(buf.d_stable_count, 0, sizeof(Size), stream);
    cudaMemsetAsync(buf.d_unstable_count, 0, sizeof(Size), stream);

    int grid = gridFor(n_cols);
    if (grid == 0)
    {
        freeGpuBuffers(buf);
        cudaSetDevice(prev_device);
        return errors::ErrorResult<GpuScanResult>::success(std::move(result));
    }

    packedScanKernel<<<grid, kScanBlockSize, 0, stream>>>(
        buf.d_columns, buf.d_column_sizes,
        buf.d_pivot_candidates, buf.d_zero_addition, buf.d_stable_columns,
        n_cols, n_words);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess)
        goto cleanup;

    packedPivotClaimKernel<<<grid, kScanBlockSize, 0, stream>>>(
        buf.d_pivot_candidates, buf.d_low_to_col, n_cols,
        buf.d_stable_list, buf.d_unstable_list,
        buf.d_stable_count, buf.d_unstable_count);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess)
        goto cleanup;

    cudaStreamSynchronize(stream);

    Size stable_count = 0;
    Size unstable_count = 0;
    cudaMemcpy(&stable_count, buf.d_stable_count, sizeof(Size), cudaMemcpyDeviceToHost);
    cudaMemcpy(&unstable_count, buf.d_unstable_count, sizeof(Size), cudaMemcpyDeviceToHost);

    result.stable_count = stable_count;
    result.unstable_count = unstable_count;

    if (stable_count > 0 && stable_count <= n_cols)
    {
        result.stable_columns.resize(stable_count);
        cudaMemcpy(result.stable_columns.data(), buf.d_stable_list,
                   stable_count * sizeof(Index), cudaMemcpyDeviceToHost);
    }
    if (unstable_count > 0 && unstable_count <= n_cols)
    {
        result.unstable_columns.resize(unstable_count);
        cudaMemcpy(result.unstable_columns.data(), buf.d_unstable_list,
                   unstable_count * sizeof(Index), cudaMemcpyDeviceToHost);
    }

    result.leftmost_column_by_row.resize(n_rows, static_cast<Index>(-1));
    Index *d_leftmost = nullptr;
    if (n_rows > 0 && stable_count > 0)
    {
        const Size lmr_bytes = n_rows * sizeof(Index);
        cudaMalloc(&d_leftmost, lmr_bytes);
        if (d_leftmost)
        {
            cudaMemsetAsync(d_leftmost, 0xFF, lmr_bytes, stream);
            int lmr_grid = gridFor(stable_count);
            if (lmr_grid > 0)
            {
                packedLeftmostOneKernel<<<lmr_grid, kScanBlockSize, 0, stream>>>(
                    buf.d_columns, buf.d_column_sizes,
                    buf.d_stable_list, stable_count,
                    d_leftmost, n_rows, n_words);
                cudaStreamSynchronize(stream);
                cudaMemcpy(result.leftmost_column_by_row.data(), d_leftmost,
                           lmr_bytes, cudaMemcpyDeviceToHost);
            }
            cudaFree(d_leftmost);
        }
    }

cleanup:
    freeGpuBuffers(buf);
    cudaSetDevice(prev_device);

    if (err != cudaSuccess)
    {
        return errors::ErrorResult<GpuScanResult>::error(errors::ErrorCode::E41_RESOURCE_LIMIT);
    }
    return errors::ErrorResult<GpuScanResult>::success(std::move(result));
}

} // namespace nerve::formats::gpu

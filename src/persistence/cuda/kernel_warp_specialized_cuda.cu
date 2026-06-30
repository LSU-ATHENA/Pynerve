// Warp-specialized CUDA kernels for persistence matrix operations.
// This unit provides concrete implementations for:
// - one-warp-per-column XOR updates,
// - one-warp-per-column pivot discovery, and
// - iterative pivot-driven column reduction.

#include "nerve/gpu/gpu_error.hpp"
#include "nerve/persistence/cuda/cuda_warp_specialized_kernels.hpp"
#include "nerve/gpu/gpu_ptx_ops.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <limits>
#include <random>
#include <vector>

#if defined(CUDART_VERSION) && CUDART_VERSION >= 11000
#include <cuda/pipeline>
#endif

namespace nerve
{
namespace persistence
{
namespace gpu
{

constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = 8;
constexpr int THREADS_PER_BLOCK = WARP_SIZE * WARPS_PER_BLOCK;
constexpr unsigned int FULL_WARP_MASK = 0xFFFFFFFFu;
constexpr int WARP_BENCHMARK_ITERATIONS = 100;

__host__ __device__ inline int clampWordCount(int requested_words, int max_words)
{
    if (requested_words < 0)
    {
        return 0;
    }
    if (requested_words > max_words)
    {
        return max_words;
    }
    return requested_words;
}

__host__ __device__ inline int blockCountForColumns(int num_columns)
{
    return (num_columns + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
}

inline bool checkedSizeProduct(size_t lhs, size_t rhs, size_t &out)
{
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs)
    {
        return false;
    }
    out = lhs * rhs;
    return true;
}

__device__ inline int warpMaxReduce(int value)
{
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
    {
        value = max(value, __shfl_down_sync(FULL_WARP_MASK, value, offset));
    }
    return value;
}

template <typename T>
__device__ inline T warpSumReduce(T value)
{
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
    {
        const T other = __shfl_down_sync(FULL_WARP_MASK, value, offset);
        const T next = value + other;
        value = isfinite(static_cast<double>(next)) ? next : static_cast<T>(INFINITY);
    }
    return value;
}

/**
 * Find the highest set-bit pivot in a packed binary column using one warp.
 * Each lane inspects a strided subset of 64-bit words, and the warp then
 * computes the maximum candidate pivot index.
 */
__device__ inline int findHighestPivotWarp(const uint64_t *__restrict__ column_words, int num_words,
                                           int lane_id)
{
    int lane_pivot = -1;
    for (int word_idx = num_words - 1 - lane_id; word_idx >= 0; word_idx -= WARP_SIZE)
    {
        const uint64_t word = column_words[word_idx];
        if (word != 0)
        {
            lane_pivot = word_idx * 64 + ptx::find_msb_u64(word);
            break;
        }
    }
    return warpMaxReduce(lane_pivot);
}

__global__ void __launch_bounds__(256)
    warpSpecializedColumnAddKernel(uint64_t *__restrict__ columns_a,
                                   const uint64_t *__restrict__ columns_b,
                                   const int *__restrict__ col_sizes, int num_words,
                                   int num_columns)
{
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int col_idx = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (col_idx >= num_columns)
    {
        return;
    }

    const int words_to_process = clampWordCount(col_sizes[col_idx], num_words);
    const int base = col_idx * num_words;
    for (int i = lane_id; i < words_to_process; i += WARP_SIZE)
    {
        columns_a[base + i] ^= columns_b[base + i];
    }
}

__global__ void __launch_bounds__(256)
    warpSpecializedPivotFindKernel(const uint64_t *__restrict__ columns,
                                   const int *__restrict__ col_sizes, int num_words,
                                   int num_columns, int *__restrict__ pivots)
{
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int col_idx = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (col_idx >= num_columns)
    {
        return;
    }

    const int words_to_process = clampWordCount(col_sizes[col_idx], num_words);
    const int base = col_idx * num_words;
    const int pivot = findHighestPivotWarp(columns + base, words_to_process, lane_id);
    if (lane_id == 0)
    {
        pivots[col_idx] = pivot;
    }
}

// Warp-level matrix-vector product for contiguous dense rows.
// This kernel does not use WMMA; it uses scalar multiply-accumulate with
// intra-warp reduction for consistent behavior across architectures.
template <typename T>
__global__ void __launch_bounds__(256)
    tensorCoreMatVecKernel(const T *__restrict__ matrix, const T *__restrict__ vector,
                           T *__restrict__ result, int rows, int cols)
{
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (row >= rows)
    {
        return;
    }

    T sum = static_cast<T>(0);
    const int row_base = row * cols;
    for (int i = lane_id; i < cols; i += WARP_SIZE)
    {
        const T lhs = matrix[row_base + i];
        const T rhs = vector[i];
        const T product = lhs * rhs;
        const T next;
        if constexpr (std::is_same_v<T, float>)
            next = ptx::fma_f32(lhs, rhs, sum);
        else
            next = sum + product;
        if (!isfinite(static_cast<double>(lhs)) || !isfinite(static_cast<double>(rhs)) ||
            !isfinite(static_cast<double>(product)) || !isfinite(static_cast<double>(next)))
        {
            sum = static_cast<T>(INFINITY);
            break;
        }
        sum = next;
    }
    sum = warpSumReduce(sum);
    if (lane_id == 0)
    {
        result[row] = isfinite(static_cast<double>(sum)) ? sum : static_cast<T>(INFINITY);
    }
}

template <>
__global__ void __launch_bounds__(256) tensorCoreMatVecKernel<__half>(
    const __half *__restrict__ matrix, const __half *__restrict__ vector,
    __half *__restrict__ result, int rows, int cols)
{
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (row >= rows)
    {
        return;
    }

    float sum = 0.0f;
    const int row_base = row * cols;
    for (int i = lane_id; i < cols; i += WARP_SIZE)
    {
        const float lhs = __half2float(matrix[row_base + i]);
        const float rhs = __half2float(vector[i]);
        const float product = lhs * rhs;
        const float next = sum + product;
        if (!isfinite(lhs) || !isfinite(rhs) || !isfinite(product) || !isfinite(next))
        {
            sum = INFINITY;
            break;
        }
        sum = next;
    }
    sum = warpSumReduce(sum);
    if (lane_id == 0)
    {
        result[row] = __float2half_rn(isfinite(sum) ? sum : INFINITY);
    }
}

/**
 * Iterative pivot-driven reduction:
 * - Use current pivot to select a source column.
 * - XOR source into target.
 * - Recompute pivot for target.
 * The loop is bounded to guard against malformed pivot maps.
 */
__global__ void __launch_bounds__(256)
    pipelinedReductionKernel(uint64_t *__restrict__ columns, const int *__restrict__ col_pivots,
                             const int *__restrict__ pivot_to_col, int num_words, int num_columns,
                             int *__restrict__ new_pivots)
{
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int col_idx = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (col_idx >= num_columns)
    {
        return;
    }

    uint64_t *const target_column = columns + col_idx * num_words;
    int pivot = col_pivots[col_idx];
    const int pivot_limit = num_words * 64;
    const int iteration_limit = max(1, num_words * 4);

    for (int iter = 0; pivot >= 0 && iter < iteration_limit; ++iter)
    {
        if (pivot >= pivot_limit)
        {
            break;
        }
        const int source_col = pivot_to_col[pivot];
        if (source_col < 0 || source_col >= num_columns)
        {
            break;
        }

        if (source_col == col_idx)
        {
            for (int i = lane_id; i < num_words; i += WARP_SIZE)
            {
                target_column[i] = 0;
            }
            pivot = -1;
            break;
        }

        const uint64_t *const source_column = columns + source_col * num_words;
        for (int i = lane_id; i < num_words; i += WARP_SIZE)
        {
            target_column[i] ^= source_column[i];
        }
        __syncwarp();
        pivot = findHighestPivotWarp(target_column, num_words, lane_id);
    }

    if (lane_id == 0)
    {
        new_pivots[col_idx] = pivot;
    }
}

// Async-copy reduction variant (needs CUDA >= 11.0)
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11000

__global__ void __launch_bounds__(256)
    warpSpecializedAsyncReductionKernel(uint64_t *__restrict__ columns,
                                        const int *__restrict__ col_pivots,
                                        const int *__restrict__ pivot_to_col, int num_words,
                                        int num_columns, int *__restrict__ new_pivots)
{
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int col_idx = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (col_idx >= num_columns)
    {
        return;
    }

    extern __shared__ uint64_t shared_col_buf[];
    uint64_t *warp_buf = shared_col_buf + warp_id * num_words;

    uint64_t *target_column = columns + col_idx * num_words;
    int pivot = col_pivots[col_idx];
    const int pivot_limit = num_words * 64;
    const int iteration_limit = max(1, num_words * 4);

    for (int iter = 0; pivot >= 0 && iter < iteration_limit; ++iter)
    {
        if (pivot >= pivot_limit)
        {
            break;
        }
        const int source_col = pivot_to_col[pivot];
        if (source_col < 0 || source_col >= num_columns)
        {
            break;
        }

        if (source_col == col_idx)
        {
            for (int i = lane_id; i < num_words; i += WARP_SIZE)
            {
                target_column[i] = 0;
            }
            pivot = -1;
            break;
        }

        // Stage the source column into shared memory via async copy pipeline
        auto pipeline = cuda::make_pipeline();
        cuda::memcpy_async(warp_buf, columns + static_cast<size_t>(source_col) * num_words,
                           static_cast<size_t>(num_words), pipeline);
        pipeline.commit();
        pipeline.wait();

        // XOR from shared memory
        for (int i = lane_id; i < num_words; i += WARP_SIZE)
        {
            target_column[i] ^= warp_buf[i];
        }
        __syncwarp();
        pivot = findHighestPivotWarp(target_column, num_words, lane_id);
    }

    if (lane_id == 0)
    {
        new_pivots[col_idx] = pivot;
    }
}

#endif // CUDART_VERSION >= 11000

void launchWarpSpecializedColumnAdd(uint64_t *d_columns_a, const uint64_t *d_columns_b,
                                    const int *d_col_sizes, int num_words, int num_columns,
                                    cudaStream_t stream)
{
    if (d_columns_a == nullptr || d_columns_b == nullptr || d_col_sizes == nullptr ||
        num_words <= 0 || num_columns <= 0)
    {
        return;
    }
    const int num_blocks = blockCountForColumns(num_columns);
    warpSpecializedColumnAddKernel<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
        d_columns_a, d_columns_b, d_col_sizes, num_words, num_columns);
    GPU_CHECK(cudaPeekAtLastError());
}

void launchWarpSpecializedPivotFind(const uint64_t *d_columns, const int *d_col_sizes,
                                    int num_words, int num_columns, int *d_pivots,
                                    cudaStream_t stream)
{
    if (d_columns == nullptr || d_col_sizes == nullptr || d_pivots == nullptr || num_words <= 0 ||
        num_columns <= 0)
    {
        return;
    }
    const int num_blocks = blockCountForColumns(num_columns);
    warpSpecializedPivotFindKernel<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
        d_columns, d_col_sizes, num_words, num_columns, d_pivots);
    GPU_CHECK(cudaPeekAtLastError());
}

void launchPipelinedReduction(uint64_t *d_columns, const int *d_col_pivots,
                              const int *d_pivot_to_col, int num_words, int num_columns,
                              int *d_new_pivots, cudaStream_t stream, bool use_async_copy)
{
    if (d_columns == nullptr || d_col_pivots == nullptr || d_pivot_to_col == nullptr ||
        d_new_pivots == nullptr || num_words <= 0 || num_columns <= 0)
    {
        return;
    }
    const int num_blocks = blockCountForColumns(num_columns);

#if defined(CUDART_VERSION) && CUDART_VERSION >= 11000
    if (use_async_copy)
    {
        size_t shared_bytes = 0;
        if (!checkedSizeProduct(static_cast<size_t>(WARPS_PER_BLOCK),
                                static_cast<size_t>(num_words) * sizeof(uint64_t), shared_bytes) ||
            shared_bytes > static_cast<size_t>(1 << 16))
        { // cap at 64KB
            use_async_copy = false;
        }
        else
        {
            warpSpecializedAsyncReductionKernel<<<num_blocks, THREADS_PER_BLOCK, shared_bytes,
                                                  stream>>>(d_columns, d_col_pivots, d_pivot_to_col,
                                                            num_words, num_columns, d_new_pivots);
            GPU_CHECK(cudaPeekAtLastError());
            return;
        }
    }
#else
    (void)use_async_copy;
#endif

    pipelinedReductionKernel<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
        d_columns, d_col_pivots, d_pivot_to_col, num_words, num_columns, d_new_pivots);
    GPU_CHECK(cudaPeekAtLastError());
}

WarpSpecializationBenchmark benchmarkWarpSpecialization(int num_columns, int num_words)
{
    WarpSpecializationBenchmark bench{};
    bench.column_add_time_ms = 0.0;
    bench.pivot_find_time_ms = 0.0;
    bench.reduction_time_ms = 0.0;
    bench.warp_speedup = 1.0;
    bench.tensor_core_speedup = 1.0;
    bench.pipelining_speedup = 1.0;
    bench.total_speedup = 1.0;

    if (num_columns <= 0 || num_words <= 0)
    {
        return bench;
    }

    size_t word_count = 0;
    size_t col_bytes = 0;
    size_t idx_bytes = 0;
    if (!checkedSizeProduct(static_cast<size_t>(num_columns), static_cast<size_t>(num_words),
                            word_count) ||
        !checkedSizeProduct(word_count, sizeof(uint64_t), col_bytes) ||
        !checkedSizeProduct(static_cast<size_t>(num_columns), sizeof(int), idx_bytes))
    {
        return bench;
    }

    uint64_t *d_a = nullptr;
    uint64_t *d_b = nullptr;
    int *d_sizes = nullptr;
    int *d_pivots = nullptr;
    cudaEvent_t start{};
    cudaEvent_t stop{};

    const bool alloc_ok =
        cudaMalloc(&d_a, col_bytes) == cudaSuccess && cudaMalloc(&d_b, col_bytes) == cudaSuccess &&
        cudaMalloc(&d_sizes, idx_bytes) == cudaSuccess &&
        cudaMalloc(&d_pivots, idx_bytes) == cudaSuccess && cudaEventCreate(&start) == cudaSuccess &&
        cudaEventCreate(&stop) == cudaSuccess;
    if (!alloc_ok)
    {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_sizes);
        cudaFree(d_pivots);
        if (start != nullptr)
        {
            cudaEventDestroy(start);
        }
        if (stop != nullptr)
        {
            cudaEventDestroy(stop);
        }
        return bench;
    }

    std::mt19937_64 rng(42);
    std::uniform_int_distribution<uint64_t> dist(0, std::numeric_limits<uint64_t>::max());
    std::vector<uint64_t> h_a(word_count);
    std::vector<uint64_t> h_b(word_count);
    std::vector<int> h_sizes(static_cast<size_t>(num_columns), num_words);
    for (size_t i = 0; i < word_count; ++i)
    {
        h_a[i] = dist(rng);
        h_b[i] = dist(rng);
    }

    if (cudaMemcpy(d_a, h_a.data(), col_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_b, h_b.data(), col_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_sizes, h_sizes.data(), idx_bytes, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_sizes);
        cudaFree(d_pivots);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return bench;
    }

    const int iterations = WARP_BENCHMARK_ITERATIONS;
    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i)
    {
        launchWarpSpecializedColumnAdd(d_a, d_b, d_sizes, num_words, num_columns, 0);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_ms = 0.0f;
    if (cudaEventElapsedTime(&elapsed_ms, start, stop) == cudaSuccess &&
        std::isfinite(elapsed_ms) && elapsed_ms >= 0.0f)
    {
        bench.column_add_time_ms = elapsed_ms / iterations;
    }

    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i)
    {
        launchWarpSpecializedPivotFind(d_a, d_sizes, num_words, num_columns, d_pivots, 0);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    if (cudaEventElapsedTime(&elapsed_ms, start, stop) == cudaSuccess &&
        std::isfinite(elapsed_ms) && elapsed_ms >= 0.0f)
    {
        bench.pivot_find_time_ms = elapsed_ms / iterations;
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_sizes);
    cudaFree(d_pivots);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return bench;
}

bool hasTensorCoreSupport()
{
    int major = 0;
    return cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0) == cudaSuccess &&
           major >= 7;
}

bool hasAsyncCopySupport()
{
    int major = 0;
    return cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0) == cudaSuccess &&
           major >= 9;
}

WarpSpecializationConfig getOptimalWarpSpecializationConfig(int num_columns, int num_words,
                                                            bool has_tensor_cores)
{
    WarpSpecializationConfig config;
    config.use_warp_specialization = true;
    config.warps_per_block = WARPS_PER_BLOCK;
    config.use_tensor_cores = has_tensor_cores && (num_columns >= 2048 && num_words >= 8);
    config.use_pipelining = (num_columns >= 128 && num_words >= 4);
    config.pipeline_stages = 2;
    config.use_async_copy = hasAsyncCopySupport();
    return config;
}

} // namespace gpu
} // namespace persistence
} // namespace nerve

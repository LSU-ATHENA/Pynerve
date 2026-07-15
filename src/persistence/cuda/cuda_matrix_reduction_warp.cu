
#include "nerve/gpu/packed_column_primitives.cuh"
#include "nerve/persistence/cuda/cuda_matrix_reduction.hpp"

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cub/cub.cuh>

namespace cg = cooperative_groups;
using namespace nerve::gpu::packed;

namespace nerve::persistence::accelerated
{
constexpr int WARP_SIZE = 32;
constexpr int MAX_DIM = 32;                         // Maximum dimension for shared memory
constexpr unsigned int FULL_WARP_MASK = 0xFFFFFFFF; // All 32 threads active in warp
struct WarpBitset
{
    uint64_t *data;
    int numWords;

    __device__ __forceinline__ WarpBitset(uint64_t *ptr, int words)
        : data(ptr)
        , numWords(words)
    {}
    __device__ __forceinline__ void add(const WarpBitset &other, cg::thread_block_tile<32> &warp)
    {
        int lane = warp.thread_rank();
        packed_column_xor(data, other.data, numWords, lane, XorStrategy::DirectXor);
    }
    __device__ __forceinline__ int getLowestOne(cg::thread_block_tile<32> &warp)
    {
        int lane = warp.thread_rank();
        return packed_column_find_lsb_warp(data, numWords, lane);
    }
    __device__ __forceinline__ int getHighestOne(cg::thread_block_tile<32> &warp)
    {
        int lane = warp.thread_rank();
        return packed_column_find_msb_warp(data, numWords, lane);
    }
    __device__ __forceinline__ bool isEmpty(cg::thread_block_tile<32> &warp)
    {
        int lane = warp.thread_rank();
        return packed_column_is_empty_warp(data, numWords, lane);
    }
};

// Attempts to claim pivot[pivot] for column colIdx using
// priority-aware atomicCAS: the LOWEST column index always wins.
// Returns the column index that ultimately owns this pivot:
// - colIdx   if we own it (or successfully claimed it)
// - other index (> 0)  if another column owns it (caller should XOR)
// Never returns -1 (use owner == colIdx as the claim-success check).
__device__ __forceinline__ int claimPivotSlot(int *__restrict__ pivotColumn, int pivot, int colIdx,
                                              int lane)
{
    int owner = -1;
    if (lane == 0)
    {
        int current;
        while (true)
        {
            current = pivotColumn[pivot];
            if (current >= 0 && current < colIdx)
            {
                // Lower-indexed column already owns this pivot.
                owner = current;
                break;
            }
            if (current == colIdx)
            {
                // We already own this pivot (shouldn't normally happen).
                owner = colIdx;
                break;
            }
            // current < 0 (unclaimed) or current > colIdx (higher index).
            // Try to claim with our index (lower or first).
            int old = atomicCAS(&pivotColumn[pivot], current, colIdx);
            if (old == current)
            {
                owner = colIdx; // Successfully claimed.
                break;
            }
            // CAS failed (another thread wrote between read and CAS).
            // Retry with the updated value.
        }
    }
    return __shfl_sync(FULL_WARP_MASK, owner, 0);
}

__global__ void reduceMatrixKernelOptimized(const uint64_t *__restrict__ boundaryMatrix,
                                            uint64_t *__restrict__ columns, int n_cols,
                                            int n_words_per_col, int *__restrict__ pivotColumn,
                                            uint64_t *__restrict__ reduced)
{
    int warpsPerBlock = blockDim.x / 32;
    int warpId = threadIdx.x / 32;
    int lane = threadIdx.x & 31;

    // Grid-stride loop: each warp processes multiple columns so that the
    // kernel covers all columns regardless of the block count.
    for (int colIdx = blockIdx.x * warpsPerBlock + warpId; colIdx < n_cols;
         colIdx += gridDim.x * warpsPerBlock)
    {
        uint64_t *myCol = &columns[colIdx * n_words_per_col];

        const int MAX_ITERATIONS = 1000;
        // NOTE: d_reduced[i] is written ONCE when column i successfully
        // claims a pivot and breaks from this loop.  This is a SNAPSHOT
        // of the column's state mid-reduction while other warps are still
        // racing.  Columns that exhaust MAX_ITERATIONS before claiming a
        // pivot NEVER write to d_reduced -- the buffer remains all zeros
        // from the initial cudaMemset.  The post-pass in HyphaReducer
        // must handle both cases:
        //   - Invalidated (pivot claimed, but a lower-indexed column
        //     also claimed it): uses h_reduced[i] as starting point
        //   - Aborted (hit MAX_ITERATIONS without claiming): must build
        //     from CSC because h_reduced[i] is all zeros
        for (int iter = 0; iter < MAX_ITERATIONS; ++iter)
        {
            int pivot = packed_column_find_msb_warp(myCol, n_words_per_col, lane);
            if (pivot < 0)
                break;

            int owner = claimPivotSlot(pivotColumn, pivot, colIdx, lane);

            if (owner == colIdx)
            {
                break;
            }
            const uint64_t *otherCol = &columns[static_cast<std::size_t>(owner) * n_words_per_col];
            for (int i = lane; i < n_words_per_col; i += 32)
            {
                myCol[i] ^= otherCol[i];
            }
        }
        for (int i = lane; i < n_words_per_col; i += 32)
        {
            reduced[colIdx * n_words_per_col + i] = myCol[i];
        }
    }
}
template <int NumWarps>
__global__ void __launch_bounds__(256)
    reduceMatrixBatchPersistent(const uint64_t *__restrict__ boundaryMatrix,
                                uint64_t *__restrict__ columns, const int *__restrict__ n_cols,
                                int n_words_per_col, int *__restrict__ pivotColumn, int batchSize,
                                int maxIterations)
{
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    int warpId = block.thread_rank() / 32;
    int lane = warp.thread_rank();
    for (int matrixIdx = blockIdx.x; matrixIdx < batchSize; matrixIdx += gridDim.x)
    {
        int colsInMatrix = n_cols[matrixIdx];
        uint64_t *matrixCols = &columns[matrixIdx * n_cols[0] * n_words_per_col];
        int max_pivot = n_words_per_col * 64;
        int *matrixPivots = &pivotColumn[matrixIdx * max_pivot];
        for (int colIdx = warpId; colIdx < colsInMatrix; colIdx += NumWarps)
        {
            uint64_t *myCol = &matrixCols[colIdx * n_words_per_col];
            WarpBitset bitset(myCol, n_words_per_col);
            for (int iter = 0; iter < maxIterations; ++iter)
            {
                int pivot = bitset.getHighestOne(warp);
                if (pivot < 0)
                    break;

                int owner = claimPivotSlot(matrixPivots, pivot, colIdx, lane);

                if (owner == colIdx)
                {
                    break;
                }

                uint64_t *otherCol = &matrixCols[owner * n_words_per_col];
                WarpBitset otherBitset(otherCol, n_words_per_col);
                bitset.add(otherBitset, warp);
            }
        }

        block.sync(); // Ensure all warps finish before next matrix
    }
}
__global__ void __launch_bounds__(256)
    clearingOptimizationKernel(const int2 *__restrict__ pairs, int n_pairs,
                               uint64_t *__restrict__ columns, int n_cols, int n_words_per_col,
                               bool *__restrict__ cleared)
{
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = warp.thread_rank();
    for (int idx = tid; idx < n_pairs; idx += blockDim.x * gridDim.x)
    {
        int2 pair = pairs[idx];
        int birth = pair.x;
        int death = pair.y;

        if (birth < 0 || death < 0 || death >= n_cols)
            continue;
        if (lane == 0)
        {
            cleared[death] = true;
        }
        uint64_t *col = &columns[death * n_words_per_col];
        for (int i = lane; i < n_words_per_col; i += warp.size())
        {
            col[i] = 0;
        }
    }
}
__global__ void __launch_bounds__(256)
    computeColumnNormCUB(const uint64_t *__restrict__ column, int n_words, int *__restrict__ norm)
{
    typedef cub::BlockReduce<int, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage tempStorage;

    int tid = threadIdx.x;
    int threadSum = 0;
    for (int i = tid; i < n_words; i += blockDim.x)
    {
        threadSum += __popcll(column[i]);
    }
    int blockSum = BlockReduce(tempStorage).Sum(threadSum);

    if (tid == 0)
    {
    }
}
class ReductionGraph
{
public:
    cudaGraph_t graph;
    cudaGraphExec_t instance;
    bool captured;

    ReductionGraph()
        : graph(nullptr)
        , instance(nullptr)
        , captured(false)
    {}

    ~ReductionGraph()
    {
        if (instance)
            cudaGraphExecDestroy(instance);
        if (graph)
            cudaGraphDestroy(graph);
    }
    cudaError_t capture(uint64_t *columns, int n_cols, int n_words, int *pivotColumn,
                        cudaStream_t stream)
    {
        if (captured)
            return cudaSuccess;

        cudaGraphCreate(&graph, 0);
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        int warpsPerBlock = 8;
        int threadsPerBlock = warpsPerBlock * 32;
        int numBlocks = (n_cols + warpsPerBlock - 1) / warpsPerBlock;

        reduceMatrixKernelOptimized<<<numBlocks, threadsPerBlock, 0, stream>>>(
            nullptr, columns, n_cols, n_words, pivotColumn, columns);
        const cudaError_t launch_status = cudaPeekAtLastError();
        if (launch_status != cudaSuccess)
        {
            cudaStreamEndCapture(stream, &graph);
            return launch_status;
        }
        cudaStreamEndCapture(stream, &graph);
        cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);
        captured = true;

        return cudaSuccess;
    }
    cudaError_t launch(cudaStream_t stream)
    {
        if (!captured)
            return cudaErrorInvalidValue;
        return cudaGraphLaunch(instance, stream);
    }
};
cudaError_t reduceMatrixOptimized(const uint64_t *boundaryMatrix, uint64_t *columns, int n_cols,
                                  int n_words_per_col, int *pivotColumn, uint64_t *reduced,
                                  cudaStream_t stream = 0)
{
    if (n_cols == 0)
        return cudaSuccess;
    cudaMemsetAsync(pivotColumn, 0xFF, static_cast<std::size_t>(n_words_per_col) * 64 * sizeof(int),
                    stream);
    int warpsPerBlock = 8;
    int threadsPerBlock = warpsPerBlock * 32;
    int numBlocks = (n_cols + warpsPerBlock - 1) / warpsPerBlock;

    // Clamp to a reasonable upper bound.  The kernel uses a grid-stride
    // loop so every column is processed regardless of block count.
    // One wave of 2x SM count gives good occupancy without excessive
    // launch overhead for very large n_cols.
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int maxBlocks = prop.multiProcessorCount * 2;
    if (numBlocks > maxBlocks)
        numBlocks = maxBlocks;

    reduceMatrixKernelOptimized<<<numBlocks, threadsPerBlock, 0, stream>>>(
        boundaryMatrix, columns, n_cols, n_words_per_col, pivotColumn, reduced);

    return cudaGetLastError();
}

// Builds packed (bit-packed) column vectors directly from CSC arrays on the GPU.
// One warp per column via grid-stride loop.  Each warp sets bits cooperatively
// using atomicOr for safety when multiple lanes target the same word.
// The caller must zero-initialise `packed` (e.g. via cudaMemsetAsync) beforehand.
__global__ void buildPackedSubmatrixKernel(const int *__restrict__ col_ptr,
                                           const int *__restrict__ row_indices,
                                           uint64_t *__restrict__ packed, int n_cols,
                                           int words_per_col)
{
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    int warpsPerBlock = blockDim.x / 32;
    int warpId = (blockIdx.x * warpsPerBlock) + (threadIdx.x / 32);
    int lane = warp.thread_rank();

    for (int col = warpId; col < n_cols; col += gridDim.x * warpsPerBlock)
    {
        int start = col_ptr[col];
        int end = col_ptr[col + 1];
        uint64_t *col_out = &packed[static_cast<std::size_t>(col) * words_per_col];

        for (int i = start + lane; i < end; i += 32)
        {
            int row = row_indices[i];
            int word = row / 64;
            int bit = row % 64;
            atomicOr(reinterpret_cast<unsigned long long *>(&col_out[word]), 1ULL << bit);
        }
    }
}

__global__ void extractPivotOfColumnKernel(const uint64_t *__restrict__ reduced, int n_cols,
                                           int words_per_col, int *__restrict__ pivot_of_col)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n_cols)
        return;

    const uint64_t *col_base = reduced + static_cast<size_t>(col) * words_per_col;
    int pivot = -1;
    for (int w = words_per_col - 1; w >= 0; --w)
    {
        uint64_t word = col_base[w];
        if (word != 0)
        {
            pivot = w * 64 + (63 - __clzll(word));
            break;
        }
    }
    pivot_of_col[col] = pivot;
}

cudaError_t launchBuildPackedFromCSC(uint64_t *d_packed, const int *d_col_ptr,
                                     const int *d_row_indices, int n_cols, int words_per_col,
                                     cudaStream_t stream)
{
    if (n_cols == 0)
        return cudaSuccess;

    int warpsPerBlock = 8;
    int threadsPerBlock = warpsPerBlock * 32;
    int numBlocks = (n_cols + warpsPerBlock - 1) / warpsPerBlock;

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    int maxBlocks = prop.multiProcessorCount * 4;
    if (numBlocks > maxBlocks)
        numBlocks = maxBlocks;

    buildPackedSubmatrixKernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
        d_col_ptr, d_row_indices, d_packed, n_cols, words_per_col);

    return cudaGetLastError();
}

cudaError_t extractPivotOfColumn(const uint64_t *reduced, int n_cols, int words_per_col,
                                 int *pivot_of_col, cudaStream_t stream)
{
    if (n_cols == 0)
        return cudaSuccess;

    int blockSize = 256;
    int numBlocks = (n_cols + blockSize - 1) / blockSize;

    extractPivotOfColumnKernel<<<numBlocks, blockSize, 0, stream>>>(reduced, n_cols, words_per_col,
                                                                    pivot_of_col);

    return cudaGetLastError();
}

cudaError_t reduceMatrixBatchOptimized(const uint64_t *const *boundaryMatrices,
                                       uint64_t **columnsArray, const int *n_cols_array,
                                       int n_words_per_col, int **pivotTables, int batchSize,
                                       cudaStream_t stream = 0)
{
    if (batchSize == 0)
        return cudaSuccess;
    constexpr int NUM_WARPS = 8;
    int threadsPerBlock = NUM_WARPS * 32;
    int numBlocks = 128; // Optimal for most GPUs
    int max_pivot = n_words_per_col * 64;
    for (int i = 0; i < batchSize; ++i)
    {
        cudaMemsetAsync(pivotTables[i], 0xFF, max_pivot * sizeof(int), stream);
    }

    reduceMatrixBatchPersistent<NUM_WARPS><<<numBlocks, threadsPerBlock, 0, stream>>>(
        boundaryMatrices[0], columnsArray[0], n_cols_array, n_words_per_col, pivotTables[0],
        batchSize, 1000);

    return cudaGetLastError();
}

} // namespace nerve::persistence::accelerated

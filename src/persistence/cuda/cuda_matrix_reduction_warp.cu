
#include "nerve/persistence/cuda/cuda_matrix_reduction.hpp"
#include "nerve/gpu/packed_column_primitives.cuh"

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cub/cub.cuh>

namespace cg = cooperative_groups;

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
        packed::packed_column_xor(data, other.data, numWords, lane,
                                           packed::XorStrategy::DirectXor);
    }
    __device__ __forceinline__ int getLowestOne(cg::thread_block_tile<32> &warp)
    {
        int lane = warp.thread_rank();
        return packed::packed_column_find_lsb_warp(data, numWords, lane);
    }
    __device__ __forceinline__ bool isEmpty(cg::thread_block_tile<32> &warp)
    {
        int lane = warp.thread_rank();
        return packed::packed_column_is_empty_warp(data, numWords, lane);
    }
};
__global__ void __launch_bounds__(256)
    reduceMatrixKernelOptimized(const uint64_t *__restrict__ boundaryMatrix,
                                uint64_t *__restrict__ columns, int n_cols, int n_words_per_col,
                                int *__restrict__ pivotColumn, uint64_t *__restrict__ reduced)
{
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    int warpId = block.thread_rank() / 32;
    int lane = warp.thread_rank();
    int colIdx = blockIdx.x * (blockDim.x / 32) + warpId;

    if (colIdx >= n_cols)
        return;
    extern __shared__ int sharedPivotCache[];
    int cacheSize = blockDim.x / 32;
    uint64_t *myCol = &columns[colIdx * n_words_per_col];
    WarpBitset bitset(myCol, n_words_per_col);
    const int MAX_ITERATIONS = 1000; // Prevent infinite loops
    for (int iter = 0; iter < MAX_ITERATIONS; ++iter)
    {
        int pivot = bitset.getLowestOne(warp);

        if (pivot < 0)
            break; // Column is empty
        int colToAdd = -1;
        if (lane == 0)
        {
            colToAdd = atomicCAS(&pivotColumn[pivot], -1, colIdx);
        }
        colToAdd = __shfl_sync(FULL_WARP_MASK, colToAdd, 0);

        if (colToAdd < 0 || colToAdd == colIdx)
        {
            if (lane == 0)
            {
                pivotColumn[pivot] = colIdx;
            }
            break;
        }
        uint64_t *otherCol = &columns[colToAdd * n_words_per_col];
        WarpBitset otherBitset(otherCol, n_words_per_col);
        bitset.add(otherBitset, warp);
    }
    for (int i = lane; i < n_words_per_col; i += warp.size())
    {
        reduced[colIdx * n_words_per_col + i] = myCol[i];
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
        int *matrixPivots = &pivotColumn[matrixIdx * n_cols[0]];
        for (int colIdx = warpId; colIdx < colsInMatrix; colIdx += NumWarps)
        {
            uint64_t *myCol = &matrixCols[colIdx * n_words_per_col];
            WarpBitset bitset(myCol, n_words_per_col);
            for (int iter = 0; iter < maxIterations; ++iter)
            {
                int pivot = bitset.getLowestOne(warp);
                if (pivot < 0)
                    break;

                int colToAdd = -1;
                if (lane == 0)
                {
                    colToAdd = atomicCAS(&matrixPivots[pivot], -1, colIdx);
                }
                colToAdd = __shfl_sync(FULL_WARP_MASK, colToAdd, 0);

                if (colToAdd < 0 || colToAdd == colIdx)
                {
                    if (lane == 0)
                        matrixPivots[pivot] = colIdx;
                    break;
                }

                uint64_t *otherCol = &matrixCols[colToAdd * n_words_per_col];
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

        int sharedMemSize = warpsPerBlock * sizeof(int); // Pivot cache

        reduceMatrixKernelOptimized<<<numBlocks, threadsPerBlock, sharedMemSize, stream>>>(
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
    cudaMemsetAsync(pivotColumn, 0xFF, n_cols * sizeof(int), stream);
    int warpsPerBlock = 8;
    int threadsPerBlock = warpsPerBlock * 32;
    int numBlocks = (n_cols + warpsPerBlock - 1) / warpsPerBlock;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int maxBlocks = prop.multiProcessorCount * 2;
    if (numBlocks > maxBlocks)
        numBlocks = maxBlocks;

    int sharedMemSize = warpsPerBlock * sizeof(int);

    reduceMatrixKernelOptimized<<<numBlocks, threadsPerBlock, sharedMemSize, stream>>>(
        boundaryMatrix, columns, n_cols, n_words_per_col, pivotColumn, reduced);

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
    for (int i = 0; i < batchSize; ++i)
    {
        cudaMemsetAsync(pivotTables[i], 0xFF, n_cols_array[i] * sizeof(int), stream);
    }

    reduceMatrixBatchPersistent<NUM_WARPS><<<numBlocks, threadsPerBlock, 0, stream>>>(
        boundaryMatrices[0], columnsArray[0], n_cols_array, n_words_per_col, pivotTables[0],
        batchSize, 1000);

    return cudaGetLastError();
}

} // namespace nerve::persistence::accelerated

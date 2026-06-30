#include "nerve/algebra/boundary.hpp"
#include "nerve/errors/errors.hpp"
#include "nerve/gpu/device_array.hpp"
#include "nerve/gpu/gpu_error.hpp"
#include "nerve/types.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstring>
#include <limits>
#include <vector>

namespace nerve
{
namespace gpu
{
namespace kernels
{

namespace
{

constexpr int kDefaultBlockSize = 256;
constexpr int kDefaultMaxPasses = 32;
constexpr int kMaxColumnsPerBlock = 1024;

__device__ __forceinline__ int dmax(int a, int b)
{
    return (a > b) ? a : b;
}

__device__ int warpReduceMax(int val)
{
#pragma unroll
    for (int offset = 32; offset > 0; offset /= 2)
    {
        val = dmax(val, __shfl_down_sync(0xffffffff, val, offset));
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

inline int gridSize(int count, int block_size)
{
    return (count + block_size - 1) / block_size;
}

inline bool checkedProduct(std::size_t lhs, std::size_t rhs, std::size_t &out) noexcept
{
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs)
        return false;
    out = lhs * rhs;
    return true;
}

inline bool checkedBytes(std::size_t count, std::size_t elem_size, std::size_t &out) noexcept
{
    return checkedProduct(count, elem_size, out);
}

inline bool fitsInt(std::size_t val) noexcept
{
    return val <= static_cast<std::size_t>(std::numeric_limits<int>::max());
}

inline errors::ErrorResult<void> launchFail(const char *msg)
{
    return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL, msg);
}

inline errors::ErrorResult<void> oom()
{
    return errors::ErrorResult<void>::error(errors::ErrorCode::E10_GPU_OOM);
}

} // anonymous namespace

// GPU Kernels

// Kernel 1: Column Density Analysis
// Counts non-zero entries per column for load-balanced partitioning.
__global__ __launch_bounds__(256) void gpuColumnDensityKernel(const int *__restrict__ col_starts,
                                                              const int *__restrict__ col_indices,
                                                              int n_columns, int max_height,
                                                              int *__restrict__ densities)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n_columns)
        return;

    int start = col_starts[col];
    int end = (col + 1 < n_columns) ? col_starts[col + 1] : start + max_height;

    int count = 0;
    for (int i = start; i < end; ++i)
    {
        if (col_indices[i] >= 0)
            ++count;
    }
    densities[col] = count;
}

// Kernel 2: Dynamic Partition from Cumulative Density
// Given cumulative densities and target blocks, computes partition boundaries
// via binary search. Each thread block computes one boundary.

__global__ __launch_bounds__(256) void gpuPartitionFromCumulativeKernel(
    const int *__restrict__ cumulative, int total_density, int n_columns, int num_blocks,
    int *__restrict__ block_starts, int *__restrict__ block_ends)
{
    int bid = blockIdx.x;
    if (bid >= num_blocks)
        return;

    int target = (bid + 1) * total_density / num_blocks;

    int lo = 0;
    int hi = n_columns - 1;
    while (lo < hi)
    {
        int mid = (lo + hi) >> 1;
        if (cumulative[mid] < target)
        {
            lo = mid + 1;
        }
        else
        {
            hi = mid;
        }
    }
    int boundary = lo;

    if (bid == 0)
    {
        block_starts[0] = 0;
        block_ends[0] = boundary;
    }
    else
    {
        block_ends[bid] = boundary;
        if (bid + 1 < num_blocks)
        {
            block_starts[bid + 1] = boundary;
        }
    }
    if (bid == num_blocks - 1)
    {
        block_ends[bid] = n_columns;
    }
}

// Kernel 3: Block-Level Reduction
// Each CUDA block reduces its assigned columns sequentially using block-local
// pivot tracking in shared memory. Threads cooperate on each column via
// warp-level reductions for pivot finding and parallel XOR addition.

template <int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE) void gpuBlockReductionKernel(
    int *__restrict__ col_data, int *__restrict__ col_indices, const int *__restrict__ col_starts,
    int block_start, int block_end, int *__restrict__ pivots, int n_columns, int max_height)
{
    int n_block_cols = block_end - block_start;
    if (n_block_cols <= 0)
        return;
    if (n_block_cols > kMaxColumnsPerBlock)
        return;

    extern __shared__ int s_shmem[];
    int *s_block_cols = s_shmem;
    int *s_block_pivots = s_shmem + n_block_cols;

    __shared__ int reduce_shmem[64];

    for (int idx = 0; idx < n_block_cols; ++idx)
    {
        int col = block_start + idx;
        int col_start = col_starts[col];
        int col_end = (col + 1 < n_columns) ? col_starts[col + 1] : col_start + max_height;
        int col_len = col_end - col_start;

        if (col_len <= 0)
        {
            if (threadIdx.x == 0)
                pivots[col] = -1;
            continue;
        }

        int *c_data = col_data + col_start;
        int *c_idx = col_indices + col_start;

        for (int pass = 0; pass < kDefaultMaxPasses; ++pass)
        {
            int local_max = -1;
            for (int i = threadIdx.x; i < col_len; i += BLOCK_SIZE)
            {
                if (c_data[i] != 0)
                {
                    local_max = dmax(local_max, c_idx[i]);
                }
            }
            int pivot = blockReduceMax(local_max, reduce_shmem);
            if (pivot < 0)
                break;

            int has_conflict = 0;
            int conflict_col = -1;
            if (threadIdx.x == 0)
            {
                for (int j = 0; j < idx; ++j)
                {
                    if (s_block_pivots[j] == pivot)
                    {
                        has_conflict = 1;
                        conflict_col = s_block_cols[j];
                        break;
                    }
                }
            }
            __syncthreads();

            if (has_conflict && threadIdx.x == 0)
            {
                s_block_cols[idx] = col;
                s_block_pivots[idx] = pivot;
            }
            __syncthreads();

            if (!has_conflict)
            {
                if (threadIdx.x == 0)
                {
                    s_block_cols[idx] = col;
                    s_block_pivots[idx] = pivot;
                    pivots[col] = pivot;
                }
                __syncthreads();
                break;
            }

            if (conflict_col >= 0)
            {
                int src_start = col_starts[conflict_col];
                int src_end = (conflict_col + 1 < n_columns) ? col_starts[conflict_col + 1]
                                                             : src_start + max_height;

                for (int i = threadIdx.x; i < (src_end - src_start); i += BLOCK_SIZE)
                {
                    int s_row = col_indices[src_start + i];
                    if (s_row < 0)
                        continue;
                    int lo = 0;
                    int hi = col_len - 1;
                    while (lo <= hi)
                    {
                        int mid = lo + ((hi - lo) >> 1);
                        if (c_idx[mid] == s_row)
                        {
                            atomicXor(c_data + mid, 1);
                            break;
                        }
                        else if (c_idx[mid] < s_row)
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
            __syncthreads();
        }

        if (threadIdx.x == 0)
        {
            int has_nonzero = 0;
            for (int i = 0; i < col_len; ++i)
            {
                if (c_data[i] != 0)
                {
                    has_nonzero = 1;
                    break;
                }
            }
            if (!has_nonzero)
                pivots[col] = -1;
        }
        __syncthreads();
    }
}

// Kernel 4: Cross-Block Merge
// Resolves pivot conflicts between blocks. Each thread block handles one
// column. Priority is given to the lowest-indexed column for each pivot.
// Uses atomicCAS on global low_to_col for conflict detection.

__global__ __launch_bounds__(256) void gpuMergeResultsKernel(
    int *__restrict__ col_data, int *__restrict__ col_indices, const int *__restrict__ col_starts,
    int *__restrict__ pivots, int *__restrict__ low_to_col, int n_columns, int max_height,
    int max_passes)
{
    int col = blockIdx.x;
    if (col >= n_columns)
        return;

    int start = col_starts[col];
    int col_end = (col + 1 < n_columns) ? col_starts[col + 1] : start + max_height;
    int col_len = col_end - start;
    if (col_len <= 0)
    {
        if (threadIdx.x == 0)
            pivots[col] = -1;
        return;
    }

    int *c_data = col_data + start;
    int *c_idx = col_indices + start;

    __shared__ int reduce_shmem[64];
    __shared__ int block_pivot;
    __shared__ int block_src;

    if (pivots[col] < 0)
        return;

    for (int pass = 0; pass < max_passes; ++pass)
    {
        int local_max = -1;
        for (int i = threadIdx.x; i < col_len; i += blockDim.x)
        {
            if (c_data[i] != 0)
            {
                local_max = dmax(local_max, c_idx[i]);
            }
        }

        int pivot_candidate = blockReduceMax(local_max, reduce_shmem);

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
                int claimed = atomicCAS(&low_to_col[block_pivot], -1, col);
                if (claimed == -1)
                {
                    pivots[col] = block_pivot;
                    return;
                }
                block_src = claimed;
            }
        }
        __syncthreads();

        int src_col = block_src;
        if (src_col < 0 || src_col == col)
        {
            break;
        }

        int src_start = col_starts[src_col];
        int src_end = (src_col + 1 < n_columns) ? col_starts[src_col + 1] : src_start + max_height;

        for (int i = threadIdx.x; i < (src_end - src_start); i += blockDim.x)
        {
            if (col_data[src_start + i] == 0)
                continue;
            int s_row = col_indices[src_start + i];
            if (s_row < 0)
                continue;
            int lo = 0;
            int hi = col_len - 1;
            while (lo <= hi)
            {
                int mid = lo + ((hi - lo) >> 1);
                if (c_idx[mid] == s_row)
                {
                    atomicXor(c_data + mid, 1);
                    break;
                }
                else if (c_idx[mid] < s_row)
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
    }

    if (threadIdx.x == 0)
    {
        pivots[col] = -1;
    }
}

// Explicit template instantiations for gpuBlockReductionKernel
template __global__ void gpuBlockReductionKernel<128>(int *, int *, const int *, int, int, int *,
                                                      int, int);
template __global__ void gpuBlockReductionKernel<256>(int *, int *, const int *, int, int, int *,
                                                      int, int);
template __global__ void gpuBlockReductionKernel<512>(int *, int *, const int *, int, int, int *,
                                                      int, int);

// DynamicPartitionReducer Host-Side Implementation

struct GpuMatrix
{
    DeviceArray<int> data;
    DeviceArray<int> indices;
    DeviceArray<int> starts;
    int n_columns = 0;
    int max_height = 0;
};

struct GpuReduceState
{
    DeviceArray<int> pivots;
    DeviceArray<int> low_to_col;
    DeviceArray<int> densities;
    DeviceArray<int> cumulative;
    DeviceArray<int2> pairs;
    DeviceArray<int> pair_count;
};

class DynamicPartitionReducer
{
public:
    DynamicPartitionReducer();
    ~DynamicPartitionReducer();

    DynamicPartitionReducer(const DynamicPartitionReducer &) = delete;
    DynamicPartitionReducer &operator=(const DynamicPartitionReducer &) = delete;

    errors::ErrorResult<void> compute(const algebra::BoundaryMatrix &boundary_matrix,
                                      std::vector<Index> &out_pivots,
                                      std::vector<std::pair<Size, Size>> &out_pairs);

    void setBlockSize(int bs);
    void setNumBlocks(int nb);
    void setMaxPasses(int mp);
    void setImbalanceTarget(float t);

    int blockSize() const;
    int numBlocks() const;

private:
    errors::ErrorResult<void> convertToGpu(const algebra::BoundaryMatrix &matrix);
    errors::ErrorResult<void> allocateState(int n_columns);
    errors::ErrorResult<void> initializeState(int n_columns);
    errors::ErrorResult<void> analyzeDensity();
    errors::ErrorResult<void> computePartition();
    errors::ErrorResult<void> blockReduction();
    errors::ErrorResult<void> mergeResults();
    errors::ErrorResult<void> extractPairs(std::vector<Index> &out_pivots,
                                           std::vector<std::pair<Size, Size>> &out_pairs);

    GpuMatrix matrix_;
    GpuReduceState state_;

    DeviceArray<int> block_starts_dev_;
    DeviceArray<int> block_ends_dev_;

    std::vector<int> densities_;
    std::vector<int> cumulative_;
    std::vector<int> block_starts_;
    std::vector<int> block_ends_;
    int total_density_ = 0;

    int block_size_ = kDefaultBlockSize;
    int num_blocks_ = 0;
    int max_passes_ = kDefaultMaxPasses;
    float imbalance_target_ = 0.1f;
};

DynamicPartitionReducer::DynamicPartitionReducer()
    : block_size_(kDefaultBlockSize)
    , num_blocks_(0)
    , max_passes_(kDefaultMaxPasses)
    , imbalance_target_(0.1f)
{}

DynamicPartitionReducer::~DynamicPartitionReducer() = default;

errors::ErrorResult<void>
DynamicPartitionReducer::compute(const algebra::BoundaryMatrix &boundary_matrix,
                                 std::vector<Index> &out_pivots,
                                 std::vector<std::pair<Size, Size>> &out_pairs)
{
    out_pivots.clear();
    out_pairs.clear();

    if (boundary_matrix.cols() == 0)
    {
        return errors::ErrorResult<void>::success();
    }

    auto convert_result = convertToGpu(boundary_matrix);
    if (convert_result.isError())
        return convert_result;

    auto alloc_result = allocateState(matrix_.n_columns);
    if (alloc_result.isError())
        return alloc_result;

    auto init_result = initializeState(matrix_.n_columns);
    if (init_result.isError())
        return init_result;

    auto density_result = analyzeDensity();
    if (density_result.isError())
        return density_result;

    auto partition_result = computePartition();
    if (partition_result.isError())
        return partition_result;

    auto block_reduce_result = blockReduction();
    if (block_reduce_result.isError())
        return block_reduce_result;

    auto merge_result = mergeResults();
    if (merge_result.isError())
        return merge_result;

    auto extract_result = extractPairs(out_pivots, out_pairs);
    return extract_result;
}

errors::ErrorResult<void>
DynamicPartitionReducer::convertToGpu(const algebra::BoundaryMatrix &boundary_matrix)
{
    Size n_cols = boundary_matrix.cols();
    Size n_rows = boundary_matrix.rows();

    int n_cols_int = 0;
    if (!fitsInt(n_cols))
        return errors::ErrorResult<void>::error(errors::ErrorCode::E41_RESOURCE_LIMIT,
                                                "GPU reduction columns exceed int range");

    int max_entries = 0;
    for (Size col = 0; col < n_cols; ++col)
    {
        int count = 0;
        for (Size row = 0; row < n_rows; ++row)
        {
            if (boundary_matrix.getMatrixEntry(row, col) != 0.0)
                ++count;
        }
        max_entries = std::max(max_entries, count);
    }

    matrix_.n_columns = n_cols_int;
    matrix_.max_height = max_entries > 0 ? max_entries : 1;

    std::size_t total_entries = 0;
    if (!checkedProduct(static_cast<std::size_t>(n_cols_int),
                        static_cast<std::size_t>(matrix_.max_height), total_entries))
        return errors::ErrorResult<void>::error(errors::ErrorCode::E41_RESOURCE_LIMIT,
                                                "GPU matrix size overflows");

    std::size_t data_bytes = 0;
    std::size_t starts_bytes = 0;
    if (!checkedBytes(total_entries, sizeof(int), data_bytes) ||
        !checkedBytes(static_cast<std::size_t>(n_cols_int), sizeof(int), starts_bytes))
        return errors::ErrorResult<void>::error(errors::ErrorCode::E41_RESOURCE_LIMIT,
                                                "GPU matrix byte count overflows");

    std::vector<int> h_data(total_entries, 0);
    std::vector<int> h_indices(total_entries, -1);
    std::vector<int> h_starts(static_cast<std::size_t>(n_cols_int), 0);

    std::size_t pos = 0;
    for (Size col = 0; col < n_cols; ++col)
    {
        h_starts[static_cast<std::size_t>(col)] = static_cast<int>(pos);
        int row_count = 0;
        for (Size row = 0; row < n_rows && row_count < matrix_.max_height; ++row)
        {
            if (boundary_matrix.getMatrixEntry(row, col) != 0.0)
            {
                h_data[pos] = 1;
                h_indices[pos] = static_cast<int>(row);
                ++pos;
                ++row_count;
            }
        }
        pos = static_cast<std::size_t>(h_starts[static_cast<std::size_t>(col)]) +
              static_cast<std::size_t>(matrix_.max_height);
    }

    try
    {
        matrix_.data = DeviceArray<int>(total_entries);
        matrix_.indices = DeviceArray<int>(total_entries);
        matrix_.starts = DeviceArray<int>(static_cast<std::size_t>(n_cols_int));
    }
    catch (...)
    {
        return oom();
    }

    cudaError_t err;
    err = cudaMemcpy(matrix_.data.get(), h_data.data(), data_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
        return launchFail("copy matrix data to device");
    err = cudaMemcpy(matrix_.indices.get(), h_indices.data(), data_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
        return launchFail("copy matrix indices to device");
    err = cudaMemcpy(matrix_.starts.get(), h_starts.data(), starts_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
        return launchFail("copy matrix starts to device");

    return errors::ErrorResult<void>::success();
}

errors::ErrorResult<void> DynamicPartitionReducer::allocateState(int n_columns)
{
    std::size_t n = static_cast<std::size_t>(n_columns);

    try
    {
        state_.pivots = DeviceArray<int>(n);
        state_.low_to_col = DeviceArray<int>(n);
        state_.densities = DeviceArray<int>(n);
        state_.cumulative = DeviceArray<int>(n);
        state_.pairs = DeviceArray<int2>(n);
        state_.pair_count = DeviceArray<int>(1);
    }
    catch (...)
    {
        return oom();
    }

    return errors::ErrorResult<void>::success();
}

errors::ErrorResult<void> DynamicPartitionReducer::initializeState(int n_columns)
{
    std::vector<int> h_init(static_cast<std::size_t>(n_columns), -1);
    cudaError_t err;

    err = cudaMemcpy(state_.pivots.get(), h_init.data(),
                     static_cast<std::size_t>(n_columns) * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
        return launchFail("init pivots");

    err = cudaMemcpy(state_.low_to_col.get(), h_init.data(),
                     static_cast<std::size_t>(n_columns) * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
        return launchFail("init low_to_col");

    err = cudaMemset(state_.pair_count.get(), 0, sizeof(int));
    if (err != cudaSuccess)
        return launchFail("init pair_count");

    return errors::ErrorResult<void>::success();
}

errors::ErrorResult<void> DynamicPartitionReducer::analyzeDensity()
{
    int n_cols = matrix_.n_columns;
    int blk = block_size_;
    int grid = gridSize(n_cols, blk);

    gpuColumnDensityKernel<<<grid, blk>>>(matrix_.starts.get(), matrix_.indices.get(), n_cols,
                                          matrix_.max_height, state_.densities.get());
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        return launchFail("gpuColumnDensityKernel");
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        return launchFail("sync gpuColumnDensityKernel");

    std::vector<int> h_densities(static_cast<std::size_t>(n_cols));
    err = cudaMemcpy(h_densities.data(), state_.densities.get(),
                     static_cast<std::size_t>(n_cols) * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
        return launchFail("read densities");

    densities_.resize(static_cast<std::size_t>(n_cols));
    cumulative_.resize(static_cast<std::size_t>(n_cols));
    total_density_ = 0;
    for (int i = 0; i < n_cols; ++i)
    {
        int d = h_densities[static_cast<std::size_t>(i)];
        if (d <= 0)
            d = 1;
        densities_[static_cast<std::size_t>(i)] = d;
        total_density_ += d;
        cumulative_[static_cast<std::size_t>(i)] = total_density_;
    }

    return errors::ErrorResult<void>::success();
}

errors::ErrorResult<void> DynamicPartitionReducer::computePartition()
{
    int n_cols = matrix_.n_columns;
    if (n_cols <= 0)
        return errors::ErrorResult<void>::success();

    cudaDeviceProp prop{};
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess)
        return launchFail("get device properties");

    int sm_count = prop.multiProcessorCount;
    int target_blocks = sm_count * 2;
    int max_blocks = std::min(target_blocks, n_cols);
    num_blocks_ = std::max(1, max_blocks);

    block_starts_.resize(static_cast<std::size_t>(num_blocks_));
    block_ends_.resize(static_cast<std::size_t>(num_blocks_));

    int target_per_block = total_density_ / num_blocks_;

    block_starts_[0] = 0;
    for (int b = 1; b < num_blocks_; ++b)
    {
        int target = b * target_per_block;
        auto it = std::lower_bound(cumulative_.begin(), cumulative_.end(), target);
        int boundary = static_cast<int>(std::distance(cumulative_.begin(), it));
        block_starts_[static_cast<std::size_t>(b)] = std::min(boundary, n_cols);
    }
    for (int b = 0; b < num_blocks_ - 1; ++b)
    {
        block_ends_[static_cast<std::size_t>(b)] = block_starts_[static_cast<std::size_t>(b + 1)];
    }
    block_ends_[static_cast<std::size_t>(num_blocks_ - 1)] = n_cols;

    try
    {
        block_starts_dev_ = DeviceArray<int>(static_cast<std::size_t>(num_blocks_));
        block_ends_dev_ = DeviceArray<int>(static_cast<std::size_t>(num_blocks_));
    }
    catch (...)
    {
        return oom();
    }

    err = cudaMemcpy(block_starts_dev_.get(), block_starts_.data(),
                     static_cast<std::size_t>(num_blocks_) * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
        return launchFail("copy block_starts to device");

    err = cudaMemcpy(block_ends_dev_.get(), block_ends_.data(),
                     static_cast<std::size_t>(num_blocks_) * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
        return launchFail("copy block_ends to device");

    return errors::ErrorResult<void>::success();
}

errors::ErrorResult<void> DynamicPartitionReducer::blockReduction()
{
    int n_cols = matrix_.n_columns;
    int max_h = matrix_.max_height;

    for (int b = 0; b < num_blocks_; ++b)
    {
        int blk_start = block_starts_[static_cast<std::size_t>(b)];
        int blk_end = block_ends_[static_cast<std::size_t>(b)];
        int n_block_cols = blk_end - blk_start;

        if (n_block_cols <= 0)
            continue;

        int blk_size = 256;
        if (n_block_cols < 32)
            blk_size = 128;
        else if (n_block_cols > 256)
            blk_size = 512;

        std::size_t shmem_bytes = static_cast<std::size_t>(n_block_cols) * 2 * sizeof(int);

        cudaError_t err;
        if (blk_size <= 128)
        {
            gpuBlockReductionKernel<128><<<1, 128, shmem_bytes>>>(
                matrix_.data.get(), matrix_.indices.get(), matrix_.starts.get(), blk_start, blk_end,
                state_.pivots.get(), n_cols, max_h);
        }
        else if (blk_size <= 256)
        {
            gpuBlockReductionKernel<256><<<1, 256, shmem_bytes>>>(
                matrix_.data.get(), matrix_.indices.get(), matrix_.starts.get(), blk_start, blk_end,
                state_.pivots.get(), n_cols, max_h);
        }
        else
        {
            gpuBlockReductionKernel<512><<<1, 512, shmem_bytes>>>(
                matrix_.data.get(), matrix_.indices.get(), matrix_.starts.get(), blk_start, blk_end,
                state_.pivots.get(), n_cols, max_h);
        }

        err = cudaGetLastError();
        if (err != cudaSuccess)
            return launchFail("gpuBlockReductionKernel");
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
            return launchFail("sync gpuBlockReductionKernel");
    }

    return errors::ErrorResult<void>::success();
}

errors::ErrorResult<void> DynamicPartitionReducer::mergeResults()
{
    int n_cols = matrix_.n_columns;
    int max_h = matrix_.max_height;
    int blk = block_size_;

    std::vector<int> h_init(static_cast<std::size_t>(n_cols), -1);
    cudaError_t err =
        cudaMemcpy(state_.low_to_col.get(), h_init.data(),
                   static_cast<std::size_t>(n_cols) * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
        return launchFail("reset low_to_col for merge");

    gpuMergeResultsKernel<<<n_cols, blk>>>(matrix_.data.get(), matrix_.indices.get(),
                                           matrix_.starts.get(), state_.pivots.get(),
                                           state_.low_to_col.get(), n_cols, max_h, max_passes_);
    err = cudaGetLastError();
    if (err != cudaSuccess)
        return launchFail("gpuMergeResultsKernel");
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        return launchFail("sync gpuMergeResultsKernel");

    return errors::ErrorResult<void>::success();
}

errors::ErrorResult<void>
DynamicPartitionReducer::extractPairs(std::vector<Index> &out_pivots,
                                      std::vector<std::pair<Size, Size>> &out_pairs)
{
    int n_cols = matrix_.n_columns;

    std::vector<int> h_pivots(static_cast<std::size_t>(n_cols), -1);
    cudaError_t err =
        cudaMemcpy(h_pivots.data(), state_.pivots.get(),
                   static_cast<std::size_t>(n_cols) * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
        return launchFail("read pivots");

    out_pivots.resize(static_cast<std::size_t>(n_cols));
    for (int i = 0; i < n_cols; ++i)
    {
        out_pivots[static_cast<std::size_t>(i)] =
            static_cast<Index>(h_pivots[static_cast<std::size_t>(i)]);
    }

    std::vector<int> h_low_to_col(static_cast<std::size_t>(n_cols), -1);
    err = cudaMemcpy(h_low_to_col.data(), state_.low_to_col.get(),
                     static_cast<std::size_t>(n_cols) * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
        return launchFail("read low_to_col");

    for (int col = 0; col < n_cols; ++col)
    {
        int pivot = h_pivots[static_cast<std::size_t>(col)];
        if (pivot >= 0 && h_low_to_col[static_cast<std::size_t>(pivot)] == col)
        {
            out_pairs.emplace_back(static_cast<Size>(pivot), static_cast<Size>(col));
        }
    }

    return errors::ErrorResult<void>::success();
}

void DynamicPartitionReducer::setBlockSize(int bs)
{
    block_size_ = std::max(32, bs);
}
void DynamicPartitionReducer::setNumBlocks(int nb)
{
    num_blocks_ = std::max(1, nb);
}
void DynamicPartitionReducer::setMaxPasses(int mp)
{
    max_passes_ = std::max(1, mp);
}
void DynamicPartitionReducer::setImbalanceTarget(float t)
{
    imbalance_target_ = std::max(0.01f, t);
}
int DynamicPartitionReducer::blockSize() const
{
    return block_size_;
}
int DynamicPartitionReducer::numBlocks() const
{
    return num_blocks_;
}

} // namespace kernels
} // namespace gpu
} // namespace nerve

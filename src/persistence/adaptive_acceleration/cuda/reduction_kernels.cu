#include "nerve/gpu/gpu_error.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace nerve::persistence::adaptive_acceleration::gpu
{

constexpr int REDUCTION_BLOCK_SIZE = 256;

template <int ComputeCapability>
struct ReductionParams
{
    static constexpr int TILE_SIZE = 16;
    static constexpr int BLOCK_SIZE = 256;
    static constexpr int WARP_SIZE = 32;
    static constexpr int SHARED_MEM_SIZE = 48 * 1024; // 48KB per SM
    static constexpr bool USE_TENSOR_CORES = ComputeCapability >= 70;
    static constexpr bool USE_WMMA = ComputeCapability >= 70;
    static constexpr bool USE_MMA = ComputeCapability >= 80;
    static constexpr bool USE_FP64_TENSOR_CORES = ComputeCapability >= 80;

    static constexpr bool USE_FP16 = ComputeCapability >= 70;
    static constexpr bool USE_BF16 = ComputeCapability >= 86;
    static constexpr bool USE_TF32 = ComputeCapability >= 80;

    static constexpr bool ENABLE_MIXED_PRECISION = true;
};

__device__ __forceinline__ int getComputeCapability()
{
#if defined(__CUDA_ARCH__)
    return __CUDA_ARCH__ / 10;
#else
    return 0;
#endif
}

__device__ __forceinline__ bool canUseTensorCores()
{
    int compute_cap = getComputeCapability();
    return compute_cap >= 70;
}

__device__ __forceinline__ bool canUseWmma()
{
    int compute_cap = getComputeCapability();
    return compute_cap >= 70;
}

__device__ __forceinline__ bool canUseMma()
{
    int compute_cap = getComputeCapability();
    return compute_cap >= 80;
}

__device__ __forceinline__ double deviceInfinityDouble()
{
    return __longlong_as_double(0x7ff0000000000000ULL);
}

__device__ __forceinline__ bool accumulateSquaredDiff(double diff, double &dist_sq)
{
    const double contribution = diff * diff;
    const double next = dist_sq + contribution;
    if (!isfinite(diff) || !isfinite(contribution) || !isfinite(next))
    {
        return false;
    }
    dist_sq = next;
    return true;
}

__host__ __device__ inline size_t matrixIndex(size_t row, size_t col, size_t leading_dim)
{
    return row * leading_dim + col;
}

__device__ void atomicClearColumn(int *__restrict__ matrix_data, int row_start, int row_end,
                                  int clear_value)
{
    for (int idx = row_start; idx < row_end; ++idx)
    {
        atomicAnd(&matrix_data[idx], clear_value);
    }
}

template <typename Precision>
__global__ __launch_bounds__(REDUCTION_BLOCK_SIZE) void parallelColumnReductionKernel(
    int *__restrict__ low_row_to_col, int *__restrict__ col_pivot, bool *__restrict__ clear_cols,
    int *__restrict__ matrix_data, int *__restrict__ row_starts, int n_rows, int n_cols)
{
    // Thread and block indices
    const int tid = threadIdx.x;
    const int block_id = blockIdx.x;
    const int block_size = blockDim.x;
    const int grid_size = gridDim.x;

    // Shared memory for pivot tracking
    __shared__ int s_pivots[REDUCTION_BLOCK_SIZE];
    __shared__ bool s_clear_flags[REDUCTION_BLOCK_SIZE];

    // Initialize shared memory
    if (tid < 256)
    {
        s_pivots[tid] = -1;
        s_clear_flags[tid] = false;
    }
    __syncthreads();

    // Process columns in parallel
    for (int col = block_id; col < n_cols; col += grid_size)
    {
        const int col_start = col * block_size + tid;

        // Find pivot for this column
        int pivot = -1;
        if (col_start < n_rows)
        {
            // Scan column for non-zero element
            for (int row = col_start; row < n_rows; ++row)
            {
                int col_start_idx = row_starts[row];
                int col_end_idx = (row + 1 < n_rows) ? row_starts[row + 1] : n_cols;

                // Check for non-zero element
                for (int idx = col_start_idx; idx < col_end_idx; ++idx)
                {
                    if (matrix_data[idx] != 0)
                    {
                        pivot = row;
                        break;
                    }
                }

                if (pivot >= 0)
                {
                    break;
                }
            }
        }

        // Store pivot in shared memory
        if (tid < 256 && col < 256)
        {
            s_pivots[col] = pivot;
            s_clear_flags[col] = (pivot >= 0);
        }
        __syncthreads();

        // Clear columns that can be cleared
        if (pivot >= 0 && pivot < n_rows)
        {
            int col_start_idx = row_starts[pivot];
            int col_end_idx = (pivot + 1 < n_rows) ? row_starts[pivot + 1] : n_cols;

            // Check if all elements in this row are zero
            bool all_zero = true;
            for (int idx = col_start_idx; idx < col_end_idx; ++idx)
            {
                if (matrix_data[idx] != 0)
                {
                    all_zero = false;
                    break;
                }
            }

            if (tid < 256 && col < 256)
            {
                s_clear_flags[col] = all_zero;
            }
        }
        __syncthreads();

        // Perform column reduction if pivot found
        if (pivot >= 0 && pivot < n_rows)
        {
            int col_start_idx = row_starts[pivot];

            // Add this column to the low_row_to_col mapping
            low_row_to_col[pivot] = col;

            // Perform reduction using atomic operations
            for (int row = 0; row < pivot; ++row)
            {
                int col_end_idx = (row + 1 < n_rows) ? row_starts[row + 1] : n_cols;

                // Add row to column if not cleared
                if (!s_clear_flags[col] || row >= pivot)
                {
                    for (int idx = col_start_idx; idx < col_end_idx; ++idx)
                    {
                        // Atomic add operation
                        int old_val = matrix_data[idx];
                        matrix_data[idx] = old_val + matrix_data[col_start_idx];
                    }
                }
            }

            // Mark column as processed
            if (tid < 256 && col < 256)
            {
                s_pivots[col] = -1; // Mark as processed
            }
        }
    }

    // Synchronize all threads
    __syncthreads();

    // Copy results back to global memory
    if (tid == 0)
    {
        const int shared_cols = (n_cols < REDUCTION_BLOCK_SIZE) ? n_cols : REDUCTION_BLOCK_SIZE;
        for (int col = 0; col < shared_cols; ++col)
        {
            int pivot = s_pivots[col];
            if (pivot >= 0)
            {
                low_row_to_col[pivot] = pivot;
            }
        }
    }
}

// Streaming matrix computation kernel for large datasets
template <typename Precision>
__global__ __launch_bounds__(REDUCTION_BLOCK_SIZE) void streamingMatrixKernel(
    const double *__restrict__ input_points, double *__restrict__ output_distances,
    size_t chunk_size, size_t point_dim, double max_radius_sq)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = chunk_size * chunk_size;
    if (i >= total)
    {
        return;
    }

    const size_t row = i / chunk_size;
    const size_t col = i % chunk_size;

    if (row == col)
    {
        output_distances[i] = 0.0;
        return;
    }
    if (row > col)
    {
        return;
    }

    double dist_sq = 0.0;
    bool valid_distance = true;
    for (size_t k = 0; k < point_dim; ++k)
    {
        const double diff = input_points[matrixIndex(row, k, point_dim)] -
                            input_points[matrixIndex(col, k, point_dim)];
        if (!accumulateSquaredDiff(diff, dist_sq))
        {
            valid_distance = false;
            break;
        }
    }

    const double distance =
        (valid_distance && dist_sq <= max_radius_sq) ? std::sqrt(dist_sq) : deviceInfinityDouble();
    output_distances[matrixIndex(row, col, chunk_size)] = distance;
    output_distances[matrixIndex(col, row, chunk_size)] = distance;
}

// Performance monitoring kernel
template <typename Precision>
__global__ __launch_bounds__(REDUCTION_BLOCK_SIZE) void performanceMonitoringKernel(
    const double *__restrict__ timing_data, size_t data_size,
    double *__restrict__ performance_metrics, size_t chunk_index)
{
    const size_t start_idx = chunk_index * 256;
    const size_t end_idx = ((start_idx + 256) < data_size) ? (start_idx + 256) : data_size;

    double local_sum = 0.0;
    for (size_t i = start_idx; i < end_idx; ++i)
    {
        if (i < data_size)
        {
            local_sum += timing_data[i];
        }
    }

    if (chunk_index == 0)
    {
        *performance_metrics = local_sum / data_size;
    }
}

// Kernel launch wrappers
template <typename Precision>
cudaError_t launchParallelColumnReduction(int *low_row_to_col, int *col_pivot, bool *clear_cols,
                                          int *matrix_data, int *row_starts, int n_rows, int n_cols,
                                          cudaStream_t stream = nullptr)
{
    if (low_row_to_col == nullptr || col_pivot == nullptr || clear_cols == nullptr ||
        matrix_data == nullptr || row_starts == nullptr || n_rows <= 0 || n_cols <= 0)
    {
        return cudaErrorInvalidValue;
    }

    const int block_size = REDUCTION_BLOCK_SIZE;
    const int grid_size = (n_cols + block_size - 1) / block_size;

    parallelColumnReductionKernel<Precision><<<grid_size, block_size, 0, stream>>>(
        low_row_to_col, col_pivot, clear_cols, matrix_data, row_starts, n_rows, n_cols);
    return cudaGetLastError();
}

template <typename Precision>
cudaError_t launchStreamingMatrix(const double *points, double *distances, size_t num_points,
                                  size_t point_dim, double max_radius, size_t chunk_size,
                                  cudaStream_t stream = nullptr)
{
    if (points == nullptr || distances == nullptr || num_points == 0 || point_dim == 0 ||
        chunk_size == 0 || !std::isfinite(max_radius) || max_radius < 0.0)
    {
        return cudaErrorInvalidValue;
    }
    const double max_radius_sq = max_radius * max_radius;
    if (!std::isfinite(max_radius_sq))
    {
        return cudaErrorInvalidValue;
    }

    const size_t num_chunks = (num_points + chunk_size - 1) / chunk_size;

    for (size_t chunk = 0; chunk < num_chunks; ++chunk)
    {
        const size_t current_chunk_size = std::min(chunk_size, num_points - chunk * chunk_size);
        const size_t grid_size =
            (current_chunk_size * current_chunk_size + REDUCTION_BLOCK_SIZE - 1) /
            REDUCTION_BLOCK_SIZE;
        streamingMatrixKernel<Precision>
            <<<static_cast<unsigned int>(grid_size), REDUCTION_BLOCK_SIZE, 0, stream>>>(
                points + chunk * chunk_size * point_dim,
                distances + chunk * chunk_size * chunk_size, current_chunk_size, point_dim,
                max_radius_sq);
        const cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            return error;
        }
    }
    return cudaSuccess;
}

template cudaError_t launchParallelColumnReduction<float>(int *low_row_to_col, int *col_pivot,
                                                          bool *clear_cols, int *matrix_data,
                                                          int *row_starts, int n_rows, int n_cols,
                                                          cudaStream_t stream);

template cudaError_t launchParallelColumnReduction<double>(int *low_row_to_col, int *col_pivot,
                                                           bool *clear_cols, int *matrix_data,
                                                           int *row_starts, int n_rows, int n_cols,
                                                           cudaStream_t stream);

template cudaError_t launchStreamingMatrix<float>(const double *points, double *distances,
                                                  size_t num_points, size_t point_dim,
                                                  double max_radius, size_t chunk_size,
                                                  cudaStream_t stream);

template cudaError_t launchStreamingMatrix<double>(const double *points, double *distances,
                                                   size_t num_points, size_t point_dim,
                                                   double max_radius, size_t chunk_size,
                                                   cudaStream_t stream);

} // namespace nerve::persistence::adaptive_acceleration::gpu

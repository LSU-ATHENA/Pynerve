
#include <cuda_runtime.h>

namespace nerve::persistence::adaptive_acceleration::gpu
{

// Distance Reduction Kernel Constants
constexpr int DISTANCE_REDUCTION_BLOCK_SIZE = 256; // Threads per block for reduction

template <typename T>
__global__ __launch_bounds__(DISTANCE_REDUCTION_BLOCK_SIZE) void matrixReductionKernelEnhanced(
    int *__restrict__ low_row_to_col, int *__restrict__ col_pivot, bool *__restrict__ clear_cols,
    int *__restrict__ matrix_data, int *__restrict__ row_starts, int n_rows, int n_cols)
{
    const int tid = threadIdx.x;
    const int block_id = blockIdx.x;
    const int blockSize = blockDim.x;
    const int gridSize = gridDim.x;

    for (int col = block_id; col < n_cols; col += gridSize)
    {
        if (tid >= blockSize)
        {
            return;
        }
        if (col >= n_cols || col < 0)
        {
            return;
        }
        if (clear_cols[col])
        {
            return;
        }

        int pivot_row = -1;
        int row_start = row_starts[col];
        int row_end = (col + 1 < n_cols) ? row_starts[col + 1] : n_rows;
        if (row_start < 0 || row_end > n_rows || row_start >= row_end)
        {
            return;
        }

        for (int idx = row_start; idx < row_end; ++idx)
        {
            if (idx >= 0 && idx < n_rows)
            {
                int row = matrix_data[idx * 2];
                if (row >= 0 && row < n_rows && row > pivot_row)
                {
                    pivot_row = row;
                }
            }
        }

        if (pivot_row >= 0)
        {
            int old_pivot = atomicCAS(&col_pivot[col], -1, pivot_row);
            if (old_pivot != -1 && old_pivot != pivot_row)
            {
                return;
            }
            if (pivot_row >= 0 && pivot_row < n_rows)
            {
                atomicExch(&low_row_to_col[pivot_row], col);
            }
        }
    }
}

template <typename T>
cudaError_t launchMatrixReductionKernelEnhanced(int *low_row_to_col, int *col_pivot,
                                                bool *clear_cols, int *matrix_data, int *row_starts,
                                                int n_rows, int n_cols, cudaStream_t stream = 0)
{
    dim3 blockSize(DISTANCE_REDUCTION_BLOCK_SIZE);
    dim3 gridSize((n_cols + blockSize.x - 1) / blockSize.x);

    matrixReductionKernelEnhanced<T><<<gridSize, blockSize, 0, stream>>>(
        low_row_to_col, col_pivot, clear_cols, matrix_data, row_starts, n_rows, n_cols);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        return error;
    }
    return cudaSuccess;
}

template cudaError_t launchMatrixReductionKernelEnhanced<float>(int *low_row_to_col, int *col_pivot,
                                                                bool *clear_cols, int *matrix_data,
                                                                int *row_starts, int n_rows,
                                                                int n_cols, cudaStream_t stream);

template cudaError_t launchMatrixReductionKernelEnhanced<double>(int *low_row_to_col,
                                                                 int *col_pivot, bool *clear_cols,
                                                                 int *matrix_data, int *row_starts,
                                                                 int n_rows, int n_cols,
                                                                 cudaStream_t stream);

} // namespace nerve::persistence::adaptive_acceleration::gpu

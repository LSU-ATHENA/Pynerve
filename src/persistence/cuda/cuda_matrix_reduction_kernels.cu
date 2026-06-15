
#include "nerve/persistence/cuda/cuda_matrix_reduction.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace nerve::persistence::accelerated
{
using namespace gpu_kernels;

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

} // namespace nerve::persistence::accelerated

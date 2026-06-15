#include <cuda_runtime.h>

__global__ void __launch_bounds__(256)
    parallelColumnReductionKernel(const int *__restrict__ column_data,
                                  const int *__restrict__ column_indices,
                                  const int *__restrict__ column_starts, int *__restrict__ pivots,
                                  int *__restrict__ pairings, int n_columns, int max_column_height)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n_columns)
        return;

    int start = column_starts[col];
    int end = (col + 1 < n_columns) ? column_starts[col + 1]
                                    : column_starts[n_columns - 1] + max_column_height;

    int pivot = -1;
    for (int i = start; i < end && column_indices[i] != -1; ++i)
    {
        if (column_data[i] != 0 && column_indices[i] > pivot)
        {
            pivot = column_indices[i];
        }
    }

    pivots[col] = pivot;

    if (pivot >= 0 && pivot < n_columns)
    {
        pairings[pivot] = col;
    }
}

__global__ void __launch_bounds__(256)
    sparseColumnAddScaledKernel(int *__restrict__ col_j_data, int *__restrict__ col_j_indices,
                                int *__restrict__ col_j_starts, const int *__restrict__ col_i_data,
                                const int *__restrict__ col_i_indices,
                                const int *__restrict__ col_i_starts, int scalar, int col_i,
                                int col_j, int max_height)
{
    if (max_height <= 0)
        return;

    const int tid = threadIdx.x;

    int i_start = (col_i_starts != nullptr) ? col_i_starts[col_i] : (col_i * max_height);
    int i_end = i_start + max_height;

    int j_start = (col_j_starts != nullptr) ? col_j_starts[col_j] : (col_j * max_height);
    int j_end = j_start + max_height;

    for (int local = tid; local < (i_end - i_start); local += blockDim.x)
    {
        const int i_row = col_i_indices[i_start + local];
        if (i_row < 0)
        {
            continue;
        }
        const int i_val = col_i_data[i_start + local] * scalar;

        for (int j_idx = j_start; j_idx < j_end; ++j_idx)
        {
            if (col_j_indices[j_idx] == i_row)
            {
                col_j_data[j_idx] = (col_j_data[j_idx] + i_val) % 2;
                break;
            }
        }
    }
}

__global__ void __launch_bounds__(256)
    cochainReductionKernel(const int *__restrict__ coboundary_data,
                           const int *__restrict__ coboundary_indices,
                           const int *__restrict__ coboundary_starts, int *__restrict__ pivots,
                           int *__restrict__ processed, int n_cochains, int max_height)
{
    int cochain = blockIdx.x * blockDim.x + threadIdx.x;
    if (cochain >= n_cochains || processed[cochain])
        return;

    int idx = n_cochains - 1 - cochain;

    int start = coboundary_starts[idx];
    int end = (idx + 1 < n_cochains) ? coboundary_starts[idx + 1]
                                     : coboundary_starts[n_cochains - 1] + max_height;

    int pivot = -1;
    for (int i = start; i < end && coboundary_indices[i] != -1; ++i)
    {
        if (coboundary_data[i] != 0 && (pivot == -1 || coboundary_indices[i] < pivot))
        {
            pivot = coboundary_indices[i];
        }
    }

    if (pivot != -1)
    {
        pivots[pivot] = idx;
        processed[idx] = 1;
    }
}

__global__ void __launch_bounds__(1)
    symmetricDifferenceKernel(const int *__restrict__ col_a_data, int col_a_size,
                              const int *__restrict__ col_b_data, int col_b_size,
                              int *__restrict__ out_result, int *__restrict__ out_count,
                              int max_result_size)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;

    int a_idx = 0;
    int b_idx = 0;
    int out_idx = 0;

    while (a_idx < col_a_size && b_idx < col_b_size && out_idx < max_result_size)
    {
        int a_value = col_a_data[a_idx];
        int b_value = col_b_data[b_idx];
        if (a_value < b_value)
        {
            out_result[out_idx++] = a_value;
            ++a_idx;
        }
        else if (b_value < a_value)
        {
            out_result[out_idx++] = b_value;
            ++b_idx;
        }
        else
        {
            ++a_idx;
            ++b_idx;
        }
    }

    while (a_idx < col_a_size && out_idx < max_result_size)
    {
        out_result[out_idx++] = col_a_data[a_idx++];
    }
    while (b_idx < col_b_size && out_idx < max_result_size)
    {
        out_result[out_idx++] = col_b_data[b_idx++];
    }

    *out_count = out_idx;
}

__global__ void __launch_bounds__(256) detectBirthSimplicesKernel(
    const int *__restrict__ boundary_data, const int *__restrict__ boundary_indices,
    const int *__restrict__ boundary_starts, const int *__restrict__ simplex_dimensions,
    bool *__restrict__ out_cleared, int *__restrict__ out_count, int n_simplices, int max_height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n_simplices)
        return;

    if (simplex_dimensions[idx] == 0)
    {
        out_cleared[idx] = true;
        atomicAdd(out_count, 1);
        return;
    }

    int start = boundary_starts[idx];
    int end = (idx + 1 < n_simplices) ? boundary_starts[idx + 1]
                                      : boundary_starts[n_simplices - 1] + max_height;

    bool has_boundary = false;
    for (int i = start; i < end && boundary_indices[i] != -1; ++i)
    {
        if (boundary_data[i] != 0)
        {
            has_boundary = true;
            break;
        }
    }

    if (!has_boundary)
    {
        out_cleared[idx] = true;
        atomicAdd(out_count, 1);
    }
    else
    {
        out_cleared[idx] = false;
    }
}

__global__ void __launch_bounds__(256)
    detectEmergentPairsKernel(const int *__restrict__ boundary_data,
                              const int *__restrict__ boundary_indices,
                              const int *__restrict__ boundary_starts,
                              bool *__restrict__ out_emergent, int *__restrict__ out_count,
                              int n_simplices, int max_height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n_simplices)
        return;

    int start = boundary_starts[idx];
    int end = (idx + 1 < n_simplices) ? boundary_starts[idx + 1]
                                      : boundary_starts[n_simplices - 1] + max_height;

    bool is_zero = true;
    for (int i = start; i < end && boundary_indices[i] != -1; ++i)
    {
        if (boundary_data[i] != 0)
        {
            is_zero = false;
            break;
        }
    }

    if (is_zero)
    {
        out_emergent[idx] = true;
        atomicAdd(out_count, 1);
    }
    else
    {
        out_emergent[idx] = false;
    }
}

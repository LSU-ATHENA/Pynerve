
#include "nerve/persistence/cuda/cuda_error_handling.hpp"
#include "nerve/persistence/cuda_matrix_reduction_private.cuh"

#include <cuda_runtime.h>

#include <algorithm>

namespace nerve::persistence::accelerated
{
using namespace gpu_kernels;

errors::ErrorResult<void> computeMatrixReductionGpu(const int *columns, const Size *column_sizes,
                                                    const double *weights, Size n_columns,
                                                    Size max_dim,
                                                    const MatrixReductionConfig &config)
{
    (void)config.use_memory_manager;

    if (!columns || !column_sizes || !weights)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E51_PH_INPUT,
                                                "Null pointer arguments");
    }

    auto validation = utils::validate_matrix_reduction_params(n_columns, max_dim, config);
    if (validation.isError())
    {
        return validation;
    }

    Size col_storage = 0;
    Size host_col_ints = 0;
    Size column_bytes = 0;
    if (!detail::checkedSizeProduct(n_columns, MAX_DIM, col_storage) ||
        !detail::checkedSizeProduct(n_columns, max_dim, host_col_ints) ||
        !detail::checkedSizeProduct(col_storage, sizeof(int), column_bytes))
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E41_RESOURCE_LIMIT,
                                                "Matrix reduction column storage overflows");
    }
    const Size copy_cols = std::min(col_storage, host_col_ints);

    int *d_columns = nullptr;
    Size *d_column_sizes = nullptr;
    double *d_weights = nullptr;
    int *d_low_row_to_col = nullptr;
    int *d_col_pivot = nullptr;

    cudaError_t st = cudaMalloc(&d_columns, column_bytes);
    if (st != cudaSuccess)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL,
                                                "cudaMalloc columns");
    }

    Size column_size_bytes = 0;
    Size weight_bytes = 0;
    Size pivot_bytes = 0;
    if (!detail::checkedSizeProduct(n_columns, sizeof(Size), column_size_bytes) ||
        !detail::checkedSizeProduct(n_columns, sizeof(double), weight_bytes) ||
        !detail::checkedSizeProduct(n_columns, sizeof(int), pivot_bytes))
    {
        cudaFree(d_columns);
        return errors::ErrorResult<void>::error(errors::ErrorCode::E41_RESOURCE_LIMIT,
                                                "Matrix reduction vector storage overflows");
    }

    st = cudaMalloc(&d_column_sizes, column_size_bytes);
    if (st != cudaSuccess)
    {
        cudaFree(d_columns);
        return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL,
                                                "cudaMalloc sizes");
    }

    st = cudaMalloc(&d_weights, weight_bytes);
    if (st != cudaSuccess)
    {
        cudaFree(d_columns);
        cudaFree(d_column_sizes);
        return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL,
                                                "cudaMalloc weights");
    }

    Size row_slots = 0;
    Size row_bytes = 0;
    if (!detail::checkedSizeProduct(n_columns, std::min(max_dim, MAX_DIM), row_slots) ||
        !detail::checkedSizeProduct(row_slots, sizeof(int), row_bytes))
    {
        cudaFree(d_columns);
        cudaFree(d_column_sizes);
        cudaFree(d_weights);
        return errors::ErrorResult<void>::error(errors::ErrorCode::E41_RESOURCE_LIMIT,
                                                "Matrix reduction row storage overflows");
    }
    st = cudaMalloc(&d_low_row_to_col, row_bytes);
    if (st != cudaSuccess)
    {
        cudaFree(d_columns);
        cudaFree(d_column_sizes);
        cudaFree(d_weights);
        return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL,
                                                "cudaMalloc low_row");
    }

    st = cudaMalloc(&d_col_pivot, pivot_bytes);
    if (st != cudaSuccess)
    {
        cudaFree(d_columns);
        cudaFree(d_column_sizes);
        cudaFree(d_weights);
        cudaFree(d_low_row_to_col);
        return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL,
                                                "cudaMalloc pivot");
    }

    Size copy_bytes = 0;
    if (!detail::checkedSizeProduct(copy_cols, sizeof(int), copy_bytes))
    {
        st = cudaErrorInvalidValue;
    }
    else
    {
        st = cudaMemcpy(d_columns, columns, copy_bytes, cudaMemcpyHostToDevice);
    }
    if (st == cudaSuccess && copy_cols < col_storage)
    {
        Size clear_bytes = 0;
        if (!detail::checkedSizeProduct(col_storage - copy_cols, sizeof(int), clear_bytes))
        {
            st = cudaErrorInvalidValue;
        }
        else
        {
            st = cudaMemset(d_columns + copy_cols, 0, clear_bytes);
        }
    }
    if (st == cudaSuccess)
    {
        st = cudaMemcpy(d_column_sizes, column_sizes, column_size_bytes, cudaMemcpyHostToDevice);
    }
    if (st == cudaSuccess)
    {
        st = cudaMemcpy(d_weights, weights, weight_bytes, cudaMemcpyHostToDevice);
    }
    if (st == cudaSuccess)
    {
        st = cudaMemset(d_low_row_to_col, 0xFF, row_bytes);
    }
    if (st == cudaSuccess)
    {
        st = cudaMemset(d_col_pivot, 0xFF, pivot_bytes);
    }

    if (st != cudaSuccess)
    {
        cudaFree(d_columns);
        cudaFree(d_column_sizes);
        cudaFree(d_weights);
        cudaFree(d_low_row_to_col);
        cudaFree(d_col_pivot);
        return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL,
                                                "cuda HtoD setup");
    }

    const errors::ErrorResult<void> launched =
        launchMatrixReductionKernel(d_columns, d_column_sizes, d_weights, d_low_row_to_col,
                                    d_col_pivot, n_columns, max_dim, config);

    cudaFree(d_columns);
    cudaFree(d_column_sizes);
    cudaFree(d_weights);
    cudaFree(d_low_row_to_col);
    cudaFree(d_col_pivot);

    if (launched.isError())
    {
        return launched;
    }

    return errors::ErrorResult<void>::ok();
}

} // namespace nerve::persistence::accelerated


#include "nerve/persistence/cuda/cuda_error_handling.hpp"
#include "nerve/persistence/cuda_matrix_reduction_private.cuh"

#include <algorithm>

namespace nerve::persistence::accelerated
{
using namespace gpu_kernels;

errors::ErrorResult<void>
launchMatrixReductionKernel(const int *d_columns, const Size *d_column_sizes,
                            const double *d_weights, int *d_low_row_to_col, int *d_col_pivot,
                            Size n_columns, Size max_dim, const MatrixReductionConfig &config)
{
    if (!d_columns || !d_column_sizes || !d_weights || !d_low_row_to_col || !d_col_pivot)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E51_PH_INPUT,
                                                "Null pointer arguments");
    }

    if (n_columns == 0 || max_dim == 0)
    {
        return errors::ErrorResult<void>::ok();
    }

    auto validation = utils::validate_matrix_reduction_params(n_columns, max_dim, config);
    if (validation.isError())
    {
        return validation;
    }

    const bool use_clearing = config.enable_clearing;
    Size total_elements = 0;
    if (!detail::checkedSizeProduct(n_columns, max_dim, total_elements))
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E41_RESOURCE_LIMIT,
                                                "Matrix reduction launch size overflows");
    }
    const bool use_streaming =
        config.enable_streaming && total_elements > STREAMING_THRESHOLD;

    unsigned tpb =
        static_cast<unsigned>(std::min<Size>(utils::get_optimal_block_size(n_columns, 1024), 1024));
    if (tpb == 0)
    {
        tpb = 256;
    }
    const dim3 blockDim(tpb, 1, 1);
    const dim3 gridDim(static_cast<unsigned>(utils::get_optimal_grid_size(n_columns, tpb)), 1, 1);

    if (use_streaming)
    {
        const Size chunk_size =
            std::max<Size>(1, std::min(config.streaming_chunk_size, total_elements));
        const Size num_chunks =
            std::max<Size>(1, utils::get_optimal_grid_size(total_elements, chunk_size));
        Size processed_columns = 0;

        for (Size chunk = 0; chunk < num_chunks; ++chunk)
        {
            const Size chunk_offset = chunk * chunk_size;
            const Size chunk_cols = std::min(
                chunk_size, n_columns > processed_columns ? n_columns - processed_columns : 0);
            if (chunk_cols == 0)
            {
                break;
            }

            matrixReductionStreamingKernel<<<gridDim, blockDim>>>(
                d_columns, d_column_sizes, d_weights, d_low_row_to_col, d_col_pivot, n_columns,
                max_dim, use_clearing, chunk_cols, chunk_offset);

            const cudaError_t launch_err = cudaGetLastError();
            if (launch_err != cudaSuccess)
            {
                return cuda_error_handling::validateKernelLaunch(
                    "matrix_reduction_streaming_kernel", std::source_location::current());
            }
            processed_columns += chunk_cols;
        }
    }
    else
    {
        matrixReductionAcceleratedKernel<<<gridDim, blockDim>>>(
            d_columns, d_column_sizes, d_weights, d_low_row_to_col, d_col_pivot, n_columns, max_dim,
            use_clearing);

        const cudaError_t launch_err = cudaGetLastError();
        if (launch_err != cudaSuccess)
        {
            return cuda_error_handling::validateKernelLaunch("matrix_reduction_accelerated_kernel",
                                                             std::source_location::current());
        }
    }

    const cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess)
    {
        return cuda_error_handling::check_cuda_operation(sync_err, "cudaDeviceSynchronize",
                                                         std::source_location::current());
    }

    return errors::ErrorResult<void>::ok();
}

} // namespace nerve::persistence::accelerated

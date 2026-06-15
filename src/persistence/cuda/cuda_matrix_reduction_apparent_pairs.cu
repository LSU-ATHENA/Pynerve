
#include "nerve/persistence/cuda/cuda_error_handling.hpp"
#include "nerve/persistence/cuda_matrix_reduction_private.cuh"

#include <algorithm>
#include <vector>

namespace nerve::persistence::accelerated
{
using namespace gpu_kernels;

errors::ErrorResult<std::vector<int>>
computeApparentPairsGpu(const int *low_row_to_col, const int *col_pivot, const double *weights,
                        Size n_columns, Size max_dim, const ApparentPairsConfig &config)
{
    if (!low_row_to_col || !col_pivot || !weights)
    {
        return errors::ErrorResult<std::vector<int>>::error(errors::ErrorCode::E51_PH_INPUT,
                                                            "Null pointer arguments");
    }
    auto config_status = config.validate();
    if (config_status.isError())
    {
        return errors::ErrorResult<std::vector<int>>::error(config_status.errorCode());
    }

    if (n_columns == 0 || max_dim == 0)
    {
        return errors::ErrorResult<std::vector<int>>::ok(std::vector<int>());
    }
    if (max_dim > MAX_DIM)
    {
        return errors::ErrorResult<std::vector<int>>::error(errors::ErrorCode::E41_RESOURCE_LIMIT,
                                                            "max_dim exceeds CUDA storage");
    }

    Size max_possible_pairs = 0;
    if (!detail::checkedSizeProduct(n_columns, max_dim, max_possible_pairs))
    {
        return errors::ErrorResult<std::vector<int>>::error(errors::ErrorCode::E41_RESOURCE_LIMIT,
                                                            "Pair count overflows");
    }
    const Size max_pairs = std::min(config.max_pairs, max_possible_pairs);

    Size low_row_slots = 0;
    Size low_row_bytes = 0;
    Size pivot_bytes = 0;
    Size weight_bytes = 0;
    Size pair_value_bytes = 0;
    if (!detail::checkedSizeProduct(n_columns, std::min(max_dim, MAX_DIM), low_row_slots) ||
        !detail::checkedSizeProduct(low_row_slots, sizeof(int), low_row_bytes) ||
        !detail::checkedSizeProduct(n_columns, sizeof(int), pivot_bytes) ||
        !detail::checkedSizeProduct(n_columns, sizeof(double), weight_bytes) ||
        !detail::checkedSizeProduct(max_pairs, sizeof(int), pair_value_bytes))
    {
        return errors::ErrorResult<std::vector<int>>::error(errors::ErrorCode::E41_RESOURCE_LIMIT,
                                                            "Apparent-pair storage overflows");
    }

    int *d_low = nullptr;
    int *d_pivot = nullptr;
    double *d_weights = nullptr;

    cudaError_t st = cudaMalloc(&d_low, low_row_bytes);
    if (st != cudaSuccess)
    {
        return errors::ErrorResult<std::vector<int>>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL,
                                                            "cudaMalloc d_low");
    }
    st = cudaMalloc(&d_pivot, pivot_bytes);
    if (st != cudaSuccess)
    {
        cudaFree(d_low);
        return errors::ErrorResult<std::vector<int>>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL,
                                                            "cudaMalloc d_pivot");
    }
    st = cudaMalloc(&d_weights, weight_bytes);
    if (st != cudaSuccess)
    {
        cudaFree(d_low);
        cudaFree(d_pivot);
        return errors::ErrorResult<std::vector<int>>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL,
                                                            "cudaMalloc d_weights");
    }

    st = cudaMemcpy(d_low, low_row_to_col, low_row_bytes, cudaMemcpyHostToDevice);
    if (st != cudaSuccess)
    {
        cudaFree(d_low);
        cudaFree(d_pivot);
        cudaFree(d_weights);
        return errors::ErrorResult<std::vector<int>>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL,
                                                            "cudaMemcpy d_low");
    }
    st = cudaMemcpy(d_pivot, col_pivot, pivot_bytes, cudaMemcpyHostToDevice);
    if (st != cudaSuccess)
    {
        cudaFree(d_low);
        cudaFree(d_pivot);
        cudaFree(d_weights);
        return errors::ErrorResult<std::vector<int>>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL,
                                                            "cudaMemcpy d_pivot");
    }
    st = cudaMemcpy(d_weights, weights, weight_bytes, cudaMemcpyHostToDevice);
    if (st != cudaSuccess)
    {
        cudaFree(d_low);
        cudaFree(d_pivot);
        cudaFree(d_weights);
        return errors::ErrorResult<std::vector<int>>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL,
                                                            "cudaMemcpy d_weights");
    }

    int *d_pair_count = nullptr;
    int *d_pair_values = nullptr;

    st = cudaMalloc(&d_pair_count, sizeof(int));
    if (st != cudaSuccess)
    {
        cudaFree(d_low);
        cudaFree(d_pivot);
        cudaFree(d_weights);
        return errors::ErrorResult<std::vector<int>>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL,
                                                            "cudaMalloc pair_count");
    }

    st = cudaMalloc(&d_pair_values, pair_value_bytes);
    if (st != cudaSuccess)
    {
        cudaFree(d_low);
        cudaFree(d_pivot);
        cudaFree(d_weights);
        cudaFree(d_pair_count);
        return errors::ErrorResult<std::vector<int>>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL,
                                                            "cudaMalloc pair_values");
    }

    st = cudaMemset(d_pair_count, 0, sizeof(int));
    if (st != cudaSuccess)
    {
        cudaFree(d_low);
        cudaFree(d_pivot);
        cudaFree(d_weights);
        cudaFree(d_pair_count);
        cudaFree(d_pair_values);
        return errors::ErrorResult<std::vector<int>>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL,
                                                            "cudaMemset pair_count");
    }

    st = cudaMemset(d_pair_values, 0, pair_value_bytes);
    if (st != cudaSuccess)
    {
        cudaFree(d_low);
        cudaFree(d_pivot);
        cudaFree(d_weights);
        cudaFree(d_pair_count);
        cudaFree(d_pair_values);
        return errors::ErrorResult<std::vector<int>>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL,
                                                            "cudaMemset pair_values");
    }

    const unsigned tpb =
        static_cast<unsigned>(std::min<Size>(utils::get_optimal_block_size(n_columns, 1024), 1024));
    const unsigned threads = tpb > 0 ? tpb : 256U;
    const dim3 blockDim(threads, 1, 1);
    const dim3 gridDim(static_cast<unsigned>(utils::get_optimal_grid_size(n_columns, threads)), 1,
                       1);

    computeApparentPairsKernel<<<gridDim, blockDim>>>(d_low, d_pivot, d_weights, n_columns,
                                                      d_pair_count, d_pair_values, max_pairs,
                                                      config.use_optimization);

    st = cudaGetLastError();
    if (st != cudaSuccess)
    {
        cudaFree(d_low);
        cudaFree(d_pivot);
        cudaFree(d_weights);
        cudaFree(d_pair_count);
        cudaFree(d_pair_values);
        return errors::ErrorResult<std::vector<int>>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL,
                                                            "kernel launch");
    }

    st = cudaDeviceSynchronize();
    if (st != cudaSuccess)
    {
        cudaFree(d_low);
        cudaFree(d_pivot);
        cudaFree(d_weights);
        cudaFree(d_pair_count);
        cudaFree(d_pair_values);
        return errors::ErrorResult<std::vector<int>>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL,
                                                            "cudaDeviceSynchronize");
    }

    int host_count = 0;
    st = cudaMemcpy(&host_count, d_pair_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_pair_count);
    d_pair_count = nullptr;
    cudaFree(d_low);
    cudaFree(d_pivot);
    cudaFree(d_weights);

    if (st != cudaSuccess)
    {
        cudaFree(d_pair_values);
        return errors::ErrorResult<std::vector<int>>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL,
                                                            "cudaMemcpy count");
    }

    if (host_count < 0)
    {
        host_count = 0;
    }
    const Size out_n =
        static_cast<Size>(host_count) < max_pairs ? static_cast<Size>(host_count) : max_pairs;

    std::vector<int> hostValues(out_n);
    if (out_n > 0)
    {
        Size out_bytes = 0;
        if (!detail::checkedSizeProduct(out_n, sizeof(int), out_bytes))
        {
            st = cudaErrorInvalidValue;
        }
        else
        {
            st = cudaMemcpy(hostValues.data(), d_pair_values, out_bytes, cudaMemcpyDeviceToHost);
        }
    }
    cudaFree(d_pair_values);

    if (st != cudaSuccess)
    {
        return errors::ErrorResult<std::vector<int>>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL,
                                                            "cudaMemcpy values");
    }

    return errors::ErrorResult<std::vector<int>>::ok(std::move(hostValues));
}

} // namespace nerve::persistence::accelerated

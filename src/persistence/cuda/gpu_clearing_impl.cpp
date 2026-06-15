
#include "nerve/algebra/boundary.hpp"
#include "nerve/errors/errors.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace nerve
{
namespace gpu
{
namespace persistence
{
namespace detail
{

// Forward declarations from cuda_clearing.cu
extern void launchDetectPositiveSimplices(const int *d_boundary_data, const int *d_boundary_indices,
                                          const int *d_boundary_starts, const int *d_dimensions,
                                          int *d_is_positive, int n_simplices, cudaStream_t stream);

extern void launchMarkClearingColumns(const int *d_is_positive, const int *d_dimensions,
                                      int *d_clear_column, int target_dimension, int n_simplices,
                                      cudaStream_t stream);

extern void launchApplyClearing(int *d_boundary_data, int *d_boundary_indices,
                                const int *d_clear_column, const int *d_starts, int n_simplices,
                                cudaStream_t stream);

extern void launchClearingPairing(const int *d_boundary_indices, const int *d_starts,
                                  const int *d_dimensions, const int *d_is_negative,
                                  int *d_birth_simplex, int n_simplices, cudaStream_t stream);

namespace
{

bool checkedBytes(std::size_t count, std::size_t element_size, std::size_t &bytes)
{
    if (count != 0 && element_size > std::numeric_limits<std::size_t>::max() / count)
    {
        return false;
    }
    bytes = count * element_size;
    return true;
}

} // namespace

// GPU-accelerated clearing optimization engine
class GPUClearingEngine
{
public:
    struct ClearingResult
    {
        std::vector<bool> positive_simplices;
        std::vector<bool> negative_simplices;
        std::vector<bool> columns_to_clear;
        std::vector<int> birth_deaths_pairs;
        size_t operations_saved;
    };

    static errors::ErrorResult<void>
    applyClearingOptimization(const ::nerve::algebra::BoundaryMatrix &boundary_matrix,
                              const std::vector<int> &simplex_dimensions,
                              const std::vector<double> &filtration_values, int target_dimension,
                              double max_filtration, ClearingResult &out_result)
    {
        if (simplex_dimensions.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E41_RESOURCE_LIMIT);
        }
        int n_simplices = static_cast<int>(simplex_dimensions.size());
        if (n_simplices == 0)
        {
            return errors::ErrorResult<void>::success();
        }
        const bool has_filtration = !filtration_values.empty();
        if (has_filtration && filtration_values.size() != simplex_dimensions.size())
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E51_PH_INPUT);
        }
        if (!std::isfinite(max_filtration))
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E20_NUM_NAN);
        }
        if (has_filtration)
        {
            for (double value : filtration_values)
            {
                if (!std::isfinite(value))
                {
                    return errors::ErrorResult<void>::error(errors::ErrorCode::E20_NUM_NAN);
                }
            }
        }

        std::vector<int> h_boundary_data;
        std::vector<int> h_boundary_indices;
        std::vector<int> hBoundaryStarts(static_cast<std::size_t>(n_simplices) + 1);

        int current_pos = 0;
        for (int i = 0; i < n_simplices; ++i)
        {
            hBoundaryStarts[i] = current_pos;

            std::vector<int> column_entries;
            for (int row = 0; row < n_simplices; ++row)
            {
                if (boundary_matrix.getMatrixEntry(row, i) != 0.0)
                {
                    column_entries.push_back(row);
                }
            }

            // Add to flat arrays
            for (int entry : column_entries)
            {
                if (current_pos == std::numeric_limits<int>::max())
                {
                    return errors::ErrorResult<void>::error(errors::ErrorCode::E41_RESOURCE_LIMIT);
                }
                h_boundary_data.push_back(1); // Z2 coefficient
                h_boundary_indices.push_back(entry);
                current_pos++;
            }
        }
        hBoundaryStarts[n_simplices] = current_pos;
        const std::size_t boundary_entry_count = h_boundary_data.size();
        if (h_boundary_data.empty())
        {
            h_boundary_data.push_back(0);
            h_boundary_indices.push_back(-1);
        }

        int *d_boundary_data = nullptr;
        int *d_boundary_indices = nullptr;
        int *d_boundary_starts = nullptr;
        int *d_dimensions = nullptr;
        int *d_is_positive = nullptr;
        int *d_clear_column = nullptr;

        cudaError_t err;
        std::size_t boundary_data_bytes = 0;
        std::size_t boundary_indices_bytes = 0;
        std::size_t starts_bytes = 0;
        std::size_t simplex_bytes = 0;
        if (!checkedBytes(h_boundary_data.size(), sizeof(int), boundary_data_bytes) ||
            !checkedBytes(h_boundary_indices.size(), sizeof(int), boundary_indices_bytes) ||
            !checkedBytes(hBoundaryStarts.size(), sizeof(int), starts_bytes) ||
            !checkedBytes(static_cast<std::size_t>(n_simplices), sizeof(int), simplex_bytes))
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E41_RESOURCE_LIMIT);
        }

        err = cudaMalloc(reinterpret_cast<void **>(&d_boundary_data), boundary_data_bytes);
        if (err != cudaSuccess)
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E10_GPU_OOM);
        }

        err = cudaMemcpy(d_boundary_data, h_boundary_data.data(), boundary_data_bytes,
                         cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            cleanup(d_boundary_data, d_boundary_indices, d_boundary_starts, d_dimensions,
                    d_is_positive, d_clear_column);
            return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }

        err = cudaMalloc(reinterpret_cast<void **>(&d_boundary_indices), boundary_indices_bytes);
        if (err != cudaSuccess)
        {
            cleanup(d_boundary_data, d_boundary_indices, d_boundary_starts, d_dimensions,
                    d_is_positive, d_clear_column);
            return errors::ErrorResult<void>::error(errors::ErrorCode::E10_GPU_OOM);
        }

        err = cudaMemcpy(d_boundary_indices, h_boundary_indices.data(), boundary_indices_bytes,
                         cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            cleanup(d_boundary_data, d_boundary_indices, d_boundary_starts, d_dimensions,
                    d_is_positive, d_clear_column);
            return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }

        err = cudaMalloc(reinterpret_cast<void **>(&d_boundary_starts), starts_bytes);
        if (err != cudaSuccess)
        {
            cleanup(d_boundary_data, d_boundary_indices, d_boundary_starts, d_dimensions,
                    d_is_positive, d_clear_column);
            return errors::ErrorResult<void>::error(errors::ErrorCode::E10_GPU_OOM);
        }

        err = cudaMemcpy(d_boundary_starts, hBoundaryStarts.data(), starts_bytes,
                         cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            cleanup(d_boundary_data, d_boundary_indices, d_boundary_starts, d_dimensions,
                    d_is_positive, d_clear_column);
            return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }

        err = cudaMalloc(reinterpret_cast<void **>(&d_dimensions), simplex_bytes);
        if (err != cudaSuccess)
        {
            cleanup(d_boundary_data, d_boundary_indices, d_boundary_starts, d_dimensions,
                    d_is_positive, d_clear_column);
            return errors::ErrorResult<void>::error(errors::ErrorCode::E10_GPU_OOM);
        }

        err = cudaMemcpy(d_dimensions, simplex_dimensions.data(), simplex_bytes,
                         cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            cleanup(d_boundary_data, d_boundary_indices, d_boundary_starts, d_dimensions,
                    d_is_positive, d_clear_column);
            return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }

        err = cudaMalloc(reinterpret_cast<void **>(&d_is_positive), simplex_bytes);
        if (err != cudaSuccess)
        {
            cleanup(d_boundary_data, d_boundary_indices, d_boundary_starts, d_dimensions,
                    d_is_positive, d_clear_column);
            return errors::ErrorResult<void>::error(errors::ErrorCode::E10_GPU_OOM);
        }

        err = cudaMalloc(reinterpret_cast<void **>(&d_clear_column), simplex_bytes);
        if (err != cudaSuccess)
        {
            cleanup(d_boundary_data, d_boundary_indices, d_boundary_starts, d_dimensions,
                    d_is_positive, d_clear_column);
            return errors::ErrorResult<void>::error(errors::ErrorCode::E10_GPU_OOM);
        }

        launchDetectPositiveSimplices(d_boundary_data, d_boundary_indices, d_boundary_starts,
                                      d_dimensions, d_is_positive, n_simplices, nullptr);

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            cleanup(d_boundary_data, d_boundary_indices, d_boundary_starts, d_dimensions,
                    d_is_positive, d_clear_column);
            return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
        {
            cleanup(d_boundary_data, d_boundary_indices, d_boundary_starts, d_dimensions,
                    d_is_positive, d_clear_column);
            return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }

        launchMarkClearingColumns(d_is_positive, d_dimensions, d_clear_column, target_dimension,
                                  n_simplices, nullptr);

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            cleanup(d_boundary_data, d_boundary_indices, d_boundary_starts, d_dimensions,
                    d_is_positive, d_clear_column);
            return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
        {
            cleanup(d_boundary_data, d_boundary_indices, d_boundary_starts, d_dimensions,
                    d_is_positive, d_clear_column);
            return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }

        std::vector<int> hIsPositive(n_simplices);
        std::vector<int> hClearColumn(n_simplices);

        err = cudaMemcpy(hIsPositive.data(), d_is_positive, simplex_bytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            cleanup(d_boundary_data, d_boundary_indices, d_boundary_starts, d_dimensions,
                    d_is_positive, d_clear_column);
            return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }

        err =
            cudaMemcpy(hClearColumn.data(), d_clear_column, simplex_bytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            cleanup(d_boundary_data, d_boundary_indices, d_boundary_starts, d_dimensions,
                    d_is_positive, d_clear_column);
            return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }

        out_result.positive_simplices.resize(n_simplices);
        out_result.columns_to_clear.resize(n_simplices);
        out_result.operations_saved = 0;

        for (int i = 0; i < n_simplices; ++i)
        {
            const bool within_filtration =
                !has_filtration || filtration_values[static_cast<std::size_t>(i)] <= max_filtration;
            out_result.positive_simplices[i] = within_filtration && (hIsPositive[i] != 0);
            out_result.columns_to_clear[i] = within_filtration && (hClearColumn[i] != 0);
            if (out_result.columns_to_clear[i])
            {
                int col_size = (i + 1 <= n_simplices)
                                   ? hBoundaryStarts[i + 1] - hBoundaryStarts[i]
                                   : static_cast<int>(boundary_entry_count) - hBoundaryStarts[i];
                const std::size_t col = static_cast<std::size_t>(std::max(0, col_size));
                const std::size_t saved =
                    col <= std::numeric_limits<std::size_t>::max() / std::max<std::size_t>(1, col)
                        ? col * col
                        : std::numeric_limits<std::size_t>::max();
                out_result.operations_saved =
                    saved > std::numeric_limits<std::size_t>::max() - out_result.operations_saved
                        ? std::numeric_limits<std::size_t>::max()
                        : out_result.operations_saved + saved;
            }
        }

        cleanup(d_boundary_data, d_boundary_indices, d_boundary_starts, d_dimensions, d_is_positive,
                d_clear_column);

        return errors::ErrorResult<void>::success();
    }

    static size_t estimateSavingsCpu(const std::vector<int> &simplex_dimensions,
                                     const ::nerve::algebra::BoundaryMatrix &boundary_matrix)
    {
        size_t savings = 0;
        if (simplex_dimensions.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        {
            return std::numeric_limits<std::size_t>::max();
        }
        int n_simplices = static_cast<int>(simplex_dimensions.size());

        for (int i = 0; i < n_simplices; ++i)
        {
            if (simplex_dimensions[i] == 0)
            {
                savings =
                    savings == std::numeric_limits<std::size_t>::max() ? savings : savings + 1;
                continue;
            }

            bool has_boundary = false;
            for (int row = 0; row < n_simplices; ++row)
            {
                if (boundary_matrix.getMatrixEntry(row, i) != 0.0)
                {
                    has_boundary = true;
                    break;
                }
            }

            if (!has_boundary)
            {
                int col_size = 0;
                for (int row = 0; row < n_simplices; ++row)
                {
                    if (boundary_matrix.getMatrixEntry(row, i) != 0.0)
                    {
                        col_size++;
                    }
                }
                const std::size_t col = static_cast<std::size_t>(col_size);
                const std::size_t saved =
                    col <= std::numeric_limits<std::size_t>::max() / std::max<std::size_t>(1, col)
                        ? col * col
                        : std::numeric_limits<std::size_t>::max();
                savings = saved > std::numeric_limits<std::size_t>::max() - savings
                              ? std::numeric_limits<std::size_t>::max()
                              : savings + saved;
            }
        }

        return savings;
    }

private:
    static void cleanup(int *d_boundary_data, int *d_boundary_indices, int *d_boundary_starts,
                        int *d_dimensions, int *d_is_positive, int *d_clear_column)
    {
        if (d_boundary_data)
            cudaFree(d_boundary_data);
        if (d_boundary_indices)
            cudaFree(d_boundary_indices);
        if (d_boundary_starts)
            cudaFree(d_boundary_starts);
        if (d_dimensions)
            cudaFree(d_dimensions);
        if (d_is_positive)
            cudaFree(d_is_positive);
        if (d_clear_column)
            cudaFree(d_clear_column);
    }
};

} // namespace detail
} // namespace persistence
} // namespace gpu
} // namespace nerve

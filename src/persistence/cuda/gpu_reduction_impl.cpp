#include "nerve/algebra/boundary.hpp"
#include "nerve/errors/detail/error_result.hpp"
#include "nerve/persistence/reduction/reducer.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <limits>
#include <string_view>
#include <vector>

namespace nerve
{
namespace gpu
{

namespace detail
{
extern void launchParallelColumnReduction(const int *column_data, const int *column_indices,
                                          const int *column_starts, int *pivots, int *pairings,
                                          int n_columns, int max_column_height,
                                          cudaStream_t stream);

extern void launchSparseColumnAdd(int *col_j_data, int *col_j_indices, const int *col_i_data,
                                  const int *col_i_indices, int col_i, int col_j, int max_height,
                                  cudaStream_t stream);

extern void launchSymmetricDifference(const int *col_a_data, int col_a_size, const int *col_b_data,
                                      int col_b_size, int *out_result, int *out_count,
                                      int max_result_size, cudaStream_t stream);
} // namespace detail

namespace
{

bool checkedProduct(std::size_t lhs, std::size_t rhs, std::size_t &out) noexcept
{
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs)
    {
        return false;
    }
    out = lhs * rhs;
    return true;
}

bool checkedByteCount(std::size_t count, std::size_t element_size, std::size_t &out) noexcept
{
    return checkedProduct(count, element_size, out);
}

bool checkedIntSize(std::size_t value, int &out) noexcept
{
    if (value > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        return false;
    }
    out = static_cast<int>(value);
    return true;
}

errors::ErrorResult<void> resourceLimit(std::string_view message)
{
    return errors::ErrorResult<void>::error(errors::ErrorCode::E41_RESOURCE_LIMIT, message);
}

} // namespace

namespace persistence
{

class ReductionEngine
{
public:
    struct SparseMatrixGPU
    {
        int *data = nullptr;
        int *indices = nullptr;
        int *starts = nullptr;
        int n_columns = 0;
        int max_height = 0;
        size_t data_size = 0;
    };

    static errors::ErrorResult<void>
    computeReduction(const algebra::BoundaryMatrix &boundary_matrix, std::vector<Index> &out_pivots,
                     std::vector<std::pair<Size, Size>> &out_pairs)
    {
        out_pivots.clear();
        out_pairs.clear();
        if (boundary_matrix.cols() == 0)
        {
            return errors::ErrorResult<void>::success();
        }

        SparseMatrixGPU gpu_matrix;
        auto convert_result = convertToGpuFormat(boundary_matrix, gpu_matrix);
        if (convert_result.isError())
        {
            cleanup(gpu_matrix);
            return convert_result;
        }

        int *d_pivots = nullptr;
        int *d_pairings = nullptr;
        auto free_device = [&]() {
            cudaFree(d_pivots);
            cudaFree(d_pairings);
            d_pivots = nullptr;
            d_pairings = nullptr;
        };
        auto fail = [&](errors::ErrorCode code) -> errors::ErrorResult<void> {
            free_device();
            cleanup(gpu_matrix);
            return errors::ErrorResult<void>::error(code);
        };

        std::size_t columns_bytes = 0;
        if (!checkedByteCount(static_cast<std::size_t>(gpu_matrix.n_columns), sizeof(int),
                              columns_bytes))
        {
            cleanup(gpu_matrix);
            return resourceLimit("GPU reduction column buffer size overflows");
        }

        cudaError_t err = cudaMalloc(&d_pivots, columns_bytes);
        if (err != cudaSuccess)
        {
            return fail(errors::ErrorCode::E10_GPU_OOM);
        }
        err = cudaMalloc(&d_pairings, columns_bytes);
        if (err != cudaSuccess)
        {
            return fail(errors::ErrorCode::E10_GPU_OOM);
        }

        std::vector<int> hPivots(static_cast<std::size_t>(gpu_matrix.n_columns), -1);
        std::vector<int> hPairings(static_cast<std::size_t>(gpu_matrix.n_columns), -1);
        err = cudaMemcpy(d_pivots, hPivots.data(), columns_bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            return fail(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }
        err = cudaMemcpy(d_pairings, hPairings.data(), columns_bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            return fail(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }

        detail::launchParallelColumnReduction(gpu_matrix.data, gpu_matrix.indices,
                                              gpu_matrix.starts, d_pivots, d_pairings,
                                              gpu_matrix.n_columns, gpu_matrix.max_height, nullptr);
        err = cudaGetLastError();
        if (err == cudaSuccess)
        {
            err = cudaDeviceSynchronize();
        }
        if (err != cudaSuccess)
        {
            return fail(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }

        err = cudaMemcpy(hPivots.data(), d_pivots, columns_bytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            return fail(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }
        err = cudaMemcpy(hPairings.data(), d_pairings, columns_bytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            return fail(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }

        out_pivots.resize(static_cast<std::size_t>(gpu_matrix.n_columns));
        for (int i = 0; i < gpu_matrix.n_columns; ++i)
        {
            out_pivots[static_cast<std::size_t>(i)] =
                static_cast<Index>(hPivots[static_cast<std::size_t>(i)]);
        }
        for (int i = 0; i < gpu_matrix.n_columns; ++i)
        {
            const int pivot = hPivots[static_cast<std::size_t>(i)];
            if (pivot < -1 || pivot >= gpu_matrix.n_columns)
            {
                return fail(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
            }
            if (pivot != -1 && hPairings[static_cast<std::size_t>(pivot)] == i)
            {
                out_pairs.push_back({static_cast<Size>(pivot), static_cast<Size>(i)});
            }
        }

        free_device();
        cleanup(gpu_matrix);
        return errors::ErrorResult<void>::success();
    }

    static errors::ErrorResult<void> performColumnAddition(const std::vector<int> &col_a,
                                                           const std::vector<int> &col_b,
                                                           std::vector<int> &out_result)
    {
        constexpr size_t GPU_THRESHOLD = 1024;
        out_result.clear();
        if (col_a.size() > std::numeric_limits<std::size_t>::max() - col_b.size())
        {
            return resourceLimit("GPU column addition input size overflows");
        }
        const size_t max_result = col_a.size() + col_b.size();
        if (max_result < GPU_THRESHOLD)
        {
            out_result.reserve(max_result);
            size_t i = 0, j = 0;
            while (i < col_a.size() && j < col_b.size())
            {
                if (col_a[i] == col_b[j])
                {
                    ++i;
                    ++j;
                }
                else if (col_a[i] < col_b[j])
                {
                    out_result.push_back(col_a[i++]);
                }
                else
                {
                    out_result.push_back(col_b[j++]);
                }
            }
            while (i < col_a.size())
            {
                out_result.push_back(col_a[i++]);
            }
            while (j < col_b.size())
            {
                out_result.push_back(col_b[j++]);
            }
            return errors::ErrorResult<void>::success();
        }

        int col_a_size = 0;
        int col_b_size = 0;
        int max_result_size = 0;
        std::size_t col_a_bytes = 0;
        std::size_t col_b_bytes = 0;
        std::size_t result_bytes = 0;
        if (!checkedIntSize(col_a.size(), col_a_size) ||
            !checkedIntSize(col_b.size(), col_b_size) ||
            !checkedIntSize(max_result, max_result_size) ||
            !checkedByteCount(col_a.size(), sizeof(int), col_a_bytes) ||
            !checkedByteCount(col_b.size(), sizeof(int), col_b_bytes) ||
            !checkedByteCount(max_result, sizeof(int), result_bytes))
        {
            return resourceLimit("GPU column addition exceeds CUDA kernel limits");
        }

        cudaError_t err;
        int *d_col_a = nullptr;
        int *d_col_b = nullptr;
        int *d_result = nullptr;
        int *d_result_count = nullptr;
        cudaStream_t stream = nullptr;
        auto free_all = [&]() {
            if (stream)
                cudaStreamDestroy(stream);
            cudaFree(d_col_a);
            cudaFree(d_col_b);
            cudaFree(d_result);
            cudaFree(d_result_count);
            stream = nullptr;
            d_col_a = nullptr;
            d_col_b = nullptr;
            d_result = nullptr;
            d_result_count = nullptr;
        };
        auto fail = [&](errors::ErrorCode code) -> errors::ErrorResult<void> {
            free_all();
            return errors::ErrorResult<void>::error(code);
        };

        err = cudaMalloc(&d_col_a, col_a_bytes);
        if (err != cudaSuccess)
            return fail(errors::ErrorCode::E10_GPU_OOM);
        err = cudaMalloc(&d_col_b, col_b_bytes);
        if (err != cudaSuccess)
            return fail(errors::ErrorCode::E10_GPU_OOM);
        err = cudaMalloc(&d_result, result_bytes);
        if (err != cudaSuccess)
            return fail(errors::ErrorCode::E10_GPU_OOM);
        err = cudaMalloc(&d_result_count, sizeof(int));
        if (err != cudaSuccess)
            return fail(errors::ErrorCode::E10_GPU_OOM);

        err = cudaMemcpy(d_col_a, col_a.data(), col_a_bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
            return fail(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        err = cudaMemcpy(d_col_b, col_b.data(), col_b_bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
            return fail(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        err = cudaMemset(d_result_count, 0, sizeof(int));
        if (err != cudaSuccess)
            return fail(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        err = cudaStreamCreate(&stream);
        if (err != cudaSuccess)
            return fail(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);

        detail::launchSymmetricDifference(d_col_a, col_a_size, d_col_b, col_b_size, d_result,
                                          d_result_count, max_result_size, stream);
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess)
            return fail(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);

        int result_count = 0;
        err = cudaMemcpy(&result_count, d_result_count, sizeof(int), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess || result_count < 0 || result_count > max_result_size)
            return fail(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);

        out_result.resize(static_cast<size_t>(result_count));
        if (result_count > 0)
        {
            std::size_t result_copy_bytes = 0;
            if (!checkedByteCount(static_cast<std::size_t>(result_count), sizeof(int),
                                  result_copy_bytes))
            {
                return fail(errors::ErrorCode::E41_RESOURCE_LIMIT);
            }
            err =
                cudaMemcpy(out_result.data(), d_result, result_copy_bytes, cudaMemcpyDeviceToHost);
            if (err != cudaSuccess)
            {
                return fail(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
            }
        }

        free_all();
        return errors::ErrorResult<void>::success();
    }

private:
    static errors::ErrorResult<void>
    convertToGpuFormat(const algebra::BoundaryMatrix &boundary_matrix, SparseMatrixGPU &out_matrix)
    {
        const Size n_cols = boundary_matrix.cols();
        const Size n_rows = boundary_matrix.rows();

        int n_cols_int = 0;
        int n_rows_int = 0;
        if (!checkedIntSize(static_cast<std::size_t>(n_cols), n_cols_int) ||
            !checkedIntSize(static_cast<std::size_t>(n_rows), n_rows_int))
        {
            return resourceLimit("GPU reduction matrix dimensions exceed int range");
        }

        std::vector<int> colCounts(static_cast<std::size_t>(n_cols), 0);
        int max_height = 0;
        for (Size col = 0; col < n_cols; ++col)
        {
            int count = 0;
            for (Size row = 0; row < n_rows; ++row)
            {
                if (boundary_matrix.getMatrixEntry(row, col) != 0.0)
                {
                    ++count;
                }
            }
            colCounts[static_cast<std::size_t>(col)] = count;
            max_height = std::max(max_height, count);
        }

        size_t total_entries = 0;
        if (!checkedProduct(static_cast<std::size_t>(n_cols), static_cast<std::size_t>(max_height),
                            total_entries))
        {
            return resourceLimit("GPU reduction packed matrix size overflows");
        }
        int total_entries_int = 0;
        if (!checkedIntSize(total_entries, total_entries_int))
        {
            return resourceLimit("GPU reduction packed matrix offsets exceed int range");
        }

        out_matrix.n_columns = n_cols_int;
        out_matrix.max_height = max_height;
        out_matrix.data_size = total_entries;
        if (total_entries == 0)
        {
            return errors::ErrorResult<void>::success();
        }

        std::vector<int> hData(total_entries, 0);
        std::vector<int> hIndices(total_entries, -1);
        std::vector<int> hStarts(static_cast<std::size_t>(n_cols), 0);
        int current_pos = 0;
        for (Size col = 0; col < n_cols; ++col)
        {
            const auto col_index = static_cast<std::size_t>(col);
            hStarts[col_index] = current_pos;
            int row_idx = 0;
            for (Size row = 0; row < n_rows; ++row)
            {
                if (boundary_matrix.getMatrixEntry(row, col) != 0.0)
                {
                    hData[static_cast<std::size_t>(current_pos)] = 1;
                    hIndices[static_cast<std::size_t>(current_pos)] = static_cast<int>(row);
                    ++current_pos;
                    ++row_idx;
                    if (row_idx >= colCounts[col_index])
                        break;
                }
            }
            for (int i = row_idx; i < max_height; ++i)
            {
                hIndices[static_cast<std::size_t>(hStarts[col_index] + i)] = -1;
            }
            const std::size_t next_pos = (col_index + 1U) * static_cast<std::size_t>(max_height);
            current_pos = static_cast<int>(next_pos);
        }

        std::size_t data_bytes = 0;
        std::size_t starts_bytes = 0;
        if (!checkedByteCount(total_entries, sizeof(int), data_bytes) ||
            !checkedByteCount(static_cast<std::size_t>(n_cols), sizeof(int), starts_bytes))
        {
            return resourceLimit("GPU reduction matrix byte count overflows");
        }

        cudaError_t err = cudaMalloc(&out_matrix.data, data_bytes);
        if (err != cudaSuccess)
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E10_GPU_OOM);
        }
        err = cudaMalloc(&out_matrix.indices, data_bytes);
        if (err != cudaSuccess)
        {
            cleanup(out_matrix);
            return errors::ErrorResult<void>::error(errors::ErrorCode::E10_GPU_OOM);
        }
        err = cudaMalloc(&out_matrix.starts, starts_bytes);
        if (err != cudaSuccess)
        {
            cleanup(out_matrix);
            return errors::ErrorResult<void>::error(errors::ErrorCode::E10_GPU_OOM);
        }
        err = cudaMemcpy(out_matrix.data, hData.data(), data_bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            cleanup(out_matrix);
            return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }
        err = cudaMemcpy(out_matrix.indices, hIndices.data(), data_bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            cleanup(out_matrix);
            return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }
        err = cudaMemcpy(out_matrix.starts, hStarts.data(), starts_bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            cleanup(out_matrix);
            return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }
        (void)n_rows_int;
        (void)total_entries_int;
        return errors::ErrorResult<void>::success();
    }

    static void cleanup(SparseMatrixGPU &matrix)
    {
        cudaFree(matrix.data);
        cudaFree(matrix.indices);
        cudaFree(matrix.starts);
        matrix.data = nullptr;
        matrix.indices = nullptr;
        matrix.starts = nullptr;
    }
};

} // namespace persistence
} // namespace gpu
} // namespace nerve

#include "nerve/gpu/device_array.hpp"
#include "nerve/gpu/gpu_error.hpp"
#include "nerve/persistence/reduction/reduction_hypha_ops.hpp"
#include "nerve/persistence/reduction/reduction_lockfree_ops.hpp"
#include "nerve/types.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

namespace nerve::persistence
{

namespace
{

constexpr int kDefaultBlockSize = 256;

int gridSize(int count, int block_size)
{
    if (count <= 0)
        return 0;
    return (count + block_size - 1) / block_size;
}

bool fitsCudaInt(std::size_t val) noexcept
{
    return val <= static_cast<std::size_t>(std::numeric_limits<int>::max());
}

bool checkedBytes(std::size_t count, std::size_t elem_size, std::size_t &out) noexcept
{
    if (count != 0 && elem_size > std::numeric_limits<std::size_t>::max() / count)
        return false;
    out = count * elem_size;
    return true;
}

struct GpuBoundaryMatrix
{
    int *data = nullptr;
    int *indices = nullptr;
    int *starts = nullptr;
    int n_columns = 0;
    int max_height = 0;
};

struct GpuScanState
{
    int *stable_columns = nullptr;
    int *pivot_candidates = nullptr;
    int *zero_addition = nullptr;
    int *low_to_col = nullptr;
    int *claimed_pivots = nullptr;
};

void cleanupGpu(GpuBoundaryMatrix &m, GpuScanState &s)
{
    if (m.data)
        cudaFree(m.data);
    if (m.indices)
        cudaFree(m.indices);
    if (m.starts)
        cudaFree(m.starts);
    if (s.stable_columns)
        cudaFree(s.stable_columns);
    if (s.pivot_candidates)
        cudaFree(s.pivot_candidates);
    if (s.zero_addition)
        cudaFree(s.zero_addition);
    if (s.low_to_col)
        cudaFree(s.low_to_col);
    if (s.claimed_pivots)
        cudaFree(s.claimed_pivots);
    m = GpuBoundaryMatrix{};
    s = GpuScanState{};
}

} // anonymous namespace

// Host-level wrappers for GPU scan kernels (defined in kernel_hypha_scan.cu)
namespace nerve::gpu::persistence::detail
{
void launchGpuBoundaryScan(const int *d_col_data, const int *d_col_indices, const int *d_col_starts,
                           int *d_stable_columns, int *d_pivot_candidates, int *d_zero_addition,
                           int n_columns, int max_height, cudaStream_t stream);

void launchGpuPivotClaim(const int *d_pivot_candidates, int *d_low_to_col, int *d_claimed_pivots,
                         int n_columns, cudaStream_t stream);
} // namespace nerve::gpu::persistence::detail

GpuScanResult HyphaReducer::gpuBoundaryScan(const algebra::BoundaryMatrix &boundary_matrix)
{
    GpuScanResult result;
    Size n_cols = boundary_matrix.cols();
    Size n_rows = boundary_matrix.rows();

    if (n_cols == 0 || !fitsCudaInt(n_cols))
    {
        return result;
    }
    int n_cols_int = static_cast<int>(n_cols);

    // Count max non-zero entries per column
    int max_height = 0;
    for (Size col = 0; col < n_cols; ++col)
    {
        int count = 0;
        for (Size row = 0; row < n_rows; ++row)
        {
            if (boundary_matrix.getMatrixEntry(row, col) != 0.0)
                ++count;
        }
        max_height = std::max(max_height, count);
    }
    if (max_height == 0)
        max_height = 1;

    std::size_t total_entries = 0;
    if (!checkedBytes(static_cast<std::size_t>(n_cols_int), static_cast<std::size_t>(max_height),
                      total_entries))
    {
        return result;
    }

    std::size_t data_bytes = 0;
    std::size_t starts_bytes = 0;
    if (!checkedBytes(total_entries, sizeof(int), data_bytes) ||
        !checkedBytes(static_cast<std::size_t>(n_cols_int), sizeof(int), starts_bytes))
    {
        return result;
    }

    // Build host-side packed matrix
    std::vector<int> h_data(total_entries, 0);
    std::vector<int> h_indices(total_entries, -1);
    std::vector<int> h_starts(static_cast<std::size_t>(n_cols_int), 0);

    std::size_t pos = 0;
    for (Size col = 0; col < n_cols; ++col)
    {
        h_starts[static_cast<std::size_t>(col)] = static_cast<int>(pos);
        int row_count = 0;
        for (Size row = 0; row < n_rows && row_count < max_height; ++row)
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
              static_cast<std::size_t>(max_height);
    }

    GpuBoundaryMatrix gpu_mat{};
    GpuScanState gpu_state{};

    cudaStream_t stream = nullptr;
    cudaError_t err;

    auto free_all = [&]() {
        cleanupGpu(gpu_mat, gpu_state);
        if (stream)
            cudaStreamDestroy(stream);
    };

    // Allocate GPU memory
    err = cudaMalloc(&gpu_mat.data, data_bytes);
    if (err != cudaSuccess)
    {
        free_all();
        return result;
    }
    err = cudaMalloc(&gpu_mat.indices, data_bytes);
    if (err != cudaSuccess)
    {
        free_all();
        return result;
    }
    err = cudaMalloc(&gpu_mat.starts, starts_bytes);
    if (err != cudaSuccess)
    {
        free_all();
        return result;
    }

    std::size_t col_bytes = static_cast<std::size_t>(n_cols_int) * sizeof(int);
    err = cudaMalloc(&gpu_state.stable_columns, col_bytes);
    if (err != cudaSuccess)
    {
        free_all();
        return result;
    }
    err = cudaMalloc(&gpu_state.pivot_candidates, col_bytes);
    if (err != cudaSuccess)
    {
        free_all();
        return result;
    }
    err = cudaMalloc(&gpu_state.zero_addition, col_bytes);
    if (err != cudaSuccess)
    {
        free_all();
        return result;
    }
    err = cudaMalloc(&gpu_state.low_to_col, col_bytes);
    if (err != cudaSuccess)
    {
        free_all();
        return result;
    }
    err = cudaMalloc(&gpu_state.claimed_pivots, col_bytes);
    if (err != cudaSuccess)
    {
        free_all();
        return result;
    }

    // Copy matrix data to GPU
    err = cudaMemcpy(gpu_mat.data, h_data.data(), data_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        free_all();
        return result;
    }
    err = cudaMemcpy(gpu_mat.indices, h_indices.data(), data_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        free_all();
        return result;
    }
    err = cudaMemcpy(gpu_mat.starts, h_starts.data(), starts_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        free_all();
        return result;
    }

    // Initialize low_to_col to -1
    std::vector<int> h_init(static_cast<std::size_t>(n_cols_int), -1);
    err = cudaMemcpy(gpu_state.low_to_col, h_init.data(), col_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        free_all();
        return result;
    }

    err = cudaStreamCreate(&stream);
    if (err != cudaSuccess)
    {
        free_all();
        return result;
    }

    // Launch GPU boundary scan kernel
    nerve::gpu::persistence::detail::launchGpuBoundaryScan(
        gpu_mat.data, gpu_mat.indices, gpu_mat.starts, gpu_state.stable_columns,
        gpu_state.pivot_candidates, gpu_state.zero_addition, n_cols_int, max_height, stream);

    err = cudaPeekAtLastError();
    if (err != cudaSuccess)
    {
        free_all();
        return result;
    }

    // Launch pivot claim kernel
    nerve::gpu::persistence::detail::launchGpuPivotClaim(
        gpu_state.pivot_candidates, gpu_state.low_to_col, gpu_state.claimed_pivots, n_cols_int,
        stream);

    err = cudaPeekAtLastError();
    if (err != cudaSuccess)
    {
        free_all();
        return result;
    }

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess)
    {
        free_all();
        return result;
    }

    // Read results back
    std::vector<int> h_stable(static_cast<std::size_t>(n_cols_int));
    std::vector<int> h_pivots(static_cast<std::size_t>(n_cols_int));
    std::vector<int> h_zero(static_cast<std::size_t>(n_cols_int));
    std::vector<int> h_claimed(static_cast<std::size_t>(n_cols_int));

    err = cudaMemcpy(h_stable.data(), gpu_state.stable_columns, col_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        free_all();
        return result;
    }
    err =
        cudaMemcpy(h_pivots.data(), gpu_state.pivot_candidates, col_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        free_all();
        return result;
    }
    err = cudaMemcpy(h_zero.data(), gpu_state.zero_addition, col_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        free_all();
        return result;
    }
    err = cudaMemcpy(h_claimed.data(), gpu_state.claimed_pivots, col_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        free_all();
        return result;
    }

    result.stable_columns = std::move(h_stable);
    result.pivot_candidates = std::move(h_pivots);
    result.zero_addition = std::move(h_zero);
    result.claimed_pivots = std::move(h_claimed);

    // Identify unstable columns: columns that have non-zero entries
    // but were not resolved by the GPU scan.
    result.unstable_columns.clear();
    for (int i = 0; i < n_cols_int; ++i)
    {
        if (result.pivot_candidates[i] >= 0 && result.claimed_pivots[i] < 0)
        {
            result.unstable_columns.push_back(i);
        }
    }

    free_all();
    return result;
}

void HyphaReducer::cpuClearingCompression(GpuScanResult &scan,
                                          const algebra::BoundaryMatrix &matrix)
{
    if (!config_.use_clearing)
        return;

    Size n_cols = matrix.cols();
    Size n_rows = matrix.rows();

    // For each claimed pivot, clear the corresponding column (if not the owner)
    // A column with a claimed pivot can be zeroed out since the pivot owner
    // will handle that row.
    for (int col = 0; col < static_cast<int>(n_cols); ++col)
    {
        int pivot = scan.claimed_pivots[col];
        if (pivot < 0)
            continue;

        // Check if the pivot was claimed by a different column (the owner)
        // The owner was set in low_to_col during the GPU scan
        // Columns that have their pivots claimed but are NOT the owner can be cleared
        if (scan.pivot_candidates[col] != pivot)
            continue;

        // This column's pivot was claimed - mark it as stable
        scan.stable_columns[col] = 1;
    }
}

std::vector<Pair> HyphaReducer::cpuSubmatrixReduction(const SubMatrix &sub)
{
    if (sub.columns.empty())
    {
        return {};
    }

    // Build the boundary matrix in the format expected by the lockfree reducer
    // Convert from SubMatrix to std::vector<std::vector<int>>
    std::vector<std::vector<int>> boundary(sub.columns.size());
    for (std::size_t i = 0; i < sub.columns.size(); ++i)
    {
        boundary[i] = sub.columns[i];
    }

    // Choose scheduling strategy based on column count
    int num_threads = std::thread::hardware_concurrency();
    if (static_cast<int>(sub.columns.size()) < config_.unstable_threshold)
    {
        num_threads = 1;
    }

    // Call the lockfree reducer (nerve namespace)
    auto nerve_pairs =
        nerve::persistence::reduceMatrixLockfree(boundary, sub.filtration_values, 2, num_threads);

    // Convert from nerve::persistence::Pair to nerve::Pair
    std::vector<Pair> result;
    result.reserve(nerve_pairs.size());
    for (const auto &p : nerve_pairs)
    {
        Pair np;
        np.birth = p.birth;
        np.death = p.death;
        np.dimension = p.dimension;
        result.push_back(np);
    }

    return result;
}

std::vector<Pair> HyphaReducer::compute(const algebra::BoundaryMatrix &matrix)
{
    Size n_cols = matrix.cols();
    if (n_cols == 0)
        return {};

    // GPU boundary scan (SIMT)
    auto scan_result = gpuBoundaryScan(matrix);

    // CPU clearing + compression (MIMD)
    cpuClearingCompression(scan_result, matrix);

    // Build submatrix of unstable columns
    SubMatrix sub;
    sub.column_map = scan_result.unstable_columns;

    // Extract sparse columns and filtration values for unstable columns
    Size n_rows = matrix.rows();
    for (int col_idx : scan_result.unstable_columns)
    {
        Size col = static_cast<Size>(col_idx);
        std::vector<int> sparse_col;
        for (Size row = 0; row < n_rows; ++row)
        {
            if (matrix.getMatrixEntry(row, col) != 0.0)
            {
                sparse_col.push_back(static_cast<int>(row));
            }
        }
        sub.columns.push_back(std::move(sparse_col));
        sub.filtration_values.push_back(matrix.getFiltrationValue(col));
    }

    // CPU SS+ reduction using lockfree backend (MIMD)
    auto pairs = cpuSubmatrixReduction(sub);

    return pairs;
}

} // namespace nerve::persistence

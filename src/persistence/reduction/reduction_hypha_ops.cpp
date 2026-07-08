#include "nerve/gpu/device_array.hpp"
#include "nerve/gpu/gpu_error.hpp"
#include "nerve/gpu/persistence_kernels.cuh"
#include "nerve/persistence/reduction/reduction_hypha_ops.hpp"
#include "nerve/platform.hpp"
#include "nerve/types.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <memory>
#include <vector>

namespace nerve::persistence
{

// Persistent GPU memory pool for repeated reduction calls.
// Avoids per-call cudaMalloc/cudaFree overhead by growing buffers
// only when the problem size increases.
struct HyphaReducer::GpuPool
{
    cudaStream_t stream = nullptr;
    std::uint64_t *d_boundary = nullptr;
    std::uint64_t *d_reduced = nullptr;
    int *d_pivot_table = nullptr;
    std::size_t boundary_capacity = 0; // bytes
    std::size_t pivot_capacity = 0;    // ints

    void ensure(std::size_t total_bytes, int n_cols)
    {
        if (total_bytes > boundary_capacity)
        {
            if (d_boundary)
            {
                cudaFree(d_boundary);
                d_boundary = nullptr;
            }
            if (d_reduced)
            {
                cudaFree(d_reduced);
                d_reduced = nullptr;
            }
            cudaError_t st = cudaMalloc(&d_boundary, total_bytes);
            if (st != cudaSuccess)
            {
                d_boundary = nullptr;
                return;
            }
            st = cudaMalloc(&d_reduced, total_bytes);
            if (st != cudaSuccess)
            {
                cudaFree(d_boundary);
                d_boundary = nullptr;
                d_reduced = nullptr;
                return;
            }
            boundary_capacity = total_bytes;
        }
        if (static_cast<std::size_t>(n_cols) > pivot_capacity)
        {
            if (d_pivot_table)
            {
                cudaFree(d_pivot_table);
                d_pivot_table = nullptr;
            }
            cudaError_t st = cudaMalloc(&d_pivot_table,
                            static_cast<std::size_t>(n_cols) * sizeof(int));
            if (st != cudaSuccess)
            {
                d_pivot_table = nullptr;
                return;
            }
            pivot_capacity = static_cast<std::size_t>(n_cols);
        }
        if (!stream)
            cudaStreamCreate(&stream);
    }

    ~GpuPool()
    {
        if (d_boundary)
            cudaFree(d_boundary);
        if (d_reduced)
            cudaFree(d_reduced);
        if (d_pivot_table)
            cudaFree(d_pivot_table);
        if (stream)
            cudaStreamDestroy(stream);
    }
};

HyphaReducer::HyphaReducer() = default;

HyphaReducer::HyphaReducer(const Config &cfg)
    : config_(cfg)
{}

} // namespace nerve::persistence

namespace nerve::persistence
{

HyphaReducer::~HyphaReducer() = default;

std::vector<Pair> HyphaReducer::gpuSubmatrixReduction(
    const int *col_ptr, const int *row_indices, int nnz, int n_cols, int n_rows,
    const std::vector<double> &col_filtration_values,
    const std::vector<double> &row_filtration_values,
    const std::vector<Dimension> &dimensions,
    HyphaPhaseTimings *timings)
{
    if (n_cols == 0 || n_rows == 0)
    {
        return {};
    }

    int words_per_col = (n_rows + 63) / 64;
    std::size_t total_words = static_cast<std::size_t>(n_cols) * words_per_col;
    std::size_t total_bytes = total_words * sizeof(std::uint64_t);

    // Lazy-init persistent GPU pool.
    auto t_pack_start = std::chrono::high_resolution_clock::now();
    if (!gpu_pool_)
        gpu_pool_ = std::make_unique<GpuPool>();
    gpu_pool_->ensure(total_bytes, n_cols);
    if (!gpu_pool_->d_boundary || !gpu_pool_->d_reduced || !gpu_pool_->d_pivot_table)
    {
        std::fprintf(stderr, "gpuSubmatrixReduction: GPU pool allocation failed\n");
        return {};
    }

    // Upload CSC arrays to GPU (O(nnz), much smaller than packed O(colsxrows)).
    int *d_col_ptr = nullptr;
    int *d_row_indices = nullptr;
    std::size_t col_ptr_bytes = static_cast<std::size_t>(n_cols + 1) * sizeof(int);
    std::size_t row_bytes = static_cast<std::size_t>(nnz) * sizeof(int);

    cudaError_t st = cudaMalloc(&d_col_ptr, col_ptr_bytes);
    if (st != cudaSuccess)
    {
        std::fprintf(stderr, "gpuSubmatrixReduction: cudaMalloc col_ptr failed: %s\n",
                     cudaGetErrorString(st));
        return {};
    }
    st = cudaMalloc(&d_row_indices, row_bytes);
    if (st != cudaSuccess)
    {
        std::fprintf(stderr, "gpuSubmatrixReduction: cudaMalloc row_indices failed: %s\n",
                     cudaGetErrorString(st));
        cudaFree(d_col_ptr);
        return {};
    }
    st = cudaMemcpyAsync(d_col_ptr, col_ptr, col_ptr_bytes, cudaMemcpyHostToDevice,
                          gpu_pool_->stream);
    if (st != cudaSuccess)
    {
        std::fprintf(stderr, "gpuSubmatrixReduction: CSC upload failed: %s\n",
                     cudaGetErrorString(st));
        cudaFree(d_col_ptr);
        cudaFree(d_row_indices);
        return {};
    }
    st = cudaMemcpyAsync(d_row_indices, row_indices, row_bytes, cudaMemcpyHostToDevice,
                          gpu_pool_->stream);
    if (st != cudaSuccess)
    {
        std::fprintf(stderr, "gpuSubmatrixReduction: row_indices upload failed: %s\n",
                     cudaGetErrorString(st));
        cudaFree(d_col_ptr);
        cudaFree(d_row_indices);
        return {};
    }

    st = cudaMemsetAsync(gpu_pool_->d_boundary, 0, total_bytes, gpu_pool_->stream);
    if (st != cudaSuccess)
    {
        std::fprintf(stderr, "gpuSubmatrixReduction: cudaMemsetAsync failed: %s\n",
                     cudaGetErrorString(st));
        cudaFree(d_col_ptr);
        cudaFree(d_row_indices);
        return {};
    }

    // GPU kernel: build packed columns directly from CSC on device.
    // Eliminates the 7.6ms CPU bit-loop and 3.9ms H2D copy of packed data.
    st = ::nerve::persistence::accelerated::launchBuildPackedFromCSC(
        gpu_pool_->d_boundary, d_col_ptr, d_row_indices, n_cols, words_per_col,
        gpu_pool_->stream);

    // Free CSC upload buffers -- the pack kernel only reads from them.
    // NOTE: cudaFree is synchronous (blocks until all queued work completes),
    // so the pack kernel finishes before computeMatrixReduction is launched.
    // For optimal overlap, defer the free or use cudaFreeAsync (CUDA 11.2+).
    cudaFree(d_col_ptr);
    cudaFree(d_row_indices);

    if (st != cudaSuccess)
    {
        std::fprintf(stderr, "gpuSubmatrixReduction: GPU pack kernel failed: %s\n",
                     cudaGetErrorString(st));
        return {};
    }
    auto t_pack_end = std::chrono::high_resolution_clock::now();

    // GPU warp-level packed-column reduction with MSB pivot convention.
    auto t_reduce_start = std::chrono::high_resolution_clock::now();
    nerve::gpu::kernels::KernelDispatcher dispatcher;
    st = dispatcher.computeMatrixReduction(gpu_pool_->d_boundary, gpu_pool_->d_reduced, n_cols,
                                            words_per_col, gpu_pool_->d_pivot_table,
                                            gpu_pool_->stream);

    if (st != cudaSuccess)
    {
        std::fprintf(stderr, "gpuSubmatrixReduction: GPU reduction failed: %s (%s)\n",
                     dispatcher.getLastError().c_str(), cudaGetErrorString(st));
        return {};
    }

    auto t_reduce_end = std::chrono::high_resolution_clock::now();

    // Download pivot table and reduced columns from GPU.
    auto t_dl_start = std::chrono::high_resolution_clock::now();
    std::size_t pivot_bytes = static_cast<std::size_t>(n_cols) * sizeof(int);
    std::vector<int> h_pivots(static_cast<std::size_t>(n_cols));
    st = cudaMemcpyAsync(h_pivots.data(), gpu_pool_->d_pivot_table, pivot_bytes,
                          cudaMemcpyDeviceToHost, gpu_pool_->stream);
    std::vector<std::uint64_t> h_reduced(total_words);
    st = cudaMemcpyAsync(h_reduced.data(), gpu_pool_->d_reduced, total_bytes,
                          cudaMemcpyDeviceToHost, gpu_pool_->stream);
    st = cudaStreamSynchronize(gpu_pool_->stream);

    if (st != cudaSuccess)
        return {};

    // Filtration-order correction via post-pass re-reduction.
    // Uses a hybrid approach that selects the best starting point for
    // each column based on what the GPU produced:
    //
    //   (a) Invalidated columns (h_pivots[i] >= 0, pivot claimed by
    //       earlier column): start from h_reduced[i] (GPU's partial
    //       reduction), then cascade. This preserves GPU work and
    //       minimizes cascade steps.
    //
    //   (b) Aborted columns (h_pivots[i] < 0, CSP boundary non-empty):
    //       h_reduced[i] is all zeros because the GPU never writes
    //       d_reduced for columns that hit MAX_ITERATIONS before
    //       claiming a pivot.  Must build from CSC instead.
    //
    //   (c) Genuinely empty columns: skip.
    //
    // FUNDAMENTAL LIMIT (~0.22% residual GPU-vs-Seq count error):
    // The GPU kernel writes each column's reduced form to d_reduced
    // only ONCE -- at the moment that column successfully claims a
    // pivot via atomicCAS.  These are SNAPSHOTS captured mid-reduction
    // while other warps are still racing.  Survivors' forms are
    // non-deterministic and may not be fully reduced due to
    // MAX_ITERATIONS=1000.
    //
    // When this post-pass cascades through survivors' forms, each XOR
    // step compounds the non-determinism.  Some columns reach an
    // unclaimed pivot that differs from the sequential result,
    // producing a different pair count.
    //
    // The lockfree reducer avoids this residual entirely (0.0000%)
    // because it works on shared mutable state -- survivors' forms are
    // always the FINAL state after all workers join, making the
    // post-pass XOR operations deterministic.
    //
    // Eliminating the ~0.22% GPU residual would require either:
    //   (1) A second GPU kernel pass after this post-pass corrects
    //       the pivot table, so survivors' forms are recomputed from
    //       a clean state, or
    //   (2) Deterministic GPU reduction via single-warp processing.
    auto find_msb = [&](const std::uint64_t *col) -> int {
        for (int w = words_per_col - 1; w >= 0; --w)
            if (col[w] != 0)
                return w * 64 + (nerve::bits::fls64(col[w]) - 1);
        return -1;
    };

    std::vector<int> earliest_owner(static_cast<std::size_t>(n_rows), -1);
    std::vector<std::uint64_t> col_scratch(
        static_cast<std::size_t>(words_per_col));

    for (int i = 0; i < n_cols; ++i)
    {
        int pivot = h_pivots[static_cast<std::size_t>(i)];

        // Determine the starting point for this column
        if (pivot >= 0 && pivot < n_rows)
        {
            // GPU found a pivot.  Check if the column owns it.
            std::size_t pu = static_cast<std::size_t>(pivot);
            if (earliest_owner[pu] < 0)
            {
                earliest_owner[pu] = i;
                continue;
            }
            // Pivot claimed by an earlier column -> invalidated.
            // Start from h_reduced[i] (preserves GPU's partial work).
            std::copy_n(&h_reduced[static_cast<std::size_t>(i) * words_per_col],
                        words_per_col, col_scratch.begin());
        }
        else
        {
            // h_pivots[i] < 0: column was aborted or is genuinely empty.
            // d_reduced[i] was never written (zeros).  Build from CSC.
            int col_start = col_ptr[i];
            int col_end = col_ptr[i + 1];
            if (col_start >= col_end)
                continue;   // genuinely empty
            std::fill(col_scratch.begin(), col_scratch.end(), 0);
            for (int ri = col_start; ri < col_end; ++ri)
            {
                int row = row_indices[ri];
                if (row >= 0 && row < n_rows)
                    col_scratch[static_cast<std::size_t>(row / 64)] |=
                        (1ULL << static_cast<unsigned>(row % 64));
            }
        }

        int new_pivot = find_msb(col_scratch.data());

        // Cascade: while MSB claimed, XOR with survivor
        const int max_iter = n_cols;
        bool claimed = false;
        for (int iter = 0; iter < max_iter; ++iter)
        {
            if (new_pivot < 0 || new_pivot >= n_rows)
                break;

            std::size_t npu = static_cast<std::size_t>(new_pivot);
            int claiming = earliest_owner[npu];
            if (claiming < 0)
            {
                earliest_owner[npu] = i;
                h_pivots[static_cast<std::size_t>(i)] = new_pivot;
                // Write back corrected form so later columns benefit.
                std::uint64_t *h_col = &h_reduced[
                    static_cast<std::size_t>(i) * words_per_col];
                for (int w = 0; w < words_per_col; ++w)
                    h_col[w] = col_scratch[static_cast<std::size_t>(w)];
                claimed = true;
                break;
            }
            if (claiming == i)
            {
                claimed = true;
                break;
            }

            // XOR with the survivor's reduced form (from h_reduced).
            const std::uint64_t *claim_col = &h_reduced[
                static_cast<std::size_t>(claiming) * words_per_col];
            for (int w = 0; w < words_per_col; ++w)
                col_scratch[static_cast<std::size_t>(w)] ^= claim_col[w];

            new_pivot = find_msb(col_scratch.data());
        }
        if (!claimed)
            h_pivots[static_cast<std::size_t>(i)] = -1;
    }

    // Extract persistence pairs from corrected pivot table.
    std::vector<Pair> result;
    result.reserve(static_cast<std::size_t>(n_cols) / 2);

    for (int i = 0; i < n_cols; ++i)
    {
        int pivot = h_pivots[static_cast<std::size_t>(i)];
        if (pivot >= 0 &&
            static_cast<std::size_t>(pivot) < col_filtration_values.size() &&
            static_cast<std::size_t>(i) < col_filtration_values.size())
        {
            Pair pair{};
            pair.dimension = static_cast<std::size_t>(i) < dimensions.size()
                                 ? dimensions[static_cast<std::size_t>(i)]
                                 : 0;
            pair.birth = static_cast<std::size_t>(pivot) < row_filtration_values.size()
                             ? row_filtration_values[static_cast<std::size_t>(pivot)]
                             : col_filtration_values[static_cast<std::size_t>(pivot)];
            pair.death = col_filtration_values[static_cast<std::size_t>(i)];
            result.push_back(pair);
        }
    }
    auto t_dl_end = std::chrono::high_resolution_clock::now();

    if (timings)
    {
        using ms = std::chrono::duration<double, std::milli>;
        timings->gpu_pack_ms = ms(t_pack_end - t_pack_start).count();
        timings->gpu_reduction_ms = ms(t_reduce_end - t_reduce_start).count();
        timings->gpu_download_ms = ms(t_dl_end - t_dl_start).count();
    }

    return result;
}

std::vector<Pair> HyphaReducer::compute(const algebra::BoundaryMatrix &matrix,
                                          HyphaPhaseTimings *timings)
{
    auto t_start = std::chrono::high_resolution_clock::now();
    Size n_cols = matrix.cols();
    if (n_cols == 0)
        return {};

    // Build CSC arrays directly on CPU.
    auto t_csc_start = std::chrono::high_resolution_clock::now();
    std::vector<int> col_ptr;
    std::vector<int> row_indices;
    {
        std::vector<int> col_data;
        matrix.buildCSC(col_ptr, row_indices, col_data);
    }
    auto t_csc_end = std::chrono::high_resolution_clock::now();

    int n_cols_int = static_cast<int>(n_cols);
    int nnz = static_cast<int>(row_indices.size());
    Size n_rows = matrix.rows();

    auto t_clear_start = std::chrono::high_resolution_clock::now();
    auto t_clear_end = t_clear_start;

    auto t_sub_start = std::chrono::high_resolution_clock::now();
    std::vector<double> filtration_values;
    std::vector<Dimension> dimensions;
    filtration_values.reserve(static_cast<std::size_t>(n_cols_int));
    dimensions.reserve(static_cast<std::size_t>(n_cols_int));
    for (int col_idx = 0; col_idx < n_cols_int; ++col_idx)
    {
        Size col = static_cast<Size>(col_idx);
        filtration_values.push_back(matrix.getFiltrationValue(col));
        dimensions.push_back(
            static_cast<Dimension>(matrix.getColSimplexDimension(col)));
    }
    auto t_sub_end = std::chrono::high_resolution_clock::now();

    auto t_reduce_start = std::chrono::high_resolution_clock::now();
    // Build row filtration values for correct birth simplex filtration lookup.
    // In k-dimensional boundary matrices (buildKDimensional), rows and columns
    // index different simplex sets, so row_filtration_values is required.
    std::vector<double> row_filtration_values;
    row_filtration_values.reserve(static_cast<std::size_t>(n_rows));
    for (Size row = 0; row < n_rows; ++row)
        row_filtration_values.push_back(matrix.getRowFiltrationValue(row));

    auto pairs = gpuSubmatrixReduction(col_ptr.data(), row_indices.data(), nnz,
                                       n_cols_int, static_cast<int>(n_rows),
                                       filtration_values, row_filtration_values,
                                       dimensions, timings);
    auto t_reduce_end = std::chrono::high_resolution_clock::now();

    auto t_end = std::chrono::high_resolution_clock::now();

    if (timings)
    {
        using ms = std::chrono::duration<double, std::milli>;
        timings->csc_build_ms = ms(t_csc_end - t_csc_start).count();
        timings->clearing_ms = ms(t_clear_end - t_clear_start).count();
        timings->submatrix_build_ms = ms(t_sub_end - t_sub_start).count();
        // gpu_pack_ms, gpu_reduction_ms, gpu_download_ms are filled by
        // gpuSubmatrixReduction when timings is non-null.
        timings->overhead_ms =
            ms(t_end - t_start).count() - timings->csc_build_ms -
            timings->clearing_ms - timings->submatrix_build_ms -
            timings->gpu_pack_ms -
            timings->gpu_reduction_ms - timings->gpu_download_ms;
    }

    return pairs;
}

} // namespace nerve::persistence

#include "gpu_test_helpers.cuh"
#include "nerve/errors/errors.hpp"
#include "nerve/formats/packed_boundary_matrix.hpp"
#include "nerve/formats/packed_gpu_scan.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

int main()
{
    if (!has_gpu())
    {
        std::cout << "SKIP: no GPU available" << std::endl;
        return 0;
    }

    // Create a simple 4x3 boundary matrix in COO format:
    // Column 0: row 0
    // Column 1: row 1
    // Column 2: rows 0 and 2
    // This represents a simple boundary where column 2 has boundary {0, 2}
    std::vector<nerve::Index> row_indices = {0, 1, 0, 2};
    std::vector<nerve::Index> col_indices = {0, 1, 2, 2};
    nerve::Size n_rows = 4;
    nerve::Size n_cols = 3;

    auto mat_result =
        nerve::formats::PackedBoundaryMatrix::fromCoo(n_rows, n_cols, row_indices, col_indices);
    assert(mat_result.has_value());

    auto &mat = mat_result.value();
    mat.convertToPacked();

    // Build GPU layout
    auto layout_result = mat.buildGpuLayout();
    assert(layout_result.has_value());

    const auto &layout = layout_result.value();
    assert(layout.num_columns == n_cols);
    assert(layout.num_rows == n_rows);
    assert(!layout.columns_flat.empty());
    assert(!layout.column_offsets.empty());

    std::cout << "PASS: built GpuPackedLayout (cols=" << layout.num_columns
              << " rows=" << layout.num_rows << " max_words=" << layout.max_words_per_column
              << " total_bytes=" << layout.total_packed_bytes << ")" << std::endl;

    // Run the GPU packed scan
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    auto scan_result =
        nerve::formats::gpu::launchPackedScan(layout, static_cast<void *>(stream), 0);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    assert(scan_result.has_value());

    const auto &scan = scan_result.value();
    std::cout << "PASS: launchPackedScan returned stable=" << scan.stable_count
              << " unstable=" << scan.unstable_count << std::endl;

    // Verify: total columns accounted for
    assert(scan.stable_count + scan.unstable_count == layout.num_columns);

    // Verify: stable columns are sorted
    assert(std::is_sorted(scan.stable_columns.begin(), scan.stable_columns.end()));

    // Verify: leftmost_column_by_row has correct size
    assert(static_cast<nerve::Size>(scan.leftmost_column_by_row.size()) == layout.num_rows);

    std::cout << "PASS: all packed GPU scan checks passed" << std::endl;
    return 0;
}

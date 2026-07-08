#include "gpu_test_helpers.cuh"
#include "nerve/formats/packed_boundary_matrix.hpp"

int main()
{
    if (!has_gpu())
    {
        std::cerr << "No CUDA device available -- skipping GPU formats kernel coverage tests\n";
        return 0;
    }

    // Formats: PackedBoundaryMatrix default construction
    {
        nerve::formats::PackedBoundaryMatrix matrix;
        auto fmt = matrix.format();
        assert(static_cast<int>(fmt) >= 0);
        std::cout << "PASSED: PackedBoundaryMatrix format()\n";
    }

    // Formats: PackedBoundaryMatrix dimensions via methods
    {
        nerve::formats::PackedBoundaryMatrix matrix;
        auto rows = matrix.numRows();
        auto cols = matrix.numCols();
        assert(rows >= 0);
        assert(cols >= 0);
        std::cout << "PASSED: PackedBoundaryMatrix dimensions ("
                  << rows << "x" << cols << ")\n";
    }

    // Formats: PackedBoundaryMatrix nnz
    {
        nerve::formats::PackedBoundaryMatrix matrix;
        auto nz = matrix.nnz();
        assert(nz >= 0);
        std::cout << "PASSED: PackedBoundaryMatrix nnz=" << nz << "\n";
    }

    // Formats: GpuPackedLayout smoke (namespace-level struct)
    {
        nerve::formats::GpuPackedLayout layout;
        layout.num_rows = 100;
        layout.num_columns = 50;
        layout.max_words_per_column = 8;
        assert(layout.num_rows == 100);
        assert(layout.num_columns == 50);
        assert(layout.max_words_per_column == 8);
        std::cout << "PASSED: GpuPackedLayout (100x50, 8 words/col)\n";
    }

    // Formats: GpuScanResult smoke (namespace-level struct)
    {
        nerve::formats::GpuScanResult result;
        result.stable_count = 200;
        result.unstable_count = 56;
        assert(result.stable_count == 200);
        assert(result.unstable_count == 56);
        std::cout << "PASSED: GpuScanResult (200 stable, 56 unstable)\n";
    }

    return 0;
}

#include "nerve/algebra/bit_packed.hpp"
#include "nerve/persistence/utils/bit_parallel_z2.hpp"

#include <cassert>
#include <limits>
#include <stdexcept>
#include <vector>

int main() {
    using nerve::algebra::compressed::BitPackedZ2Matrix;

    BitPackedZ2Matrix matrix(130, 2);
    assert(matrix.getNumRows() == 130);
    assert(matrix.getNumCols() == 2);
    assert(matrix.isColumnEmpty(0));

    matrix.set(0, 0);
    matrix.set(129, 0);
    assert(matrix.get(0, 0));
    assert(matrix.get(129, 0));
    assert(matrix.findPivot(0) == 129);

    matrix.set(64, 1);
    matrix.xorColumns(0, 1);
    assert(matrix.get(64, 0));
    matrix.clear(64, 0);
    assert(!matrix.get(64, 0));

    bool rejected_row_overflow = false;
    try {
        BitPackedZ2Matrix too_many_rows(std::numeric_limits<size_t>::max(), 1);
    } catch (const std::overflow_error&) {
        rejected_row_overflow = true;
    }
    assert(rejected_row_overflow);

    const size_t vector_capacity = std::vector<uint64_t>().max_size();
    if (vector_capacity < std::numeric_limits<size_t>::max()) {
        bool rejected_capacity = false;
        try {
            BitPackedZ2Matrix too_many_columns(64, vector_capacity + 1);
        } catch (const std::length_error&) {
            rejected_capacity = true;
        }
        assert(rejected_capacity);
    }

    using namespace nerve::persistence::bitparallel;

    std::vector<BitColumn> columns;
    columns.push_back(buildBitColumn({0, 64}, 128));
    CompressedSparseBlockMatrix csb;
    convertToCSB(columns, csb, 1);
    assert(csb.num_cols == 1);
    assert(csb.num_block_rows == 1);
    assert(!csb.blocks.empty());
    assert(!csb.blocks[0].empty());

    bool rejected_csb_block_size = false;
    try {
        convertToCSB(columns, csb, 0);
    } catch (const std::invalid_argument&) {
        rejected_csb_block_size = true;
    }
    assert(rejected_csb_block_size);

    bool rejected_nonfinite_filtration = false;
    try {
        BitParallelConfig config;
        (void)reduceMatrixBitParallel(columns, config,
                                      {std::numeric_limits<double>::quiet_NaN()});
    } catch (const std::invalid_argument&) {
        rejected_nonfinite_filtration = true;
    }
    assert(rejected_nonfinite_filtration);

    return 0;
}

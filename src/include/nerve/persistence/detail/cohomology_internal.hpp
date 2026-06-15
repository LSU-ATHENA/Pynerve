
#pragma once

#include "nerve/algebra/cellular.hpp"
#include "nerve/core_types.hpp"

#include <vector>

namespace nerve::persistence::detail
{

struct RrefResult
{
    std::vector<std::vector<double>> matrix;
    std::vector<Size> pivot_columns;
};

RrefResult computeRref(std::vector<std::vector<double>> matrix);
Size matrixRank(const std::vector<std::vector<double>> &matrix);
std::vector<std::vector<double>> nullspaceBasis(const std::vector<std::vector<double>> &matrix);
std::vector<std::vector<double>> submatrix(const std::vector<std::vector<double>> &matrix,
                                           const std::vector<Index> &row_indices,
                                           const std::vector<Index> &col_indices);
std::vector<double> multiplyMatrixVector(const std::vector<std::vector<double>> &matrix,
                                         const std::vector<double> &vector);
Index findCellIndex(const algebra::CellularComplex &complex, const algebra::Cell &cell);

} // namespace nerve::persistence::detail

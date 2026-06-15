#pragma once

#include "nerve/algebra/cellular.hpp"
#include "nerve/core_types.hpp"

#include <cstddef>
#include <vector>

namespace nerve::persistence
{
namespace detail
{

// Result of RREF computation.
struct RrefResult
{
    std::vector<std::vector<double>> matrix;
    std::vector<std::size_t> pivot_columns;
    std::size_t rank = 0;
};

// Compute RREF (Row Reduced Echelon Form).
RrefResult computeRref(std::vector<std::vector<double>> matrix);

// Shared linear-algebra helper routines used by cohomology codepaths.
Size matrixRank(const std::vector<std::vector<double>> &matrix);
std::vector<std::vector<double>> nullspaceBasis(const std::vector<std::vector<double>> &matrix);
std::vector<std::vector<double>> submatrix(const std::vector<std::vector<double>> &matrix,
                                           const std::vector<Index> &row_indices,
                                           const std::vector<Index> &col_indices);
std::vector<double> multiplyMatrixVector(const std::vector<std::vector<double>> &matrix,
                                         const std::vector<double> &vector);
Index findCellIndex(const algebra::CellularComplex &complex, const algebra::Cell &cell);

} // namespace detail

// RREF computation for cohomology.
class CohomologyRREF
{
public:
    CohomologyRREF() = default;

    // Compute RREF of boundary matrix.
    std::vector<std::vector<double>>
    computeRREF(const std::vector<std::vector<double>> &matrix) const;

    // Extract pivot columns.
    std::vector<std::size_t> findPivots(const std::vector<std::vector<double>> &rref_matrix) const;

    // Compute rank from RREF.
    std::size_t computeRank(const std::vector<std::vector<double>> &rref_matrix) const;

    // Get null space basis from RREF.
    std::vector<std::vector<double>>
    computeNullSpace(const std::vector<std::vector<double>> &rref_matrix) const;
};

} // namespace nerve::persistence

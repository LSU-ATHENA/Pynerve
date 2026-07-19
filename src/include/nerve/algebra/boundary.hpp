
#pragma once
#include "nerve/algebra/complex.hpp"
#include "nerve/algebra/simplex.hpp"
#include "nerve/core/policy/ownership_checks.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core/rng/determinism_contract.hpp"
#include "nerve/core_types.hpp"
#include "nerve/errors/errors.hpp"

#include <limits>
#include <map>
#include <set>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace nerve::algebra
{

using Pair = std::pair<Index, Index>;

struct Point
{
    float x, y, z;
    Point(float x_ = 0, float y_ = 0, float z_ = 0)
        : x(x_)
        , y(y_)
        , z(z_)
    {}
};

struct MatrixEntry
{
    Size row;
    Size col;
    double value;
    MatrixEntry(Size r, Size c, double v)
        : row(r)
        , col(c)
        , value(v)
    {}
};
class BoundaryMatrix
{
public:
    BoundaryMatrix() = default;
    explicit BoundaryMatrix(const SimplicialComplex &complex);
    BoundaryMatrix(const SimplicialComplex &complex, Size dimension);
    void buildFromComplex(const SimplicialComplex &complex);
    void buildKDimensional(const SimplicialComplex &complex, Size k);

    [[nodiscard]] Size rows() const noexcept;
    [[nodiscard]] Size cols() const noexcept;
    [[nodiscard]] Size dimension() const noexcept;
    [[nodiscard]] bool isEmpty() const noexcept;
    std::vector<double> multiply(core::BufferView<const double> vector) const;
    std::vector<double> transposeMultiply(core::BufferView<const double> vector) const;
    BoundaryMatrix transpose() const;
    std::vector<double> applyBoundary(core::BufferView<const double> chain) const;
    std::vector<double> applyCoboundary(core::BufferView<const double> cochain) const;
    static_assert(core::ownership_utils::is_view_type<core::BufferView<const double>>::value,
                  "Matrix operations should use non-owning views");
    static_assert(core::ownership_utils::validateHotPathApi<core::BufferView<const double>>(),
                  "Hot-path matrix operations must use zero-copy views");
    std::vector<Index> boundaryOfSimplex(const Simplex &simplex) const;
    std::vector<Index> coboundaryOfSimplex(const Simplex &simplex) const;
    std::vector<Simplex> simplicesInRow(Size row) const;
    std::vector<Simplex> simplicesInCol(Size col) const;
    void reduceToReducedRowEchelon(const core::DeterminismContract &contract = {});
    std::vector<Index> findPivotColumns(const core::DeterminismContract &contract = {});
    std::vector<Index> findPivotRows(const core::DeterminismContract &contract = {});
    errors::ErrorResult<std::vector<std::pair<Index, Index>>>
    computePersistencePairs(const core::DeterminismContract &contract = {});
    errors::ErrorResult<std::vector<Index>>
    findEssentialCycles(const core::DeterminismContract &contract = {});
    double getRowFiltrationValue(Size row) const;
    const std::vector<Index> &lastLowRowToCol() const;
    [[nodiscard]] std::string matrixString() const;
    [[nodiscard]] std::string nonzeroPatternString() const;
    Size numNonzeros() const noexcept;
    double sparsityRatio() const noexcept;
    int maxColumnHeight() const noexcept;
    int columnHeight(Size col) const;
    double getCoefficient(Size row, Size col) const;
    double getMatrixEntry(Size row, Size col) const;
    double getFiltrationValue(Size col) const;
    void setFiltrationValue(Size col, double value);
    int getRowSimplexDimension(Size row) const;
    int getColSimplexDimension(Size col) const;
    Index getColumnIndexForRowSimplex(Size row) const;

    // Build CSC (Compressed Sparse Column) arrays for GPU transfer.
    // col_ptr: size = cols_ + 1, start offset of each column
    // row_indices: size = numNonzeros(), row index per entry
    // values: size = numNonzeros(), coefficient per entry (always 1.0 in Z2)
    void buildCSC(std::vector<int> &col_ptr, std::vector<int> &row_indices,
                  std::vector<int> &values) const;

private:
    std::map<std::pair<Size, Size>, double> entries_;
    Size rows_ = 0;
    Size cols_ = 0;
    Size dimension_ = 0;
    std::vector<int> col_heights_;
    int max_column_height_ = 0;
    std::unordered_map<Simplex, Index, Simplex::Hash> simplex_to_col_;
    std::unordered_map<Simplex, Index, Simplex::Hash> simplex_to_row_;
    std::vector<Simplex> col_to_simplex_;
    std::vector<Simplex> row_to_simplex_;
    std::vector<double> filtration_values_;
    std::vector<double> row_filtration_values_;
    std::vector<Index> last_low_row_to_col_;
    void buildIndexMaps(const SimplicialComplex &complex);
    std::vector<MatrixEntry> computeBoundaryEntries(const SimplicialComplex &complex);
    int computeBoundaryCoefficient(const Simplex &simplex, const Simplex &face);
    void validateMatrix() const;
    void setEntry(Size row, Size col, double value);
    double getEntry(Size row, Size col) const;
    void swapRows(Size row1, Size row2);
};
class ChainComplex
{
public:
    ChainComplex(const SimplicialComplex &complex);
    const BoundaryMatrix &boundary(Size k) const;
    BoundaryMatrix &boundary(Size k);
    Size rank(Size k) const;
    Size bettiNumber(Size k) const;
    Size maxDimension() const noexcept;
    std::vector<double> applyBoundary(Size k, core::BufferView<const double> chain) const;
    std::vector<double> applyCoboundary(Size k, core::BufferView<const double> cochain) const;
    static_assert(core::ownership_utils::is_view_type<core::BufferView<const double>>::value,
                  "Chain operations should use non-owning views");
    static_assert(core::ownership_utils::validateHotPathApi<core::BufferView<const double>>(),
                  "Hot-path chain operations must use zero-copy views");
    std::vector<std::pair<Index, Index>> compute();
    std::vector<Pair> computePersistenceDiagram();
    std::vector<Size> computeBettiNumbers();

private:
    std::vector<BoundaryMatrix> boundary_matrices_;
    Size max_dimension_;
    void buildAllBoundaries(const SimplicialComplex &complex);
    Size computeMatrixRank(const BoundaryMatrix &matrix) const;
};
std::vector<Simplex> computeBoundaryFaces(const Simplex &simplex);
std::vector<int> computeBoundaryCoefficients(const Simplex &simplex);
bool isBoundaryCycle(const std::vector<double> &chain, const BoundaryMatrix &boundary);
bool isCoboundaryCycle(const std::vector<double> &cochain, const BoundaryMatrix &boundary);
} // namespace nerve::algebra

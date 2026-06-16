
#pragma once
#include "nerve/algebra/simplex.hpp"
#include "nerve/core/rng/determinism_contract.hpp"
#include "nerve/core_types.hpp"
#include "nerve/errors/errors.hpp"

#include <compare>
#include <memory>
#include <unordered_map>
#include <vector>

namespace nerve::algebra
{

class Cell
{
public:
    Cell() = default;
    Cell(int dimension);
    Cell(int dimension, const std::vector<Index> &boundary);

    [[nodiscard]] int dimension() const noexcept { return dimension_; }
    [[nodiscard]] const std::vector<Index> &boundary() const noexcept { return boundary_; }

    void addBoundaryCell(Index cell_index);
    void removeBoundaryCell(Index cell_index);
    void clearBoundary();

    [[nodiscard]] bool operator==(const Cell &other) const { return boundary_ == other.boundary_; }
    [[nodiscard]] bool operator<(const Cell &other) const { return boundary_ < other.boundary_; }

    struct Hash
    {
        [[nodiscard]] Size operator()(const Cell &cell) const noexcept;
    };

private:
    int dimension_ = -1;
    std::vector<Index> boundary_;
};
class CellularComplex
{
public:
    CellularComplex() = default;
    [[nodiscard]] Index addCell(const Cell &cell);
    void removeCell(Index cell_index);
    void clear();

    [[nodiscard]] Size numCells() const noexcept;
    [[nodiscard]] int maxDimension() const;
    [[nodiscard]] std::vector<Index> cellsOfDimension(int dimension) const;
    [[nodiscard]] const Cell &getCell(Index index) const;
    [[nodiscard]] std::vector<Index> getBoundary(const Cell &cell) const;
    [[nodiscard]] std::vector<Index> getCoboundary(const Cell &cell) const;
    [[nodiscard]] std::vector<Index> boundary(const Cell &cell) const;
    [[nodiscard]] std::vector<Index> coboundary(const Cell &cell) const;
    [[nodiscard]] std::vector<std::vector<double>> computeBoundaryMatrix() const;
    [[nodiscard]] std::vector<std::vector<double>> computeCoboundaryMatrix() const;

    [[nodiscard]] errors::ErrorResult<std::vector<int>>
    computeEulerCharacteristic(const core::DeterminismContract &contract = {}) const;
    [[nodiscard]] errors::ErrorResult<std::vector<int>>
    computeBettiNumbers(const core::DeterminismContract &contract = {}) const;

private:
    std::vector<Cell> cells_;
    std::unordered_map<Cell, Index, Cell::Hash> cell_to_index_;
    mutable std::vector<std::vector<Index>> coboundary_cache_;
    mutable bool coboundary_cache_valid_ = false;
    void invalidateCoboundaryCache() const;
    void rebuildCoboundaryCache() const;
};
class CellularChainComplex
{
public:
    CellularChainComplex() = default;
    explicit CellularChainComplex(const CellularComplex &complex);
    std::vector<std::vector<double>> getBoundaryMatrix() const;
    std::vector<std::vector<double>> getCoboundaryMatrix() const;
    errors::ErrorResult<std::vector<std::vector<int>>>
    computeHomology(const core::DeterminismContract &contract = {}) const;
    errors::ErrorResult<std::vector<int>>
    computeBettiNumbers(const core::DeterminismContract &contract = {}) const;

private:
    std::vector<std::vector<double>> boundary_matrix_;
    std::vector<std::vector<double>> coboundary_matrix_;
};
class CWComplex
{
public:
    CWComplex() = default;
    void addSimplices(const std::vector<Simplex> &simplices);
    void addSimplex(const Simplex &simplex);
    Size numSimplices() const;
    int maxDimension() const;
    const Simplex &getSimplex(Index index) const;
    std::vector<Index> getSimplicesOfDimension(int dimension) const;
    std::vector<Index> getStar(const Simplex &simplex) const;
    std::vector<Index> getLink(const Simplex &simplex) const;
    std::vector<int> computeHomology() const;
    std::vector<int> computeBettiNumbers() const;

private:
    std::vector<Simplex> simplices_;
    std::unordered_map<Simplex, Index, Simplex::Hash> simplex_to_index_;
    mutable std::vector<std::vector<Index>> star_cache_;
    mutable std::vector<std::vector<Index>> link_cache_;
    mutable bool caches_valid_ = false;
    void invalidateCaches() const;
    void rebuildCaches() const;
};
} // namespace nerve::algebra

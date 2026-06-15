
#pragma once

#include "nerve/algebra/cellular.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core_types.hpp"

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace nerve::persistence
{

class PersistentCohomology
{
public:
    PersistentCohomology() = default;
    explicit PersistentCohomology(const algebra::CellularComplex &complex);
    std::vector<std::vector<int>> computeCohomology() const;
    std::vector<int> computeCohomologyGroups() const;
    std::vector<int> computeBettiNumbers() const;
    std::vector<Pair> computePersistentCohomology(
        const std::vector<std::pair<algebra::Cell, double>> &filtration) const;
    std::vector<std::vector<double>> computeKernel(int dimension) const;
    std::vector<std::vector<double>> computeImage(int dimension) const;
    std::vector<std::vector<double>> computeCokernel(int dimension) const;
    std::vector<std::vector<double>>
    computeCupProduct(const std::vector<std::vector<double>> &alpha,
                      const std::vector<std::vector<double>> &beta) const;
    void setCoefficientField(int p);
    void setAlgorithm(const std::string &algorithm);
    void setVerbose(bool verbose);

private:
    algebra::CellularComplex complex_;
    int coefficient_field_;
    std::string algorithm_;
    bool verbose_;
    std::vector<std::vector<double>> computeCoboundaryMatrix() const;
    std::vector<std::vector<double>>
    transposeMatrix(const std::vector<std::vector<double>> &matrix) const;
    std::vector<std::vector<double>>
    computeKernelSpace(const std::vector<std::vector<double>> &matrix) const;
};

class DualPersistentHomology
{
public:
    DualPersistentHomology() = default;
    explicit DualPersistentHomology(const algebra::CellularComplex &complex);
    std::vector<Pair>
    computeDualPersistence(const std::vector<std::pair<algebra::Cell, double>> &filtration) const;
    std::vector<std::vector<double>> computeKernelBasis(int dimension) const;
    std::vector<std::vector<double>> computeCokernelBasis(int dimension) const;
    std::vector<std::vector<double>>
    computeDualMap(const std::vector<std::vector<double>> &matrix) const;
    std::vector<std::vector<double>>
    computeAdjointMap(const std::vector<std::vector<double>> &matrix) const;
    std::vector<double> computeSpectrum(int dimension) const;
    std::vector<std::vector<double>> computeEigenvectors(int dimension) const;

private:
    algebra::CellularComplex complex_;
    PersistentCohomology cohomology_;
    std::vector<std::vector<double>> computeLaplacian(int dimension) const;
    std::vector<std::vector<double>> computeDiracOperator() const;
};

class CohomologyRing
{
public:
    CohomologyRing() = default;
    explicit CohomologyRing(const algebra::CellularComplex &complex);
    std::vector<std::vector<double>> cupProduct(const std::vector<std::vector<double>> &alpha,
                                                const std::vector<std::vector<double>> &beta) const;
    std::vector<std::vector<double>> steenrodSquare(const std::vector<std::vector<double>> &alpha,
                                                    int i) const;
    std::vector<std::vector<std::vector<double>>> computeMultiplicationTable() const;
    std::vector<std::vector<double>> computeRingGenerators() const;
    std::vector<int> computePoincarePolynomial() const;
    std::vector<std::vector<int>> computeBettiNumbers() const;

private:
    algebra::CellularComplex complex_;
    std::vector<std::vector<double>> coboundary_matrix_;
    std::vector<std::vector<double>>
    computeCupProductMatrix(const std::vector<std::vector<double>> &alpha,
                            const std::vector<std::vector<double>> &beta) const;
};

class SheafCohomology
{
public:
    SheafCohomology() = default;
    explicit SheafCohomology(const algebra::CellularComplex &complex);
    void assignSection(const algebra::Cell &cell, const std::vector<double> &section);
    void assignCochain(const std::vector<algebra::Cell> &cells, const std::vector<double> &values);
    std::vector<std::vector<double>> computeSheafCohomology() const;
    std::vector<std::vector<double>> computeLocalCohomology(const algebra::Cell &cell) const;
    std::vector<std::vector<double>>
    computeSheafMorphism(const std::vector<std::vector<double>> &sections) const;
    std::vector<double> computeStalk(const algebra::Cell &cell) const;
    std::vector<std::vector<double>> computeGerms(const algebra::Cell &cell) const;

private:
    algebra::CellularComplex complex_;
    std::unordered_map<algebra::Cell, std::vector<double>, algebra::Cell::Hash> sections_;
    std::unordered_map<algebra::Cell, std::vector<double>, algebra::Cell::Hash> cochains_;
    std::vector<std::vector<double>> computeSheafCoboundary() const;
    std::vector<std::vector<double>> computeRestrictionMaps() const;
};

// Detail namespace for internal helpers. Definitions live in
// src/persistence/cohomology/cohomology_rref_ops.cpp.
namespace detail
{
Index findCellIndex(const algebra::CellularComplex &complex, const algebra::Cell &cell);
Size matrixRank(const std::vector<std::vector<double>> &matrix);
std::vector<std::vector<double>> nullspaceBasis(const std::vector<std::vector<double>> &matrix);
} // namespace detail

} // namespace nerve::persistence

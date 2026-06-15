#pragma once

#include "nerve/algebra/cellular.hpp"
#include "nerve/core_types.hpp"

#include <cstddef>
#include <vector>

namespace nerve::persistence
{

// Main cohomology computation class.
class Cohomology
{
public:
    Cohomology() = default;
    explicit Cohomology(const algebra::CellularComplex &complex);

    // Compute cohomology groups.
    std::vector<std::vector<int>> computeCohomologyGroups() const;

    // Compute Betti numbers.
    std::vector<int> computeBettiNumbers() const;

    // Compute cohomology with coefficients.
    std::vector<std::vector<int>> computeCohomologyWithCoefficients(int p) const;

    // Persistent cohomology APIs.
    std::vector<Pair> computePersistentCohomology(
        const std::vector<std::pair<algebra::Cell, double>> &filtration) const;
    std::vector<std::vector<double>> computeKernel(int dimension) const;
    std::vector<std::vector<double>> computeCokernel(int dimension) const;
    std::vector<std::vector<double>> computeLaplacian(int dimension) const;

private:
    algebra::CellularComplex complex_;
};

// Forward declarations - defined in cohomology_ops.hpp.
class DualPersistentHomology;
class CohomologyRing;

} // namespace nerve::persistence

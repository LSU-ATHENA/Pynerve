#pragma once

#include "nerve/algebra/cellular.hpp"
#include "nerve/core_types.hpp"

#include <cstddef>
#include <vector>

namespace nerve::persistence
{

// Involuted homology computation (for persistent cohomology).
class InvolutedHomology
{
public:
    InvolutedHomology() = default;
    explicit InvolutedHomology(const algebra::CellularComplex &complex);

    // Compute involuted homology groups.
    std::vector<std::vector<int>> computeInvolutedHomology() const;

    // Compute involution map.
    std::vector<std::vector<double>> computeInvolutionMap() const;

    // Check if complex has involution.
    bool hasInvolution() const;

private:
    algebra::CellularComplex complex_;
};

} // namespace nerve::persistence

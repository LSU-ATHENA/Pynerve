
#pragma once

#include "nerve/persistence/core/core_types.hpp"

#include <vector>

namespace nerve::persistence
{

/// Betti numbers from barcode infinite bars (standard VR pipeline convention).
std::vector<Size> bettiNumbersFromPairs(const std::vector<Pair> &pairs);

/// Shannon entropy of normalized non-negative weights (0 if degenerate).
double shannonEntropyNormalized(const std::vector<double> &weights);

} // namespace nerve::persistence

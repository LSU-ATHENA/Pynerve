
#pragma once

#include "nerve/algebra/complex.hpp"
#include "nerve/core_types.hpp"
#include "nerve/persistence/core/core_types.hpp"

#include <cstdint>
#include <vector>

namespace nerve::streaming::detail
{

using Pair = persistence::Pair; // Use persistence Pair type

std::vector<Pair> computeWitnessModePairs(const algebra::SimplicialComplex &complex,
                                          Size max_dimension, uint32_t deterministic_seed);

std::vector<Pair> computeSparseModePairs(const algebra::SimplicialComplex &complex,
                                         Size max_dimension, uint32_t deterministic_seed);

} // namespace nerve::streaming::detail

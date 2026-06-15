
#pragma once

#include "nerve/core_types.hpp"
#include "nerve/persistence/core/core_types.hpp"

#include <vector>

namespace nerve::streaming::detail
{

using Pair = persistence::Pair; // Use persistence Pair type

std::vector<Pair> canonicalPairs(const persistence::Diagram &diagram);

double diagramSupDistance(const persistence::Diagram &lhs, const persistence::Diagram &rhs);

} // namespace nerve::streaming::detail

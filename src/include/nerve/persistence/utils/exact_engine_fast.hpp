#pragma once

#include "nerve/persistence/utils/exact_engine.hpp"

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace nerve::persistence
{

ExactPersistenceResult
computeExactCohomologyZ2Fast(int n, int max_dim, double thr,
                             const std::vector<std::vector<int>> &neighbors,
                             const std::unordered_map<std::uint64_t, double> &edge_w);

} // namespace nerve::persistence

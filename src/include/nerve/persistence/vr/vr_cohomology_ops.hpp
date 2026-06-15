#pragma once

#include "nerve/core_types.hpp"
#include "nerve/persistence/utils/exact_engine.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"

#include <vector>

namespace nerve::persistence
{

ExactPersistenceResult computeCohomologyVR(const std::vector<double> &points, Size n, Size dim,
                                           double max_radius);

} // namespace nerve::persistence

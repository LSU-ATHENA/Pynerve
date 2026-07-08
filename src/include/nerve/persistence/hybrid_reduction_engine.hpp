#pragma once

#include "nerve/core_types.hpp"
#include "nerve/persistence/accelerated/heterogeneous_fast_vr.hpp"

namespace nerve::persistence::accelerated
{

errors::ErrorResult<std::vector<Pair>>
reduceHybridExact(core::BufferView<const double>points, size_t point_dim,
                  const HeterogeneousFastVR::Config &config,
                  const core::DeterminismContract &contract);

} // namespace nerve::persistence::accelerated

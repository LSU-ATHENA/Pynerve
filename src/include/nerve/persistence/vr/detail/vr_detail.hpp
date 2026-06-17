#pragma once
#include "nerve/common/accelerated_types.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"

namespace nerve::persistence::accelerated
{
namespace detail
{
double bytesToMb(size_t bytes);
double estimateProblemOps(size_t n_points, size_t point_dim, size_t max_dim);
} // namespace detail
namespace factory
{
errors::ErrorResult<std::unique_ptr<AcceleratedVREngine>>
createAccelerationRuntimeEngine(size_t n_points, size_t point_dim, double max_radius);
errors::ErrorResult<std::unique_ptr<AcceleratedVREngine>>
createProductionEngine(const VRConfig &config);
} // namespace factory
namespace utils
{
size_t estimateMemoryRequirements(size_t n_points, size_t point_dim, size_t max_dim,
                                  const VRConfig &config);
bool isAccelerationBeneficial(size_t n_points, size_t point_dim, double max_radius);
} // namespace utils
} // namespace nerve::persistence::accelerated

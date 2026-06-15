#pragma once

#include "nerve/math/persistence_metrics/point2d.hpp"

#include <functional>
#include <vector>

namespace nerve::gpu::bottleneck
{

void compute_bottleneck_distance_gpu(const std::vector<nerve::math::Point2D> &d1,
                                     const std::vector<nerve::math::Point2D> &d2,
                                     std::function<void(double)> callback);

} // namespace nerve::gpu::bottleneck

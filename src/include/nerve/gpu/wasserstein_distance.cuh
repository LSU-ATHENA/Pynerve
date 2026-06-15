#pragma once

#include "nerve/math/persistence_metrics/point2d.hpp"

#include <functional>
#include <vector>

namespace nerve::gpu::wasserstein
{

void compute_sinkhorn_distance_gpu(const std::vector<nerve::math::Point2D> &d1,
                                   const std::vector<nerve::math::Point2D> &d2, double p,
                                   double epsilon_reg, int max_iterations,
                                   std::function<void(double)> callback);

void compute_auction_distance_gpu(const std::vector<nerve::math::Point2D> &d1,
                                  const std::vector<nerve::math::Point2D> &d2, double p,
                                  std::function<void(double)> callback);

} // namespace nerve::gpu::wasserstein

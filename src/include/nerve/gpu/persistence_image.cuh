#pragma once

#include <functional>
#include <vector>

namespace nerve::gpu::persistence_image
{

void compute_persistence_image_gpu(
    const std::vector<float> &births, const std::vector<float> &deaths, int resolution, float sigma,
    std::function<void(const std::vector<std::vector<double>> &)> callback);

} // namespace nerve::gpu::persistence_image

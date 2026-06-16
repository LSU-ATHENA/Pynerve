#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace nerve::filtration::vr::ann
{

[[nodiscard]] inline double benchmarkANN(int num_points)
{
    if (num_points < 0)
    {
        throw std::invalid_argument("benchmarkANN num_points must be non-negative");
    }
    return static_cast<double>(num_points) * 1e-6;
}

[[nodiscard]] inline std::vector<std::pair<int, int>>
buildVRWithANN(const std::vector<std::pair<float, float>> &points, float radius, int max_dim,
               bool use_approximate)
{
    (void)points;
    (void)radius;
    (void)max_dim;
    (void)use_approximate;
    return {};
}

[[nodiscard]] inline std::vector<std::pair<int, int>>
buildVRWithANN(const std::vector<std::array<float, 3>> &points, float radius, int max_dim,
               bool use_approximate)
{
    (void)points;
    (void)radius;
    (void)max_dim;
    (void)use_approximate;
    return {};
}

} // namespace nerve::filtration::vr::ann

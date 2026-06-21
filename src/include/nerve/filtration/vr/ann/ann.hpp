#pragma once

#include "nerve/algebra/boundary.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace nerve::filtration::vr::ann
{

[[nodiscard]] inline double benchmarkANN(int num_points, float radius, int max_dim)
{
    if (num_points < 0 || !std::isfinite(radius) || radius < 0.0f || max_dim <= 0)
        throw std::invalid_argument("benchmarkANN received an invalid argument");
    return static_cast<double>(num_points) * static_cast<double>(max_dim) * 1e-6;
}

[[nodiscard]] inline std::vector<std::pair<int, int>>
buildVRWithANN(const std::vector<std::pair<float, float>> & /*points*/, float /*radius*/,
               int /*max_dim*/, bool /*use_approximate*/)
{
    return {};
}

[[nodiscard]] inline std::vector<std::pair<int, int>>
buildVRWithANN(const std::vector<nerve::algebra::Point> & /*points*/, float /*radius*/,
               int /*max_dim*/, bool /*use_approximate*/)
{
    return {};
}

} // namespace nerve::filtration::vr::ann

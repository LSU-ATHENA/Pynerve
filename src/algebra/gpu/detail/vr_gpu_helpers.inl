#pragma once

#include "nerve/algebra/complex.hpp"
#include "nerve/errors/errors.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <string_view>
#include <vector>

namespace nerve::gpu::algebra::detail
{

constexpr std::size_t GPU_THRESHOLD = 4096;

struct VRSimplex
{
    std::vector<int> vertices;
    double filtration_value;
    int dimension;
};

bool checkedProduct(std::size_t lhs, std::size_t rhs, std::size_t &out) noexcept
{
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs)
    {
        return false;
    }
    out = lhs * rhs;
    return true;
}

bool checkedByteCount(std::size_t count, std::size_t element_size, std::size_t &out) noexcept
{
    return checkedProduct(count, element_size, out);
}

bool checkedIntSize(std::size_t value, int &out) noexcept
{
    if (value > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        return false;
    }
    out = static_cast<int>(value);
    return true;
}

errors::ErrorResult<void> resourceLimit(std::string_view message)
{
    return errors::ErrorResult<void>::error(errors::ErrorCode::E41_RESOURCE_LIMIT, message);
}

errors::ErrorResult<void> invalidSimplices(std::string_view message)
{
    return errors::ErrorResult<void>::error(errors::ErrorCode::E88_INVALID_SIMPLICES, message);
}

double checkedPointDistance(const std::vector<std::vector<double>> &points, int lhs, int rhs)
{
    double distance_sq = 0.0;
    for (std::size_t d = 0; d < points[static_cast<std::size_t>(lhs)].size(); ++d)
    {
        const double diff =
            points[static_cast<std::size_t>(lhs)][d] - points[static_cast<std::size_t>(rhs)][d];
        const double contribution = diff * diff;
        const double next_distance_sq = distance_sq + contribution;
        if (!std::isfinite(diff) || !std::isfinite(contribution) ||
            !std::isfinite(next_distance_sq))
        {
            return std::numeric_limits<double>::infinity();
        }
        distance_sq = next_distance_sq;
    }
    return std::sqrt(distance_sq);
}

} // namespace nerve::gpu::algebra::detail

#pragma once

#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/types.hpp"

#include <algorithm>
#include <cmath>
#include <tuple>
#include <vector>

namespace nerve::test
{

constexpr double kDefaultTol = 1e-10;

inline nerve::core::BufferView<const double> view_of(const std::vector<double> &v)
{
    return {v.data(), v.size()};
}

inline std::vector<nerve::Pair> canonical(std::vector<nerve::Pair> pairs)
{
    std::sort(pairs.begin(), pairs.end(), [](const nerve::Pair &a, const nerve::Pair &b) {
        return std::tuple(a.dimension, a.birth, a.death) <
               std::tuple(b.dimension, b.birth, b.death);
    });
    return pairs;
}

inline bool pairs_equal(const nerve::Pair &a, const nerve::Pair &b, double tol = kDefaultTol)
{
    if (a.dimension != b.dimension)
        return false;
    if (std::abs(a.birth - b.birth) > tol)
        return false;
    if (a.isInfinite() && b.isInfinite())
        return true;
    if (a.isInfinite() || b.isInfinite())
        return false;
    return std::abs(a.death - b.death) < tol;
}

} // namespace nerve::test

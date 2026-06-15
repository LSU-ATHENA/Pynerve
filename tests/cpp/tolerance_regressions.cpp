#include "math/tolerance.hpp"

#include <cassert>
#include <limits>

int main()
{
    const double points[] = {-2.0, 0.5, 4.0, 1.0};
    const auto valid = nerve::math::Tolerance::estimateScale(points, 2, 2);
    assert(valid.data_scale == 4.0);

    const auto empty = nerve::math::Tolerance::estimateScale(points, 0, 2);
    assert(empty.data_scale == 1.0);

    const auto zero_dim = nerve::math::Tolerance::estimateScale(points, 2, 0);
    assert(zero_dim.data_scale == 1.0);

    const auto null_points = nerve::math::Tolerance::estimateScale(nullptr, 1, 1);
    assert(null_points.data_scale == 1.0);

    const double invalid_points[] = {0.0, std::numeric_limits<double>::infinity()};
    const auto nonfinite = nerve::math::Tolerance::estimateScale(invalid_points, 1, 2);
    assert(nonfinite.data_scale == 1.0);

    const auto overflow = nerve::math::Tolerance::estimateScale(
        points, std::numeric_limits<nerve::Size>::max() / 2 + 1, 3);
    assert(overflow.data_scale == 1.0);

    return 0;
}

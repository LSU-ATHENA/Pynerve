#include "math/precision.hpp"

#include <cassert>
#include <cmath>
#include <limits>
#include <vector>

int main() {
    nerve::math::PrecisionAwareDistance distance;

    const double points[] = {0.0, 0.0, 3.0, 4.0};
    const auto matrix = distance.computeDistanceMatrix(points, 2, 2);
    assert(matrix.isOk());
    assert(matrix.value().size() == 4);
    assert(matrix.value()[1] == 5.0);
    assert(matrix.value()[2] == 5.0);

    const auto empty = distance.computeDistanceMatrix(points, 0, 2);
    assert(empty.isErr());

    const auto zero_dim = distance.computeDistanceMatrix(points, 2, 0);
    assert(zero_dim.isErr());

    const auto null_points = distance.computeDistanceMatrix(nullptr, 2, 2);
    assert(null_points.isErr());

    const double invalid_points[] = {0.0, std::numeric_limits<double>::quiet_NaN()};
    const auto nonfinite = distance.computeDistanceMatrix(invalid_points, 1, 2);
    assert(nonfinite.isErr());

    const auto coordinate_overflow = distance.computeDistanceMatrix(
        points, std::numeric_limits<nerve::Size>::max() / 2 + 1, 3);
    assert(coordinate_overflow.isErr());

    const auto matrix_overflow = distance.computeDistanceMatrix(
        points, std::numeric_limits<nerve::Size>::max() / 2 + 1, 1);
    assert(matrix_overflow.isErr());

    const nerve::Size matrix_capacity = std::vector<double>().max_size();
    nerve::Size oversized_matrix_points =
        static_cast<nerve::Size>(std::sqrt(static_cast<long double>(matrix_capacity))) + 1;
    while (oversized_matrix_points <= matrix_capacity / oversized_matrix_points) {
        ++oversized_matrix_points;
    }
    const auto matrix_capacity_limit =
        distance.computeDistanceMatrix(points, oversized_matrix_points, 1);
    assert(matrix_capacity_limit.isErr());

    return 0;
}

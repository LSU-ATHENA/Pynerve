#include "distance/simd_distance.hpp"
#include "nerve/metrics/lazy_distance.hpp"

#include <cassert>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace
{

nerve::Size compressedSideAboveVectorCapacity()
{
    const nerve::Size capacity = std::vector<double>().max_size();
    nerve::Size side =
        static_cast<nerve::Size>(
            (1.0L + std::sqrt(1.0L + 8.0L * static_cast<long double>(capacity))) / 2.0L) +
        1;
    while (side > 1 && side <= std::numeric_limits<nerve::Size>::max() / (side - 1) &&
           (side * (side - 1)) / 2 <= capacity)
    {
        ++side;
    }
    return side;
}

} // namespace

int main()
{
    nerve::distance::DistanceMatrix packed(3);
    assert(packed.size() == 3);
    packed(0, 2) = 2.5;
    assert(packed(2, 0) == 2.5);
    assert(packed(0, 1) == 0.0);

    bool rejected_capacity = false;
    try
    {
        nerve::distance::DistanceMatrix too_large(compressedSideAboveVectorCapacity());
    }
    catch (const std::length_error &)
    {
        rejected_capacity = true;
    }
    assert(rejected_capacity);

    nerve::distance::DistanceComputer computer;
    const double points[] = {0.0, 0.0, 3.0, 4.0};
    const auto matrix = computer.buildMatrix(points, 2, 2);
    assert(matrix.size() == 2);
    assert(matrix(0, 1) == 5.0);

    const double invalid_points[] = {0.0, std::numeric_limits<double>::infinity()};
    bool rejected_nonfinite = false;
    try
    {
        (void)computer.buildMatrix(invalid_points, 1, 2);
    }
    catch (const std::invalid_argument &)
    {
        rejected_nonfinite = true;
    }
    assert(rejected_nonfinite);

    const double overflow_points[] = {0.0, std::numeric_limits<double>::max()};
    bool rejected_distance_overflow = false;
    try
    {
        (void)computer.buildMatrix(overflow_points, 2, 1);
    }
    catch (const std::overflow_error &)
    {
        rejected_distance_overflow = true;
    }
    assert(rejected_distance_overflow);

    bool rejected_coordinate_overflow = false;
    try
    {
        (void)computer.buildMatrix(points, 2, std::numeric_limits<nerve::Size>::max() / 2 + 1);
    }
    catch (const std::length_error &)
    {
        rejected_coordinate_overflow = true;
    }
    assert(rejected_coordinate_overflow);

    const double lazy_points[] = {0.0, 0.0, 3.0, 4.0};
    nerve::metrics::lazy::LazyDistanceMatrix lazy_matrix(std::span<const double>(lazy_points, 4), 2,
                                                         2);
    assert(lazy_matrix.getDistance(0, 1) == 5.0);
    assert(lazy_matrix.isWithinRadius(0, 1, 5.0));

    bool rejected_lazy_nonfinite = false;
    try
    {
        const double invalid_lazy_points[] = {0.0, std::numeric_limits<double>::quiet_NaN()};
        nerve::metrics::lazy::LazyDistanceMatrix invalid_lazy(
            std::span<const double>(invalid_lazy_points, 2), 1, 2);
    }
    catch (const std::invalid_argument &)
    {
        rejected_lazy_nonfinite = true;
    }
    assert(rejected_lazy_nonfinite);

    bool rejected_lazy_overflow = false;
    try
    {
        nerve::metrics::lazy::LazyDistanceMatrix overflow_lazy(
            std::span<const double>(overflow_points, 2), 2, 1);
        (void)overflow_lazy.getDistance(0, 1);
    }
    catch (const std::overflow_error &)
    {
        rejected_lazy_overflow = true;
    }
    assert(rejected_lazy_overflow);

    bool rejected_lazy_radius = false;
    try
    {
        (void)lazy_matrix.isWithinRadius(0, 1, std::numeric_limits<double>::quiet_NaN());
    }
    catch (const std::invalid_argument &)
    {
        rejected_lazy_radius = true;
    }
    assert(rejected_lazy_radius);

    bool rejected_lazy_cosine_zero = false;
    try
    {
        const double cosine_points[] = {0.0, 0.0, 1.0, 0.0};
        nerve::metrics::lazy::LazyDistanceMatrix cosine_lazy(
            std::span<const double>(cosine_points, 4), 2, 2, "cosine");
        (void)cosine_lazy.getDistance(0, 1);
    }
    catch (const std::invalid_argument &)
    {
        rejected_lazy_cosine_zero = true;
    }
    assert(rejected_lazy_cosine_zero);

    nerve::metrics::lazy::SparseDistanceMatrix sparse_matrix(
        std::span<const double>(lazy_points, 4), 2, 2, 5.0);
    assert(sparse_matrix.isEdge(0, 1));
    assert(sparse_matrix.getDistance(0, 1) == 5.0);
    bool rejected_sparse_index = false;
    try
    {
        (void)sparse_matrix.getDistance(2, 2);
    }
    catch (const std::out_of_range &)
    {
        rejected_sparse_index = true;
    }
    assert(rejected_sparse_index);

    bool rejected_sparse_threshold = false;
    try
    {
        nerve::metrics::lazy::SparseDistanceMatrix invalid_sparse_threshold(
            std::span<const double>(lazy_points, 4), 2, 2,
            std::numeric_limits<double>::quiet_NaN());
    }
    catch (const std::invalid_argument &)
    {
        rejected_sparse_threshold = true;
    }
    assert(rejected_sparse_threshold);

    bool rejected_sparse_overflow = false;
    try
    {
        nerve::metrics::lazy::SparseDistanceMatrix overflow_sparse(
            std::span<const double>(overflow_points, 2), 2, 1, 1.0);
    }
    catch (const std::overflow_error &)
    {
        rejected_sparse_overflow = true;
    }
    assert(rejected_sparse_overflow);

    bool rejected_sparse_metric = false;
    try
    {
        nerve::metrics::lazy::SparseDistanceMatrix invalid_sparse_metric(
            std::span<const double>(lazy_points, 4), 2, 2, 1.0, "invalid");
    }
    catch (const std::invalid_argument &)
    {
        rejected_sparse_metric = true;
    }
    assert(rejected_sparse_metric);

    const auto lazy_vr =
        nerve::metrics::lazy::buildVRLazy(std::span<const double>(lazy_points, 4), 2, 2, 5.0, 1);
    assert(lazy_vr.size() == 3);

    bool rejected_lazy_vr_radius = false;
    try
    {
        (void)nerve::metrics::lazy::buildVRLazy(std::span<const double>(lazy_points, 4), 2, 2,
                                                std::numeric_limits<double>::quiet_NaN(), 1);
    }
    catch (const std::invalid_argument &)
    {
        rejected_lazy_vr_radius = true;
    }
    assert(rejected_lazy_vr_radius);

    bool rejected_lazy_vr_dimension = false;
    try
    {
        (void)nerve::metrics::lazy::buildVRLazy(std::span<const double>(lazy_points, 4), 2, 2, 1.0,
                                                -1);
    }
    catch (const std::invalid_argument &)
    {
        rejected_lazy_vr_dimension = true;
    }
    assert(rejected_lazy_vr_dimension);

    bool rejected_lazy_vr_overflow = false;
    try
    {
        (void)nerve::metrics::lazy::buildVRLazy(std::span<const double>(overflow_points, 2), 2, 1,
                                                1.0, 1);
    }
    catch (const std::overflow_error &)
    {
        rejected_lazy_vr_overflow = true;
    }
    assert(rejected_lazy_vr_overflow);

    return 0;
}

#include "math/persistence_metrics/algorithms.hpp"
#include "metrics/bottleneck_exact.hpp"

#include <cassert>
#include <limits>
#include <vector>

int main()
{
    using nerve::math::PersistenceMetrics;
    using nerve::math::Point2D;

    const std::vector<Point2D> first{{0.0, 1.0}};
    const std::vector<Point2D> second{{0.0, 2.0}};

    const auto bottleneck = PersistenceMetrics::bottleneckDistance(first, second);
    assert(bottleneck.isOk());
    assert(bottleneck.value() == 1.0);

    const std::vector<Point2D> invalid_points{{0.0, std::numeric_limits<double>::quiet_NaN()}};
    const auto invalid_bottleneck = PersistenceMetrics::bottleneckDistance(invalid_points, second);
    assert(invalid_bottleneck.isErr());
    assert(invalid_bottleneck.error().value() ==
           static_cast<int>(nerve::error::TDAErrorCode::NaNInInput));

    const auto invalid_wasserstein_points =
        PersistenceMetrics::wassersteinDistance(first, invalid_points);
    assert(invalid_wasserstein_points.isErr());
    assert(invalid_wasserstein_points.error().value() ==
           static_cast<int>(nerve::error::TDAErrorCode::NaNInInput));

    const auto invalid_wasserstein_power = PersistenceMetrics::wassersteinDistance(
        first, second, std::numeric_limits<double>::quiet_NaN());
    assert(invalid_wasserstein_power.isErr());

    const std::vector<Point2D> overflow_points{
        {-std::numeric_limits<double>::max(), std::numeric_limits<double>::max()}};
    const auto overflow_bottleneck = PersistenceMetrics::bottleneckDistance(overflow_points, {});
    assert(overflow_bottleneck.isErr());

    const auto extracted = PersistenceMetrics::extractPointsFromDiagram(
        {{std::numeric_limits<double>::quiet_NaN(), 1.0},
         {2.0, std::numeric_limits<double>::infinity()},
         {3.0, 4.0}});
    assert(extracted.size() == 2);
    assert(extracted[0].x == 2.0);
    assert(extracted[0].y == 2.0);
    assert(extracted[1].x == 3.0);
    assert(extracted[1].y == 4.0);

    using nerve::metrics::BottleneckDistance;
    using nerve::metrics::Diagram;
    using nerve::metrics::WassersteinDistance;

    const Diagram exact_first{{0.0, 1.0, 0}};
    const Diagram exact_second{{0.0, 2.0, 0}};
    const auto exact_bottleneck = BottleneckDistance::compute(exact_first, exact_second);
    assert(exact_bottleneck.isOk());
    assert(exact_bottleneck.value() == 1.0);

    const Diagram essential_first{{0.0, std::numeric_limits<double>::infinity(), 0}};
    const Diagram essential_second{{1.0, std::numeric_limits<double>::infinity(), 0}};
    const auto essential_bottleneck =
        BottleneckDistance::compute(essential_first, essential_second);
    assert(essential_bottleneck.isOk());
    assert(essential_bottleneck.value() == 0.0);

    const Diagram invalid_exact{
        {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), 0}};
    assert(BottleneckDistance::compute(invalid_exact, {}).isErr());

    const Diagram exact_overflow{
        {-std::numeric_limits<double>::max(), std::numeric_limits<double>::max(), 0}};
    assert(BottleneckDistance::compute(exact_overflow, {}).isErr());
    assert(WassersteinDistance::compute(exact_overflow, {}).isErr());

    return 0;
}

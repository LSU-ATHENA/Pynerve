#include "nerve/persistence/utils/incremental_updates.hpp"
#include "nerve/persistence/utils/incremental_updates_internal.hpp"

#include <cassert>
#include <limits>

int main() {
    using namespace nerve;
    using namespace nerve::persistence;

    auto complex = buildIncrementalComplex(PointCloud{{0.0}, {1.0}, {2.0}}, 2);
    assert(complex.ok());
    assert(complex.value().numSimplices() > 0);

    auto invalid_point = buildIncrementalComplex(
        PointCloud{{0.0}, {std::numeric_limits<double>::quiet_NaN()}}, 1);
    assert(!invalid_point.ok());
    assert(invalid_point.errorCode() == ErrorCode::E54_PH4_INVALID_INPUT);

    auto overflow_distance =
        buildIncrementalComplex(PointCloud{{0.0}, {std::numeric_limits<double>::max()}}, 1);
    assert(!overflow_distance.ok());
    assert(overflow_distance.errorCode() == ErrorCode::E54_PH4_INVALID_INPUT);

    IncrementalPersistence incremental;
    auto propagated =
        incremental.incrementalAddPoint(PointCloud{{0.0}}, {std::numeric_limits<double>::max()}, 1);
    assert(!propagated.ok());
    assert(propagated.errorCode() == ErrorCode::E54_PH4_INVALID_INPUT);

    return 0;
}

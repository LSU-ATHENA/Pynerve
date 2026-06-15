#include "nerve/persistence/streaming/streaming_reducer.hpp"

#include <cassert>
#include <limits>
#include <span>
#include <vector>

namespace {

nerve::persistence::streaming::StreamingColumnGenerator makeGenerator(
    const std::vector<double>& points) {
    return nerve::persistence::streaming::StreamingColumnGenerator(
        std::span<const double>(points.data(), points.size()), 2, 2, 2.0);
}

}  // namespace

int main() {
    const std::vector<double> points{0.0, 0.0, 1.0, 0.0};
    auto valid = makeGenerator(points);
    assert(valid.getNumSimplices() > 0);

    const std::vector<double> nonfinite{
        0.0, 0.0, std::numeric_limits<double>::quiet_NaN(), 0.0};
    auto rejected_nonfinite = makeGenerator(nonfinite);
    assert(rejected_nonfinite.getNumSimplices() == 0);

    const std::vector<double> overflow_prone{
        0.0, 0.0, std::numeric_limits<double>::max(), 0.0};
    auto rejected_overflow = makeGenerator(overflow_prone);
    assert(rejected_overflow.getNumSimplices() == 0);

    return 0;
}

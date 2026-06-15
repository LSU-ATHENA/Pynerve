#include "nerve/persistence/vr/vr_fast_ops.hpp"

#include <cassert>
#include <cstddef>
#include <vector>

int main()
{
    // Verify the old computeVrPersistence entry point still links and runs.
    constexpr std::size_t kNumPoints = 3;
    constexpr std::size_t kPointDim = 2;
    const std::vector<double> points = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0};
    nerve::common::VRConfig config;
    config.max_dim = 0;
    config.max_radius = 10.0;

    const auto result =
        nerve::persistence::computeVrPersistence(points.data(), kNumPoints, kPointDim, config);
    assert(!result.empty());
    return 0;
}

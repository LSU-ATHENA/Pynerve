#include "nerve/common/accelerated_types.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/persistence/core/core_types.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <vector>

namespace
{

nerve::common::VRConfig makeConfig(nerve::common::VRAlgorithmSelection algo)
{
    nerve::common::VRConfig cfg;
    cfg.algorithm = algo;
    cfg.max_dim = 1;
    cfg.max_radius = 2.0;
    return cfg;
}

nerve::core::BufferView<const double> viewOf(const std::vector<double> &v)
{
    return nerve::core::BufferView<const double>(v.data(), v.size());
}

} // namespace

TEST(AdaptiveAccelerationPerformanceTest, ComputeVrPersistenceFastUnderThreshold)
{
    const std::vector<double> points{0.0, 0.0, 1.0, 0.0, 0.5, 0.866, 1.0, 1.0, 0.0, 1.0};
    constexpr std::size_t dim = 2;

    const auto start = std::chrono::steady_clock::now();
    const auto result = nerve::persistence::computeVrPersistenceFastResult(
        viewOf(points), dim, makeConfig(nerve::common::VRAlgorithmSelection::EXACT_STANDARD));
    const auto end = std::chrono::steady_clock::now();

    ASSERT_TRUE(result.isSuccess());
    const auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    EXPECT_LT(elapsed_ms, 5000);
    EXPECT_FALSE(result.value().empty());
}

TEST(AdaptiveAccelerationPerformanceTest, ExactAlgorithmCompletes)
{
    const std::vector<double> points{0.0, 0.0, 1.0, 0.0, 0.0, 1.0};
    constexpr std::size_t dim = 2;

    const auto result = nerve::persistence::computeVrPersistenceFastResult(
        viewOf(points), dim, makeConfig(nerve::common::VRAlgorithmSelection::EXACT_STANDARD));
    EXPECT_TRUE(result.isSuccess());
    if (result.isSuccess())
    {
        EXPECT_FALSE(result.value().empty());
    }
}

TEST(AdaptiveAccelerationPerformanceTest, FastSimdAlgorithmCompletes)
{
    const std::vector<double> points{0.0, 0.0, 1.0, 0.0, 0.0, 1.0};
    constexpr std::size_t dim = 2;

    const auto result = nerve::persistence::computeVrPersistenceFastResult(
        viewOf(points), dim, makeConfig(nerve::common::VRAlgorithmSelection::FAST_SIMD));
    EXPECT_TRUE(result.isSuccess());
    if (result.isSuccess())
    {
        EXPECT_FALSE(result.value().empty());
    }
}

TEST(AdaptiveAccelerationPerformanceTest, AutoAlgorithmCompletes)
{
    const std::vector<double> points{0.0, 0.0, 1.0, 0.0, 0.0, 1.0};
    constexpr std::size_t dim = 2;

    const auto result = nerve::persistence::computeVrPersistenceFastResult(
        viewOf(points), dim, makeConfig(nerve::common::VRAlgorithmSelection::AUTO));
    EXPECT_TRUE(result.isSuccess());
    if (result.isSuccess())
    {
        EXPECT_FALSE(result.value().empty());
    }
}

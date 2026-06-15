#include "nerve/common/accelerated_types.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/errors/errors.hpp"
#include "nerve/persistence/adaptive_acceleration/adaptive_acceleration_engine.hpp"
#include "nerve/persistence/core/core_types.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <vector>

namespace
{

nerve::common::VRConfig testConfig()
{
    nerve::common::VRConfig cfg;
    cfg.algorithm = nerve::common::VRAlgorithmSelection::EXACT_STANDARD;
    cfg.max_dim = 1;
    cfg.max_radius = 2.0;
    return cfg;
}

nerve::core::BufferView<const double> viewOf(const std::vector<double> &v)
{
    return nerve::core::BufferView<const double>(v.data(), v.size());
}

} // namespace

TEST(SOTAMemoryTest, RepeatedEngineCreateDestroy)
{
    for (int i = 0; i < 10; ++i)
    {
        auto engine =
            nerve::persistence::adaptive_acceleration::AdaptiveAccelerationVrEngine::create(
                testConfig());
        EXPECT_TRUE(engine.isSuccess());
    }
}

TEST(SOTAMemoryTest, RepeatedComputeCallsNoLeak)
{
    const std::vector<double> points{0.0, 0.0, 1.0, 0.0, 0.5, 0.866};
    constexpr std::size_t dim = 2;

    auto engine = nerve::persistence::adaptive_acceleration::AdaptiveAccelerationVrEngine::create(
        testConfig());
    ASSERT_TRUE(engine.isSuccess());

    std::vector<double> local_points = points;
    for (int i = 0; i < 5; ++i)
    {
        auto result = engine.value()->computeVrPersistence(local_points, dim, testConfig());
        EXPECT_TRUE(result.isSuccess());
        local_points.push_back(static_cast<double>(i));
        local_points.push_back(static_cast<double>(i + 1));
    }
}

TEST(SOTAMemoryTest, LargeAllocationHandled)
{
    constexpr std::size_t n_points = 8;
    constexpr std::size_t dim = 3;
    std::vector<double> large_points(n_points * dim, 1.0);

    const auto result =
        nerve::persistence::computeVrPersistenceFastResult(viewOf(large_points), dim, testConfig());
    EXPECT_TRUE(result.isSuccess());
}

TEST(SOTAMemoryTest, EngineDestructorReleasesResources)
{
    auto engine = nerve::persistence::adaptive_acceleration::AdaptiveAccelerationVrEngine::create(
        testConfig());
    ASSERT_TRUE(engine.isSuccess());

    {
        std::vector<double> pts{0.0, 0.0, 1.0, 0.0, 0.5, 0.866};
        auto res = engine.value()->computeVrPersistence(pts, 2, testConfig());
        EXPECT_TRUE(res.isSuccess());
    }

    engine.value()->getPerformanceStats();

    engine.value().reset();
    EXPECT_EQ(engine.value(), nullptr);
}

TEST(SOTAMemoryTest, FactoryProducesValidEngines)
{
    auto optimal = nerve::persistence::adaptive_acceleration::AdaptiveAccelerationEngineFactory::
        createOptimal();
    EXPECT_TRUE(optimal.isSuccess());

    const std::string use_case = "persistent_homology";
    auto for_use = nerve::persistence::adaptive_acceleration::AdaptiveAccelerationEngineFactory::
        createForUseCase(use_case);
    EXPECT_TRUE(for_use.isSuccess());

    auto custom = nerve::persistence::adaptive_acceleration::AdaptiveAccelerationEngineFactory::
        createWithConfig(testConfig());
    EXPECT_TRUE(custom.isSuccess());
}

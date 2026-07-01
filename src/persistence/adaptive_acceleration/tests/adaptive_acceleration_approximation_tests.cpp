#include "nerve/common/accelerated_types.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/persistence/adaptive_acceleration/adaptive_algorithm_selector.hpp"
#include "nerve/persistence/adaptive_acceleration/streaming/approximate_processor.hpp"
#include "nerve/persistence/core/core_types.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>

namespace
{

nerve::common::VRConfig exactVRConfig()
{
    nerve::common::VRConfig cfg;
    cfg.algorithm = nerve::common::VRAlgorithmSelection::EXACT_STANDARD;
    cfg.max_dim = 1;
    cfg.max_radius = 3.0;
    return cfg;
}

nerve::core::BufferView<const double> viewOf(const std::vector<double> &v)
{
    return nerve::core::BufferView<const double>(v.data(), v.size());
}

} // namespace

#ifdef NERVE_ENABLE_APPROXIMATION

TEST(SOTAApproximationTest, ApproximateWithinEpsilon)
{
    const std::vector<double> points{0.0, 0.0, 1.0, 0.0, 0.5, 0.866, 1.0, 1.0, 0.0, 1.0};
    constexpr std::size_t dim = 2;
    constexpr double epsilon = 0.5;

    nerve::persistence::adaptive_acceleration::approximation::ApproximationConfig aconfig;
    aconfig.level = nerve::persistence::adaptive_acceleration::approximation::ApproximationLevel::
        HIGH_PRECISION;
    aconfig.max_error = epsilon;
    aconfig.enable_progressive_refinement = false;

    auto processor =
        nerve::persistence::adaptive_acceleration::approximation::ApproximateProcessor::create(
            aconfig);
    ASSERT_TRUE(processor.isSuccess());

    auto approx_result =
        processor.value()->computeApproximate(viewOf(points), dim,
                                              nerve::persistence::adaptive_acceleration::
                                                  approximation::ApproximationLevel::HIGH_PRECISION,
                                              exactVRConfig());
    EXPECT_TRUE(approx_result.isSuccess());

    const auto exact_result =
        nerve::persistence::computeVrPersistenceFastResult(viewOf(points), dim, exactVRConfig());
    ASSERT_TRUE(exact_result.isSuccess());

    const auto &approx_pairs = approx_result.value();
    const auto &exact_pairs = exact_result.value();
    EXPECT_EQ(approx_pairs.size(), exact_pairs.size());

    for (std::size_t i = 0; i < std::min(approx_pairs.size(), exact_pairs.size()); ++i)
    {
        EXPECT_NEAR(approx_pairs[i].birth, exact_pairs[i].birth, epsilon);
        EXPECT_NEAR(approx_pairs[i].death, exact_pairs[i].death, epsilon);
    }

    const auto &stats = processor.value()->getApproximationStats();
    EXPECT_GE(stats.speedup_factor, 0.0);
    EXPECT_GE(stats.approximation_error, 0.0);
}

TEST(SOTAApproximationTest, DistilledVrApproximation)
{
    const std::vector<double> points{0.0, 0.0, 1.0, 0.0, 0.5, 0.866};
    constexpr std::size_t dim = 2;

    nerve::persistence::adaptive_acceleration::approximation::ApproximationConfig aconfig;
    aconfig.level = nerve::persistence::adaptive_acceleration::approximation::ApproximationLevel::
        MEDIUM_PRECISION;
    aconfig.max_error = 0.5;

    auto processor =
        nerve::persistence::adaptive_acceleration::approximation::ApproximateProcessor::create(
            aconfig);
    ASSERT_TRUE(processor.isSuccess());

    auto approx_result = processor.value()->computeApproximate(
        viewOf(points), dim,
        nerve::persistence::adaptive_acceleration::approximation::ApproximationLevel::
            MEDIUM_PRECISION,
        exactVRConfig());
    EXPECT_TRUE(approx_result.isSuccess());

    if (approx_result.isSuccess())
    {
        EXPECT_FALSE(approx_result.value().empty());
        const auto &stats = processor.value()->getApproximationStats();
        EXPECT_GE(stats.approximation_error, 0.0);
        EXPECT_TRUE(stats.converged || !aconfig.enable_progressive_refinement);
    }
}

TEST(SOTAApproximationTest, ApproximateFactory)
{
    nerve::persistence::adaptive_acceleration::approximation::ApproximationConfig aconfig;
    aconfig.level =
        nerve::persistence::adaptive_acceleration::approximation::ApproximationLevel::EXACT;

    auto exact_proc =
        nerve::persistence::adaptive_acceleration::approximation::ApproximateProcessorFactory::
            createForLevel(nerve::persistence::adaptive_acceleration::approximation::
                               ApproximationLevel::HIGH_PRECISION,
                           aconfig);
    EXPECT_TRUE(exact_proc.isSuccess());

    auto exploratory = nerve::persistence::adaptive_acceleration::approximation::
        ApproximateProcessorFactory::createForExploratory(aconfig);
    EXPECT_TRUE(exploratory.isSuccess());

    auto high_prec = nerve::persistence::adaptive_acceleration::approximation::
        ApproximateProcessorFactory::createForHighPrecision(aconfig);
    EXPECT_TRUE(high_prec.isSuccess());
}

TEST(SOTAApproximationTest, ErrorBoundsCalculator)
{
    const std::vector<double> points{0.0, 0.0, 1.0, 0.0, 0.5, 0.866};
    constexpr std::size_t dim = 2;

    const auto exact_result =
        nerve::persistence::computeVrPersistenceFastResult(viewOf(points), dim, exactVRConfig());
    ASSERT_TRUE(exact_result.isSuccess());

    std::vector<nerve::persistence::Pair> empty_pairs;
    auto bounds = nerve::persistence::adaptive_acceleration::approximation::ErrorBoundsCalculator::
        calculateErrorBounds(
            exact_result.value(), empty_pairs,
            nerve::persistence::adaptive_acceleration::approximation::ApproximationLevel::EXACT);
    EXPECT_TRUE(bounds.isSuccess());
}

#else

TEST(SOTAApproximationTest, ApproximationNotEnabled)
{
    GTEST_SKIP() << "NERVE_ENABLE_APPROXIMATION is not defined";
}

#endif

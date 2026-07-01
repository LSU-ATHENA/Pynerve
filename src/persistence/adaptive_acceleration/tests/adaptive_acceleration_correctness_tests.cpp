#include "nerve/common/accelerated_types.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/errors/errors.hpp"
#include "nerve/persistence/core/core_types.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <limits>
#include <vector>

namespace
{

nerve::common::VRConfig testConfig()
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

TEST(SOTACorrectnessTest, BirthLessThanDeathForAllPairs)
{
    const std::vector<double> points{0.0, 0.0, 1.0, 0.0, 0.5, 0.866, 1.0, 1.0, 0.0, 1.0, 0.3, 0.7};
    constexpr std::size_t dim = 2;

    const auto result =
        nerve::persistence::computeVrPersistenceFastResult(viewOf(points), dim, testConfig());
    ASSERT_TRUE(result.isSuccess());

    for (const auto &pair : result.value())
    {
        EXPECT_LT(pair.birth, pair.death) << "birth must be strictly less than death";
    }
}

TEST(SOTACorrectnessTest, NonNegativePersistence)
{
    const std::vector<double> points{0.0, 0.0, 1.0, 0.0, 0.5, 0.866, 1.0, 1.0, 0.0, 1.0};
    constexpr std::size_t dim = 2;

    const auto result =
        nerve::persistence::computeVrPersistenceFastResult(viewOf(points), dim, testConfig());
    ASSERT_TRUE(result.isSuccess());

    for (const auto &pair : result.value())
    {
        EXPECT_GE(pair.lifetime(), 0.0);
    }
}

TEST(SOTACorrectnessTest, EmptyInput)
{
    const std::vector<double> points{};
    constexpr std::size_t dim = 2;

    const auto result =
        nerve::persistence::computeVrPersistenceFastResult(viewOf(points), dim, testConfig());
    EXPECT_TRUE(result.isError());
    EXPECT_NE(result.errorCode(), nerve::errors::ErrorCode::SUCCESS);

    const auto emptyResult =
        nerve::persistence::computeVrPersistenceFast(viewOf(points), dim, testConfig());
    EXPECT_TRUE(emptyResult.empty());
}

TEST(SOTACorrectnessTest, SinglePoint)
{
    const std::vector<double> points{0.0, 0.0};
    constexpr std::size_t dim = 2;

    const auto result =
        nerve::persistence::computeVrPersistenceFastResult(viewOf(points), dim, testConfig());
    ASSERT_TRUE(result.isSuccess());
    EXPECT_TRUE(result.value().empty());
}

TEST(SOTACorrectnessTest, TwoPoints)
{
    const std::vector<double> points{0.0, 0.0, 1.0, 0.0};
    constexpr std::size_t dim = 2;

    const auto result =
        nerve::persistence::computeVrPersistenceFastResult(viewOf(points), dim, testConfig());
    ASSERT_TRUE(result.isSuccess());
    EXPECT_FALSE(result.value().empty());
    for (const auto &pair : result.value())
    {
        EXPECT_LE(pair.birth, pair.death);
        EXPECT_GE(pair.dimension, 0);
    }
}

TEST(SOTACorrectnessTest, AllPairsHaveValidDimension)
{
    const std::vector<double> points{0.0, 0.0, 1.0, 0.0, 0.5, 0.866, 1.0,
                                     1.0, 0.0, 1.0, 0.2, 0.3, 0.8,   0.9};
    constexpr std::size_t dim = 2;

    const auto result =
        nerve::persistence::computeVrPersistenceFastResult(viewOf(points), dim, testConfig());
    ASSERT_TRUE(result.isSuccess());

    for (const auto &pair : result.value())
    {
        EXPECT_GE(pair.dimension, 0);
        EXPECT_LE(pair.dimension, 1);
    }
}

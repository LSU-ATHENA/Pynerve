#include "nerve/common/accelerated_types.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/persistence/core/core_types.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
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

bool pairsApproxEqual(const std::vector<nerve::persistence::Pair> &a,
                      const std::vector<nerve::persistence::Pair> &b, double eps)
{
    if (a.size() != b.size())
        return false;
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        if (std::abs(a[i].birth - b[i].birth) > eps)
            return false;
        if (std::abs(a[i].death - b[i].death) > eps)
            return false;
        if (a[i].dimension != b[i].dimension)
            return false;
    }
    return true;
}

} // namespace

TEST(SOTAIntegrationTest, AutoAlgorithmProducesCorrectOutput)
{
    const std::vector<double> points{0.0, 0.0, 1.0, 0.0, 0.5, 0.866, 0.0, 1.0};
    constexpr std::size_t dim = 2;

    const auto result = nerve::persistence::computeVrPersistenceFastResult(
        viewOf(points), dim, makeConfig(nerve::common::VRAlgorithmSelection::AUTO));

    ASSERT_TRUE(result.isSuccess());
    EXPECT_FALSE(result.value().empty());

    for (const auto &pair : result.value())
    {
        EXPECT_LE(pair.birth, pair.death);
        EXPECT_GE(pair.dimension, 0);
    }
}

TEST(SOTAIntegrationTest, CrossValidateExactVsFast)
{
    const std::vector<double> points{0.0, 0.0, 1.0, 0.0, 0.5, 0.866};
    constexpr std::size_t dim = 2;

    const auto exact = nerve::persistence::computeVrPersistenceFastResult(
        viewOf(points), dim, makeConfig(nerve::common::VRAlgorithmSelection::EXACT_STANDARD));
    const auto fast = nerve::persistence::computeVrPersistenceFastResult(
        viewOf(points), dim, makeConfig(nerve::common::VRAlgorithmSelection::FAST_SIMD));

    ASSERT_TRUE(exact.isSuccess());
    ASSERT_TRUE(fast.isSuccess());
    EXPECT_EQ(exact.value().size(), fast.value().size());
}

TEST(SOTAIntegrationTest, CrossValidateExactVsAuto)
{
    const std::vector<double> points{0.0, 0.0, 1.0, 0.0, 0.5, 0.866, 1.0, 1.0, 0.0, 1.0};
    constexpr std::size_t dim = 2;

    const auto exact = nerve::persistence::computeVrPersistenceFastResult(
        viewOf(points), dim, makeConfig(nerve::common::VRAlgorithmSelection::EXACT_STANDARD));
    const auto automatic = nerve::persistence::computeVrPersistenceFastResult(
        viewOf(points), dim, makeConfig(nerve::common::VRAlgorithmSelection::AUTO));

    ASSERT_TRUE(exact.isSuccess());
    ASSERT_TRUE(automatic.isSuccess());

    EXPECT_TRUE(pairsApproxEqual(exact.value(), automatic.value(), 1e-10))
        << "EXACT and AUTO results differ beyond tolerance";
}

TEST(SOTAIntegrationTest, CrossValidateFastVsAuto)
{
    const std::vector<double> points{0.0, 0.0, 1.0, 0.0, 0.5, 0.866, 1.0, 1.0, 0.0, 1.0, 0.2, 0.3};
    constexpr std::size_t dim = 2;

    const auto fast = nerve::persistence::computeVrPersistenceFastResult(
        viewOf(points), dim, makeConfig(nerve::common::VRAlgorithmSelection::FAST_SIMD));
    const auto automatic = nerve::persistence::computeVrPersistenceFastResult(
        viewOf(points), dim, makeConfig(nerve::common::VRAlgorithmSelection::AUTO));

    ASSERT_TRUE(fast.isSuccess());
    ASSERT_TRUE(automatic.isSuccess());
    EXPECT_TRUE(pairsApproxEqual(fast.value(), automatic.value(), 1e-10))
        << "FAST_SIMD and AUTO results differ beyond tolerance";
}

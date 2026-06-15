#include "nerve/persistence/vr/vr_fast_ops.hpp"
#include "nerve/persistence/vr/vr_fast_simd_ops.hpp"
#include "nerve/persistence/vr/vr_large_witness_ops.hpp"
#include "nerve/persistence/vr/vr_medium_hybrid_ops.hpp"

#include <cassert>
#include <limits>
#include <vector>

namespace
{

nerve::common::VRConfig exactConfig()
{
    nerve::common::VRConfig config;
    config.algorithm = nerve::common::VRAlgorithmSelection::EXACT_STANDARD;
    config.max_dim = 1;
    config.max_radius = 2.0;
    return config;
}

nerve::core::BufferView<const double> viewOf(const std::vector<double> &values)
{
    return nerve::core::BufferView<const double>(values.data(), values.size());
}

void assertInvalidFastResult(const std::vector<double> &points)
{
    const auto result =
        nerve::persistence::computeVrPersistenceFastResult(viewOf(points), 2, exactConfig());
    assert(result.isError());
    assert(result.errorCode() == nerve::errors::ErrorCode::E54_PH4_INVALID_INPUT);
    assert(nerve::persistence::computeVrPersistenceFast(viewOf(points), 2, exactConfig()).empty());
}

void assertInvalidSimdResult(const std::vector<double> &points)
{
    auto cfg = exactConfig();
    cfg.algorithm = nerve::common::VRAlgorithmSelection::FAST_SIMD;
    assert(nerve::persistence::computeVrPersistenceFastSimd(viewOf(points), 2, cfg).empty());
}

void assertInvalidMediumHybridResult(const std::vector<double> &points)
{
    auto cfg = exactConfig();
    cfg.algorithm = nerve::common::VRAlgorithmSelection::MEDIUM_HYBRID;
    assert(nerve::persistence::computeVrPersistenceMediumHybrid(viewOf(points), 2, cfg).empty());
}

void assertInvalidLargeWitnessResult(const std::vector<double> &points)
{
    assert(nerve::persistence::computeVrPersistenceLargeWitness(viewOf(points), 2, exactConfig())
               .empty());
}

} // namespace

int main()
{
    const std::vector<double> points{0.0, 0.0, 1.0, 0.0};
    const auto valid =
        nerve::persistence::computeVrPersistenceFastResult(viewOf(points), 2, exactConfig());
    assert(valid.isSuccess());
    assert(!valid.value().empty());

    const std::vector<double> nonfinite{0.0, 0.0, std::numeric_limits<double>::quiet_NaN(), 0.0};
    assertInvalidFastResult(nonfinite);
    assertInvalidSimdResult(nonfinite);
    assertInvalidMediumHybridResult(nonfinite);
    assertInvalidLargeWitnessResult(nonfinite);

    const std::vector<double> overflow_prone{0.0, 0.0, std::numeric_limits<double>::max(), 0.0};
    assertInvalidFastResult(overflow_prone);
    assertInvalidSimdResult(overflow_prone);
    assertInvalidMediumHybridResult(overflow_prone);
    assertInvalidLargeWitnessResult(overflow_prone);

    std::vector<double> medium_overflow(1024 * 2, 0.0);
    medium_overflow[2] = std::numeric_limits<double>::max();
    assertInvalidMediumHybridResult(medium_overflow);

    std::vector<double> witness_overflow(10001 * 2, 0.0);
    witness_overflow[2] = std::numeric_limits<double>::max();
    assertInvalidLargeWitnessResult(witness_overflow);

    return 0;
}

#include "nerve/common/accelerated_types.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/errors/errors.hpp"
#include "nerve/persistence/core/core_types.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"
#include "nerve/persistence/vr/vr_fast_simd_ops.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <tuple>
#include <vector>

namespace
{

using nerve::Dimension;
using nerve::common::VRConfig;
using nerve::core::BufferView;
using nerve::persistence::Pair;
using nerve::persistence::VRAlgorithmSelection;

BufferView<const double> view_of(const std::vector<double> &v)
{
    return {v.data(), v.size()};
}

std::vector<Pair> canonical(std::vector<Pair> pairs)
{
    std::sort(pairs.begin(), pairs.end(), [](const Pair &a, const Pair &b) {
        return std::tuple(a.dimension, a.birth, a.death) <
               std::tuple(b.dimension, b.birth, b.death);
    });
    return pairs;
}

void assert_pairs_equal(const std::vector<Pair> &expected, const std::vector<Pair> &actual)
{
    assert(expected.size() == actual.size());
    for (std::size_t i = 0; i < expected.size(); ++i)
    {
        assert(expected[i].dimension == actual[i].dimension);
        assert(std::abs(expected[i].birth - actual[i].birth) < 1e-12);
        if (std::isinf(expected[i].death))
            assert(std::isinf(actual[i].death));
        else
            assert(std::abs(expected[i].death - actual[i].death) < 1e-12);
    }
}

size_t count_finite_by_dim(const std::vector<Pair> &pairs, Dimension dim)
{
    size_t c = 0;
    for (const auto &p : pairs)
        if (p.dimension == dim && !p.isInfinite())
            ++c;
    return c;
}

size_t count_essential_by_dim(const std::vector<Pair> &pairs, Dimension dim)
{
    size_t c = 0;
    for (const auto &p : pairs)
        if (p.dimension == dim && p.isInfinite())
            ++c;
    return c;
}

bool has_pair(const std::vector<Pair> &pairs, Dimension dim, double birth, double death,
              double tol = 1e-10)
{
    for (const auto &p : pairs)
    {
        if (p.dimension != dim)
            continue;
        if (std::abs(p.birth - birth) > tol)
            continue;
        if (std::isinf(death) && p.isInfinite())
            return true;
        if (std::isinf(death) || p.isInfinite())
            continue;
        if (std::abs(p.death - death) < tol)
            return true;
    }
    return false;
}

VRConfig fast_exact_config()
{
    VRConfig c;
    c.max_radius = 2.0;
    c.max_dim = 1;
    c.algorithm = VRAlgorithmSelection::EXACT_STANDARD;
    return c;
}

VRConfig fast_simd_config()
{
    VRConfig c;
    c.max_radius = 2.0;
    c.max_dim = 1;
    c.algorithm = VRAlgorithmSelection::FAST_SIMD;
    return c;
}

} // namespace

int main()
{
    // Basic fast exact path produces non-empty result
    {
        const std::vector<double> points{0.0, 0.0, 1.0, 0.0, 0.0, 1.0};
        const auto pairs =
            nerve::persistence::computeVrPersistenceFast(view_of(points), 2, fast_exact_config());
        assert(!pairs.empty());
    }

    // Fast exact path via Result API returns success
    {
        const std::vector<double> points{0.0, 0.0, 1.0, 0.0};
        const auto result = nerve::persistence::computeVrPersistenceFastResult(view_of(points), 2,
                                                                               fast_exact_config());
        assert(result.isSuccess());
        assert(!result.value().empty());
    }

    // Fast SIMD path produces non-empty for valid input
    {
        const std::vector<double> points{0.0, 0.0, 1.0, 0.0, 0.0, 1.0};
        const auto pairs = nerve::persistence::computeVrPersistenceFastSimd(view_of(points), 2,
                                                                            fast_simd_config());
        assert(!pairs.empty());
    }

    // Fast exact and SIMD paths give same diagram for small set
    {
        const std::vector<double> points{0.0, 0.0, 1.0, 0.0, 0.0, 1.0};
        const auto exact = canonical(
            nerve::persistence::computeVrPersistenceFast(view_of(points), 2, fast_exact_config()));
        const auto simd = canonical(nerve::persistence::computeVrPersistenceFastSimd(
            view_of(points), 2, fast_simd_config()));
        assert_pairs_equal(exact, simd);
    }

    // Two-point correctness via fast exact path
    {
        const std::vector<double> pts{0.0, 0.0, 1.0, 0.0};
        auto cfg = fast_exact_config();
        cfg.max_radius = 1.5;
        cfg.max_dim = 1;
        const auto pairs =
            canonical(nerve::persistence::computeVrPersistenceFast(view_of(pts), 2, cfg));
        assert(pairs.size() == 2);
        assert(count_finite_by_dim(pairs, 0) == 1);
        assert(count_essential_by_dim(pairs, 0) == 1);
        assert(has_pair(pairs, 0, 0.0, 1.0));
    }

    // Empty input produces empty diagram via both paths
    {
        const std::vector<double> empty;
        assert(nerve::persistence::computeVrPersistenceFast(view_of(empty), 2, fast_exact_config())
                   .empty());
        assert(
            nerve::persistence::computeVrPersistenceFastSimd(view_of(empty), 2, fast_simd_config())
                .empty());
    }

    // Single point produces 1 H0 essential via both paths
    {
        const std::vector<double> point{0.0, 0.0};
        const auto exact = canonical(
            nerve::persistence::computeVrPersistenceFast(view_of(point), 2, fast_exact_config()));
        const auto simd = canonical(nerve::persistence::computeVrPersistenceFastSimd(
            view_of(point), 2, fast_simd_config()));
        assert(exact.size() == 1);
        assert(simd.size() == 1);
        assert(has_pair(exact, 0, 0.0, std::numeric_limits<double>::infinity()));
        assert(has_pair(simd, 0, 0.0, std::numeric_limits<double>::infinity()));
    }

    // NaN input rejected by both paths
    {
        const std::vector<double> invalid{0.0, 0.0, std::numeric_limits<double>::quiet_NaN(), 0.0};
        assert(
            nerve::persistence::computeVrPersistenceFast(view_of(invalid), 2, fast_exact_config())
                .empty());
        assert(nerve::persistence::computeVrPersistenceFastSimd(view_of(invalid), 2,
                                                                fast_simd_config())
                   .empty());
    }

    // INF input rejected by both paths
    {
        const std::vector<double> invalid{0.0, 0.0, std::numeric_limits<double>::infinity(), 0.0};
        assert(
            nerve::persistence::computeVrPersistenceFast(view_of(invalid), 2, fast_exact_config())
                .empty());
        assert(nerve::persistence::computeVrPersistenceFastSimd(view_of(invalid), 2,
                                                                fast_simd_config())
                   .empty());
    }

    // Result API returns error for NaN
    {
        const std::vector<double> invalid{0.0, 0.0, std::numeric_limits<double>::quiet_NaN(), 0.0};
        const auto result = nerve::persistence::computeVrPersistenceFastResult(view_of(invalid), 2,
                                                                               fast_exact_config());
        assert(result.isError());
    }

    // Fast path determinism
    {
        const std::vector<double> pts{0.0, 0.0, 1.0, 0.0, 0.5, 0.866, 0.2, 0.8};
        const auto run1 = canonical(
            nerve::persistence::computeVrPersistenceFast(view_of(pts), 2, fast_exact_config()));
        const auto run2 = canonical(
            nerve::persistence::computeVrPersistenceFast(view_of(pts), 2, fast_exact_config()));
        assert_pairs_equal(run1, run2);
    }

    // All finite pairs satisfy birth <= death
    {
        const std::vector<double> pts{0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5};
        const auto pairs =
            nerve::persistence::computeVrPersistenceFast(view_of(pts), 2, fast_exact_config());
        for (const auto &p : pairs)
        {
            if (!p.isInfinite())
                assert(p.birth <= p.death + 1e-14);
            assert(p.dimension >= 0);
        }
    }

    // Fast SIMD runtime queries do not crash
    {
        nerve::persistence::isAvx512Available();
        nerve::persistence::getOptimalSimdBlockSize(3);
    }

    return 0;
}

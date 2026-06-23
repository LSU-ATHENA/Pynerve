#include "nerve/common/accelerated_types.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core/rng/determinism_contract.hpp"
#include "nerve/core_types.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <random>
#include <tuple>
#include <vector>

namespace
{

using nerve::Dimension;
using nerve::Field;
using nerve::Size;
using nerve::common::VRConfig;
using nerve::core::BufferView;
using nerve::persistence::Pair;
using nerve::persistence::VRAlgorithmSelection;

constexpr double kTol = 1e-10;

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

bool pairs_equal(const Pair &a, const Pair &b)
{
    if (a.dimension != b.dimension)
        return false;
    if (std::abs(a.birth - b.birth) > kTol)
        return false;
    if (a.isInfinite() && b.isInfinite())
        return true;
    if (a.isInfinite() || b.isInfinite())
        return false;
    return std::abs(a.death - b.death) < kTol;
}

bool assert_same_pairs(const std::vector<Pair> &expected, const std::vector<Pair> &actual)
{
    const auto c1 = canonical(expected);
    const auto c2 = canonical(actual);
    if (c1.size() != c2.size())
    {
        std::cerr << "pair count mismatch: " << c1.size() << " vs " << c2.size() << "\n";
        return false;
    }
    for (std::size_t i = 0; i < c1.size(); ++i)
    {
        if (!pairs_equal(c1[i], c2[i]))
        {
            std::cerr << "pair " << i << " differs: dim=" << c1[i].dimension
                      << " birth=" << c1[i].birth << " death=" << c1[i].death
                      << " vs dim=" << c2[i].dimension << " birth=" << c2[i].birth
                      << " death=" << c2[i].death << "\n";
            return false;
        }
    }
    return true;
}

bool check_same_config_three_times_identical()
{
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    constexpr int kN = 8;
    constexpr int kDim = 2;
    std::vector<double> pts(static_cast<std::size_t>(kN) * kDim);
    for (auto &v : pts)
        v = dist(rng);

    VRConfig cfg;
    cfg.max_radius = 1.5;
    cfg.max_dim = 2;
    cfg.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    const auto run1 = nerve::persistence::computeVrPersistenceFast(view_of(pts), kDim, cfg);
    const auto run2 = nerve::persistence::computeVrPersistenceFast(view_of(pts), kDim, cfg);
    const auto run3 = nerve::persistence::computeVrPersistenceFast(view_of(pts), kDim, cfg);

    if (!assert_same_pairs(run1, run2))
    {
        std::cerr << "determinism: run1 vs run2 differ\n";
        return false;
    }
    if (!assert_same_pairs(run1, run3))
    {
        std::cerr << "determinism: run1 vs run3 differ\n";
        return false;
    }

    return true;
}

bool check_different_seeds_produce_different_results()
{
    std::mt19937 rng1(42);
    std::mt19937 rng2(9999);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    constexpr int kN = 6;
    constexpr int kDim = 2;
    std::vector<double> pts1(static_cast<std::size_t>(kN) * kDim);
    std::vector<double> pts2(static_cast<std::size_t>(kN) * kDim);
    for (auto &v : pts1)
        v = dist(rng1);
    for (auto &v : pts2)
        v = dist(rng2);

    VRConfig cfg;
    cfg.max_radius = 2.0;
    cfg.max_dim = 2;
    cfg.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    const auto res1 = nerve::persistence::computeVrPersistenceFast(view_of(pts1), kDim, cfg);
    const auto res2 = nerve::persistence::computeVrPersistenceFast(view_of(pts2), kDim, cfg);

    if (res1.size() == res2.size())
    {
        bool all_same = true;
        const auto c1 = canonical(res1);
        const auto c2 = canonical(res2);
        for (std::size_t i = 0; i < c1.size() && all_same; ++i)
        {
            if (!pairs_equal(c1[i], c2[i]))
                all_same = false;
        }
        if (all_same)
        {
            std::cerr << "determinism: different seeds produced identical results\n";
            return false;
        }
    }

    return true;
}

bool check_exact_determinism_across_algorithms()
{
    std::mt19937 rng(77);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    constexpr int kN = 8;
    constexpr int kDim = 2;
    std::vector<double> pts(static_cast<std::size_t>(kN) * kDim);
    for (auto &v : pts)
        v = dist(rng);

    VRConfig exact_cfg;
    exact_cfg.max_radius = 1.5;
    exact_cfg.max_dim = 2;
    exact_cfg.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    VRConfig simd_cfg = exact_cfg;
    simd_cfg.algorithm = VRAlgorithmSelection::FAST_SIMD;

    const auto exact1 = nerve::persistence::computeVrPersistenceFast(view_of(pts), kDim, exact_cfg);
    const auto exact2 = nerve::persistence::computeVrPersistenceFast(view_of(pts), kDim, exact_cfg);
    const auto simd1 = nerve::persistence::computeVrPersistenceFast(view_of(pts), kDim, simd_cfg);
    const auto simd2 = nerve::persistence::computeVrPersistenceFast(view_of(pts), kDim, simd_cfg);

    if (!assert_same_pairs(exact1, exact2))
    {
        std::cerr << "EXACT self-determinism failed\n";
        return false;
    }
    if (!assert_same_pairs(simd1, simd2))
    {
        std::cerr << "FAST_SIMD self-determinism failed\n";
        return false;
    }

    return true;
}

bool check_determinism_across_varying_max_radius()
{
    std::mt19937 rng(1234);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    constexpr int kN = 6;
    constexpr int kDim = 2;
    std::vector<double> pts(static_cast<std::size_t>(kN) * kDim);
    for (auto &v : pts)
        v = dist(rng);

    constexpr double kRadii[] = {0.5, 1.0, 1.5, 2.0, 5.0};

    for (double radius : kRadii)
    {
        VRConfig cfg;
        cfg.max_radius = radius;
        cfg.max_dim = 2;
        cfg.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

        const auto run1 = nerve::persistence::computeVrPersistenceFast(view_of(pts), kDim, cfg);
        const auto run2 = nerve::persistence::computeVrPersistenceFast(view_of(pts), kDim, cfg);

        if (!assert_same_pairs(run1, run2))
        {
            std::cerr << "determinism across max_radius=" << radius << " failed\n";
            return false;
        }

        for (const auto &p : run1)
        {
            if (!p.isInfinite() && p.death > radius + kTol)
            {
                std::cerr << "determinism: death > max_radius for radius=" << radius << "\n";
                return false;
            }
        }
    }

    return true;
}

bool check_determinism_contract_enforcer()
{
    auto contract = nerve::core::DeterminismEnforcer::createContract(
        nerve::core::DeterminismLevel::BASIC, "stress_determinism_test");
    if (!contract.isValid())
    {
        std::cerr << "determinism contract should be valid\n";
        return false;
    }

    bool can_satisfy = nerve::core::DeterminismEnforcer::canSatisfyContract(contract);
    if (!can_satisfy)
    {
        std::cerr << "cannot satisfy determinism contract\n";
        return false;
    }

    return true;
}

} // namespace

int main()
{
    if (!check_same_config_three_times_identical())
    {
        std::cerr << "FAIL: same config 3x identical\n";
        return 1;
    }
    if (!check_different_seeds_produce_different_results())
    {
        std::cerr << "FAIL: different seeds produce different results\n";
        return 1;
    }
    if (!check_exact_determinism_across_algorithms())
    {
        std::cerr << "FAIL: determinism across algorithms\n";
        return 1;
    }
    if (!check_determinism_across_varying_max_radius())
    {
        std::cerr << "FAIL: determinism across max_radius\n";
        return 1;
    }
    if (!check_determinism_contract_enforcer())
    {
        std::cerr << "FAIL: determinism contract enforcer\n";
        return 1;
    }
    return 0;
}

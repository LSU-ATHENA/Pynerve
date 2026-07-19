#include "nerve/common/accelerated_types.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core_types.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"
#include "test_utils.hpp"

#include <cstddef>
#include <iostream>
#include <random>
#include <vector>

namespace
{

using nerve::Size;
using nerve::common::VRConfig;
using nerve::persistence::Pair;
using nerve::persistence::VRAlgorithmSelection;
using namespace nerve::test;

constexpr double kTol = 1e-10;

bool check_large_cloud_2d()
{
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-10.0, 10.0);

    constexpr int kN = 50;
    constexpr int kDim = 2;
    std::vector<double> pts(static_cast<std::size_t>(kN) * kDim);
    for (auto &v : pts)
        v = dist(rng);

    VRConfig cfg;
    cfg.max_radius = 5.0;
    cfg.max_dim = 2;
    cfg.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    const auto pairs = nerve::persistence::computeVrPersistenceFast(view_of(pts), kDim, cfg);

    if (pairs.empty())
    {
        std::cerr << "large 2D cloud: no pairs\n";
        return false;
    }

    for (const auto &p : pairs)
    {
        if (!p.isInfinite())
        {
            if (p.birth > p.death + kTol)
            {
                std::cerr << "large 2D: birth>death " << p.birth << " > " << p.death << "\n";
                return false;
            }
            if (p.lifetime() < -kTol)
            {
                std::cerr << "large 2D: negative persistence\n";
                return false;
            }
        }
        if (p.dimension < 0)
        {
            std::cerr << "large 2D: negative dimension\n";
            return false;
        }
    }

    Size h0_essential = 0;
    for (const auto &p : pairs)
        if (p.dimension == 0 && p.isInfinite())
            ++h0_essential;
    if (h0_essential != 1)
    {
        std::cerr << "large 2D: expected 1 H0 essential, got " << h0_essential << "\n";
        return false;
    }

    return true;
}

bool check_large_cloud_3d()
{
    std::mt19937 rng(123);
    std::uniform_real_distribution<double> dist(-5.0, 5.0);

    constexpr int kN = 60;
    constexpr int kDim = 3;
    std::vector<double> pts(static_cast<std::size_t>(kN) * kDim);
    for (auto &v : pts)
        v = dist(rng);

    VRConfig cfg;
    cfg.max_radius = 4.0;
    cfg.max_dim = 2;
    cfg.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    const auto pairs = nerve::persistence::computeVrPersistenceFast(view_of(pts), kDim, cfg);

    if (pairs.empty())
    {
        std::cerr << "large 3D cloud: no pairs\n";
        return false;
    }

    for (const auto &p : pairs)
    {
        if (!p.isInfinite())
        {
            if (p.birth > p.death + kTol)
            {
                std::cerr << "large 3D: birth>death\n";
                return false;
            }
            if (p.lifetime() < -kTol)
            {
                std::cerr << "large 3D: negative persistence\n";
                return false;
            }
        }
    }

    Size h0_essential = 0;
    for (const auto &p : pairs)
        if (p.dimension == 0 && p.isInfinite())
            ++h0_essential;
    if (h0_essential != 1)
    {
        std::cerr << "large 3D: expected 1 H0 essential, got " << h0_essential << "\n";
        return false;
    }

    return true;
}

bool check_medium_cloud_exact_vs_simd()
{
    std::mt19937 rng(456);
    std::uniform_real_distribution<double> dist(-2.0, 2.0);

    constexpr int kN = 20;
    constexpr int kDim = 2;
    std::vector<double> pts(static_cast<std::size_t>(kN) * kDim);
    for (auto &v : pts)
        v = dist(rng);

    VRConfig exact_cfg;
    exact_cfg.max_radius = 3.0;
    exact_cfg.max_dim = 2;
    exact_cfg.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    VRConfig simd_cfg = exact_cfg;
    simd_cfg.algorithm = VRAlgorithmSelection::FAST_SIMD;

    const auto exact = nerve::persistence::computeVrPersistenceFast(view_of(pts), kDim, exact_cfg);
    const auto simd = nerve::persistence::computeVrPersistenceFast(view_of(pts), kDim, simd_cfg);

    if (exact.size() != simd.size())
    {
        std::cerr << "medium EXACT/SIMD size mismatch: " << exact.size() << " vs " << simd.size()
                  << "\n";
        return false;
    }

    return true;
}

bool check_multiple_random_seeds()
{
    constexpr int kSeeds[] = {42, 123, 456, 789, 1024};
    constexpr int kN = 25;
    constexpr int kDim = 2;

    for (int seed : kSeeds)
    {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> dist(-3.0, 3.0);

        std::vector<double> pts(static_cast<std::size_t>(kN) * kDim);
        for (auto &v : pts)
            v = dist(rng);

        VRConfig cfg;
        cfg.max_radius = 2.5;
        cfg.max_dim = 2;
        cfg.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

        const auto pairs = nerve::persistence::computeVrPersistenceFast(view_of(pts), kDim, cfg);

        if (pairs.empty())
        {
            std::cerr << "seed " << seed << ": empty pairs\n";
            return false;
        }

        for (const auto &p : pairs)
        {
            if (!p.isInfinite() && p.birth > p.death + kTol)
            {
                std::cerr << "seed " << seed << ": birth>death\n";
                return false;
            }
            if (p.dimension < 0)
            {
                std::cerr << "seed " << seed << ": negative dimension\n";
                return false;
            }
        }
    }

    return true;
}

bool check_invariant_nonnegative_persistence_across_sizes()
{
    std::mt19937 rng(999);
    std::uniform_real_distribution<double> dist(-2.0, 2.0);

    constexpr int kSizes[] = {10, 30, 50, 70, 100};
    constexpr int kDim = 2;

    for (int n : kSizes)
    {
        std::vector<double> pts(static_cast<std::size_t>(n) * kDim);
        for (auto &v : pts)
            v = dist(rng);

        VRConfig cfg;
        cfg.max_radius = 3.0;
        cfg.max_dim = 2;
        cfg.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

        const auto pairs = nerve::persistence::computeVrPersistenceFast(view_of(pts), kDim, cfg);

        for (const auto &p : pairs)
        {
            if (!p.isInfinite())
            {
                if (p.lifetime() < -kTol)
                {
                    std::cerr << "n=" << n << ": negative persistence " << p.lifetime() << "\n";
                    return false;
                }
            }
        }
    }

    return true;
}

} // namespace

int main()
{
    if (!check_large_cloud_2d())
    {
        std::cerr << "FAIL: large 2D cloud\n";
        return 1;
    }
    if (!check_large_cloud_3d())
    {
        std::cerr << "FAIL: large 3D cloud\n";
        return 1;
    }
    if (!check_medium_cloud_exact_vs_simd())
    {
        std::cerr << "FAIL: medium EXACT vs SIMD\n";
        return 1;
    }
    if (!check_multiple_random_seeds())
    {
        std::cerr << "FAIL: multiple random seeds\n";
        return 1;
    }
    if (!check_invariant_nonnegative_persistence_across_sizes())
    {
        std::cerr << "FAIL: non-negative persistence across sizes\n";
        return 1;
    }
    return 0;
}

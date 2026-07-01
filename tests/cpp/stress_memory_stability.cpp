#include "nerve/common/accelerated_types.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core_types.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"
#include "test_utils.hpp"

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

using nerve::common::VRConfig;
using nerve::persistence::Pair;
using nerve::persistence::VRAlgorithmSelection;
using namespace nerve::test;

constexpr double kTol = 1e-10;


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
            std::cerr << "pair " << i << " differs\n";
            return false;
        }
    }
    return true;
}

bool check_repeated_computation_same_input()
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

    const auto first = nerve::persistence::computeVrPersistenceFast(view_of(pts), kDim, cfg);

    for (int iter = 0; iter < 100; ++iter)
    {
        const auto result = nerve::persistence::computeVrPersistenceFast(view_of(pts), kDim, cfg);

        if (result.empty())
        {
            std::cerr << "memory stability: empty result at iteration " << iter << "\n";
            return false;
        }

        if (!assert_same_pairs(first, result))
        {
            std::cerr << "memory stability: result changed at iteration " << iter << "\n";
            return false;
        }

        for (const auto &p : result)
        {
            if (!p.isInfinite())
            {
                if (p.birth > p.death + kTol)
                {
                    std::cerr << "memory stability: birth>death at iteration " << iter << "\n";
                    return false;
                }
                if (p.lifetime() < -kTol)
                {
                    std::cerr << "memory stability: negative persistence at iteration " << iter
                              << "\n";
                    return false;
                }
            }
            if (p.dimension < 0)
            {
                std::cerr << "memory stability: negative dimension at iteration " << iter << "\n";
                return false;
            }
        }
    }

    return true;
}

bool check_repeated_computation_varying_clouds()
{
    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> dist(-2.0, 2.0);

    for (int iter = 0; iter < 100; ++iter)
    {
        int n = 5 + (iter % 10);
        int dim = 1 + (iter % 3);
        std::vector<double> pts(static_cast<std::size_t>(n) * dim);
        for (auto &v : pts)
            v = dist(rng);

        VRConfig cfg;
        cfg.max_radius = 2.0;
        cfg.max_dim = 2;
        cfg.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

        const auto result = nerve::persistence::computeVrPersistenceFast(view_of(pts), dim, cfg);

        for (const auto &p : result)
        {
            if (!p.isInfinite())
            {
                if (p.birth > p.death + kTol)
                {
                    std::cerr << "varying: birth>death at iteration " << iter << "\n";
                    return false;
                }
                if (p.lifetime() < -kTol)
                {
                    std::cerr << "varying: negative persistence at iteration " << iter << "\n";
                    return false;
                }
            }
        }
    }

    return true;
}

bool check_repeated_different_configs()
{
    std::mt19937 rng(6789);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    constexpr int kN = 6;
    constexpr int kDim = 2;
    std::vector<double> pts(static_cast<std::size_t>(kN) * kDim);
    for (auto &v : pts)
        v = dist(rng);

    for (int iter = 0; iter < 50; ++iter)
    {
        VRConfig cfg;
        cfg.max_radius = 0.5 + static_cast<double>(iter % 10) * 0.3;
        cfg.max_dim = 1 + (iter % 3);
        cfg.algorithm = (iter % 2 == 0) ? VRAlgorithmSelection::EXACT_STANDARD
                                        : VRAlgorithmSelection::FAST_SIMD;

        const auto result = nerve::persistence::computeVrPersistenceFast(view_of(pts), kDim, cfg);

        for (const auto &p : result)
        {
            if (!p.isInfinite())
            {
                if (p.lifetime() < -kTol)
                {
                    std::cerr << "configs: negative persistence at iteration " << iter << "\n";
                    return false;
                }
            }
        }
    }

    return true;
}

bool check_repeated_with_larger_clouds()
{
    std::mt19937 rng(555);
    std::uniform_real_distribution<double> dist(-5.0, 5.0);

    for (int iter = 0; iter < 30; ++iter)
    {
        int n = 10 + iter;
        std::vector<double> pts(static_cast<std::size_t>(n) * 2);
        for (auto &v : pts)
            v = dist(rng);

        VRConfig cfg;
        cfg.max_radius = 3.0;
        cfg.max_dim = 2;
        cfg.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

        const auto result = nerve::persistence::computeVrPersistenceFast(view_of(pts), 2, cfg);

        for (const auto &p : result)
        {
            if (!p.isInfinite())
            {
                if (p.lifetime() < -kTol)
                {
                    std::cerr << "larger: negative persistence at iteration " << iter << "\n";
                    return false;
                }
                if (p.birth > p.death + kTol)
                {
                    std::cerr << "larger: birth>death at iteration " << iter << "\n";
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
    if (!check_repeated_computation_same_input())
    {
        std::cerr << "FAIL: repeated same input\n";
        return 1;
    }
    if (!check_repeated_computation_varying_clouds())
    {
        std::cerr << "FAIL: repeated varying clouds\n";
        return 1;
    }
    if (!check_repeated_different_configs())
    {
        std::cerr << "FAIL: repeated different configs\n";
        return 1;
    }
    if (!check_repeated_with_larger_clouds())
    {
        std::cerr << "FAIL: repeated larger clouds\n";
        return 1;
    }
    return 0;
}

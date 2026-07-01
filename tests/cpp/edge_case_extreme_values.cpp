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

using nerve::Dimension;
using nerve::Size;
using nerve::common::VRConfig;
using nerve::persistence::Pair;
using nerve::persistence::VRAlgorithmSelection;
using namespace nerve::test;

constexpr double kTol = 1e-10;


bool check_max_radius_zero_only_vertices()
{
    const std::vector<double> pts = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0};
    VRConfig cfg;
    cfg.max_radius = 0.0;
    cfg.max_dim = 2;
    cfg.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    const auto pairs = nerve::persistence::computeVrPersistenceFast(view_of(pts), 2, cfg);

    Size h0_essential = 0;
    for (const auto &p : pairs)
    {
        if (p.dimension == 0 && p.isInfinite())
            ++h0_essential;
        if (p.dimension != 0)
        {
            std::cerr << "radius=0: found dim=" << p.dimension << " pair\n";
            return false;
        }
    }
    if (h0_essential != 3)
    {
        std::cerr << "radius=0: expected 3 H0 essential, got " << h0_essential << "\n";
        return false;
    }

    return true;
}

bool check_max_radius_large_all_trivial()
{
    const std::vector<double> pts = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0};
    VRConfig cfg;
    cfg.max_radius = 1e10;
    cfg.max_dim = 2;
    cfg.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    const auto pairs = nerve::persistence::computeVrPersistenceFast(view_of(pts), 2, cfg);

    Size h0_essential = 0;
    for (const auto &p : pairs)
    {
        if (p.dimension == 0 && p.isInfinite())
            ++h0_essential;
        if (!p.isInfinite() && p.death > 1e10 + kTol)
        {
            std::cerr << "radius=1e10: death exceeds radius\n";
            return false;
        }
    }
    if (h0_essential != 1)
    {
        std::cerr << "radius=1e10: expected 1 H0 essential, got " << h0_essential << "\n";
        return false;
    }

    return true;
}

bool check_varying_max_dim_on_same_points()
{
    const std::vector<double> pts = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0};
    VRConfig cfg;
    cfg.max_radius = 2.0;
    cfg.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    std::vector<std::vector<Pair>> results_by_dim;
    for (Dimension d = 0; d <= 3; ++d)
    {
        cfg.max_dim = d;
        results_by_dim.push_back(
            nerve::persistence::computeVrPersistenceFast(view_of(pts), 2, cfg));
    }

    for (Dimension d = 0; d < 3; ++d)
    {
        if (results_by_dim[d].size() > results_by_dim[d + 1].size())
        {
            std::cerr << "varying max_dim: results decreased from dim=" << d << " to " << d + 1
                      << "\n";
            return false;
        }
    }

    for (const auto &res : results_by_dim)
    {
        for (const auto &p : res)
        {
            if (!p.isInfinite())
            {
                if (p.lifetime() < -kTol)
                {
                    std::cerr << "varying max_dim: negative persistence\n";
                    return false;
                }
            }
        }
    }

    return true;
}

bool check_very_small_radius()
{
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    constexpr int kN = 10;
    constexpr int kDim = 2;
    std::vector<double> pts(static_cast<std::size_t>(kN) * kDim);
    for (auto &v : pts)
        v = dist(rng);

    constexpr double kSmallRadius = 0.01;
    VRConfig cfg;
    cfg.max_radius = kSmallRadius;
    cfg.max_dim = 2;
    cfg.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    const auto pairs = nerve::persistence::computeVrPersistenceFast(view_of(pts), kDim, cfg);

    Size h0_essential = 0;
    for (const auto &p : pairs)
    {
        if (p.dimension == 0 && p.isInfinite())
            ++h0_essential;
        if (!p.isInfinite())
        {
            if (p.death > kSmallRadius + kTol)
            {
                std::cerr << "small radius: death > radius\n";
                return false;
            }
        }
        if (p.dimension > 0 && !p.isInfinite())
        {
            if (p.birth > kSmallRadius + kTol)
            {
                std::cerr << "small radius: H1 birth > radius\n";
                return false;
            }
        }
    }
    if (h0_essential != static_cast<Size>(kN))
    {
        std::cerr << "small radius: expected " << kN << " H0 essential, got " << h0_essential
                  << "\n";
        return false;
    }

    return true;
}

bool check_very_large_radius_all_connected()
{
    const std::vector<double> pts = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0};
    VRConfig cfg;
    cfg.max_radius = 1e6;
    cfg.max_dim = 2;
    cfg.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    const auto pairs = nerve::persistence::computeVrPersistenceFast(view_of(pts), 2, cfg);

    Size h0_essential = 0;
    for (const auto &p : pairs)
        if (p.dimension == 0 && p.isInfinite())
            ++h0_essential;
    if (h0_essential != 1)
    {
        std::cerr << "large radius: expected 1 H0 essential, got " << h0_essential << "\n";
        return false;
    }

    for (const auto &p : pairs)
    {
        if (p.dimension > 0 && p.isInfinite())
        {
            std::cerr << "large radius: unexpected infinite H" << p.dimension << " pair\n";
            return false;
        }
    }

    return true;
}

} // namespace

int main()
{
    if (!check_max_radius_zero_only_vertices())
    {
        std::cerr << "FAIL: max_radius=0 only vertices\n";
        return 1;
    }
    if (!check_max_radius_large_all_trivial())
    {
        std::cerr << "FAIL: max_radius=1e10 all trivial\n";
        return 1;
    }
    if (!check_varying_max_dim_on_same_points())
    {
        std::cerr << "FAIL: varying max_dim on same points\n";
        return 1;
    }
    if (!check_very_small_radius())
    {
        std::cerr << "FAIL: very small radius\n";
        return 1;
    }
    if (!check_very_large_radius_all_connected())
    {
        std::cerr << "FAIL: very large radius all connected\n";
        return 1;
    }
    return 0;
}

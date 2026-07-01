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
#include <tuple>
#include <vector>

namespace
{

using nerve::Size;
using nerve::common::VRConfig;
using nerve::persistence::Pair;
using nerve::persistence::VRAlgorithmSelection;
using namespace nerve::test;

constexpr double kTol = 1e-10;


bool check_all_points_identical()
{
    const std::vector<double> pts = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    VRConfig cfg;
    cfg.max_radius = 2.0;
    cfg.max_dim = 2;
    cfg.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    const auto pairs = nerve::persistence::computeVrPersistenceFast(view_of(pts), 2, cfg);

    Size h0_essential = 0;
    for (const auto &p : pairs)
    {
        if (p.dimension == 0 && p.isInfinite())
            ++h0_essential;
        if (!p.isInfinite())
        {
            if (p.birth > p.death + kTol)
            {
                std::cerr << "identical points: birth>death\n";
                return false;
            }
            if (p.lifetime() < -kTol)
            {
                std::cerr << "identical points: negative persistence\n";
                return false;
            }
        }
    }
    if (h0_essential != 1)
    {
        std::cerr << "identical points: expected 1 H0 essential, got " << h0_essential << "\n";
        return false;
    }

    return true;
}

bool check_partially_duplicate_points()
{
    const std::vector<double> pts = {0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0};
    VRConfig cfg;
    cfg.max_radius = 2.0;
    cfg.max_dim = 2;
    cfg.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    const auto pairs = nerve::persistence::computeVrPersistenceFast(view_of(pts), 2, cfg);

    Size h0_essential = 0;
    for (const auto &p : pairs)
    {
        if (p.dimension == 0 && p.isInfinite())
            ++h0_essential;
        if (!p.isInfinite())
        {
            if (p.lifetime() < -kTol)
            {
                std::cerr << "partial duplicates: negative persistence\n";
                return false;
            }
        }
    }
    if (h0_essential != 1)
    {
        std::cerr << "partial duplicates: expected 1 H0 essential, got " << h0_essential << "\n";
        return false;
    }

    return true;
}

bool check_very_close_points()
{
    constexpr double kEpsilon = 1e-4;
    const std::vector<double> pts = {0.0, 0.0, kEpsilon, kEpsilon};
    VRConfig cfg;
    cfg.max_radius = 1.0;
    cfg.max_dim = 2;
    cfg.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    const auto pairs = nerve::persistence::computeVrPersistenceFast(view_of(pts), 2, cfg);

    Size h0_essential = 0;
    for (const auto &p : pairs)
    {
        if (p.dimension == 0 && p.isInfinite())
            ++h0_essential;
        if (!p.isInfinite())
        {
            if (p.lifetime() < -kTol)
            {
                std::cerr << "close points: negative persistence\n";
                return false;
            }
        }
    }
    if (h0_essential != 1)
    {
        std::cerr << "close points: expected 1 H0 essential, got " << h0_essential << "\n";
        return false;
    }

    bool found_connection = false;
    for (const auto &p : pairs)
    {
        if (p.dimension == 0 && !p.isInfinite())
        {
            found_connection = true;
            if (std::abs(p.death - kEpsilon * std::sqrt(2.0)) > kTol * 100)
            {
                std::cerr << "close points: unexpected death value " << p.death << " vs expected ~"
                          << kEpsilon * std::sqrt(2.0) << "\n";
                return false;
            }
        }
    }
    if (!found_connection)
    {
        std::cerr << "close points: expected H0 finite bar connecting them\n";
        return false;
    }

    return true;
}

bool check_triplicate_points()
{
    const std::vector<double> pts = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    VRConfig cfg;
    cfg.max_radius = 1.0;
    cfg.max_dim = 1;
    cfg.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    const auto pairs = nerve::persistence::computeVrPersistenceFast(view_of(pts), 2, cfg);

    Size h0_essential = 0;
    for (const auto &p : pairs)
    {
        if (p.dimension == 0 && p.isInfinite())
            ++h0_essential;
    }
    if (h0_essential != 1)
    {
        std::cerr << "triplicate: expected 1 H0 essential, got " << h0_essential << "\n";
        return false;
    }

    return true;
}

bool check_duplicate_points_numerical_stability()
{
    const std::vector<double> pts = {0.0, 0.0, 0.0, 1e-12, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0};
    VRConfig cfg;
    cfg.max_radius = 2.0;
    cfg.max_dim = 2;
    cfg.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    const auto pairs = nerve::persistence::computeVrPersistenceFast(view_of(pts), 3, cfg);

    for (const auto &p : pairs)
    {
        if (!p.isInfinite())
        {
            if (std::isnan(p.birth) || std::isnan(p.death))
            {
                std::cerr << "numerical stability: NaN in pair\n";
                return false;
            }
            if (std::isinf(p.birth) || std::isinf(p.death))
            {
                std::cerr << "numerical stability: Inf in pair\n";
                return false;
            }
            if (p.lifetime() < -kTol)
            {
                std::cerr << "numerical stability: negative persistence\n";
                return false;
            }
        }
    }

    Size h0_essential = 0;
    for (const auto &p : pairs)
    {
        if (p.dimension == 0 && p.isInfinite())
            ++h0_essential;
    }
    if (h0_essential < 1)
    {
        std::cerr << "numerical stability: expected at least 1 H0 essential, got " << h0_essential
                  << "\n";
        return false;
    }

    return true;
}

} // namespace

int main()
{
    if (!check_all_points_identical())
    {
        std::cerr << "FAIL: all points identical\n";
        return 1;
    }
    if (!check_partially_duplicate_points())
    {
        std::cerr << "FAIL: partially duplicate points\n";
        return 1;
    }
    if (!check_very_close_points())
    {
        std::cerr << "FAIL: very close points\n";
        return 1;
    }
    if (!check_triplicate_points())
    {
        std::cerr << "FAIL: triplicate points\n";
        return 1;
    }
    if (!check_duplicate_points_numerical_stability())
    {
        std::cerr << "FAIL: duplicate points numerical stability\n";
        return 1;
    }
    return 0;
}

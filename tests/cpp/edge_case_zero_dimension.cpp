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

using nerve::Size;
using nerve::common::VRConfig;
using nerve::persistence::Pair;
using nerve::persistence::VRAlgorithmSelection;
using namespace nerve::test;

constexpr double kTol = 1e-10;


bool check_max_dim0_only_h0()
{
    const std::vector<double> pts = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0};
    VRConfig cfg;
    cfg.max_radius = 2.0;
    cfg.max_dim = 0;
    cfg.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    const auto pairs = nerve::persistence::computeVrPersistenceFast(view_of(pts), 2, cfg);

    for (const auto &p : pairs)
    {
        if (p.dimension != 0)
        {
            std::cerr << "max_dim=0: found dim=" << p.dimension << " pair\n";
            return false;
        }
        if (!p.isInfinite())
        {
            if (p.birth > p.death + kTol)
            {
                std::cerr << "max_dim=0: birth>death\n";
                return false;
            }
        }
    }

    Size h0_essential = 0;
    for (const auto &p : pairs)
        if (p.dimension == 0 && p.isInfinite())
            ++h0_essential;
    if (h0_essential < 1)
    {
        std::cerr << "max_dim=0: expected at least 1 H0 essential, got " << h0_essential << "\n";
        return false;
    }

    return true;
}

bool check_max_dim1_has_h0_and_h1()
{
    const std::vector<double> pts = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0};
    VRConfig cfg;
    cfg.max_radius = 2.0;
    cfg.max_dim = 1;
    cfg.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    const auto pairs = nerve::persistence::computeVrPersistenceFast(view_of(pts), 2, cfg);

    bool has_h0 = false;
    bool has_h1 = false;
    for (const auto &p : pairs)
    {
        if (p.dimension == 0)
            has_h0 = true;
        if (p.dimension == 1)
            has_h1 = true;
        if (p.dimension > 1)
        {
            std::cerr << "max_dim=1: found dim=" << p.dimension << " pair\n";
            return false;
        }
    }

    if (!has_h0)
    {
        std::cerr << "max_dim=1: missing H0 pairs\n";
        return false;
    }
    if (!has_h1)
    {
        std::cerr << "max_dim=1: missing H1 pairs (square has H1)\n";
        return false;
    }

    return true;
}

bool check_max_dim2_produces_valid_output()
{
    // Tetrahedron: all faces and the 3-simplex appear at r = sqrt(2).
    // With max_dim=2 the 3-simplex is still present in the VR complex
    // (max_dim controls persistence computation, not simplex inclusion),
    // so the 2-cycle is killed immediately -- zero persistence, filtered.
    // This test verifies that the algorithm does not crash and
    // produces at least H0 (the only guaranteed output at max_dim=2).
    const std::vector<double> pts = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    VRConfig cfg;
    cfg.max_radius = 2.0;
    cfg.max_dim = 2;
    cfg.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    const auto pairs = nerve::persistence::computeVrPersistenceFast(view_of(pts), 3, cfg);

    bool has_h0 = false;
    for (const auto &p : pairs)
    {
        if (p.dimension == 0)
            has_h0 = true;
    }

    if (!has_h0)
    {
        std::cerr << "max_dim=2: missing H0\n";
        return false;
    }

    return true;
}

bool check_lower_dim_is_subset_of_higher_dim()
{
    const std::vector<double> pts = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};

    VRConfig cfg_dim1;
    cfg_dim1.max_radius = 2.0;
    cfg_dim1.max_dim = 1;
    cfg_dim1.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    VRConfig cfg_dim2;
    cfg_dim2.max_radius = 2.0;
    cfg_dim2.max_dim = 2;
    cfg_dim2.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    const auto pairs1 = nerve::persistence::computeVrPersistenceFast(view_of(pts), 3, cfg_dim1);
    const auto pairs2 = nerve::persistence::computeVrPersistenceFast(view_of(pts), 3, cfg_dim2);

    std::vector<Pair> pairs1_dim0;
    for (const auto &p : pairs1)
        if (p.dimension == 0)
            pairs1_dim0.push_back(p);

    std::vector<Pair> pairs2_dim0;
    for (const auto &p : pairs2)
        if (p.dimension == 0)
            pairs2_dim0.push_back(p);

    const auto c1 = canonical(pairs1_dim0);
    const auto c2 = canonical(pairs2_dim0);

    for (std::size_t i = 0; i < c1.size(); ++i)
    {
        bool found = false;
        for (std::size_t j = 0; j < c2.size(); ++j)
        {
            if (c1[i].dimension == c2[j].dimension && std::abs(c1[i].birth - c2[j].birth) < kTol &&
                ((c1[i].isInfinite() && c2[j].isInfinite()) ||
                 (!c1[i].isInfinite() && !c2[j].isInfinite() &&
                  std::abs(c1[i].death - c2[j].death) < kTol)))
            {
                found = true;
                break;
            }
        }
        if (!found)
        {
            std::cerr << "dim=0 pair from max_dim=1 not found in max_dim=2 results\n";
            return false;
        }
    }

    return true;
}

bool check_random_subset_property()
{
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> pts(static_cast<std::size_t>(6) * 2);
    for (auto &v : pts)
        v = dist(rng);

    VRConfig cfg0, cfg1, cfg2;
    cfg0.max_radius = 2.0;
    cfg0.max_dim = 0;
    cfg0.algorithm = VRAlgorithmSelection::EXACT_STANDARD;
    cfg1.max_radius = 2.0;
    cfg1.max_dim = 1;
    cfg1.algorithm = VRAlgorithmSelection::EXACT_STANDARD;
    cfg2.max_radius = 2.0;
    cfg2.max_dim = 2;
    cfg2.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    const auto res0 = nerve::persistence::computeVrPersistenceFast(view_of(pts), 2, cfg0);
    const auto res1 = nerve::persistence::computeVrPersistenceFast(view_of(pts), 2, cfg1);
    const auto res2 = nerve::persistence::computeVrPersistenceFast(view_of(pts), 2, cfg2);

    for (const auto &p : res0)
    {
        if (p.dimension != 0)
        {
            std::cerr << "random subset: max_dim=0 produced dim=" << p.dimension << "\n";
            return false;
        }
    }

    if (res0.size() > res1.size())
    {
        std::cerr << "random subset: max_dim=0 produced more pairs than max_dim=1\n";
        return false;
    }
    if (res1.size() > res2.size())
    {
        std::cerr << "random subset: max_dim=1 produced more pairs than max_dim=2\n";
        return false;
    }

    return true;
}

} // namespace

int main()
{
    if (!check_max_dim0_only_h0())
    {
        std::cerr << "FAIL: max_dim=0 only H0\n";
        return 1;
    }
    if (!check_max_dim1_has_h0_and_h1())
    {
        std::cerr << "FAIL: max_dim=1 has H0 and H1\n";
        return 1;
    }
    if (!check_max_dim2_produces_valid_output())
    {
        std::cerr << "FAIL: max_dim=2 produces valid output\n";
        return 1;
    }
    if (!check_lower_dim_is_subset_of_higher_dim())
    {
        std::cerr << "FAIL: lower dim subset of higher\n";
        return 1;
    }
    if (!check_random_subset_property())
    {
        std::cerr << "FAIL: random subset property\n";
        return 1;
    }
    return 0;
}

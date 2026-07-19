#include "nerve/common/accelerated_types.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core_types.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"
#include "test_utils.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

namespace
{

using nerve::Size;
using nerve::common::VRConfig;
using nerve::persistence::Pair;
using nerve::persistence::VRAlgorithmSelection;
using namespace nerve::test;

constexpr double kTol = 1e-10;

bool check_single_point_max_dim0()
{
    const std::vector<double> pt = {0.0, 0.0};
    VRConfig cfg;
    cfg.max_radius = 1.0;
    cfg.max_dim = 0;
    cfg.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    const auto pairs = nerve::persistence::computeVrPersistenceFast(view_of(pt), 2, cfg);

    Size h0_essential = 0;
    for (const auto &p : pairs)
    {
        if (p.dimension == 0 && p.isInfinite())
            ++h0_essential;
        if (p.dimension != 0)
        {
            std::cerr << "single point max_dim=0: unexpected dim=" << p.dimension << "\n";
            return false;
        }
    }
    if (h0_essential != 1)
    {
        std::cerr << "single point max_dim=0: expected 1 H0 essential, got " << h0_essential
                  << "\n";
        return false;
    }

    return true;
}

bool check_single_point_max_dim1()
{
    const std::vector<double> pt = {0.0, 0.0};
    VRConfig cfg;
    cfg.max_radius = 1.0;
    cfg.max_dim = 1;
    cfg.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    const auto pairs = nerve::persistence::computeVrPersistenceFast(view_of(pt), 2, cfg);

    Size h0_essential = 0;
    for (const auto &p : pairs)
    {
        if (p.dimension == 0 && p.isInfinite())
            ++h0_essential;
        if (p.dimension > 0)
        {
            std::cerr << "single point max_dim=1: unexpected dim=" << p.dimension << "\n";
            return false;
        }
    }
    if (h0_essential != 1)
    {
        std::cerr << "single point max_dim=1: expected 1 H0 essential, got " << h0_essential
                  << "\n";
        return false;
    }

    return true;
}

bool check_single_point_max_dim2()
{
    const std::vector<double> pt = {0.0, 0.0};
    VRConfig cfg;
    cfg.max_radius = 1.0;
    cfg.max_dim = 2;
    cfg.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    const auto pairs = nerve::persistence::computeVrPersistenceFast(view_of(pt), 2, cfg);

    Size h0_essential = 0;
    for (const auto &p : pairs)
    {
        if (p.dimension == 0 && p.isInfinite())
            ++h0_essential;
        if (p.dimension > 0)
        {
            std::cerr << "single point max_dim=2: unexpected dim=" << p.dimension << "\n";
            return false;
        }
    }
    if (h0_essential != 1)
    {
        std::cerr << "single point max_dim=2: expected 1 H0 essential, got " << h0_essential
                  << "\n";
        return false;
    }

    return true;
}

bool check_single_point_max_dim3()
{
    const std::vector<double> pt = {0.0, 0.0, 0.0};
    VRConfig cfg;
    cfg.max_radius = 1.0;
    cfg.max_dim = 3;
    cfg.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    const auto pairs = nerve::persistence::computeVrPersistenceFast(view_of(pt), 3, cfg);

    Size h0_essential = 0;
    for (const auto &p : pairs)
    {
        if (p.dimension == 0 && p.isInfinite())
            ++h0_essential;
        if (p.dimension > 0)
        {
            std::cerr << "single point max_dim=3: unexpected dim=" << p.dimension << "\n";
            return false;
        }
    }
    if (h0_essential != 1)
    {
        std::cerr << "single point max_dim=3: expected 1 H0 essential, got " << h0_essential
                  << "\n";
        return false;
    }

    return true;
}

bool check_two_points_1d_space()
{
    const std::vector<double> pts = {0.0, 1.0};
    VRConfig cfg;
    cfg.max_radius = 2.0;
    cfg.max_dim = 2;
    cfg.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    const auto pairs = nerve::persistence::computeVrPersistenceFast(view_of(pts), 1, cfg);

    Size h0_essential = 0;
    for (const auto &p : pairs)
    {
        if (p.dimension == 0 && p.isInfinite())
            ++h0_essential;
        if (!p.isInfinite() && p.dimension == 0 && std::abs(p.death - 1.0) > kTol)
        {
            std::cerr << "two points 1D: expected death=1.0, got " << p.death << "\n";
            return false;
        }
    }
    if (h0_essential != 1)
    {
        std::cerr << "two points 1D: expected 1 H0 essential, got " << h0_essential << "\n";
        return false;
    }

    return true;
}

bool check_two_points_2d_space()
{
    const std::vector<double> pts = {0.0, 0.0, 1.0, 0.0};
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
    }
    if (h0_essential != 1)
    {
        std::cerr << "two points 2D: expected 1 H0 essential, got " << h0_essential << "\n";
        return false;
    }

    return true;
}

bool check_two_points_3d_space()
{
    const std::vector<double> pts = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
    VRConfig cfg;
    cfg.max_radius = 2.0;
    cfg.max_dim = 2;
    cfg.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    const auto pairs = nerve::persistence::computeVrPersistenceFast(view_of(pts), 3, cfg);

    Size h0_essential = 0;
    for (const auto &p : pairs)
    {
        if (p.dimension == 0 && p.isInfinite())
            ++h0_essential;
    }
    if (h0_essential != 1)
    {
        std::cerr << "two points 3D: expected 1 H0 essential, got " << h0_essential << "\n";
        return false;
    }

    return true;
}

bool check_collinear_points_3d()
{
    const std::vector<double> pts = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0};
    VRConfig cfg;
    cfg.max_radius = 2.0;
    cfg.max_dim = 2;
    cfg.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    const auto pairs = nerve::persistence::computeVrPersistenceFast(view_of(pts), 3, cfg);

    Size h0_essential = 0;
    for (const auto &p : pairs)
    {
        if (p.dimension == 0 && p.isInfinite())
            ++h0_essential;
        if (!p.isInfinite())
        {
            if (p.lifetime() < -kTol)
            {
                std::cerr << "collinear 3D: negative persistence\n";
                return false;
            }
        }
    }
    if (h0_essential != 1)
    {
        std::cerr << "collinear 3D: expected 1 H0 essential, got " << h0_essential << "\n";
        return false;
    }

    return true;
}

bool check_coplanar_points_3d()
{
    const std::vector<double> pts = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0};
    VRConfig cfg;
    cfg.max_radius = 2.0;
    cfg.max_dim = 2;
    cfg.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    const auto pairs = nerve::persistence::computeVrPersistenceFast(view_of(pts), 3, cfg);

    Size h1_essential = 0;
    for (const auto &p : pairs)
    {
        if (p.dimension == 1 && p.isInfinite())
            ++h1_essential;
        if (!p.isInfinite())
        {
            if (p.lifetime() < -kTol)
            {
                std::cerr << "coplanar 3D: negative persistence\n";
                return false;
            }
        }
    }
    if (h1_essential != 0)
    {
        std::cerr << "coplanar 3D: expected 0 H1 essential, got " << h1_essential << "\n";
        return false;
    }

    return true;
}

} // namespace

int main()
{
    if (!check_single_point_max_dim0())
    {
        std::cerr << "FAIL: single point max_dim=0\n";
        return 1;
    }
    if (!check_single_point_max_dim1())
    {
        std::cerr << "FAIL: single point max_dim=1\n";
        return 1;
    }
    if (!check_single_point_max_dim2())
    {
        std::cerr << "FAIL: single point max_dim=2\n";
        return 1;
    }
    if (!check_single_point_max_dim3())
    {
        std::cerr << "FAIL: single point max_dim=3\n";
        return 1;
    }
    if (!check_two_points_1d_space())
    {
        std::cerr << "FAIL: two points 1D space\n";
        return 1;
    }
    if (!check_two_points_2d_space())
    {
        std::cerr << "FAIL: two points 2D space\n";
        return 1;
    }
    if (!check_two_points_3d_space())
    {
        std::cerr << "FAIL: two points 3D space\n";
        return 1;
    }
    if (!check_collinear_points_3d())
    {
        std::cerr << "FAIL: collinear points 3D\n";
        return 1;
    }
    if (!check_coplanar_points_3d())
    {
        std::cerr << "FAIL: coplanar points 3D\n";
        return 1;
    }
    return 0;
}

#include "nerve/common/accelerated_types.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core_types.hpp"
#include "nerve/persistence/core/core_types.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"
#include "test_utils.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

namespace
{

using nerve::Dimension;
using nerve::common::VRConfig;
using nerve::persistence::Pair;
using nerve::persistence::VRAlgorithmSelection;
using namespace nerve::test;

// Helpers


size_t count_pairs_by_dim(const std::vector<Pair> &pairs, Dimension dim)
{
    size_t count = 0;
    for (const auto &p : pairs)
    {
        if (p.dimension == dim)
            ++count;
    }
    return count;
}

size_t count_essential_by_dim(const std::vector<Pair> &pairs, Dimension dim)
{
    size_t count = 0;
    for (const auto &p : pairs)
    {
        if (p.dimension == dim && p.isInfinite())
            ++count;
    }
    return count;
}

size_t count_finite_by_dim(const std::vector<Pair> &pairs, Dimension dim)
{
    size_t count = 0;
    for (const auto &p : pairs)
    {
        if (p.dimension == dim && !p.isInfinite())
            ++count;
    }
    return count;
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

// Case 1: Two points at distance 1
//   Point set: {(0,0), (1,0)}
//   Expected: H0 has one finite pair (0, 1.0) and one essential (0, inf)

bool check_two_points()
{
    const std::vector<double> pts = {0.0, 0.0, 1.0, 0.0};
    VRConfig config;
    config.max_radius = 1.5;
    config.max_dim = 1;
    config.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    const auto pairs =
        canonical(nerve::persistence::computeVrPersistenceFast(view_of(pts), 2, config));

    // Exactly 2 pairs: one finite (death=1.0), one essential
    if (pairs.size() != 2)
    {
        std::cerr << "two_points: expected 2 pairs, got " << pairs.size() << "\n";
        return false;
    }

    // H0: 1 finite, 1 essential
    if (count_finite_by_dim(pairs, 0) != 1)
    {
        std::cerr << "two_points: expected 1 finite H0 pair\n";
        return false;
    }
    if (count_essential_by_dim(pairs, 0) != 1)
    {
        std::cerr << "two_points: expected 1 essential H0 pair\n";
        return false;
    }

    // Finite pair: birth=0, death=1.0 (full edge distance)
    if (!has_pair(pairs, 0, 0.0, 1.0))
    {
        std::cerr << "two_points: expected H0 pair (0, 1.0)\n";
        return false;
    }

    return true;
}

// Case 2: Equilateral triangle with edge length 1
//   Point set: {(0,0), (1,0), (0.5, sqrt(3)/2)}
//   Expected: H0=2 finite (0,1) + 1 essential, H1=1 trivial (1,1)

bool check_equilateral_triangle()
{
    const double r3o2 = std::sqrt(3.0) / 2.0;
    const std::vector<double> pts = {0.0, 0.0, 1.0, 0.0, 0.5, r3o2};
    VRConfig config;
    config.max_radius = 2.0;
    config.max_dim = 2;
    config.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    const auto pairs =
        canonical(nerve::persistence::computeVrPersistenceFast(view_of(pts), 2, config));

    // H0: 2 finite merges at t=1 (edges), 1 essential
    if (count_finite_by_dim(pairs, 0) != 2)
    {
        std::cerr << "triangle: expected 2 finite H0, got " << count_finite_by_dim(pairs, 0)
                  << "\n";
        return false;
    }
    if (count_essential_by_dim(pairs, 0) != 1)
    {
        std::cerr << "triangle: expected 1 H0 essential\n";
        return false;
    }

    // All finite H0 should have birth=0, death=1 (full edge distance)
    for (const auto &p : pairs)
    {
        if (p.dimension == 0 && !p.isInfinite())
        {
            if (std::abs(p.birth) > 1e-12 || std::abs(p.death - 1.0) > 1e-10)
            {
                std::cerr << "triangle: H0 expected (0, 1), got (" << p.birth << ", " << p.death
                          << ")\n";
                return false;
            }
        }
    }

    return true;
}

// Case 3: Square (4 points on unit square)
//   Point set: {(0,0), (1,0), (0,1), (1,1)}
//   Expected: H0=1 essential, H1=1 non-trivial finite (1, sqrt2)

bool check_square()
{
    const std::vector<double> pts = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0};
    VRConfig config;
    config.max_radius = 2.0;
    config.max_dim = 2;
    config.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    const auto pairs =
        canonical(nerve::persistence::computeVrPersistenceFast(view_of(pts), 2, config));

    // H0: 1 essential
    if (count_essential_by_dim(pairs, 0) != 1)
    {
        std::cerr << "square: expected 1 H0 essential, got " << count_essential_by_dim(pairs, 0)
                  << "\n";
        return false;
    }

    // H1: at least 1 non-trivial pair with birth=1, death=sqrt2
    const double sqrt2 = std::sqrt(2.0);
    bool found_non_trivial_h1 = false;
    for (const auto &p : pairs)
    {
        if (p.dimension == 1 && !p.isInfinite())
        {
            if (std::abs(p.birth - 1.0) < 0.01 && std::abs(p.death - sqrt2) < 0.01)
            {
                found_non_trivial_h1 = true;
            }
        }
    }
    if (!found_non_trivial_h1)
    {
        std::cerr << "square: expected H1 non-trivial (1, " << sqrt2 << ")\n";
        return false;
    }

    return true;
}

// Case 4: Three collinear points
//   Point set: {(0,0), (0.5,0), (1,0)}
//   Expected: H0 has 2 finite bars (death=0.5 for each merge) and 1 essential

bool check_three_collinear_points()
{
    const std::vector<double> pts = {0.0, 0.0, 0.5, 0.0, 1.0, 0.0};
    VRConfig config;
    config.max_radius = 1.0;
    config.max_dim = 1;
    config.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    const auto pairs =
        canonical(nerve::persistence::computeVrPersistenceFast(view_of(pts), 2, config));

    // H0: 2 finite + 1 essential
    if (count_finite_by_dim(pairs, 0) != 2)
    {
        std::cerr << "collinear: expected 2 finite H0 pairs, got " << count_finite_by_dim(pairs, 0)
                  << "\n";
        return false;
    }
    if (count_essential_by_dim(pairs, 0) != 1)
    {
        std::cerr << "collinear: expected 1 H0 essential\n";
        return false;
    }

    // Both finite merges at t=0.5 (adjacent edges)
    if (!has_pair(pairs, 0, 0.0, 0.5))
    {
        std::cerr << "collinear: missing H0 (0, 0.5)\n";
        return false;
    }

    return true;
}

// Case 5: Six points forming a hexagon (circle approximation)
//   Points equally spaced on unit circle
//   Expected: H1 essential at radius just above edge length

bool check_hexagon_circle()
{
    std::vector<double> pts;
    constexpr int kN = 6;
    for (int i = 0; i < kN; ++i)
    {
        const double angle = 2.0 * M_PI * i / kN;
        pts.push_back(std::cos(angle));
        pts.push_back(std::sin(angle));
    }

    // Use max_radius just above edge length (side = 1.0 for unit hexagon)
    // to see H1 essential (cycle) without triangles filling
    VRConfig config;
    config.max_radius = 1.5;
    config.max_dim = 2;
    config.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    const auto pairs =
        canonical(nerve::persistence::computeVrPersistenceFast(view_of(pts), 2, config));

    // Must have exactly 1 H0 essential
    if (count_essential_by_dim(pairs, 0) != 1)
    {
        std::cerr << "hexagon: expected 1 H0 essential, got " << count_essential_by_dim(pairs, 0)
                  << "\n";
        return false;
    }

    // H1 should exist as an essential class (the hexagon encloses a hole)
    // The 1-cycle forms at t=1 (all edges present) and persists since no
    // 2-simplex fills it within radius 1.5
    if (count_essential_by_dim(pairs, 1) != 1)
    {
        std::cerr << "hexagon: expected 1 H1 essential, got " << count_essential_by_dim(pairs, 1)
                  << "\n";
        for (const auto &p : pairs)
        {
            if (p.dimension == 1)
                std::cerr << "  H1: birth=" << p.birth << " death=" << p.death << "\n";
        }
        return false;
    }

    return true;
}

// Case 6: Empty point set
//   Expected: zero pairs

bool check_empty_point_set()
{
    const std::vector<double> pts;
    VRConfig config;
    config.max_radius = 1.0;
    config.max_dim = 2;
    config.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    const auto pairs = nerve::persistence::computeVrPersistenceFast(view_of(pts), 2, config);

    if (!pairs.empty())
    {
        std::cerr << "empty: expected 0 pairs, got " << pairs.size() << "\n";
        return false;
    }
    return true;
}

// Case 7: Single point
//   Expected: 1 H0 essential pair

bool check_single_point()
{
    const std::vector<double> pts = {0.0, 0.0};
    VRConfig config;
    config.max_radius = 1.0;
    config.max_dim = 2;
    config.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    const auto pairs = nerve::persistence::computeVrPersistenceFast(view_of(pts), 2, config);

    if (pairs.size() != 1)
    {
        std::cerr << "single_point: expected 1 pair, got " << pairs.size() << "\n";
        return false;
    }
    if (!has_pair(pairs, 0, 0.0, std::numeric_limits<double>::infinity()))
    {
        std::cerr << "single_point: expected H0 essential (0, inf)\n";
        return false;
    }
    return true;
}

// Case 8: Regular tetrahedron (all edges = 1)
//   All simplices appear at t=1 -> all higher homology is trivial
//   H0: 3 finite (0,1) + 1 essential
//   H1: 3 trivial (1,1)
//   H2: 1 trivial (1,1)

bool check_tetrahedron()
{
    const double r3o2 = std::sqrt(3.0) / 2.0;
    const double r3o6 = std::sqrt(3.0) / 6.0;
    const double r6o3 = std::sqrt(6.0) / 3.0;

    const std::vector<double> pts = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, r3o2, 0.0, 0.5, r3o6, r6o3};

    VRConfig config;
    config.max_radius = 1.5;
    config.max_dim = 3;
    config.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    const auto pairs =
        canonical(nerve::persistence::computeVrPersistenceFast(view_of(pts), 3, config));

    // H0: 3 finite merges + 1 essential
    if (count_finite_by_dim(pairs, 0) != 3)
    {
        std::cerr << "tetrahedron: expected 3 finite H0, got " << count_finite_by_dim(pairs, 0)
                  << "\n";
        return false;
    }
    if (count_essential_by_dim(pairs, 0) != 1)
    {
        std::cerr << "tetrahedron: expected 1 H0 essential, got "
                  << count_essential_by_dim(pairs, 0) << "\n";
        return false;
    }

    // All finite H0 at birth=0, death=1
    for (const auto &p : pairs)
    {
        if (p.dimension == 0 && !p.isInfinite())
        {
            if (std::abs(p.birth) > 1e-12 || std::abs(p.death - 1.0) > 1e-10)
            {
                std::cerr << "tetrahedron: H0 expected (0, 1), got (" << p.birth << ", " << p.death
                          << ")\n";
                return false;
            }
        }
    }

    return true;
}

// Case 9: Parallel algorithms equivalence
//   Single-threaded and multi-threaded should give same pairs

bool check_parallel_equivalence()
{
    constexpr int kN = 20;
    constexpr int kDim = 2;
    std::vector<double> pts(static_cast<size_t>(kN) * kDim);
    for (int i = 0; i < kN; ++i)
    {
        pts[static_cast<size_t>(i) * kDim] = static_cast<double>(i % 5) * 0.3;
        pts[static_cast<size_t>(i) * kDim + 1] = static_cast<double>(i / 5) * 0.3;
    }

    VRConfig config;
    config.max_radius = 1.0;
    config.max_dim = 2;
    config.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    const auto single =
        canonical(nerve::persistence::computeVrPersistenceFast(view_of(pts), 2, config));

    // Run again with different algorithm path
    VRConfig simd_config = config;
    simd_config.algorithm = VRAlgorithmSelection::FAST_SIMD;
    const auto simd =
        canonical(nerve::persistence::computeVrPersistenceFast(view_of(pts), 2, simd_config));

    if (single.size() != simd.size())
    {
        std::cerr << "parallel_eq: EXACT has " << single.size() << " pairs, FAST_SIMD has "
                  << simd.size() << "\n";
        return false;
    }

    for (size_t i = 0; i < single.size(); ++i)
    {
        if (single[i].dimension != simd[i].dimension ||
            std::abs(single[i].birth - simd[i].birth) > 1e-10)
        {
            std::cerr << "parallel_eq: pair " << i << " differs\n";
            return false;
        }
        const bool single_inf = single[i].isInfinite();
        const bool simd_inf = simd[i].isInfinite();
        if (single_inf != simd_inf)
            return false;
        if (!single_inf && std::abs(single[i].death - simd[i].death) > 1e-10)
        {
            return false;
        }
    }

    return true;
}

// Case 10: Exact medium configuration equivalence

bool check_medium_hybrid_equivalence()
{
    const std::vector<double> pts = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0};
    VRConfig config;
    config.max_radius = 2.0;
    config.max_dim = 2;
    config.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    const auto exact =
        canonical(nerve::persistence::computeVrPersistenceFast(view_of(pts), 2, config));

    // With AUTO selection, result should also be correct
    VRConfig auto_config = config;
    auto_config.algorithm = VRAlgorithmSelection::AUTO;
    const auto auto_result =
        canonical(nerve::persistence::computeVrPersistenceFast(view_of(pts), 2, auto_config));

    if (exact.size() != auto_result.size())
    {
        std::cerr << "medium_eq: size mismatch " << exact.size() << " vs " << auto_result.size()
                  << "\n";
        return false;
    }

    for (size_t i = 0; i < exact.size(); ++i)
    {
        if (exact[i].dimension != auto_result[i].dimension ||
            std::abs(exact[i].birth - auto_result[i].birth) > 1e-10)
        {
            std::cerr << "medium_eq: pair " << i << " differs\n";
            return false;
        }
        const bool exact_inf = exact[i].isInfinite();
        const bool auto_inf = auto_result[i].isInfinite();
        if (exact_inf != auto_inf)
            return false;
        if (!exact_inf && std::abs(exact[i].death - auto_result[i].death) > 1e-10)
        {
            return false;
        }
    }

    return true;
}

} // namespace

int main()
{
    // Edge cases
    if (!check_empty_point_set())
    {
        std::cerr << "FAIL: empty point set\n";
        return 1;
    }
    if (!check_single_point())
    {
        std::cerr << "FAIL: single point\n";
        return 1;
    }

    // Known shapes
    if (!check_two_points())
    {
        std::cerr << "FAIL: two points\n";
        return 1;
    }
    if (!check_three_collinear_points())
    {
        std::cerr << "FAIL: three collinear points\n";
        return 1;
    }
    if (!check_equilateral_triangle())
    {
        std::cerr << "FAIL: equilateral triangle\n";
        return 1;
    }
    if (!check_square())
    {
        std::cerr << "FAIL: square\n";
        return 1;
    }
    if (!check_hexagon_circle())
    {
        std::cerr << "FAIL: hexagon circle\n";
        return 1;
    }
    if (!check_tetrahedron())
    {
        std::cerr << "FAIL: tetrahedron\n";
        return 1;
    }

    // Algorithm equivalence
    if (!check_parallel_equivalence())
    {
        std::cerr << "FAIL: parallel equivalence\n";
        return 1;
    }
    if (!check_medium_hybrid_equivalence())
    {
        std::cerr << "FAIL: medium hybrid equivalence\n";
        return 1;
    }

    return 0;
}

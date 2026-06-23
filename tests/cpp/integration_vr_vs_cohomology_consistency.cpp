#include "nerve/common/accelerated_types.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core_types.hpp"
#include "nerve/persistence/vr/vr_cohomology_ops.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
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

bool check_triangle_homology_vs_cohomology()
{
    const std::vector<double> tri = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0};
    VRConfig cfg;
    cfg.max_radius = 2.0;
    cfg.max_dim = 2;
    cfg.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    const auto homology = nerve::persistence::computeVrPersistenceFast(view_of(tri), 2, cfg);
    const auto coho = nerve::persistence::computeCohomologyVR(tri, 3, 2, 2.0);

    if (!assert_same_pairs(homology, coho.pairs))
    {
        std::cerr << "FAIL: triangle homology vs cohomology\n";
        return false;
    }
    Size h0_essential = 0;
    for (const auto &p : homology)
        if (p.dimension == 0 && p.isInfinite())
            ++h0_essential;
    if (h0_essential != 1)
    {
        std::cerr << "triangle: expected 1 H0 essential, got " << h0_essential << "\n";
        return false;
    }
    return true;
}

bool check_square_homology_vs_cohomology()
{
    const std::vector<double> sq = {0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0};
    VRConfig cfg;
    cfg.max_radius = 2.0;
    cfg.max_dim = 2;
    cfg.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    const auto homology = nerve::persistence::computeVrPersistenceFast(view_of(sq), 2, cfg);
    const auto coho = nerve::persistence::computeCohomologyVR(sq, 4, 2, 2.0);

    if (!assert_same_pairs(homology, coho.pairs))
    {
        std::cerr << "FAIL: square homology vs cohomology\n";
        return false;
    }
    Size h1_essential = 0;
    for (const auto &p : homology)
        if (p.dimension == 1 && p.isInfinite())
            ++h1_essential;
    if (h1_essential != 0)
    {
        std::cerr << "square: expected 0 H1 essential, got " << h1_essential << "\n";
        return false;
    }
    return true;
}

bool check_tetrahedron_homology_vs_cohomology()
{
    const std::vector<double> tet = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    VRConfig cfg;
    cfg.max_radius = 2.0;
    cfg.max_dim = 3;
    cfg.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    const auto homology = nerve::persistence::computeVrPersistenceFast(view_of(tet), 3, cfg);
    const auto coho = nerve::persistence::computeCohomologyVR(tet, 4, 3, 2.0);

    if (!assert_same_pairs(homology, coho.pairs))
    {
        std::cerr << "FAIL: tetrahedron homology vs cohomology\n";
        return false;
    }
    Size h2_essential = 0;
    for (const auto &p : homology)
        if (p.dimension == 2 && p.isInfinite())
            ++h2_essential;
    if (h2_essential != 0)
    {
        std::cerr << "tetrahedron: expected 0 H2 essential, got " << h2_essential << "\n";
        return false;
    }
    return true;
}

bool check_triangle_exact_vs_fast_simd()
{
    const std::vector<double> tri = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0};
    VRConfig exact_cfg;
    exact_cfg.max_radius = 2.0;
    exact_cfg.max_dim = 2;
    exact_cfg.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    VRConfig simd_cfg = exact_cfg;
    simd_cfg.algorithm = VRAlgorithmSelection::FAST_SIMD;

    const auto exact = nerve::persistence::computeVrPersistenceFast(view_of(tri), 2, exact_cfg);
    const auto simd = nerve::persistence::computeVrPersistenceFast(view_of(tri), 2, simd_cfg);

    if (!assert_same_pairs(exact, simd))
    {
        std::cerr << "FAIL: triangle EXACT vs FAST_SIMD mismatch\n";
        return false;
    }
    return true;
}

bool check_square_exact_vs_fast_simd()
{
    const std::vector<double> sq = {0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0};
    VRConfig exact_cfg;
    exact_cfg.max_radius = 2.0;
    exact_cfg.max_dim = 2;
    exact_cfg.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    VRConfig simd_cfg = exact_cfg;
    simd_cfg.algorithm = VRAlgorithmSelection::FAST_SIMD;

    const auto exact = nerve::persistence::computeVrPersistenceFast(view_of(sq), 2, exact_cfg);
    const auto simd = nerve::persistence::computeVrPersistenceFast(view_of(sq), 2, simd_cfg);

    if (!assert_same_pairs(exact, simd))
    {
        std::cerr << "FAIL: square EXACT vs FAST_SIMD mismatch\n";
        return false;
    }
    return true;
}

bool check_tetrahedron_exact_vs_fast_simd()
{
    const std::vector<double> tet = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    VRConfig exact_cfg;
    exact_cfg.max_radius = 2.0;
    exact_cfg.max_dim = 3;
    exact_cfg.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    VRConfig simd_cfg = exact_cfg;
    simd_cfg.algorithm = VRAlgorithmSelection::FAST_SIMD;

    const auto exact = nerve::persistence::computeVrPersistenceFast(view_of(tet), 3, exact_cfg);
    const auto simd = nerve::persistence::computeVrPersistenceFast(view_of(tet), 3, simd_cfg);

    if (!assert_same_pairs(exact, simd))
    {
        std::cerr << "FAIL: tetrahedron EXACT vs FAST_SIMD mismatch\n";
        return false;
    }
    return true;
}

} // namespace

int main()
{
    if (!check_triangle_homology_vs_cohomology())
    {
        std::cerr << "FAIL: triangle homology vs cohomology\n";
        return 1;
    }
    if (!check_square_homology_vs_cohomology())
    {
        std::cerr << "FAIL: square homology vs cohomology\n";
        return 1;
    }
    if (!check_tetrahedron_homology_vs_cohomology())
    {
        std::cerr << "FAIL: tetrahedron homology vs cohomology\n";
        return 1;
    }
    if (!check_triangle_exact_vs_fast_simd())
    {
        std::cerr << "FAIL: triangle EXACT vs FAST_SIMD\n";
        return 1;
    }
    if (!check_square_exact_vs_fast_simd())
    {
        std::cerr << "FAIL: square EXACT vs FAST_SIMD\n";
        return 1;
    }
    if (!check_tetrahedron_exact_vs_fast_simd())
    {
        std::cerr << "FAIL: tetrahedron EXACT vs FAST_SIMD\n";
        return 1;
    }
    return 0;
}

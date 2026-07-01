#include "nerve/common/accelerated_types.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core_types.hpp"
#include "nerve/persistence/vr/vr_cohomology_ops.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

namespace
{

using nerve::Dimension;
using nerve::Size;
using nerve::common::VRAlgorithmSelection;
using nerve::common::VRConfig;
using nerve::core::BufferView;
using nerve::persistence::Pair;

constexpr double kStrictTol = 1e-10;

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

bool pairs_match(const Pair &a, const Pair &b, double tol = kStrictTol)
{
    if (a.dimension != b.dimension)
        return false;
    if (std::abs(a.birth - b.birth) > tol)
        return false;
    if (a.isInfinite() && b.isInfinite())
        return true;
    if (a.isInfinite() || b.isInfinite())
        return false;
    return std::abs(a.death - b.death) < tol;
}

template <typename Container>
bool assert_pairs_equal(const Container &expected, const Container &actual,
                        const std::string &label, double tol = kStrictTol)
{
    const auto c1 = canonical(
        std::vector<Pair>(expected.begin(), expected.end()));
    const auto c2 = canonical(
        std::vector<Pair>(actual.begin(), actual.end()));
    if (c1.size() != c2.size())
    {
        std::cerr << "FAIL [" << label << "]: pair count mismatch -- "
                  << c1.size() << " vs " << c2.size() << "\n";
        for (const auto &p : c1)
            std::cerr << "  expected: dim=" << p.dimension << " birth=" << p.birth
                      << " death=" << p.death << "\n";
        for (const auto &p : c2)
            std::cerr << "  actual:   dim=" << p.dimension << " birth=" << p.birth
                      << " death=" << p.death << "\n";
        return false;
    }
    for (std::size_t i = 0; i < c1.size(); ++i)
    {
        if (!pairs_match(c1[i], c2[i], tol))
        {
        std::cerr << "FAIL [" << label << "]: pair " << i
                  << " differs -- expected dim=" << c1[i].dimension
                      << " birth=" << c1[i].birth << " death=" << c1[i].death
                      << ", got dim=" << c2[i].dimension
                      << " birth=" << c2[i].birth << " death=" << c2[i].death << "\n";
            return false;
        }
    }
    return true;
}

/// Count essential (infinite-death) pairs in a given dimension.
Size count_essential(const std::vector<Pair> &pairs, Dimension dim)
{
    Size n = 0;
    for (const auto &p : pairs)
        if (p.dimension == dim && p.isInfinite())
            ++n;
    return n;
}

/// Count finite (non-infinite) pairs in a given dimension.
Size count_finite(const std::vector<Pair> &pairs, Dimension dim)
{
    Size n = 0;
    for (const auto &p : pairs)
        if (p.dimension == dim && !p.isInfinite())
            ++n;
    return n;
}

/// Check that no pairs have negative or extremely tiny persistence.
bool all_persistence_nonnegative(const std::vector<Pair> &pairs,
                                 const std::string &label)
{
    for (const auto &p : pairs)
    {
        if (!p.isInfinite() && p.death < p.birth - kStrictTol)
        {
            std::cerr << "FAIL [" << label << "]: negative persistence -- dim="
                      << p.dimension << " birth=" << p.birth
                      << " death=" << p.death << "\n";
            return false;
        }
    }
    return true;
}

/// Standard 3-simplex vertices: (0,0,0), (1,0,0), (0,1,0), (0,0,1).
/// Edge lengths:  3 edges of length 1,  3 edges of length sqrt(2) ~ 1.414.
/// VR topology:
///   r < 1          : 4 components                     -> betti-0 = 4
///   1 <= r < sqrt(2)     : 1 component (star graph)         -> betti-0 = 1
///   r >= sqrt(2)         : contractible 3-ball              -> betti-0 = 1, betti-1 = 0, betti-2 = 0
/// Faces and the 3-simplex all appear at r = sqrt(2), so every
/// H1/H2 pair has zero persistence and should be filtered.
bool check_tetrahedron_dim2_cohomology()
{
    const std::vector<double> tet = {
        0.0, 0.0, 0.0,  // v0
        1.0, 0.0, 0.0,  // v1
        0.0, 1.0, 0.0,  // v2
        0.0, 0.0, 1.0   // v3
    };

    {
        VRConfig cfg;
        cfg.max_radius = 2.0;
        cfg.max_dim = 2;
        cfg.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

        const auto homology = nerve::persistence::computeVrPersistenceFast(
            view_of(tet), 2, cfg);
        const auto coho = nerve::persistence::computeCohomologyVR(tet, 4, 2, 2.0);

        if (!assert_pairs_equal(homology, coho.pairs,
                                "tet-maxdim2-homology-vs-cohomology"))
            return false;

        if (!all_persistence_nonnegative(homology,
                                         "tet-maxdim2-persistence-sign"))
            return false;

        // H0: exactly 1 essential (connected component that never dies)
        if (count_essential(homology, 0) != 1)
        {
            std::cerr << "FAIL tet-maxdim2: expected 1 H0 essential, got "
                      << count_essential(homology, 0) << "\n";
            return false;
        }

        // H1/H2: no essential classes in a contractible space
        if (count_essential(homology, 2) != 0)
        {
            std::cerr << "FAIL tet-maxdim2: expected 0 H2 essential, got "
                      << count_essential(homology, 2) << "\n";
            return false;
        }
    }

    {
        VRConfig cfg;
        cfg.max_radius = 2.0;
        cfg.max_dim = 3;
        cfg.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

        const auto homology = nerve::persistence::computeVrPersistenceFast(
            view_of(tet), 3, cfg);
        const auto coho = nerve::persistence::computeCohomologyVR(tet, 4, 3, 2.0);

        if (!assert_pairs_equal(homology, coho.pairs,
                                "tet-maxdim3-homology-vs-cohomology"))
            return false;

        if (!all_persistence_nonnegative(homology,
                                         "tet-maxdim3-persistence-sign"))
            return false;

        if (count_essential(homology, 0) != 1)
        {
            std::cerr << "FAIL tet-maxdim3: expected 1 H0 essential, got "
                      << count_essential(homology, 0) << "\n";
            return false;
        }

        for (Dimension d = 1; d <= 3; ++d)
        {
            if (count_essential(homology, d) != 0)
            {
                std::cerr << "FAIL tet-maxdim3: expected 0 H" << d
                          << " essential, got " << count_essential(homology, d) << "\n";
                return false;
            }
        }
    }

    return true;
}

/// Square vertices: (0,0), (1,0), (1,1), (0,1).
/// Edge lengths:  side edges = 1,  diagonals = sqrt(2) ~ 1.414.
/// VR topology:
///   r < 1          : 4 components                     -> betti-0 = 4
///   1 <= r < sqrt(2)     : 1 component, 1 H1 cycle         -> betti-0 = 1, betti-1 = 1
///   r >= sqrt(2)         : filled square (diagonals)        -> betti-0 = 1, betti-1 = 0, betti-2 = 0
/// At max_radius = 2.0, the H1 cycle dies at r = sqrt(2) -> finite pair only.
bool check_square_dim2_cohomology()
{
    const std::vector<double> sq = {0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0};

    VRConfig cfg;
    cfg.max_radius = 2.0;
    cfg.max_dim = 2;
    cfg.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    const auto homology = nerve::persistence::computeVrPersistenceFast(
        view_of(sq), 2, cfg);
    const auto coho = nerve::persistence::computeCohomologyVR(sq, 4, 2, 2.0);

    if (!assert_pairs_equal(homology, coho.pairs,
                            "square-dim2-homology-vs-cohomology"))
        return false;

    if (!all_persistence_nonnegative(homology, "square-dim2-persistence-sign"))
        return false;

    // H0: exactly 1 essential
    if (count_essential(homology, 0) != 1)
    {
        std::cerr << "FAIL square-dim2: expected 1 H0 essential, got "
                  << count_essential(homology, 0) << "\n";
        return false;
    }

    // H1: at least 1 pair (the cycle dies when diagonals appear at r=sqrt(2))
    if (count_essential(homology, 1) + count_finite(homology, 1) == 0)
    {
        std::cerr << "FAIL square-dim2: expected at least 1 H1 pair, got 0\n";
        return false;
    }

    // H2: no essential classes
    if (count_essential(homology, 2) != 0)
    {
        std::cerr << "FAIL square-dim2: expected 0 H2 essential, got "
                  << count_essential(homology, 2) << "\n";
        return false;
    }

    return true;
}

/// 5 points in 3D: origin + 4 axis-aligned points at distance 1.
/// This exercises dim-2 cohomology on a non-trivial 3D point cloud.
/// Only invariants checked: homology-cohomology agreement,
/// non-negative persistence, H0 essential = 1.
bool check_octahedron_scaffold_dim2()
{
    const std::vector<double> pts = {
        0.0, 0.0, 0.0,   // v0 - origin
        1.0, 0.0, 0.0,   // v1
        -1.0, 0.0, 0.0,  // v2
        0.0, 1.0, 0.0,   // v3
        0.0, 0.0, 1.0    // v4
    };

    VRConfig cfg;
    cfg.max_radius = 3.0;
    cfg.max_dim = 2;
    cfg.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    const auto homology = nerve::persistence::computeVrPersistenceFast(
        view_of(pts), 3, cfg);
    const auto coho = nerve::persistence::computeCohomologyVR(pts, 5, 3, 3.0);

    // Cohomology may produce dim-3 pairs; filter to dim <= 2 for comparison
    std::vector<Pair> coho_dim2;
    for (const auto &p : coho.pairs)
        if (p.dimension <= 2)
            coho_dim2.push_back(p);

    if (!assert_pairs_equal(homology, coho_dim2,
                            "octahedron-dim2-homology-vs-cohomology"))
        return false;

    if (!all_persistence_nonnegative(homology,
                                     "octahedron-dim2-persistence-sign"))
        return false;

    // H0: exactly 1 essential (all points eventually connect)
    if (count_essential(homology, 0) != 1)
    {
        std::cerr << "FAIL octahedron-dim2: expected 1 H0 essential, got "
                  << count_essential(homology, 0) << "\n";
        return false;
    }

    return true;
}

} // namespace

int main()
{
    bool ok = true;

    std::cout << "dim-2 cohomology: tetrahedron ..." << std::endl;
    if (!check_tetrahedron_dim2_cohomology())
    {
        std::cerr << "FAIL: tetrahedron dim-2 cohomology\n";
        ok = false;
    }
    else
    {
        std::cout << "  PASS\n";
    }

    std::cout << "dim-2 cohomology: square ..." << std::endl;
    if (!check_square_dim2_cohomology())
    {
        std::cerr << "FAIL: square dim-2 cohomology\n";
        ok = false;
    }
    else
    {
        std::cout << "  PASS\n";
    }

    std::cout << "dim-2 cohomology: octahedron scaffold ..." << std::endl;
    if (!check_octahedron_scaffold_dim2())
    {
        std::cerr << "FAIL: octahedron-scaffold dim-2 cohomology\n";
        ok = false;
    }
    else
    {
        std::cout << "  PASS\n";
    }

    return ok ? 0 : 1;
}

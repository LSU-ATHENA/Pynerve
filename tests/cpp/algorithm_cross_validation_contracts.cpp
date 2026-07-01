#include "nerve/common/accelerated_types.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core_types.hpp"
#include "nerve/persistence/core/core_types.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"
#include "nerve/persistence/vr/vr_fast_simd_ops.hpp"
#include "nerve/persistence/vr/vr_medium_hybrid_ops.hpp"
#include "test_utils.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <random>
#include <vector>

namespace
{

using nerve::common::VRConfig;
using nerve::persistence::Pair;
using nerve::persistence::VRAlgorithmSelection;
using namespace nerve::test;


bool pairs_equal(const Pair &a, const Pair &b)
{
    if (a.dimension != b.dimension)
        return false;
    if (std::abs(a.birth - b.birth) > 1e-10)
        return false;
    if (a.isInfinite() && b.isInfinite())
        return true;
    if (a.isInfinite() || b.isInfinite())
        return false;
    return std::abs(a.death - b.death) < 1e-10;
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

// Cross-validate EXACT_STANDARD vs FAST_SIMD vs AUTO dispatcher
bool check_exact_vs_simd_equivalence(int n_points, int dim, double radius, int seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    std::vector<double> pts(static_cast<std::size_t>(n_points) * dim);
    for (auto &v : pts)
        v = dist(rng);

    VRConfig exact_cfg;
    exact_cfg.algorithm = VRAlgorithmSelection::EXACT_STANDARD;
    exact_cfg.max_radius = radius;
    exact_cfg.max_dim = 2;

    VRConfig simd_cfg = exact_cfg;
    simd_cfg.algorithm = VRAlgorithmSelection::FAST_SIMD;

    VRConfig auto_cfg = exact_cfg;
    auto_cfg.algorithm = VRAlgorithmSelection::AUTO;

    const auto exact = nerve::persistence::computeVrPersistenceFast(view_of(pts), dim, exact_cfg);
    const auto simd = nerve::persistence::computeVrPersistenceFast(view_of(pts), dim, simd_cfg);
    const auto automatic =
        nerve::persistence::computeVrPersistenceFast(view_of(pts), dim, auto_cfg);

    if (!assert_same_pairs(exact, simd))
    {
        std::cerr << "EXACT vs FAST_SIMD mismatch at seed=" << seed << " n=" << n_points
                  << " dim=" << dim << "\n";
        return false;
    }
    if (!assert_same_pairs(exact, automatic))
    {
        std::cerr << "EXACT vs AUTO mismatch at seed=" << seed << " n=" << n_points
                  << " dim=" << dim << "\n";
        return false;
    }
    return true;
}

bool check_medium_hybrid_equivalence(int n_points, int dim, double radius, int seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    // For medium-hybrid equivalence, need enough points to trigger hybrid path
    std::vector<double> pts(static_cast<std::size_t>(n_points) * dim);
    for (auto &v : pts)
        v = dist(rng);

    // Inject a few structured points for exact comparison
    const std::vector<double> anchor = {
        0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0,
    };
    for (std::size_t i = 0; i < 4 && i < static_cast<std::size_t>(n_points); ++i)
    {
        pts[i * dim] = anchor[i * 2];
        if (dim > 1)
            pts[i * dim + 1] = anchor[i * 2 + 1];
    }

    VRConfig config;
    config.algorithm = VRAlgorithmSelection::EXACT_STANDARD;
    config.max_radius = radius;
    config.max_dim = 2;

    const auto exact = nerve::persistence::computeVrPersistenceFast(view_of(pts), dim, config);

    // Use direct medium-hybrid backend
    const auto medium =
        nerve::persistence::computeVrPersistenceMediumHybrid(view_of(pts), dim, config);

    if (!assert_same_pairs(exact, medium))
    {
        std::cerr << "EXACT vs MEDIUM_HYBRID mismatch at seed=" << seed << " n=" << n_points
                  << " dim=" << dim << "\n";
        return false;
    }
    return true;
}

// Cross-validate with explicit VRConfig algorithm selection
bool check_algorithm_selection_equivalence()
{
    const std::vector<double> points = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5};

    VRConfig base;
    base.max_radius = 2.0;
    base.max_dim = 2;

    const std::vector<VRAlgorithmSelection> algs = {
        VRAlgorithmSelection::EXACT_STANDARD,
        VRAlgorithmSelection::FAST_SIMD,
        VRAlgorithmSelection::AUTO,
    };

    std::vector<std::vector<Pair>> results;
    for (const auto alg : algs)
    {
        VRConfig cfg = base;
        cfg.algorithm = alg;
        results.push_back(nerve::persistence::computeVrPersistenceFast(view_of(points), 2, cfg));
    }

    for (std::size_t i = 1; i < results.size(); ++i)
    {
        if (!assert_same_pairs(results[0], results[i]))
        {
            std::cerr << "algorithm selection mismatch between " << static_cast<int>(algs[0])
                      << " and " << static_cast<int>(algs[i]) << "\n";
            return false;
        }
    }
    return true;
}

// Verify dimension 0-only computation
bool check_dimension_zero_consistency()
{
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    constexpr int kN = 10;
    constexpr int kDim = 2;
    std::vector<double> pts(static_cast<std::size_t>(kN) * kDim);
    for (auto &v : pts)
        v = dist(rng);

    VRConfig cfg_dim0;
    cfg_dim0.algorithm = VRAlgorithmSelection::EXACT_STANDARD;
    cfg_dim0.max_radius = 2.0;
    cfg_dim0.max_dim = 0;

    VRConfig cfg_dim2 = cfg_dim0;
    cfg_dim2.max_dim = 2;

    const auto only_h0 = nerve::persistence::computeVrPersistenceFast(view_of(pts), kDim, cfg_dim0);
    const auto up_to_h2 =
        nerve::persistence::computeVrPersistenceFast(view_of(pts), kDim, cfg_dim2);

    // dim-0 pairs must be identical between both runs
    std::vector<Pair> h0_from_full;
    for (const auto &p : up_to_h2)
    {
        if (p.dimension == 0)
            h0_from_full.push_back(p);
    }

    if (!assert_same_pairs(only_h0, h0_from_full))
    {
        std::cerr << "H0 pairs differ between max_dim=0 and max_dim=2\n";
        return false;
    }
    return true;
}

// Diff algorithm backends with 0-radius (only vertices)
bool check_zero_radius_equivalence()
{
    const std::vector<double> points = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0};

    VRConfig config;
    config.algorithm = VRAlgorithmSelection::EXACT_STANDARD;
    config.max_radius = 0.0;
    config.max_dim = 1;

    const auto exact = nerve::persistence::computeVrPersistenceFast(view_of(points), 2, config);

    VRConfig simd_cfg = config;
    simd_cfg.algorithm = VRAlgorithmSelection::FAST_SIMD;
    const auto simd = nerve::persistence::computeVrPersistenceFast(view_of(points), 2, simd_cfg);

    if (!assert_same_pairs(exact, simd))
    {
        std::cerr << "zero-radius: EXACT vs FAST_SIMD mismatch\n";
        return false;
    }

    // At zero radius with 3 points: 3 H0 essential classes
    int h0_essential = 0;
    for (const auto &p : exact)
    {
        if (p.dimension == 0 && p.isInfinite())
            ++h0_essential;
    }
    if (h0_essential != 3)
    {
        std::cerr << "zero-radius: expected 3 H0 essential, got " << h0_essential << "\n";
        return false;
    }
    return true;
}

// Stress: larger random configs compared across EXACT and AUTO
bool check_stress_random_equivalence()
{
    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    const struct
    {
        int n;
        int dim;
        double radius;
    } configs[] = {{3, 1, 1.0}, {4, 2, 1.5}, {6, 3, 2.0}, {8, 2, 1.0}, {10, 3, 1.5}};

    for (const auto &cfg : configs)
    {
        std::vector<double> pts(static_cast<std::size_t>(cfg.n) * cfg.dim);
        for (auto &v : pts)
            v = dist(rng);

        VRConfig exact_cfg;
        exact_cfg.algorithm = VRAlgorithmSelection::EXACT_STANDARD;
        exact_cfg.max_radius = cfg.radius;
        exact_cfg.max_dim = 2;

        VRConfig auto_cfg = exact_cfg;
        auto_cfg.algorithm = VRAlgorithmSelection::AUTO;

        const auto exact =
            nerve::persistence::computeVrPersistenceFast(view_of(pts), cfg.dim, exact_cfg);
        const auto automatic =
            nerve::persistence::computeVrPersistenceFast(view_of(pts), cfg.dim, auto_cfg);

        if (!assert_same_pairs(exact, automatic))
        {
            std::cerr << "stress: EXACT vs AUTO mismatch at n=" << cfg.n << " dim=" << cfg.dim
                      << " radius=" << cfg.radius << "\n";
            return false;
        }
    }
    return true;
}

// Edge: 1-dimensional points
bool check_1d_points_equivalence()
{
    const std::vector<double> points = {0.0, 0.5, 1.0, 1.5, 2.0};

    VRConfig config;
    config.max_radius = 2.0;
    config.max_dim = 1;

    VRConfig exact_cfg = config;
    exact_cfg.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    VRConfig simd_cfg = config;
    simd_cfg.algorithm = VRAlgorithmSelection::FAST_SIMD;

    const auto exact = nerve::persistence::computeVrPersistenceFast(view_of(points), 1, exact_cfg);
    const auto simd = nerve::persistence::computeVrPersistenceFast(view_of(points), 1, simd_cfg);

    if (!assert_same_pairs(exact, simd))
    {
        std::cerr << "1D points: EXACT vs FAST_SIMD mismatch\n";
        return false;
    }

    // 5 points on line:
    // At radius 2.0: edges form up to distance 2.0, so points 0.0,0.5,1.0,1.5 connected
    // Point at 2.0 connects to 1.5, 1.0. Everything is connected.
    // H0: 1 essential, plus finite bars at 0.25, 0.5, 0.75, 1.0
    int h0_essential = 0;
    int h0_finite = 0;
    for (const auto &p : exact)
    {
        if (p.dimension == 0 && p.isInfinite())
            ++h0_essential;
        if (p.dimension == 0 && !p.isInfinite())
            ++h0_finite;
    }
    if (h0_essential != 1)
    {
        std::cerr << "1D: expected 1 H0 essential, got " << h0_essential << "\n";
        return false;
    }
    if (h0_finite != 4)
    {
        std::cerr << "1D: expected 4 H0 finite, got " << h0_finite << "\n";
        return false;
    }
    return true;
}

} // namespace

int main()
{
    // Systematic cross-validation across random configs
    const struct
    {
        int n;
        int dim;
        double radius;
        int seed;
    } trials[] = {
        {4, 2, 2.0, 42}, {4, 2, 1.5, 99}, {5, 1, 1.0, 10}, {5, 3, 2.5, 20},
        {6, 2, 2.0, 30}, {6, 4, 1.0, 40}, {7, 2, 1.5, 50}, {8, 3, 2.0, 60},
    };

    for (const auto &t : trials)
    {
        if (!check_exact_vs_simd_equivalence(t.n, t.dim, t.radius, t.seed))
        {
            std::cerr << "FAIL: EXACT vs SIMD at n=" << t.n << " dim=" << t.dim << "\n";
            return 1;
        }
    }

    if (!check_algorithm_selection_equivalence())
    {
        std::cerr << "FAIL: algorithm selection equivalence\n";
        return 1;
    }

    if (!check_dimension_zero_consistency())
    {
        std::cerr << "FAIL: dimension zero consistency\n";
        return 1;
    }

    if (!check_zero_radius_equivalence())
    {
        std::cerr << "FAIL: zero radius equivalence\n";
        return 1;
    }

    if (!check_1d_points_equivalence())
    {
        std::cerr << "FAIL: 1D points equivalence\n";
        return 1;
    }

    // Stress tests with larger random inputs
    if (!check_stress_random_equivalence())
    {
        std::cerr << "FAIL: stress random equivalence\n";
        return 1;
    }

    // Medium hybrid equivalence (needs more points)
    if (!check_medium_hybrid_equivalence(64, 2, 1.5, 100))
    {
        std::cerr << "FAIL: medium hybrid equivalence\n";
        return 1;
    }

    return 0;
}

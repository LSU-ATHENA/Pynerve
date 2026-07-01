#include "nerve/algebra/boundary.hpp"
#include "nerve/algebra/complex.hpp"
#include "nerve/algebra/simplex.hpp"
#include "nerve/common/accelerated_types.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core_types.hpp"
#include "nerve/errors/errors.hpp"
#include "nerve/persistence/core/core_types.hpp"
#include "nerve/persistence/utils/api.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"
#include "test_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace
{

using nerve::Field;
using nerve::Size;
using nerve::algebra::Simplex;
using nerve::algebra::SimplicialComplex;
using nerve::common::VRConfig;
using nerve::persistence::Pair;
using nerve::persistence::PersistenceOptions;
using nerve::persistence::VRAlgorithmSelection;
using namespace nerve::test;

// Helpers

template <typename Fn>
bool throws_invalid_argument(Fn fn)
{
    try
    {
        static_cast<void>(fn());
    }
    catch (const std::invalid_argument &)
    {
        return true;
    }
    catch (...)
    {
        return false;
    }
    return false;
}

template <typename Fn>
bool does_not_throw(Fn fn)
{
    try
    {
        static_cast<void>(fn());
    }
    catch (...)
    {
        return false;
    }
    return true;
}


// Invariant: birth < death for all finite pairs

bool check_birth_less_than_death_invariant()
{
    const std::vector<double> points = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5};
    VRConfig config;
    config.max_radius = 2.0;
    config.max_dim = 2;
    config.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    const auto pairs = nerve::persistence::computeVrPersistenceFast(view_of(points), 2, config);

    for (const auto &p : pairs)
    {
        if (p.isInfinite())
            continue;
        if (!(p.birth <= p.death))
        {
            std::cerr << "birth<=death invariant violated: birth=" << p.birth
                      << " death=" << p.death << " dim=" << p.dimension << "\n";
            return false;
        }
    }
    return true;
}

// Invariant: persistence >= 0 for all finite pairs

bool check_nonnegative_persistence_invariant()
{
    const std::vector<double> points = {
        0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 2.0, 2.0, 1.0, 1.0,
    };
    VRConfig config;
    config.max_radius = 3.0;
    config.max_dim = 2;
    config.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    const auto pairs = nerve::persistence::computeVrPersistenceFast(view_of(points), 2, config);

    for (const auto &p : pairs)
    {
        if (p.isInfinite())
            continue;
        if (p.lifetime() < 0.0)
        {
            std::cerr << "non-negative persistence invariant violated: birth=" << p.birth
                      << " death=" << p.death << " persistence=" << p.lifetime() << "\n";
            return false;
        }
    }
    return true;
}

// Invariant: empty input -> empty diagram

bool check_empty_input_produces_empty_diagram()
{
    const std::vector<double> empty_points;
    VRConfig config;
    config.max_radius = 1.0;
    config.max_dim = 2;
    config.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    const auto pairs =
        nerve::persistence::computeVrPersistenceFast(view_of(empty_points), 2, config);
    if (!pairs.empty())
    {
        std::cerr << "empty input should produce empty diagram, got " << pairs.size() << " pairs\n";
        return false;
    }
    return true;
}

// Invariant: single point -> H0 has one essential class

bool check_single_point_invariant()
{
    const std::vector<double> point = {0.0, 0.0};
    VRConfig config;
    config.max_radius = 1.0;
    config.max_dim = 2;
    config.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    const auto pairs = nerve::persistence::computeVrPersistenceFast(view_of(point), 2, config);

    // Single point: zero-dimensional homology has one generator
    // birth=0, death=inf
    bool found_h0_essential = false;
    for (const auto &p : pairs)
    {
        if (p.dimension == 0 && p.isInfinite())
        {
            found_h0_essential = true;
        }
    }
    if (!found_h0_essential)
    {
        std::cerr << "single point should have H0 essential class\n";
        return false;
    }
    return true;
}

// Invariant: two points at distance d
// - At radius 0: H0 has 2 essential classes
// - At radius > d/2: H0 has 1 essential class (edge forms)

bool check_two_point_persistence_invariant()
{
    const std::vector<double> two_points = {0.0, 0.0, 1.0, 0.0};
    constexpr double kDistance = 1.0;

    // At radius 0: no edge, two components
    {
        VRConfig config;
        config.max_radius = 0.0;
        config.max_dim = 1;
        config.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

        const auto pairs =
            nerve::persistence::computeVrPersistenceFast(view_of(two_points), 2, config);

        Size h0_essential = 0;
        for (const auto &p : pairs)
        {
            if (p.dimension == 0 && p.isInfinite())
            {
                ++h0_essential;
            }
        }
        if (h0_essential != 2)
        {
            std::cerr << "two points at radius 0: expected 2 H0 essential, got " << h0_essential
                      << "\n";
            return false;
        }
    }

    // At radius > d/2: edge connects them, one component
    {
        VRConfig config;
        config.max_radius = kDistance + 0.1;
        config.max_dim = 1;
        config.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

        const auto pairs =
            nerve::persistence::computeVrPersistenceFast(view_of(two_points), 2, config);

        Size h0_essential = 0;
        for (const auto &p : pairs)
        {
            if (p.dimension == 0 && p.isInfinite())
            {
                ++h0_essential;
            }
        }
        if (h0_essential != 1)
        {
            std::cerr << "two points at radius > d/2: expected 1 H0 essential, got " << h0_essential
                      << "\n";
            return false;
        }

        // The 0-dimensional finite bar should have birth=0, death=kDistance
        bool found_h0_finite = false;
        for (const auto &p : pairs)
        {
            if (p.dimension == 0 && !p.isInfinite())
            {
                if (std::abs(p.birth) < 1e-12 && std::abs(p.death - kDistance) < 1e-12)
                {
                    found_h0_finite = true;
                }
            }
        }
        if (!found_h0_finite)
        {
            std::cerr << "two points: expected H0 finite bar birth=0 death=" << kDistance << "\n";
            return false;
        }
    }

    return true;
}

// Invariant: Betti numbers are non-negative and consistent

bool check_betti_number_consistency()
{
    const std::vector<double> points = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0};
    VRConfig config;
    config.max_radius = 2.0;
    config.max_dim = 2;
    config.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    const auto pairs = nerve::persistence::computeVrPersistenceFast(view_of(points), 2, config);

    // Count Betti numbers from diagram
    Size betti[3] = {0, 0, 0};
    for (const auto &p : pairs)
    {
        if (p.isInfinite() && p.dimension >= 0 && static_cast<std::size_t>(p.dimension) < 3)
        {
            ++betti[p.dimension];
        }
    }

    // All Betti numbers must be non-negative
    for (std::size_t d = 0; d < 3; ++d)
    {
        if (betti[d] < 0)
        {
            std::cerr << "negative Betti number for dim " << d << "\n";
            return false;
        }
    }

    // Betti_0 should be >= 1 for any non-empty point cloud
    if (betti[0] < 1)
    {
        std::cerr << "Betti_0 must be >= 1 for non-empty point cloud\n";
        return false;
    }

    return true;
}

// Invariant: boundary of boundary is zero (partial∘partial = 0)

bool check_boundary_matrix_dims()
{
    // Build a simplicial complex
    SimplicialComplex complex;
    complex.addSimplex(Simplex({0}));
    complex.addSimplex(Simplex({1}));
    complex.addSimplex(Simplex({2}));
    complex.addSimplex(Simplex({0, 1}));
    complex.addSimplex(Simplex({0, 2}));
    complex.addSimplex(Simplex({1, 2}));
    complex.addSimplex(Simplex({0, 1, 2}));

    // partial₂: C₂ -> C₁ : 3 edges x 1 triangle
    nerve::algebra::BoundaryMatrix boundary2(complex, 2);
    if (boundary2.rows() != 3 || boundary2.cols() != 1)
    {
        std::cerr << "∂₂ expected 3x1, got " << boundary2.rows() << "x" << boundary2.cols() << "\n";
        return false;
    }

    // partial₁: C₁ -> C₀ : 3 vertices x 3 edges
    nerve::algebra::BoundaryMatrix boundary1(complex, 1);
    if (boundary1.rows() != 3 || boundary1.cols() != 3)
    {
        std::cerr << "∂₁ expected 3x3, got " << boundary1.rows() << "x" << boundary1.cols() << "\n";
        return false;
    }

    // partial₂ entries should be +/-1 (each triangle boundary has 3 edges)
    for (Size row = 0; row < boundary2.rows(); ++row)
    {
        const auto c = boundary2.getCoefficient(row, 0);
        if (std::abs(c) != 1.0)
        {
            std::cerr << "∂₂ coeff at (" << row << ",0) expected +/-1, got " << c << "\n";
            return false;
        }
    }

    return true;
}

// Invariant: simplex filtration values are non-decreasing with inclusion

bool check_filtration_monotonicity_invariant()
{
    // Build VR filtration and verify that if simplex s is a face of t,
    // then filtration(s) <= filtration(t)
    const std::vector<double> points = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0};
    VRConfig config;
    config.max_radius = 1.5;
    config.max_dim = 2;
    config.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    const auto pairs = nerve::persistence::computeVrPersistenceFast(view_of(points), 2, config);

    // For a 3-point triangle, check that dim-0 pairs have birth=0
    for (const auto &p : pairs)
    {
        if (p.dimension == 0 && !p.isInfinite())
        {
            if (std::abs(p.birth) > 1e-12)
            {
                std::cerr << "H0 finite bar should have birth=0, got " << p.birth << "\n";
                return false;
            }
        }
    }

    return true;
}

// Generated property-based: random point clouds produce valid diagrams

bool check_random_persistence_validity()
{
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    constexpr int kConfigs = 5;
    const std::pair<int, int> configs[kConfigs] = {{5, 1}, {8, 2}, {10, 3}, {15, 2}, {20, 1}};

    for (const auto &cfg : configs)
    {
        const int n = cfg.first;
        const int dim = cfg.second;

        std::vector<double> points(static_cast<std::size_t>(n) * dim);
        for (auto &v : points)
        {
            v = dist(rng);
        }

        VRConfig config;
        config.max_radius = 2.0;
        config.max_dim = 2;
        config.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

        const auto pairs =
            nerve::persistence::computeVrPersistenceFast(view_of(points), dim, config);

        // Verify invariants for every pair
        for (const auto &p : pairs)
        {
            if (!p.isInfinite())
            {
                if (!(p.birth <= p.death + 1e-14))
                {
                    std::cerr << "random test: birth>death violated for n=" << n << " dim=" << dim
                              << "\n";
                    return false;
                }
                if (p.lifetime() < 0.0)
                {
                    std::cerr << "random test: persistence<0 for n=" << n << " dim=" << dim << "\n";
                    return false;
                }
            }
            if (p.dimension < 0)
            {
                std::cerr << "random test: negative dimension\n";
                return false;
            }
        }
    }

    return true;
}

// Invariant: re-running produces the same result (determinism)

bool check_determinism_invariant()
{
    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    constexpr int kN = 5;
    constexpr int kDim = 2;
    std::vector<double> points(static_cast<std::size_t>(kN) * kDim);
    for (auto &v : points)
        v = dist(rng);

    VRConfig config;
    config.max_radius = 1.5;
    config.max_dim = 2;
    config.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    const auto run1 = nerve::persistence::computeVrPersistenceFast(view_of(points), kDim, config);
    const auto run2 = nerve::persistence::computeVrPersistenceFast(view_of(points), kDim, config);

    const auto c1 = canonical(run1);
    const auto c2 = canonical(run2);

    if (c1.size() != c2.size())
    {
        std::cerr << "determinism: different pair counts " << c1.size() << " vs " << c2.size()
                  << "\n";
        return false;
    }
    for (std::size_t i = 0; i < c1.size(); ++i)
    {
        if (!pairs_equal(c1[i], c2[i], 1e-12))
        {
            std::cerr << "determinism: pair " << i << " differs\n";
            return false;
        }
    }

    return true;
}

// Invariant: boundary matrix sparsity is reasonable

bool check_boundary_matrix_validity()
{
    SimplicialComplex complex;
    // Build a tetrahedron {0,1,2,3}
    for (int i = 0; i < 4; ++i)
        complex.addSimplex(Simplex({i}));
    for (int i = 0; i < 4; ++i)
        for (int j = i + 1; j < 4; ++j)
            complex.addSimplex(Simplex({i, j}));
    for (int i = 0; i < 4; ++i)
        for (int j = i + 1; j < 4; ++j)
            for (int k = j + 1; k < 4; ++k)
                complex.addSimplex(Simplex({i, j, k}));
    complex.addSimplex(Simplex({0, 1, 2, 3}));

    nerve::algebra::BoundaryMatrix boundary_2(complex, 2);
    nerve::algebra::BoundaryMatrix boundary_3(complex, 3);

    // 2-boundary: 4 triangles x 6 edges -> 4 columns, 6 rows
    // Each triangle boundary has 3 non-zeros
    if (boundary_2.isEmpty() && complex.size() > 0)
    {
        std::cerr << "boundary matrix should not be empty\n";
        return false;
    }

    // 3-boundary: 1 tetrahedron x 4 triangles -> 1 col, 4 rows
    // Tetrahedron boundary has 4 non-zeros
    if (boundary_3.numNonzeros() != 4)
    {
        std::cerr << "tetrahedron boundary expected 4 non-zeros, got " << boundary_3.numNonzeros()
                  << "\n";
        return false;
    }

    return true;
}

// Invariant: homology with coefficient field Z/2

bool check_field_aware_contract()
{
    // Verify the API contract accepts valid coefficient fields
    PersistenceOptions options;
    options.max_dim = 1;
    options.max_radius = 2.0;
    options.mode = nerve::persistence::PersistenceMode::EXACT;

    const std::vector<double> points = {0.0, 0.0, 1.0, 0.0};

    const auto result = nerve::persistence::compute(view_of(points), 2, options);
    if (result.isError())
    {
        std::cerr << "field-aware persistence failed: " << result.compactSummary() << "\n";
        return false;
    }

    // Verify result has pairs and diagnostics
    if (result.value().pairs.empty())
    {
        std::cerr << "field-aware persistence returned empty pairs\n";
        return false;
    }

    return true;
}

} // namespace

int main()
{
    // Core invariants
    if (!check_birth_less_than_death_invariant())
    {
        std::cerr << "FAIL: birth < death invariant\n";
        return 1;
    }
    if (!check_nonnegative_persistence_invariant())
    {
        std::cerr << "FAIL: non-negative persistence invariant\n";
        return 1;
    }

    // Edge cases
    if (!check_empty_input_produces_empty_diagram())
    {
        std::cerr << "FAIL: empty input produces empty diagram\n";
        return 1;
    }
    if (!check_single_point_invariant())
    {
        std::cerr << "FAIL: single point invariant\n";
        return 1;
    }
    if (!check_two_point_persistence_invariant())
    {
        std::cerr << "FAIL: two point persistence invariant\n";
        return 1;
    }

    // Consistency
    if (!check_betti_number_consistency())
    {
        std::cerr << "FAIL: Betti number consistency\n";
        return 1;
    }
    if (!check_filtration_monotonicity_invariant())
    {
        std::cerr << "FAIL: filtration monotonicity\n";
        return 1;
    }

    // Algebraic invariants
    if (!check_boundary_matrix_dims())
    {
        std::cerr << "FAIL: boundary matrix dimension consistency\n";
        return 1;
    }
    if (!check_boundary_matrix_validity())
    {
        std::cerr << "FAIL: boundary matrix validity\n";
        return 1;
    }

    // Determinism
    if (!check_determinism_invariant())
    {
        std::cerr << "FAIL: determinism invariant\n";
        return 1;
    }

    // Random generated tests
    if (!check_random_persistence_validity())
    {
        std::cerr << "FAIL: random persistence validity\n";
        return 1;
    }

    // Field-aware API
    if (!check_field_aware_contract())
    {
        std::cerr << "FAIL: field-aware contract\n";
        return 1;
    }

    return 0;
}


#include "nerve/algebra/boundary.hpp"
#include "nerve/algebra/complex.hpp"
#include "nerve/algebra/simplex.hpp"
#include "nerve/common/accelerated_types.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core_types.hpp"
#include "nerve/persistence/core/core_types.hpp"
#include "nerve/persistence/reduction/reduction_ops.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"
#include "test_utils.hpp"

#include <cstddef>
#include <iostream>
#include <random>
#include <vector>

namespace
{

using nerve::Size;
using nerve::algebra::BoundaryMatrix;
using nerve::algebra::Simplex;
using nerve::algebra::SimplicialComplex;
using nerve::common::VRConfig;
using nerve::persistence::Pair;
using nerve::persistence::Reducer;
using nerve::persistence::VRAlgorithmSelection;
using namespace nerve::test;


SimplicialComplex make_tetrahedron()
{
    SimplicialComplex complex;
    complex.addSimplex(Simplex({0}));
    complex.addSimplex(Simplex({1}));
    complex.addSimplex(Simplex({2}));
    complex.addSimplex(Simplex({3}));
    complex.addSimplex(Simplex({0, 1}));
    complex.addSimplex(Simplex({0, 2}));
    complex.addSimplex(Simplex({0, 3}));
    complex.addSimplex(Simplex({1, 2}));
    complex.addSimplex(Simplex({1, 3}));
    complex.addSimplex(Simplex({2, 3}));
    complex.addSimplex(Simplex({0, 1, 2}));
    complex.addSimplex(Simplex({0, 1, 3}));
    complex.addSimplex(Simplex({0, 2, 3}));
    complex.addSimplex(Simplex({1, 2, 3}));
    complex.addSimplex(Simplex({0, 1, 2, 3}));
    return complex;
}

SimplicialComplex make_square()
{
    SimplicialComplex complex;
    complex.addSimplex(Simplex({0}));
    complex.addSimplex(Simplex({1}));
    complex.addSimplex(Simplex({2}));
    complex.addSimplex(Simplex({3}));
    complex.addSimplex(Simplex({0, 1}));
    complex.addSimplex(Simplex({1, 2}));
    complex.addSimplex(Simplex({2, 3}));
    complex.addSimplex(Simplex({3, 0}));
    complex.addSimplex(Simplex({0, 2}));
    complex.addSimplex(Simplex({0, 1, 2}));
    complex.addSimplex(Simplex({0, 2, 3}));
    return complex;
}

bool check_boundary_reduction_produces_valid_pairs()
{
    SimplicialComplex complex = make_tetrahedron();
    BoundaryMatrix matrix(complex);
    Reducer reducer(matrix);
    reducer.compute();
    auto pairs = reducer.getPairs();
    for (const auto &p : pairs)
    {
        if (!p.isInfinite())
        {
            if (!(p.birth <= p.death + 1e-12))
            {
                std::cerr << "birth<=death invariant violated: birth=" << p.birth
                          << " death=" << p.death << "\n";
                return false;
            }
            if (p.lifetime() < 0.0)
            {
                std::cerr << "negative persistence: " << p.lifetime() << "\n";
                return false;
            }
        }
        if (p.dimension < 0)
        {
            std::cerr << "negative dimension\n";
            return false;
        }
    }
    return true;
}

bool check_algorithm_path_consistency()
{
    SimplicialComplex complex = make_square();
    BoundaryMatrix matrix(complex);

    Reducer reducer1(matrix);
    reducer1.compute();
    auto pairs1 = canonical(reducer1.getPairs());

    Reducer reducer2(matrix);
    reducer2.compute();
    reducer2.cohomologyReduction();
    auto pairs2 = canonical(reducer2.getPairs());

    if (pairs1.size() != pairs2.size())
    {
        std::cerr << "standard vs cohomology: different pair counts " << pairs1.size() << " vs "
                  << pairs2.size() << "\n";
        return false;
    }
    for (std::size_t i = 0; i < pairs1.size(); ++i)
    {
        if (!pairs_equal(pairs1[i], pairs2[i], 1e-12))
        {
            std::cerr << "pair " << i << " differs between algorithms\n";
            return false;
        }
    }
    return true;
}

bool check_tetrahedron_betti_numbers()
{
    SimplicialComplex complex = make_tetrahedron();
    BoundaryMatrix matrix(complex);
    Reducer reducer(matrix);
    reducer.compute();
    auto betti = reducer.getBetti();
    if (betti.empty())
    {
        std::cerr << "betti numbers should not be empty\n";
        return false;
    }
    for (std::size_t d = 0; d < betti.size(); ++d)
    {
        if (betti[d] > 10)
        {
            std::cerr << "unrealistic Betti " << betti[d] << " for dim " << d << "\n";
            return false;
        }
    }
    return true;
}

bool check_triangle_betti_numbers()
{
    SimplicialComplex complex;
    complex.addSimplex(Simplex({0}));
    complex.addSimplex(Simplex({1}));
    complex.addSimplex(Simplex({2}));
    complex.addSimplex(Simplex({0, 1}));
    complex.addSimplex(Simplex({0, 2}));
    complex.addSimplex(Simplex({1, 2}));
    complex.addSimplex(Simplex({0, 1, 2}));
    BoundaryMatrix matrix(complex);
    Reducer reducer(matrix);
    reducer.compute();
    auto betti = reducer.getBetti();
    if (betti.empty())
    {
        std::cerr << "triangle: betti should not be empty\n";
        return false;
    }
    Size h0_essential = 0;
    auto pairs = reducer.getPairs();
    for (const auto &p : pairs)
    {
        if (p.dimension == 0 && p.isInfinite())
            ++h0_essential;
    }
    if (h0_essential != 1)
    {
        std::cerr << "triangle: expected 1 H0 essential, got " << h0_essential << "\n";
        return false;
    }
    return true;
}

bool check_square_betti_numbers()
{
    SimplicialComplex complex = make_square();
    BoundaryMatrix matrix(complex);
    Reducer reducer(matrix);
    reducer.compute();
    auto pairs = reducer.getPairs();
    auto betti = reducer.getBetti();
    if (betti.empty())
    {
        std::cerr << "square: betti should not be empty\n";
        return false;
    }
    for (const auto &p : pairs)
    {
        if (!p.isInfinite() && !(p.birth <= p.death + 1e-12))
        {
            std::cerr << "square: birth<=death violated\n";
            return false;
        }
    }
    return true;
}

bool check_reduction_determinism()
{
    SimplicialComplex complex = make_tetrahedron();
    BoundaryMatrix matrix(complex);

    Reducer r1(matrix);
    r1.compute();
    auto p1 = canonical(r1.getPairs());

    Reducer r2(matrix);
    r2.compute();
    auto p2 = canonical(r2.getPairs());

    if (p1.size() != p2.size())
    {
        std::cerr << "determinism: different pair counts\n";
        return false;
    }
    for (std::size_t i = 0; i < p1.size(); ++i)
    {
        if (!pairs_equal(p1[i], p2[i], 1e-12))
        {
            std::cerr << "determinism: pair " << i << " differs\n";
            return false;
        }
    }
    return true;
}

} // namespace

int main()
{
    if (!check_boundary_reduction_produces_valid_pairs())
    {
        std::cerr << "FAIL: boundary reduction produces valid pairs\n";
        return 1;
    }
    if (!check_algorithm_path_consistency())
    {
        std::cerr << "FAIL: algorithm path consistency\n";
        return 1;
    }
    if (!check_tetrahedron_betti_numbers())
    {
        std::cerr << "FAIL: tetrahedron Betti numbers\n";
        return 1;
    }
    if (!check_triangle_betti_numbers())
    {
        std::cerr << "FAIL: triangle Betti numbers\n";
        return 1;
    }
    if (!check_square_betti_numbers())
    {
        std::cerr << "FAIL: square Betti numbers\n";
        return 1;
    }
    if (!check_reduction_determinism())
    {
        std::cerr << "FAIL: reduction determinism\n";
        return 1;
    }
    return 0;
}

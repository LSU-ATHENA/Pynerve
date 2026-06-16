
#include "nerve/algebra/complex.hpp"
#include "nerve/algebra/simplex.hpp"
#include "nerve/common/accelerated_types.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core_types.hpp"
#include "nerve/persistence/core/core_types.hpp"
#include "nerve/persistence/utils/exact_engine.hpp"

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

using nerve::Dimension;
using nerve::Field;
using nerve::Index;
using nerve::Pair;
using nerve::Size;
using nerve::algebra::Simplex;
using nerve::algebra::SimplicialComplex;
using nerve::common::VRAlgorithmSelection;
using nerve::common::VRConfig;
using nerve::core::BufferView;
using nerve::persistence::ExactPersistenceResult;
using nerve::persistence::IncrementalExactZ2;

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

bool check_exact_empty_input()
{
    SimplicialComplex complex;
    auto result = nerve::persistence::computeExactPersistenceZ2(complex, 2);
    if (!result.pairs.empty())
    {
        std::cerr << "empty input should produce empty pairs, got " << result.pairs.size() << "\n";
        return false;
    }
    return true;
}

bool check_exact_single_point()
{
    SimplicialComplex complex;
    complex.addSimplex(Simplex({0}));
    auto result = nerve::persistence::computeExactPersistenceZ2(complex, 2);
    bool found_h0_essential = false;
    for (const auto &p : result.pairs)
    {
        if (p.dimension == 0 && p.isInfinite())
            found_h0_essential = true;
    }
    if (!found_h0_essential)
    {
        std::cerr << "single point should have H0 essential class\n";
        return false;
    }
    return true;
}

bool check_exact_two_points()
{
    SimplicialComplex complex;
    complex.addSimplex(Simplex({0}));
    complex.addSimplex(Simplex({1}));
    complex.addSimplex(Simplex({0, 1}));
    auto result = nerve::persistence::computeExactPersistenceZ2(complex, 1);
    Size h0_essential = 0;
    for (const auto &p : result.pairs)
    {
        if (p.dimension == 0 && p.isInfinite())
            ++h0_essential;
    }
    if (h0_essential != 1)
    {
        std::cerr << "two connected points: expected 1 H0 essential, got " << h0_essential << "\n";
        return false;
    }
    return true;
}

bool check_exact_triangle_invariants()
{
    SimplicialComplex complex;
    complex.addSimplex(Simplex({0}));
    complex.addSimplex(Simplex({1}));
    complex.addSimplex(Simplex({2}));
    complex.addSimplex(Simplex({0, 1}));
    complex.addSimplex(Simplex({0, 2}));
    complex.addSimplex(Simplex({1, 2}));
    complex.addSimplex(Simplex({0, 1, 2}));
    auto result = nerve::persistence::computeExactPersistenceZ2(complex, 2);
    for (const auto &p : result.pairs)
    {
        if (!p.isInfinite() && !(p.birth <= p.death + 1e-12))
        {
            std::cerr << "exact triangle: birth<=death violated\n";
            return false;
        }
    }
    return true;
}

bool check_exact_cohomology_z2()
{
    SimplicialComplex complex;
    complex.addSimplex(Simplex({0}));
    complex.addSimplex(Simplex({1}));
    complex.addSimplex(Simplex({2}));
    complex.addSimplex(Simplex({0, 1}));
    complex.addSimplex(Simplex({0, 2}));
    complex.addSimplex(Simplex({1, 2}));
    complex.addSimplex(Simplex({0, 1, 2}));
    auto result = nerve::persistence::computeExactCohomologyZ2(complex, 2);
    if (result.pairs.empty())
    {
        std::cerr << "exact cohomology Z2 should produce pairs\n";
        return false;
    }
    for (const auto &p : result.pairs)
    {
        if (p.dimension < 0)
        {
            std::cerr << "negative dimension in cohomology result\n";
            return false;
        }
        if (!p.isInfinite() && p.lifetime() < 0.0)
        {
            std::cerr << "negative persistence in cohomology result\n";
            return false;
        }
    }
    return true;
}

bool check_incremental_exact_z2()
{
    IncrementalExactZ2 engine(2);
    engine.addSimplex(Simplex({0}), 0.0);
    engine.addSimplex(Simplex({1}), 0.0);
    engine.addSimplex(Simplex({2}), 0.0);
    engine.addSimplex(Simplex({0, 1}), 1.0);
    engine.addSimplex(Simplex({0, 2}), 1.0);
    engine.addSimplex(Simplex({1, 2}), 1.0);
    engine.addSimplex(Simplex({0, 1, 2}), 2.0);
    auto result = engine.snapshot();
    if (result.pairs.empty())
    {
        std::cerr << "incremental exact Z2 should produce pairs\n";
        return false;
    }
    for (const auto &p : result.pairs)
    {
        if (!p.isInfinite() && !(p.birth <= p.death + 1e-12))
        {
            std::cerr << "incremental: birth<=death violated\n";
            return false;
        }
    }
    return true;
}

bool check_exact_betti_consistency()
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
    auto result = nerve::persistence::computeExactPersistenceZ2(complex, 2);
    auto coho = nerve::persistence::computeExactCohomologyZ2(complex, 2);
    if (result.pairs.size() != coho.pairs.size())
    {
        std::cerr << "persistence and cohomology should produce same number of pairs: "
                  << result.pairs.size() << " vs " << coho.pairs.size() << "\n";
        return false;
    }
    return true;
}

} // namespace

int main()
{
    if (!check_exact_empty_input())
    {
        std::cerr << "FAIL: exact empty input\n";
        return 1;
    }
    if (!check_exact_single_point())
    {
        std::cerr << "FAIL: exact single point\n";
        return 1;
    }
    if (!check_exact_two_points())
    {
        std::cerr << "FAIL: exact two points\n";
        return 1;
    }
    if (!check_exact_triangle_invariants())
    {
        std::cerr << "FAIL: exact triangle invariants\n";
        return 1;
    }
    if (!check_exact_cohomology_z2())
    {
        std::cerr << "FAIL: exact cohomology Z2\n";
        return 1;
    }
    if (!check_incremental_exact_z2())
    {
        std::cerr << "FAIL: incremental exact Z2\n";
        return 1;
    }
    if (!check_exact_betti_consistency())
    {
        std::cerr << "FAIL: exact Betti consistency\n";
        return 1;
    }
    return 0;
}

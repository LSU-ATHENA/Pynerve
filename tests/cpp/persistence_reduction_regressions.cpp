#include "nerve/algebra/boundary.hpp"
#include "nerve/algebra/complex.hpp"
#include "nerve/algebra/simplex.hpp"
#include "nerve/common/accelerated_types.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core_types.hpp"
#include "nerve/persistence/reduction/reduction_ops.hpp"
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

using nerve::algebra::BoundaryMatrix;
using nerve::algebra::Simplex;
using nerve::algebra::SimplicialComplex;
using nerve::persistence::Pair;
using nerve::persistence::Reducer;
using namespace nerve::test;



bool check_reducer_construction()
{
    SimplicialComplex complex;
    complex.addSimplex(Simplex({0}));
    complex.addSimplex(Simplex({1}));
    complex.addSimplex(Simplex({0, 1}));

    BoundaryMatrix boundary(complex, 1);
    Reducer reducer(boundary);
    (void)reducer;
    return true;
}

bool check_reducer_compute()
{
    SimplicialComplex complex;
    complex.addSimplex(Simplex({0}));
    complex.addSimplex(Simplex({1}));
    complex.addSimplex(Simplex({2}));
    complex.addSimplex(Simplex({0, 1}));
    complex.addSimplex(Simplex({1, 2}));
    complex.addSimplex(Simplex({0, 2}));
    complex.addSimplex(Simplex({0, 1, 2}));

    BoundaryMatrix boundary(complex, 2);
    Reducer reducer(boundary);
    reducer.compute();

    if (!reducer.hasPersistencePairs())
    {
        std::cerr << "Reducer::compute should produce persistence pairs\n";
        return false;
    }

    const auto &pairs = reducer.getPairs();
    if (pairs.empty())
    {
        std::cerr << "expected at least one persistence pair\n";
        return false;
    }
    return true;
}

bool check_determine_optimal_mode()
{
    SimplicialComplex complex;
    complex.addSimplex(Simplex({0}));
    complex.addSimplex(Simplex({1}));
    complex.addSimplex(Simplex({0, 1}));

    BoundaryMatrix boundary(complex, 1);
    Reducer reducer(boundary);

    auto mode_small = reducer.determineOptimalMode(10);
    auto mode_medium = reducer.determineOptimalMode(1000);
    auto mode_large = reducer.determineOptimalMode(100000);

    (void)mode_small;
    (void)mode_medium;
    (void)mode_large;
    return true;
}

bool check_apparent_pair_detection()
{
    SimplicialComplex complex;
    complex.addSimplex(Simplex({0}));
    complex.addSimplex(Simplex({1}));
    complex.addSimplex(Simplex({2}));
    complex.addSimplex(Simplex({0, 1}));
    complex.addSimplex(Simplex({1, 2}));
    complex.addSimplex(Simplex({0, 2}));
    complex.addSimplex(Simplex({0, 1, 2}));

    BoundaryMatrix boundary(complex, 2);
    Reducer reducer(boundary);

    std::vector<bool> cleared(boundary.cols(), false);
    auto apparent = reducer.findApparentPairs(cleared);

    if (apparent.empty())
    {
        return true;
    }
    for (const auto &ap : apparent)
    {
        if (ap.birth_index >= ap.death_index)
        {
            std::cerr << "apparent pair birth_index >= death_index\n";
            return false;
        }
    }
    return true;
}

bool check_cohomology_reduction_produces_pairs()
{
    SimplicialComplex complex;
    complex.addSimplex(Simplex({0}));
    complex.addSimplex(Simplex({1}));
    complex.addSimplex(Simplex({2}));
    complex.addSimplex(Simplex({0, 1}));
    complex.addSimplex(Simplex({1, 2}));
    complex.addSimplex(Simplex({0, 2}));
    complex.addSimplex(Simplex({0, 1, 2}));

    BoundaryMatrix boundary(complex, 2);
    Reducer reducer(boundary);
    reducer.cohomologyReduction();

    if (!reducer.hasPersistencePairs())
    {
        std::cerr << "cohomologyReduction should produce pairs\n";
        return false;
    }
    return true;
}

bool check_betti_numbers_non_negative()
{
    SimplicialComplex complex;
    complex.addSimplex(Simplex({0}));
    complex.addSimplex(Simplex({1}));
    complex.addSimplex(Simplex({2}));
    complex.addSimplex(Simplex({0, 1}));
    complex.addSimplex(Simplex({1, 2}));
    complex.addSimplex(Simplex({0, 2}));
    complex.addSimplex(Simplex({0, 1, 2}));

    BoundaryMatrix boundary(complex, 2);
    Reducer reducer(boundary);
    reducer.compute();

    const auto &betti = reducer.getBetti();
    for (size_t i = 0; i < betti.size(); ++i)
    {
        if (betti[i] < 0)
        {
            std::cerr << "negative Betti number at index " << i << "\n";
            return false;
        }
    }
    return true;
}

bool check_essentials_non_negative()
{
    SimplicialComplex complex;
    complex.addSimplex(Simplex({0}));
    complex.addSimplex(Simplex({1}));
    complex.addSimplex(Simplex({2}));
    complex.addSimplex(Simplex({0, 1}));
    complex.addSimplex(Simplex({1, 2}));
    complex.addSimplex(Simplex({0, 2}));
    complex.addSimplex(Simplex({0, 1, 2}));

    BoundaryMatrix boundary(complex, 2);
    Reducer reducer(boundary);
    reducer.compute();

    const auto &essentials = reducer.getEssentials();
    for (size_t i = 0; i < essentials.size(); ++i)
    {
        if (essentials[i] < 0)
        {
            std::cerr << "negative essential index at " << i << "\n";
            return false;
        }
    }
    return true;
}

} // namespace

int main()
{
    if (!check_reducer_construction())
    {
        std::cerr << "FAIL: Reducer construction\n";
        return 1;
    }
    if (!check_reducer_compute())
    {
        std::cerr << "FAIL: Reducer compute\n";
        return 1;
    }
    if (!check_determine_optimal_mode())
    {
        std::cerr << "FAIL: determineOptimalMode\n";
        return 1;
    }
    if (!check_apparent_pair_detection())
    {
        std::cerr << "FAIL: apparent pair detection\n";
        return 1;
    }
    if (!check_cohomology_reduction_produces_pairs())
    {
        std::cerr << "FAIL: cohomologyReduction produces pairs\n";
        return 1;
    }
    if (!check_betti_numbers_non_negative())
    {
        std::cerr << "FAIL: Betti numbers non-negative\n";
        return 1;
    }
    if (!check_essentials_non_negative())
    {
        std::cerr << "FAIL: essentials non-negative\n";
        return 1;
    }
    return 0;
}


#include "nerve/algebra/boundary.hpp"
#include "nerve/algebra/complex.hpp"
#include "nerve/algebra/simplex.hpp"
#include "nerve/common/accelerated_types.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core_types.hpp"
#include "nerve/persistence/core/core_types.hpp"
#include "nerve/persistence/reduction/reduction_ops.hpp"
#include "test_utils.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

namespace
{

using nerve::Pair;
using nerve::Size;
using nerve::algebra::BoundaryMatrix;
using nerve::algebra::Simplex;
using nerve::algebra::SimplicialComplex;
using nerve::common::VRAlgorithmSelection;
using nerve::common::VRConfig;
using nerve::persistence::Reducer;
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

SimplicialComplex make_triangle()
{
    SimplicialComplex complex;
    complex.addSimplex(Simplex({0}));
    complex.addSimplex(Simplex({1}));
    complex.addSimplex(Simplex({2}));
    complex.addSimplex(Simplex({0, 1}));
    complex.addSimplex(Simplex({0, 2}));
    complex.addSimplex(Simplex({1, 2}));
    complex.addSimplex(Simplex({0, 1, 2}));
    return complex;
}

bool check_determine_optimal_mode()
{
    Reducer reducer(BoundaryMatrix{});
    auto mode_small = reducer.determineOptimalMode(100);
    auto mode_medium = reducer.determineOptimalMode(5000);
    auto mode_large = reducer.determineOptimalMode(100000);
    static_cast<void>(mode_small);
    static_cast<void>(mode_medium);
    static_cast<void>(mode_large);
    return true;
}

bool check_get_filtration_value()
{
    SimplicialComplex complex = make_triangle();
    BoundaryMatrix matrix(complex, 1);
    Reducer reducer(matrix);
    for (Size i = 0; i < matrix.cols(); ++i)
    {
        double val = reducer.getFiltrationValue(static_cast<int>(i));
        if (std::isnan(val))
        {
            std::cerr << "filtration value should not be NaN at col " << i << "\n";
            return false;
        }
    }
    return true;
}

bool check_get_simplex_dimension()
{
    SimplicialComplex complex = make_tetrahedron();
    BoundaryMatrix matrix(complex, 2);
    Reducer reducer(matrix);
    for (Size i = 0; i < matrix.cols(); ++i)
    {
        int dim = reducer.getSimplexDimension(static_cast<int>(i));
        if (dim < 0 || dim > 3)
        {
            std::cerr << "unexpected simplex dimension " << dim << " at col " << i << "\n";
            return false;
        }
    }
    return true;
}

bool check_cohomology_reduction_completes()
{
    SimplicialComplex complex = make_tetrahedron();
    BoundaryMatrix matrix(complex, 2);
    Reducer reducer(matrix);
    reducer.compute();
    reducer.cohomologyReduction();
    if (!reducer.hasPersistencePairs())
    {
        std::cerr << "cohomology reduction should produce pairs\n";
        return false;
    }
    auto pairs = reducer.getPairs();
    if (pairs.empty())
    {
        std::cerr << "cohomology reduction returned empty pairs\n";
        return false;
    }
    return true;
}

bool check_accelerated_reduction_completes()
{
    SimplicialComplex complex = make_tetrahedron();
    BoundaryMatrix matrix(complex, 2);
    Reducer reducer(matrix);
    reducer.compute();
    reducer.acceleratedReduction();
    if (!reducer.hasPersistencePairs())
    {
        std::cerr << "accelerated reduction should produce pairs\n";
        return false;
    }
    return true;
}

bool check_apparent_pair_finding()
{
    SimplicialComplex complex = make_tetrahedron();
    BoundaryMatrix matrix(complex, 2);
    Reducer reducer(matrix);
    std::vector<bool> cleared(matrix.cols(), false);
    auto apparent = reducer.findApparentPairs(cleared);
    for (const auto &ap : apparent)
    {
        if (ap.birth_index >= matrix.cols() || ap.death_index >= matrix.cols())
        {
            std::cerr << "apparent pair indices out of range\n";
            return false;
        }
    }
    return true;
}

bool check_reducer_pairs_and_betti()
{
    SimplicialComplex complex = make_tetrahedron();
    BoundaryMatrix matrix(complex, 2);
    Reducer reducer(matrix);
    reducer.compute();
    auto pairs = reducer.getPairs();
    auto betti = reducer.getBetti();
    auto essentials = reducer.getEssentials();
    if (pairs.empty() && betti.empty() && essentials.empty())
    {
        std::cerr << "compute should produce at least some output\n";
        return false;
    }
    for (const auto &p : pairs)
    {
        if (!p.isInfinite() && !(p.birth <= p.death + 1e-12))
        {
            std::cerr << "birth<=death invariant violated\n";
            return false;
        }
    }
    return true;
}

} // namespace

int main()
{
    if (!check_determine_optimal_mode())
    {
        std::cerr << "FAIL: determine optimal mode\n";
        return 1;
    }
    if (!check_get_filtration_value())
    {
        std::cerr << "FAIL: get filtration value\n";
        return 1;
    }
    if (!check_get_simplex_dimension())
    {
        std::cerr << "FAIL: get simplex dimension\n";
        return 1;
    }
    if (!check_cohomology_reduction_completes())
    {
        std::cerr << "FAIL: cohomology reduction completes\n";
        return 1;
    }
    if (!check_accelerated_reduction_completes())
    {
        std::cerr << "FAIL: accelerated reduction completes\n";
        return 1;
    }
    if (!check_apparent_pair_finding())
    {
        std::cerr << "FAIL: apparent pair finding\n";
        return 1;
    }
    if (!check_reducer_pairs_and_betti())
    {
        std::cerr << "FAIL: reducer pairs and betti\n";
        return 1;
    }
    return 0;
}

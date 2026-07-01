
#include "nerve/algebra/boundary.hpp"
#include "nerve/algebra/complex.hpp"
#include "nerve/algebra/simplex.hpp"
#include "nerve/persistence/reduction/reduction_hypha_ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>

#ifdef NERVE_HAS_CUDA

namespace
{

using nerve::Pair;
using nerve::algebra::BoundaryMatrix;
using nerve::algebra::Simplex;
using nerve::algebra::SimplicialComplex;
using nerve::persistence::HyphaReducer;

bool check_hypha_construction_default()
{
    HyphaReducer hr;
    (void)hr;
    return true;
}

bool check_hypha_construction_with_config()
{
    HyphaReducer::Config cfg;
    cfg.scan_ratio = 0.5f;
    cfg.unstable_threshold = 500;
    cfg.use_clearing = true;
    HyphaReducer hr(cfg);
    (void)hr;
    return true;
}

bool check_hypha_config_get_set()
{
    HyphaReducer hr;
    HyphaReducer::Config cfg;
    cfg.scan_ratio = 0.25f;
    cfg.unstable_threshold = 2000;
    cfg.use_clearing = false;
    hr.setConfig(cfg);
    auto retrieved = hr.config();
    if (std::abs(retrieved.scan_ratio - 0.25f) > 1e-6f)
        return false;
    if (retrieved.unstable_threshold != 2000)
        return false;
    if (retrieved.use_clearing)
        return false;
    return true;
}

SimplicialComplex make_triangle_complex()
{
    SimplicialComplex complex;
    complex.addSimplex(Simplex({0}));
    complex.addSimplex(Simplex({1}));
    complex.addSimplex(Simplex({2}));
    complex.addSimplex(Simplex({0, 1}));
    complex.addSimplex(Simplex({1, 2}));
    complex.addSimplex(Simplex({0, 2}));
    complex.addSimplex(Simplex({0, 1, 2}));
    return complex;
}

bool check_hypha_reduction_completes()
{
    auto complex = make_triangle_complex();
    BoundaryMatrix bm(complex, 2);
    HyphaReducer hr;
    auto pairs = hr.compute(bm);
    if (pairs.empty())
    {
        std::cerr << "hypha reduction produced no pairs\n";
        return false;
    }
    return true;
}

bool check_hypha_pair_invariants()
{
    auto complex = make_triangle_complex();
    BoundaryMatrix bm(complex, 2);
    HyphaReducer hr;
    auto pairs = hr.compute(bm);
    for (const auto &p : pairs)
    {
        if (!p.isInfinite() && !(p.birth <= p.death + 1e-12))
        {
            std::cerr << "birth<=death violated\n";
            return false;
        }
        if (!p.isInfinite() && p.lifetime() < -1e-12)
        {
            std::cerr << "negative persistence\n";
            return false;
        }
        if (p.dimension < 0)
        {
            std::cerr << "negative dimension\n";
            return false;
        }
    }
    return true;
}

bool check_hypha_determinism()
{
    auto complex = make_triangle_complex();
    BoundaryMatrix bm(complex, 2);
    HyphaReducer hr;
    auto run1 = hr.compute(bm);
    auto run2 = hr.compute(bm);
    if (run1.size() != run2.size())
    {
        std::cerr << "determinism size mismatch\n";
        return false;
    }
    return true;
}

bool check_hypha_empty_matrix()
{
    SimplicialComplex complex;
    BoundaryMatrix bm(complex);
    HyphaReducer hr;
    auto pairs = hr.compute(bm);
    return true;
}

} // namespace

int main()
{
    if (!check_hypha_construction_default())
    {
        std::cerr << "FAIL: hypha construction default\n";
        return 1;
    }
    if (!check_hypha_construction_with_config())
    {
        std::cerr << "FAIL: hypha construction with config\n";
        return 1;
    }
    if (!check_hypha_config_get_set())
    {
        std::cerr << "FAIL: hypha config get/set\n";
        return 1;
    }
    if (!check_hypha_reduction_completes())
    {
        std::cerr << "FAIL: hypha reduction completes\n";
        return 1;
    }
    if (!check_hypha_pair_invariants())
    {
        std::cerr << "FAIL: hypha pair invariants\n";
        return 1;
    }
    if (!check_hypha_determinism())
    {
        std::cerr << "FAIL: hypha determinism\n";
        return 1;
    }
    if (!check_hypha_empty_matrix())
    {
        std::cerr << "FAIL: hypha empty matrix\n";
        return 1;
    }
    return 0;
}

#else
int main()
{
    return 0;
}
#endif
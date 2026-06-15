
#ifdef NERVE_HAS_MPI
#include "nerve/persistence/reduction/reduction_mpi_ops.hpp"
#endif

#include <iostream>

#ifdef NERVE_HAS_MPI

#include "nerve/algebra/boundary.hpp"
#include "nerve/algebra/complex.hpp"
#include "nerve/algebra/simplex.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <tuple>
#include <vector>

namespace
{

using nerve::Size;
using nerve::algebra::BoundaryMatrix;
using nerve::algebra::Simplex;
using nerve::algebra::SimplicialComplex;
using nerve::persistence::MpiDistributedReducer;
using nerve::persistence::Pair;

bool check_mpi_reducer_construction()
{
    SimplicialComplex complex;
    complex.addSimplex(Simplex({0}));
    complex.addSimplex(Simplex({1}));
    complex.addSimplex(Simplex({0, 1}));
    BoundaryMatrix bm(complex, 1);
    MpiDistributedReducer reducer;
    (void)reducer;
    return true;
}

bool check_mpi_reduction_produces_pairs()
{
    SimplicialComplex complex;
    complex.addSimplex(Simplex({0}));
    complex.addSimplex(Simplex({1}));
    complex.addSimplex(Simplex({2}));
    complex.addSimplex(Simplex({0, 1}));
    complex.addSimplex(Simplex({1, 2}));
    complex.addSimplex(Simplex({0, 2}));
    complex.addSimplex(Simplex({0, 1, 2}));
    BoundaryMatrix bm(complex, 2);
    MpiDistributedReducer reducer;
    reducer.compute(bm);
    auto pairs = reducer.getPairs();
    if (pairs.empty())
    {
        std::cerr << "expected pairs from MPI reduction\n";
        return false;
    }
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
    }
    return true;
}

bool check_mpi_betti_numbers()
{
    SimplicialComplex complex;
    complex.addSimplex(Simplex({0}));
    complex.addSimplex(Simplex({1}));
    complex.addSimplex(Simplex({2}));
    complex.addSimplex(Simplex({0, 1}));
    complex.addSimplex(Simplex({1, 2}));
    complex.addSimplex(Simplex({0, 2}));
    complex.addSimplex(Simplex({0, 1, 2}));
    BoundaryMatrix bm(complex, 2);
    MpiDistributedReducer reducer;
    reducer.compute(bm);
    auto betti = reducer.getBetti();
    if (betti.empty())
    {
        std::cerr << "expected non-empty Betti numbers\n";
        return false;
    }
    for (size_t i = 0; i < betti.size(); ++i)
    {
        if (betti[i] < 0)
        {
            std::cerr << "negative Betti number at " << i << "\n";
            return false;
        }
    }
    return true;
}

bool check_mpi_essentials_valid()
{
    SimplicialComplex complex;
    complex.addSimplex(Simplex({0}));
    complex.addSimplex(Simplex({1}));
    complex.addSimplex(Simplex({2}));
    complex.addSimplex(Simplex({0, 1}));
    complex.addSimplex(Simplex({1, 2}));
    complex.addSimplex(Simplex({0, 2}));
    complex.addSimplex(Simplex({0, 1, 2}));
    BoundaryMatrix bm(complex, 2);
    MpiDistributedReducer reducer;
    reducer.compute(bm);
    auto essentials = reducer.getEssentials();
    for (size_t i = 0; i < essentials.size(); ++i)
    {
        if (essentials[i] < 0)
        {
            std::cerr << "negative essential index\n";
            return false;
        }
    }
    return true;
}

} // namespace

#endif

int main()
{
#ifdef NERVE_HAS_MPI
    if (!check_mpi_reducer_construction())
    {
        std::cerr << "FAIL: mpi reducer construction\n";
        return 1;
    }
    if (!check_mpi_reduction_produces_pairs())
    {
        std::cerr << "FAIL: mpi reduction produces pairs\n";
        return 1;
    }
    if (!check_mpi_betti_numbers())
    {
        std::cerr << "FAIL: mpi betti numbers\n";
        return 1;
    }
    if (!check_mpi_essentials_valid())
    {
        std::cerr << "FAIL: mpi essentials valid\n";
        return 1;
    }
#endif
    return 0;
}

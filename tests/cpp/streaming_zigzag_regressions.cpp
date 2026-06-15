
#include "nerve/core_types.hpp"
#include "nerve/persistence/core/core_types.hpp"
#include "nerve/streaming/incremental.hpp"
#include "nerve/streaming/zigzag_filters.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

namespace
{

using nerve::Size;
using nerve::algebra::Simplex;
using nerve::algebra::SimplicialComplex;
using nerve::persistence::Diagram;
using nerve::persistence::Pair;
using nerve::streaming::ZigzagPH;

constexpr double TOL = 1e-10;

std::mt19937_64 make_rng()
{
    return std::mt19937_64(42);
}

SimplicialComplex make_triangle_complex()
{
    SimplicialComplex cplx;
    cplx.addSimplex(Simplex({0}));
    cplx.addSimplex(Simplex({1}));
    cplx.addSimplex(Simplex({2}));
    cplx.addSimplex(Simplex({0, 1}));
    cplx.addSimplex(Simplex({0, 2}));
    cplx.addSimplex(Simplex({1, 2}));
    cplx.addSimplex(Simplex({0, 1, 2}));
    return cplx;
}

bool check_zigzag_construction_default()
{
    ZigzagPH zph;
    if (zph.currentStep() != 0)
    {
        std::cerr << "new zigzag PH should start at step 0\n";
        return false;
    }
    return true;
}

bool check_zigzag_add_simplex()
{
    ZigzagPH zph(2);
    zph.addSimplex(Simplex({0}));
    zph.addSimplex(Simplex({1}));
    zph.addSimplex(Simplex({0, 1}));
    auto pairs = zph.getCurrentPersistence();
    if (pairs.getPairs().empty() && pairs.count() == 0)
    {
        std::cerr << "zigzag should produce persistence pairs\n";
        return false;
    }
    return true;
}

bool check_zigzag_add_remove_sequence()
{
    ZigzagPH zph(2);
    zph.addSimplex(Simplex({0}));
    zph.addSimplex(Simplex({1}));
    zph.addSimplex(Simplex({0, 1}));
    zph.removeSimplex(Simplex({1}));
    auto current = zph.getCurrentPersistence();
    for (const auto &p : current.getPairs())
    {
        if (!p.isInfinite() && p.lifetime() < 0.0)
        {
            std::cerr << "negative persistence in zigzag\n";
            return false;
        }
    }
    return true;
}

bool check_zigzag_filtration_step()
{
    ZigzagPH zph(2);
    std::vector<std::pair<Simplex, double>> step1 = {
        {Simplex({0}), 0.0}, {Simplex({1}), 0.0}, {Simplex({0, 1}), 1.0}};
    zph.addFiltrationStep(step1);
    auto diagram = zph.getCurrentPersistence();
    if (diagram.getPairs().empty() && diagram.count() == 0)
    {
        std::cerr << "filtration step should produce pairs\n";
        return false;
    }
    return true;
}

bool check_zigzag_get_zigzag_persistence()
{
    ZigzagPH zph(2);
    zph.addSimplex(Simplex({0}));
    zph.addSimplex(Simplex({1}));
    zph.addSimplex(Simplex({0, 1}));
    auto history = zph.getZigzagPersistence();
    if (history.empty())
    {
        std::cerr << "zigzag history should not be empty\n";
        return false;
    }
    for (const auto &step_pairs : history)
    {
        for (const auto &p : step_pairs)
        {
            if (!p.isInfinite() && p.lifetime() < 0.0)
            {
                std::cerr << "negative persistence in zigzag history\n";
                return false;
            }
        }
    }
    return true;
}

bool check_zigzag_reset()
{
    ZigzagPH zph(2);
    zph.addSimplex(Simplex({0}));
    zph.addSimplex(Simplex({1}));
    zph.addSimplex(Simplex({0, 1}));
    zph.reset();
    if (zph.currentStep() != 0)
    {
        std::cerr << "after reset, step should be 0\n";
        return false;
    }
    return true;
}

bool check_zigzag_filters_witness_mode()
{
    auto cplx = make_triangle_complex();
    auto pairs = nerve::streaming::detail::computeWitnessModePairs(cplx, 2, 42);
    for (const auto &p : pairs)
    {
        if (!p.isInfinite() && p.lifetime() < 0.0)
        {
            std::cerr << "witness mode: negative persistence\n";
            return false;
        }
    }
    return true;
}

bool check_zigzag_filters_sparse_mode()
{
    auto cplx = make_triangle_complex();
    auto pairs = nerve::streaming::detail::computeSparseModePairs(cplx, 2, 42);
    for (const auto &p : pairs)
    {
        if (!p.isInfinite() && p.lifetime() < 0.0)
        {
            std::cerr << "sparse mode: negative persistence\n";
            return false;
        }
    }
    return true;
}

} // namespace

int main()
{
    if (!check_zigzag_construction_default())
    {
        std::cerr << "FAIL: zigzag construction\n";
        return 1;
    }
    if (!check_zigzag_add_simplex())
    {
        std::cerr << "FAIL: zigzag add simplex\n";
        return 1;
    }
    if (!check_zigzag_add_remove_sequence())
    {
        std::cerr << "FAIL: zigzag add/remove\n";
        return 1;
    }
    if (!check_zigzag_filtration_step())
    {
        std::cerr << "FAIL: zigzag filtration step\n";
        return 1;
    }
    if (!check_zigzag_get_zigzag_persistence())
    {
        std::cerr << "FAIL: zigzag persistence\n";
        return 1;
    }
    if (!check_zigzag_reset())
    {
        std::cerr << "FAIL: zigzag reset\n";
        return 1;
    }
    if (!check_zigzag_filters_witness_mode())
    {
        std::cerr << "FAIL: witness mode\n";
        return 1;
    }
    if (!check_zigzag_filters_sparse_mode())
    {
        std::cerr << "FAIL: sparse mode\n";
        return 1;
    }
    return 0;
}

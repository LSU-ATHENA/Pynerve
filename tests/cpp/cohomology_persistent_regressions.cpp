
#include "nerve/algebra/cellular.hpp"
#include "nerve/common/accelerated_types.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core_types.hpp"
#include "nerve/persistence/cohomology/cohomology.hpp"
#include "nerve/persistence/cohomology/cohomology_ops.hpp"
#include "nerve/persistence/cohomology/persistent_cohomology.hpp"
#include "nerve/persistence/core/core_types.hpp"
#include "test_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <tuple>
#include <vector>

namespace
{

using nerve::Index;
using nerve::Size;
using nerve::algebra::Cell;
using nerve::algebra::CellularComplex;
using nerve::common::VRConfig;
using nerve::persistence::CohomologyConfig;
using nerve::persistence::CohomologyResult;
using nerve::persistence::Pair;
using nerve::persistence::PersistentCohomologyComputer;
using namespace nerve::test;



CellularComplex make_triangle_complex()
{
    CellularComplex complex;
    Index v0 = complex.addCell(Cell(0, {0}));
    Index v1 = complex.addCell(Cell(0, {1}));
    Index v2 = complex.addCell(Cell(0, {2}));
    Index e0 = complex.addCell(Cell(1, {v0, v1}));
    Index e1 = complex.addCell(Cell(1, {v0, v2}));
    Index e2 = complex.addCell(Cell(1, {v1, v2}));
    complex.addCell(Cell(2, {e0, e1, e2}));
    return complex;
}

CellularComplex make_square_complex()
{
    CellularComplex complex;
    Index v0 = complex.addCell(Cell(0, {0}));
    Index v1 = complex.addCell(Cell(0, {1}));
    Index v2 = complex.addCell(Cell(0, {2}));
    Index v3 = complex.addCell(Cell(0, {3}));
    Index e0 = complex.addCell(Cell(1, {v0, v1}));
    Index e1 = complex.addCell(Cell(1, {v1, v2}));
    Index e2 = complex.addCell(Cell(1, {v2, v3}));
    Index e3 = complex.addCell(Cell(1, {v3, v0}));
    complex.addCell(Cell(2, {e0, e1, e2, e3}));
    return complex;
}

std::vector<std::pair<Cell, double>> make_filtration_from_complex(const CellularComplex &complex,
                                                                  double vf, double ef, double tf)
{
    std::vector<std::pair<Cell, double>> filtration;
    auto cells0 = complex.cellsOfDimension(0);
    for (auto ci : cells0)
        filtration.emplace_back(complex.getCell(ci), vf);
    auto cells1 = complex.cellsOfDimension(1);
    for (auto ci : cells1)
        filtration.emplace_back(complex.getCell(ci), ef);
    auto cells2 = complex.cellsOfDimension(2);
    for (auto ci : cells2)
        filtration.emplace_back(complex.getCell(ci), tf);
    return filtration;
}

bool check_cohomology_computer_basic_filtration()
{
    CellularComplex complex = make_triangle_complex();
    PersistentCohomologyComputer computer(complex);
    auto filtration = make_filtration_from_complex(complex, 0.0, 1.0, 2.0);
    auto pairs = computer.computePersistentCohomology(filtration);
    if (pairs.empty())
    {
        std::cerr << "basic filtration should produce pairs\n";
        return false;
    }
    for (const auto &p : pairs)
    {
        if (!p.isInfinite() && !(p.birth <= p.death + 1e-12))
        {
            std::cerr << "birth<=death invariant violated: birth=" << p.birth
                      << " death=" << p.death << "\n";
            return false;
        }
        if (!p.isInfinite() && p.lifetime() < 0.0)
        {
            std::cerr << "negative persistence: birth=" << p.birth << " death=" << p.death << "\n";
            return false;
        }
    }
    return true;
}

bool check_coboundary_computation()
{
    CellularComplex complex = make_triangle_complex();
    auto matrix = complex.computeCoboundaryMatrix();
    if (matrix.empty())
    {
        std::cerr << "coboundary matrix should not be empty\n";
        return false;
    }
    Size n = static_cast<Size>(complex.numCells());
    if (matrix.size() != n)
    {
        std::cerr << "coboundary matrix rows=" << matrix.size() << " expected " << n << "\n";
        return false;
    }
    return true;
}

bool check_clearing_tracker_behavior()
{
    CellularComplex complex = make_triangle_complex();
    PersistentCohomologyComputer computer(complex);
    auto filtration = make_filtration_from_complex(complex, 0.0, 1.0, 2.0);

    CohomologyConfig config;
    config.use_clearing = true;
    config.use_apparent_pairs = false;

    auto pairs = computer.computePersistentCohomology(filtration);
    Size h0_essential = 0;
    for (const auto &p : pairs)
    {
        if (p.dimension == 0 && p.isInfinite())
            ++h0_essential;
    }
    if (h0_essential != 1)
    {
        std::cerr << "clearing: expected 1 H0 essential, got " << h0_essential << "\n";
        return false;
    }
    return true;
}

bool check_cohomology_invariants()
{
    CellularComplex complex = make_square_complex();
    PersistentCohomologyComputer computer(complex);
    auto filtration = make_filtration_from_complex(complex, 0.0, 1.0, 1.5);
    auto pairs = computer.computePersistentCohomology(filtration);
    for (const auto &p : pairs)
    {
        if (!p.isInfinite())
        {
            if (!(p.birth <= p.death + 1e-12))
            {
                std::cerr << "birth<=death violated: " << p.birth << " > " << p.death << "\n";
                return false;
            }
            if (p.lifetime() < -1e-12)
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

bool check_cohomology_barcode()
{
    CellularComplex complex = make_triangle_complex();
    PersistentCohomologyComputer computer(complex);
    auto filtration = make_filtration_from_complex(complex, 0.0, 1.0, 1.5);
    auto pairs = computer.computePersistentCohomology(filtration);
    auto barcode = computer.getBarcode();
    if (barcode.empty())
    {
        std::cerr << "barcode should not be empty\n";
        return false;
    }
    return true;
}

bool check_cohomology_compute_for_dimension()
{
    CellularComplex complex = make_triangle_complex();
    PersistentCohomologyComputer computer(complex);
    auto filtration = make_filtration_from_complex(complex, 0.0, 1.0, 2.0);
    auto dim0 = computer.computeForDimension(filtration, 0);
    auto dim1 = computer.computeForDimension(filtration, 1);
    auto dim2 = computer.computeForDimension(filtration, 2);
    Size total = dim0.size() + dim1.size() + dim2.size();
    auto all = computer.computePersistentCohomology(filtration);
    if (total != all.size())
    {
        std::cerr << "dimension-specific count " << total << " != all count " << all.size() << "\n";
        return false;
    }
    return true;
}

} // namespace

int main()
{
    if (!check_cohomology_computer_basic_filtration())
    {
        std::cerr << "FAIL: cohomology computer basic filtration\n";
        return 1;
    }
    if (!check_coboundary_computation())
    {
        std::cerr << "FAIL: coboundary computation\n";
        return 1;
    }
    if (!check_clearing_tracker_behavior())
    {
        std::cerr << "FAIL: clearing tracker behavior\n";
        return 1;
    }
    if (!check_cohomology_invariants())
    {
        std::cerr << "FAIL: cohomology invariants\n";
        return 1;
    }
    if (!check_cohomology_barcode())
    {
        std::cerr << "FAIL: cohomology barcode\n";
        return 1;
    }
    if (!check_cohomology_compute_for_dimension())
    {
        std::cerr << "FAIL: cohomology compute for dimension\n";
        return 1;
    }
    return 0;
}

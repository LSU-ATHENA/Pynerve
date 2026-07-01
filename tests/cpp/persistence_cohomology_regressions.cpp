#include "nerve/algebra/cellular.hpp"
#include "nerve/common/accelerated_types.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core_types.hpp"
#include "nerve/persistence/cohomology/cohomology.hpp"
#include "nerve/persistence/cohomology/persistent_cohomology.hpp"
#include "test_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <random>
#include <vector>

namespace
{

using nerve::algebra::Cell;
using nerve::algebra::CellularComplex;
using nerve::persistence::Cohomology;
using nerve::persistence::CohomologyResult;
using nerve::persistence::Pair;
using nerve::persistence::PersistentCohomologyComputer;
using namespace nerve::test;


bool check_persistent_cohomology_computer_constructs()
{
    PersistentCohomologyComputer computer;
    (void)computer;
    return true;
}

bool check_compute_persistent_cohomology_returns_pairs()
{
    CellularComplex complex;
    complex.addCell(Cell(0, {0}));
    complex.addCell(Cell(0, {1}));
    complex.addCell(Cell(1, {0, 1}));

    std::vector<std::pair<Cell, double>> filtration = {
        {Cell(0, {0}), 0.0}, {Cell(0, {1}), 0.0}, {Cell(1, {0, 1}), 1.0}};

    PersistentCohomologyComputer computer(complex);
    auto pairs = computer.computePersistentCohomology(filtration);

    if (pairs.empty())
    {
        std::cerr << "computePersistentCohomology returned empty pairs\n";
        return false;
    }

    return true;
}

bool check_compute_persistent_cohomology_birth_less_than_death()
{
    CellularComplex complex;
    complex.addCell(Cell(0, {0}));
    complex.addCell(Cell(0, {1}));
    complex.addCell(Cell(1, {0, 1}));

    std::vector<std::pair<Cell, double>> filtration = {
        {Cell(0, {0}), 0.0}, {Cell(0, {1}), 0.0}, {Cell(1, {0, 1}), 1.0}};

    PersistentCohomologyComputer computer(complex);
    auto pairs = computer.computePersistentCohomology(filtration);

    for (const auto &p : pairs)
    {
        if (!p.isInfinite() && !(p.birth <= p.death + 1e-14))
        {
            std::cerr << "birth <= death violated: birth=" << p.birth << " death=" << p.death
                      << "\n";
            return false;
        }
    }
    return true;
}

bool check_compute_for_dimension_returns_pairs()
{
    CellularComplex complex;
    complex.addCell(Cell(0, {0}));
    complex.addCell(Cell(0, {1}));
    complex.addCell(Cell(0, {2}));
    complex.addCell(Cell(1, {0, 1}));
    complex.addCell(Cell(1, {1, 2}));
    complex.addCell(Cell(1, {0, 2}));
    complex.addCell(Cell(2, {0, 1, 2}));

    std::vector<std::pair<Cell, double>> filtration = {
        {Cell(0, {0}), 0.0},      {Cell(0, {1}), 0.0},    {Cell(0, {2}), 0.0},
        {Cell(1, {0, 1}), 1.0},   {Cell(1, {1, 2}), 1.0}, {Cell(1, {0, 2}), 1.0},
        {Cell(2, {0, 1, 2}), 2.0}};

    PersistentCohomologyComputer computer(complex);

    bool ok = true;
    for (int d = 0; d <= 2; ++d)
    {
        auto dim_pairs = computer.computeForDimension(filtration, d);
        for (const auto &p : dim_pairs)
        {
            if (p.dimension != d)
            {
                std::cerr << "dimension mismatch in computeForDimension(" << d << "): got dim "
                          << p.dimension << "\n";
                ok = false;
            }
        }
    }
    return ok;
}

bool check_cohomology_betti_numbers()
{
    CellularComplex complex;
    complex.addCell(Cell(0, {0}));
    complex.addCell(Cell(0, {1}));
    complex.addCell(Cell(0, {2}));
    complex.addCell(Cell(1, {0, 1}));
    complex.addCell(Cell(1, {1, 2}));
    complex.addCell(Cell(1, {0, 2}));

    Cohomology cohom(complex);
    auto betti = cohom.computeBettiNumbers();

    if (betti.empty())
    {
        std::cerr << "Betti numbers empty\n";
        return false;
    }
    if (betti[0] < 0)
    {
        std::cerr << "negative Betti number\n";
        return false;
    }
    return true;
}

bool check_cohomology_persistent_cohomology()
{
    CellularComplex complex;
    complex.addCell(Cell(0, {0}));
    complex.addCell(Cell(0, {1}));
    complex.addCell(Cell(1, {0, 1}));

    std::vector<std::pair<Cell, double>> filtration = {
        {Cell(0, {0}), 0.0}, {Cell(0, {1}), 0.0}, {Cell(1, {0, 1}), 1.0}};

    Cohomology cohom(complex);
    auto pairs = cohom.computePersistentCohomology(filtration);

    if (pairs.empty())
    {
        std::cerr << "Cohomology::computePersistentCohomology returned empty\n";
        return false;
    }
    return true;
}

bool check_nonnegative_persistence()
{
    CellularComplex complex;
    complex.addCell(Cell(0, {0}));
    complex.addCell(Cell(0, {1}));
    complex.addCell(Cell(0, {2}));
    complex.addCell(Cell(1, {0, 1}));
    complex.addCell(Cell(1, {1, 2}));
    complex.addCell(Cell(1, {0, 2}));
    complex.addCell(Cell(2, {0, 1, 2}));

    std::vector<std::pair<Cell, double>> filtration = {
        {Cell(0, {0}), 0.0},      {Cell(0, {1}), 0.0},    {Cell(0, {2}), 0.0},
        {Cell(1, {0, 1}), 0.5},   {Cell(1, {1, 2}), 0.5}, {Cell(1, {0, 2}), 0.5},
        {Cell(2, {0, 1, 2}), 1.0}};

    PersistentCohomologyComputer computer(complex);
    auto pairs = computer.computePersistentCohomology(filtration);

    for (const auto &p : pairs)
    {
        if (!p.isInfinite() && p.lifetime() < 0.0)
        {
            std::cerr << "negative persistence: birth=" << p.birth << " death=" << p.death
                      << " lifetime=" << p.lifetime() << "\n";
            return false;
        }
    }
    return true;
}

bool check_determinism()
{
    CellularComplex complex;
    complex.addCell(Cell(0, {0}));
    complex.addCell(Cell(0, {1}));
    complex.addCell(Cell(0, {2}));
    complex.addCell(Cell(1, {0, 1}));
    complex.addCell(Cell(1, {1, 2}));
    complex.addCell(Cell(1, {0, 2}));
    complex.addCell(Cell(2, {0, 1, 2}));

    std::vector<std::pair<Cell, double>> filtration = {
        {Cell(0, {0}), 0.0},      {Cell(0, {1}), 0.0},    {Cell(0, {2}), 0.0},
        {Cell(1, {0, 1}), 0.5},   {Cell(1, {1, 2}), 0.5}, {Cell(1, {0, 2}), 0.5},
        {Cell(2, {0, 1, 2}), 1.0}};

    PersistentCohomologyComputer computer(complex);
    auto run1 = canonical(computer.computePersistentCohomology(filtration));
    auto run2 = canonical(computer.computePersistentCohomology(filtration));

    if (run1.size() != run2.size())
    {
        std::cerr << "determinism: size mismatch " << run1.size() << " vs " << run2.size() << "\n";
        return false;
    }
    for (size_t i = 0; i < run1.size(); ++i)
    {
        if (run1[i].dimension != run2[i].dimension ||
            std::abs(run1[i].birth - run2[i].birth) > 1e-12 ||
            run1[i].isInfinite() != run2[i].isInfinite())
        {
            std::cerr << "determinism: pair " << i << " differs\n";
            return false;
        }
        if (!run1[i].isInfinite() && std::abs(run1[i].death - run2[i].death) > 1e-12)
        {
            std::cerr << "determinism: death differs at pair " << i << "\n";
            return false;
        }
    }
    return true;
}

} // namespace

int main()
{
    if (!check_persistent_cohomology_computer_constructs())
    {
        std::cerr << "FAIL: PersistentCohomologyComputer construction\n";
        return 1;
    }
    if (!check_compute_persistent_cohomology_returns_pairs())
    {
        std::cerr << "FAIL: computePersistentCohomology returns pairs\n";
        return 1;
    }
    if (!check_compute_persistent_cohomology_birth_less_than_death())
    {
        std::cerr << "FAIL: birth < death invariant\n";
        return 1;
    }
    if (!check_compute_for_dimension_returns_pairs())
    {
        std::cerr << "FAIL: computeForDimension returns pairs\n";
        return 1;
    }
    if (!check_cohomology_betti_numbers())
    {
        std::cerr << "FAIL: Cohomology Betti numbers\n";
        return 1;
    }
    if (!check_cohomology_persistent_cohomology())
    {
        std::cerr << "FAIL: Cohomology persistent cohomology\n";
        return 1;
    }
    if (!check_nonnegative_persistence())
    {
        std::cerr << "FAIL: non-negative persistence\n";
        return 1;
    }
    if (!check_determinism())
    {
        std::cerr << "FAIL: determinism\n";
        return 1;
    }
    return 0;
}

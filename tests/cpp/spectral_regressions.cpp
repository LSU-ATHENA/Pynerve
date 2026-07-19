#include "nerve/algebra/complex.hpp"
#include "nerve/algebra/simplex.hpp"
#include "nerve/common/accelerated_types.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core_types.hpp"
#include "nerve/spectral/laplacian.hpp"
#include "nerve/spectral/persistent_laplacian.hpp"
#include "test_utils.hpp"

#include <cstddef>
#include <iostream>
#include <random>
#include <vector>

namespace
{

using nerve::algebra::Simplex;
using nerve::algebra::SimplicialComplex;
using nerve::spectral::Laplacian;
using namespace nerve::test;

bool check_laplacian_construction()
{
    SimplicialComplex complex;
    complex.addSimplex(Simplex({0}));
    complex.addSimplex(Simplex({1}));
    complex.addSimplex(Simplex({2}));

    Laplacian lap(complex);
    if (lap.size() <= 0)
    {
        std::cerr << "Laplacian size should be > 0\n";
        return false;
    }
    return true;
}

bool check_laplacian_get_laplacian_dimensions()
{
    SimplicialComplex complex;
    complex.addSimplex(Simplex({0}));
    complex.addSimplex(Simplex({1}));
    complex.addSimplex(Simplex({2}));
    complex.addSimplex(Simplex({0, 1}));
    complex.addSimplex(Simplex({1, 2}));
    complex.addSimplex(Simplex({0, 2}));

    Laplacian lap(complex);
    auto l0 = lap.getLaplacian(0);
    auto l1 = lap.getLaplacian(1);

    if (l0.empty())
    {
        std::cerr << "L0 should not be empty\n";
        return false;
    }
    if (l1.empty())
    {
        std::cerr << "L1 should not be empty\n";
        return false;
    }
    return true;
}

bool check_laplacian_up_laplacian()
{
    SimplicialComplex complex;
    complex.addSimplex(Simplex({0}));
    complex.addSimplex(Simplex({1}));
    complex.addSimplex(Simplex({0, 1}));

    Laplacian lap(complex);
    auto up = lap.getUpLaplacian(0);
    (void)up;
    return true;
}

bool check_laplacian_down_laplacian()
{
    SimplicialComplex complex;
    complex.addSimplex(Simplex({0}));
    complex.addSimplex(Simplex({1}));
    complex.addSimplex(Simplex({2}));
    complex.addSimplex(Simplex({0, 1}));
    complex.addSimplex(Simplex({1, 2}));
    complex.addSimplex(Simplex({0, 2}));

    Laplacian lap(complex);
    auto down = lap.getDownLaplacian(1);
    (void)down;
    return true;
}

bool check_laplacian_hodge_laplacian()
{
    SimplicialComplex complex;
    complex.addSimplex(Simplex({0}));
    complex.addSimplex(Simplex({1}));
    complex.addSimplex(Simplex({2}));
    complex.addSimplex(Simplex({0, 1}));
    complex.addSimplex(Simplex({1, 2}));
    complex.addSimplex(Simplex({0, 2}));

    Laplacian lap(complex);
    auto hodge = lap.getHodgeLaplacian(1);
    (void)hodge;
    return true;
}

bool check_laplacian_eigenvalues_non_negative()
{
    SimplicialComplex complex;
    complex.addSimplex(Simplex({0}));
    complex.addSimplex(Simplex({1}));
    complex.addSimplex(Simplex({2}));
    complex.addSimplex(Simplex({0, 1}));
    complex.addSimplex(Simplex({1, 2}));
    complex.addSimplex(Simplex({0, 2}));
    complex.addSimplex(Simplex({0, 1, 2}));

    Laplacian lap(complex);
    auto evals = lap.eigenvalues(0);
    for (auto e : evals)
    {
        if (e < -1e-10)
        {
            std::cerr << "negative eigenvalue: " << e << "\n";
            return false;
        }
    }
    return true;
}

bool check_laplacian_spectral_gap()
{
    SimplicialComplex complex;
    complex.addSimplex(Simplex({0}));
    complex.addSimplex(Simplex({1}));
    complex.addSimplex(Simplex({2}));
    complex.addSimplex(Simplex({0, 1}));
    complex.addSimplex(Simplex({1, 2}));
    complex.addSimplex(Simplex({0, 2}));
    complex.addSimplex(Simplex({0, 1, 2}));

    Laplacian lap(complex);
    auto gap = lap.computeSpectralGap(0);
    if (gap.empty())
    {
        std::cerr << "spectral gap should not be empty\n";
        return false;
    }
    for (auto g : gap)
    {
        if (g < 0.0)
        {
            std::cerr << "negative spectral gap: " << g << "\n";
            return false;
        }
    }
    return true;
}

bool check_laplacian_spectrum()
{
    SimplicialComplex complex;
    complex.addSimplex(Simplex({0}));
    complex.addSimplex(Simplex({1}));
    complex.addSimplex(Simplex({0, 1}));

    Laplacian lap(complex);
    auto spec = lap.spectrum(0);
    (void)spec;
    return true;
}

bool check_laplacian_max_dimension()
{
    SimplicialComplex complex;
    complex.addSimplex(Simplex({0}));
    complex.addSimplex(Simplex({1}));
    complex.addSimplex(Simplex({0, 1}));

    Laplacian lap(complex);
    auto md = lap.maxDimension();
    if (md < 0)
    {
        std::cerr << "max dimension should be >= 0\n";
        return false;
    }
    return true;
}

} // namespace

int main()
{
    if (!check_laplacian_construction())
    {
        std::cerr << "FAIL: Laplacian construction\n";
        return 1;
    }
    if (!check_laplacian_get_laplacian_dimensions())
    {
        std::cerr << "FAIL: getLaplacian dimensions\n";
        return 1;
    }
    if (!check_laplacian_up_laplacian())
    {
        std::cerr << "FAIL: getUpLaplacian\n";
        return 1;
    }
    if (!check_laplacian_down_laplacian())
    {
        std::cerr << "FAIL: getDownLaplacian\n";
        return 1;
    }
    if (!check_laplacian_hodge_laplacian())
    {
        std::cerr << "FAIL: getHodgeLaplacian\n";
        return 1;
    }
    if (!check_laplacian_eigenvalues_non_negative())
    {
        std::cerr << "FAIL: eigenvalues non-negative\n";
        return 1;
    }
    if (!check_laplacian_spectral_gap())
    {
        std::cerr << "FAIL: computeSpectralGap\n";
        return 1;
    }
    if (!check_laplacian_spectrum())
    {
        std::cerr << "FAIL: spectrum\n";
        return 1;
    }
    if (!check_laplacian_max_dimension())
    {
        std::cerr << "FAIL: maxDimension\n";
        return 1;
    }
    return 0;
}

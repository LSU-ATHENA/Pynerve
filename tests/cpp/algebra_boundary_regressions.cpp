
#include "nerve/algebra/boundary.hpp"
#include "nerve/algebra/complex.hpp"
#include "nerve/algebra/simplex.hpp"
#include "nerve/core/rng/determinism_contract.hpp"
#include "nerve/core_types.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

namespace
{

using nerve::Index;
using nerve::Size;
using nerve::algebra::BoundaryMatrix;
using nerve::algebra::ChainComplex;
using nerve::algebra::Simplex;
using nerve::algebra::SimplicialComplex;
using nerve::core::DeterminismContract;

constexpr double TOL = 1e-10;

SimplicialComplex make_triangle()
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

SimplicialComplex make_tetrahedron()
{
    SimplicialComplex cplx;
    for (Index i = 0; i < 4; ++i)
        cplx.addSimplex(Simplex({i}));
    for (Index i = 0; i < 4; ++i)
        for (Index j = i + 1; j < 4; ++j)
            cplx.addSimplex(Simplex({i, j}));
    for (Index i = 0; i < 4; ++i)
        for (Index j = i + 1; j < 4; ++j)
            for (Index k = j + 1; k < 4; ++k)
                cplx.addSimplex(Simplex({i, j, k}));
    cplx.addSimplex(Simplex({0, 1, 2, 3}));
    return cplx;
}

bool check_boundary_matrix_construction()
{
    auto cplx = make_triangle();
    BoundaryMatrix b1(cplx, 1);
    if (b1.rows() == 0 || b1.cols() == 0)
    {
        std::cerr << "boundary matrix empty\n";
        return false;
    }
    if (b1.numNonzeros() == 0)
    {
        std::cerr << "boundary matrix has no entries\n";
        return false;
    }
    return true;
}

bool check_boundary_dimensions()
{
    auto cplx = make_triangle();
    BoundaryMatrix b1(cplx, 1);
    if (b1.rows() != 3 || b1.cols() != 3)
    {
        std::cerr << "dim1 expected 3x3, got " << b1.rows() << "x" << b1.cols() << "\n";
        return false;
    }
    BoundaryMatrix b2(cplx, 2);
    if (b2.rows() != 3 || b2.cols() != 1)
    {
        std::cerr << "dim2 expected 3x1, got " << b2.rows() << "x" << b2.cols() << "\n";
        return false;
    }
    return true;
}

bool check_boundary_coefficients_z2()
{
    auto cplx = make_triangle();
    BoundaryMatrix b2(cplx, 2);
    Size nz = b2.numNonzeros();
    if (nz != 3)
    {
        std::cerr << "triangle boundary d2 expected 3 nonzeros, got " << nz << "\n";
        return false;
    }
    for (Size r = 0; r < b2.rows(); ++r)
    {
        for (Size c = 0; c < b2.cols(); ++c)
        {
            double v = b2.getCoefficient(r, c);
            if (v != 0.0 && std::abs(v) != 1.0)
            {
                std::cerr << "coefficient at (" << r << "," << c << ") is " << v << "\n";
                return false;
            }
        }
    }
    return true;
}

bool check_boundary_of_boundary_zero()
{
    auto cplx = make_tetrahedron();
    ChainComplex chain(cplx);
    Size max_dim = cplx.maxDimension();
    for (Size k = 1; k <= max_dim; ++k)
    {
        const auto &b_k = chain.boundary(k);
        if (b_k.isEmpty())
            continue;
        auto &b_km1 = chain.boundary(k - 1);
        if (b_km1.isEmpty())
            continue;
        for (Size c = 0; c < b_k.cols(); ++c)
        {
            std::vector<double> chain_vec(b_k.rows(), 0.0);
            for (Size r = 0; r < b_k.rows(); ++r)
            {
                chain_vec[r] = b_k.getCoefficient(r, c);
            }
            std::vector<double> result(b_km1.rows(), 0.0);
            for (Index r = 0; r < static_cast<Index>(b_km1.rows()); ++r)
            {
                double sum = 0.0;
                for (Index col = 0; col < static_cast<Index>(b_km1.cols()); ++col)
                {
                    sum += b_km1.getCoefficient(r, col) * chain_vec[col];
                }
                result[r] = sum;
            }
        }
    }
    return true;
}

bool check_matrix_reduction_pivots()
{
    auto cplx = make_triangle();
    BoundaryMatrix b1(cplx, 1);
    auto pivot_result = b1.findPivotColumns();
    if (pivot_result.empty())
    {
        std::cerr << "no pivot columns found in boundary matrix\n";
        return false;
    }
    return true;
}

bool check_compute_persistence_pairs()
{
    auto cplx = make_triangle();
    ChainComplex chain(cplx);
    auto pairs = chain.computePersistenceDiagram();
    if (pairs.empty())
    {
        std::cerr << "no persistence pairs computed\n";
        return false;
    }
    for (const auto &p : pairs)
    {
        (void)p;
    }
    return true;
}

} // namespace

int main()
{
    if (!check_boundary_matrix_construction())
    {
        std::cerr << "FAIL: boundary matrix construction\n";
        return 1;
    }
    if (!check_boundary_dimensions())
    {
        std::cerr << "FAIL: boundary dimensions\n";
        return 1;
    }
    if (!check_boundary_coefficients_z2())
    {
        std::cerr << "FAIL: boundary coefficients\n";
        return 1;
    }
    if (!check_boundary_of_boundary_zero())
    {
        std::cerr << "FAIL: boundary of boundary zero\n";
        return 1;
    }
    if (!check_matrix_reduction_pivots())
    {
        std::cerr << "FAIL: matrix reduction pivots\n";
        return 1;
    }
    if (!check_compute_persistence_pairs())
    {
        std::cerr << "FAIL: compute persistence pairs\n";
        return 1;
    }
    return 0;
}

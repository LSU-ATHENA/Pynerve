
#include "nerve/algebra/complex.hpp"
#include "nerve/algebra/simplex.hpp"
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

using nerve::Dimension;
using nerve::Index;
using nerve::Size;
using nerve::algebra::Simplex;
using nerve::algebra::SimplicialComplex;

constexpr double TOL = 1e-10;

std::mt19937_64 make_rng()
{
    return std::mt19937_64(42);
}

bool check_simplicial_complex_construction()
{
    SimplicialComplex cplx;
    if (cplx.size() != 0)
    {
        std::cerr << "new complex should be empty\n";
        return false;
    }
    return true;
}

bool check_add_simplices()
{
    SimplicialComplex cplx;
    cplx.addSimplex(Simplex({0}));
    cplx.addSimplex(Simplex({1}));
    cplx.addSimplex(Simplex({0, 1}));
    if (cplx.size() != 3)
    {
        std::cerr << "expected 3 simplices, got " << cplx.size() << "\n";
        return false;
    }
    return true;
}

bool check_add_simplices_multi_dim()
{
    SimplicialComplex cplx;
    cplx.addSimplex(Simplex({0}));
    cplx.addSimplex(Simplex({1}));
    cplx.addSimplex(Simplex({2}));
    cplx.addSimplex(Simplex({0, 1}));
    cplx.addSimplex(Simplex({0, 2}));
    cplx.addSimplex(Simplex({1, 2}));
    cplx.addSimplex(Simplex({0, 1, 2}));
    if (cplx.numSimplices() != 7)
    {
        std::cerr << "triangle expected 7 simplices, got " << cplx.numSimplices() << "\n";
        return false;
    }
    if (cplx.maxDimension() != 2)
    {
        std::cerr << "expected max dimension 2, got " << cplx.maxDimension() << "\n";
        return false;
    }
    return true;
}

bool check_find_faces()
{
    Simplex tri({0, 1, 2});
    auto faces = tri.faces();
    if (faces.size() != 7)
    {
        std::cerr << "triangle expected 7 faces, got " << faces.size() << "\n";
        return false;
    }
    return true;
}

bool check_simplex_faces_dimension()
{
    Simplex edge({0, 1});
    auto faces = edge.faces();
    Size face_count = 0;
    for (const auto &f : faces)
    {
        if (f.dimension() == 0)
            ++face_count;
    }
    if (face_count != 2)
    {
        std::cerr << "edge expected 2 vertex faces, got " << face_count << "\n";
        return false;
    }
    return true;
}

bool check_complex_properties()
{
    SimplicialComplex cplx;
    cplx.addSimplex(Simplex({0, 1, 2}));
    cplx.addSimplex(Simplex({0}));
    cplx.addSimplex(Simplex({1}));
    cplx.addSimplex(Simplex({2}));
    cplx.addSimplex(Simplex({0, 1}));
    cplx.addSimplex(Simplex({0, 2}));
    cplx.addSimplex(Simplex({1, 2}));
    if (cplx.maxDimension() != 2)
        return false;
    auto dim1 = cplx.simplicesOfDimension(1);
    if (dim1.size() != 3)
    {
        std::cerr << "expected 3 edges, got " << dim1.size() << "\n";
        return false;
    }
    return true;
}

bool check_chain_complex_triangle()
{
    SimplicialComplex cplx;
    cplx.addSimplex(Simplex({0}));
    cplx.addSimplex(Simplex({1}));
    cplx.addSimplex(Simplex({2}));
    cplx.addSimplex(Simplex({0, 1}));
    cplx.addSimplex(Simplex({0, 2}));
    cplx.addSimplex(Simplex({1, 2}));
    cplx.addSimplex(Simplex({0, 1, 2}));
    nerve::algebra::ChainComplex chain(cplx);
    auto betti = chain.computeBettiNumbers();
    if (betti.empty())
    {
        std::cerr << "no betti numbers computed\n";
        return false;
    }
    return true;
}

} // namespace

int main()
{
    if (!check_simplicial_complex_construction())
    {
        std::cerr << "FAIL: complex construction\n";
        return 1;
    }
    if (!check_add_simplices())
    {
        std::cerr << "FAIL: add simplices\n";
        return 1;
    }
    if (!check_add_simplices_multi_dim())
    {
        std::cerr << "FAIL: add multi-dim simplices\n";
        return 1;
    }
    if (!check_find_faces())
    {
        std::cerr << "FAIL: find faces\n";
        return 1;
    }
    if (!check_simplex_faces_dimension())
    {
        std::cerr << "FAIL: simplex faces dimension\n";
        return 1;
    }
    if (!check_complex_properties())
    {
        std::cerr << "FAIL: complex properties\n";
        return 1;
    }
    if (!check_chain_complex_triangle())
    {
        std::cerr << "FAIL: chain complex triangle\n";
        return 1;
    }
    return 0;
}

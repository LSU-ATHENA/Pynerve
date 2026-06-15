
#include "nerve/algebra/detail/algebra_detail.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <vector>

namespace
{

using nerve::Dimension;
using nerve::Index;
using nerve::Size;
using nerve::algebra::BoundaryMatrix;
using nerve::algebra::ChainComplex;
using nerve::algebra::SIMDDistanceCalculator;
using nerve::algebra::Simplex;
using nerve::algebra::SimplexSet;
using nerve::algebra::SimplicialComplex;

constexpr double TOL = 1e-10;

bool check_simplex_operations()
{
    Simplex s({0, 1, 2});
    if (s.dimension() != 2)
    {
        return false;
    }
    if (s.numVertices() != 3)
    {
        return false;
    }
    if (s[0] != 0 || s[1] != 1 || s[2] != 2)
    {
        return false;
    }

    if (!s.contains(1))
    {
        return false;
    }
    if (s.contains(5))
    {
        return false;
    }

    auto faces = s.faces();
    if (faces.size() != 3)
    {
        return false;
    }

    Simplex edge({0, 1});
    if (!edge.isFaceOf(s))
    {
        return false;
    }

    Simplex not_face({0, 5});
    if (not_face.isFaceOf(s))
    {
        return false;
    }

    if (s.faceWithoutVertex(0).dimension() != 1)
    {
        return false;
    }

    auto kfaces = s.kFaces(1);
    if (kfaces.empty())
    {
        return false;
    }

    return true;
}

bool check_simplicial_complex_construction()
{
    SimplicialComplex cplx;
    cplx.addSimplex(Simplex({0}));
    cplx.addSimplex(Simplex({1}));
    cplx.addSimplex(Simplex({2}));
    cplx.addSimplex(Simplex({0, 1}));
    cplx.addSimplex(Simplex({0, 2}));
    cplx.addSimplex(Simplex({1, 2}));
    cplx.addSimplex(Simplex({0, 1, 2}));

    if (cplx.size() != 7)
    {
        return false;
    }
    if (cplx.numSimplices() != 7)
    {
        return false;
    }
    if (cplx.maxDimension() != 2)
    {
        return false;
    }

    auto dim0 = cplx.simplicesOfDimension(0);
    if (dim0.size() != 3)
    {
        return false;
    }

    auto dim1 = cplx.simplicesOfDimension(1);
    if (dim1.size() != 3)
    {
        return false;
    }

    auto dim2 = cplx.simplicesOfDimension(2);
    if (dim2.size() != 1)
    {
        return false;
    }

    cplx.removeSimplex(Simplex({0, 1, 2}));
    if (cplx.size() != 6)
    {
        return false;
    }

    SimplicialComplex empty_cplx;
    if (empty_cplx.size() != 0)
    {
        return false;
    }
    if (empty_cplx.maxDimension() != -2)
    {
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

    ChainComplex chain(cplx);

    const auto &b1 = chain.boundary(1);
    if (b1.isEmpty())
    {
        return false;
    }
    if (b1.rows() != 3 || b1.cols() != 3)
    {
        return false;
    }

    const auto &b2 = chain.boundary(2);
    if (b2.isEmpty())
    {
        return false;
    }
    if (b2.rows() != 3 || b2.cols() != 1)
    {
        return false;
    }

    Size r1 = chain.rank(1);
    (void)r1;

    return true;
}

bool check_distance_simd_match_scalar()
{
    SIMDDistanceCalculator calc;

    double a[] = {1.0, 2.0, 3.0, 4.0};
    double b[] = {5.0, 6.0, 7.0, 8.0};

    double simd_dist = calc.euclideanDistance(a, b, 4);
    double scalar_dist = calc.euclideanDistanceScalar(a, b, 4);

    if (std::abs(simd_dist - scalar_dist) > TOL)
    {
        return false;
    }

    double manhattan_simd = calc.manhattanDistance(a, b, 4);
    double manhattan_scalar = calc.manhattanDistanceScalar(a, b, 4);

    if (std::abs(manhattan_simd - manhattan_scalar) > TOL)
    {
        return false;
    }

    double cosine_simd = calc.cosineDistance(a, b, 4);
    double cosine_scalar = calc.cosineDistanceScalar(a, b, 4);

    if (std::abs(cosine_simd - cosine_scalar) > TOL)
    {
        return false;
    }

    std::vector<double> points = {0.0, 0.0, 3.0, 0.0, 0.0, 4.0};
    auto batch = calc.batchEuclideanDistances(points.data(), 3, 2);
    if (batch.size() != 3)
    {
        return false;
    }
    if (std::abs(batch[0] - 3.0) > TOL)
    {
        return false;
    }
    if (std::abs(batch[2] - 5.0) > TOL)
    {
        return false;
    }

    double zero_a[] = {0.0, 0.0, 0.0};
    double zero_b[] = {0.0, 0.0, 0.0};
    double zero_cos = calc.cosineDistance(zero_a, zero_b, 3);
    if (std::abs(zero_cos - 1.0) > TOL)
    {
        return false;
    }

    return true;
}

bool check_geometry_volume_known_shapes()
{
    Simplex triangle({0, 1, 2});
    std::vector<std::vector<double>> tri_coords = {{0.0, 0.0}, {3.0, 0.0}, {0.0, 4.0}};
    double area = triangle.volume(tri_coords);
    if (std::abs(area - 6.0) > 1e-8)
    {
        return false;
    }

    Simplex tetra({0, 1, 2, 3});
    std::vector<std::vector<double>> tet_coords = {
        {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
    double vol = tetra.volume(tet_coords);
    double expected = 1.0 / 6.0;
    if (std::abs(vol - expected) > 1e-8)
    {
        return false;
    }

    Simplex line({0, 1});
    std::vector<std::vector<double>> line_coords = {{0.0}, {5.0}};
    double length = line.volume(line_coords);
    if (std::abs(length - 5.0) > TOL)
    {
        return false;
    }

    Simplex point({0});
    std::vector<std::vector<double>> pt_coords = {{42.0}};
    double point_vol = point.volume(pt_coords);
    if (std::abs(point_vol) > TOL)
    {
        return false;
    }

    Simplex degenerate({0, 1, 2});
    std::vector<std::vector<double>> deg_coords = {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
    double deg_vol = degenerate.volume(deg_coords);
    if (deg_vol > TOL)
    {
        return false;
    }

    return true;
}

} // namespace

int main()
{
    int failures = 0;

    auto run = [&](const char *name, bool ok) {
        if (!ok)
        {
            std::cerr << "FAIL: " << name << "\n";
            ++failures;
        }
        else
        {
            std::cout << "PASS: " << name << "\n";
        }
    };

    run("simplex_operations", check_simplex_operations());
    run("simplicial_complex_construction", check_simplicial_complex_construction());
    run("chain_complex_triangle", check_chain_complex_triangle());
    run("distance_simd_match_scalar", check_distance_simd_match_scalar());
    run("geometry_volume_known_shapes", check_geometry_volume_known_shapes());

    return failures > 0 ? 1 : 0;
}

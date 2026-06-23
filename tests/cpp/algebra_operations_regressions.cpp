
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

using nerve::Index;
using nerve::Size;
using nerve::algebra::Simplex;

constexpr double TOL = 1e-10;

std::mt19937_64 make_rng()
{
    return std::mt19937_64(2024);
}

bool check_simplex_construction()
{
    Simplex s({0, 1, 2});
    if (s.dimension() != 2)
    {
        std::cerr << "expected dim 2, got " << s.dimension() << "\n";
        return false;
    }
    if (s.numVertices() != 3)
    {
        std::cerr << "expected 3 verts, got " << s.numVertices() << "\n";
        return false;
    }
    return true;
}

bool check_simplex_vertex_access()
{
    Simplex s({3, 1, 2});
    auto verts = s.vertices();
    if (verts.size() != 3)
        return false;
    if (!s.contains(1) || !s.contains(2) || !s.contains(3))
    {
        std::cerr << "vertex containment check failed\n";
        return false;
    }
    if (s.contains(0))
    {
        std::cerr << "should not contain vertex 0\n";
        return false;
    }
    return true;
}

bool check_face_enumeration()
{
    Simplex tri({0, 1, 2});
    auto faces = tri.kFaces(static_cast<Size>(1));
    if (faces.size() != 3)
    {
        std::cerr << "expected 3 edges, got " << faces.size() << "\n";
        return false;
    }
    return true;
}

bool check_coface_enumeration()
{
    Simplex edge({0, 1});
    std::vector<Simplex> complex = {Simplex({0}),      Simplex({1}),    Simplex({2}),
                                    Simplex({0, 1}),   Simplex({0, 2}), Simplex({1, 2}),
                                    Simplex({0, 1, 2})};
    auto cofaces = edge.cofaces(complex);
    bool found_tri = false;
    for (const auto &c : cofaces)
    {
        if (c.dimension() == 2 && c.contains(0) && c.contains(1) && c.contains(2))
            found_tri = true;
    }
    if (!found_tri)
    {
        std::cerr << "edge (0,1) should coface triangle (0,1,2)\n";
        return false;
    }
    return true;
}

bool check_simplex_sorting()
{
    Simplex a({0, 2});
    Simplex b({0, 1});
    if (!(b < a))
    {
        std::cerr << "sorting order incorrect\n";
        return false;
    }
    return true;
}

bool check_simplex_equality()
{
    Simplex a({0, 1, 2});
    Simplex b({2, 1, 0});
    b.sortVertices();
    if (!(a == b))
    {
        std::cerr << "simplices should be equal after sort\n";
        return false;
    }
    return true;
}

bool check_generate_all_faces()
{
    Simplex tri({0, 1, 2});
    auto all = nerve::algebra::generateAllFaces(tri);
    if (all.size() != 3)
    {
        std::cerr << "expected 3 faces, got " << all.size() << "\n";
        return false;
    }
    return true;
}

bool check_simplex_join()
{
    Simplex a({0, 1});
    Simplex b({2, 3});
    auto j = nerve::algebra::join(a, b);
    if (j.dimension() != 3)
    {
        std::cerr << "join dimension expected 3, got " << j.dimension() << "\n";
        return false;
    }
    return true;
}

} // namespace

int main()
{
    if (!check_simplex_construction())
    {
        std::cerr << "FAIL: simplex construction\n";
        return 1;
    }
    if (!check_simplex_vertex_access())
    {
        std::cerr << "FAIL: vertex access\n";
        return 1;
    }
    if (!check_face_enumeration())
    {
        std::cerr << "FAIL: face enumeration\n";
        return 1;
    }
    if (!check_coface_enumeration())
    {
        std::cerr << "FAIL: coface enumeration\n";
        return 1;
    }
    if (!check_simplex_sorting())
    {
        std::cerr << "FAIL: simplex sorting\n";
        return 1;
    }
    if (!check_simplex_equality())
    {
        std::cerr << "FAIL: simplex equality\n";
        return 1;
    }
    if (!check_generate_all_faces())
    {
        std::cerr << "FAIL: generate all faces\n";
        return 1;
    }
    if (!check_simplex_join())
    {
        std::cerr << "FAIL: simplex join\n";
        return 1;
    }
    return 0;
}

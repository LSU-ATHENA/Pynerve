
#include "nerve/algebra/boundary.hpp"
#include "nerve/algebra/complex.hpp"
#include "nerve/algebra/simplex.hpp"
#include "nerve/persistence/reduction/reduction_hypha_ops.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

#ifdef NERVE_HAS_CUDA

#include <cuda_runtime.h>

namespace
{

bool hasCudaAtRuntime()
{
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return err == cudaSuccess && device_count > 0;
}

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
    HyphaReducer hr(cfg);
    (void)hr;
    return true;
}

bool check_hypha_config_get_set()
{
    HyphaReducer hr;
    HyphaReducer::Config cfg;
    hr.setConfig(cfg);
    (void)hr.config();
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

SimplicialComplex make_tetrahedron_complex()
{
    // Tetrahedron: 4 vertices, 6 edges, 4 triangles, 1 tetrahedron
    // Vertices: 0,1,2,3
    // Edges: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
    // Triangles: (0,1,2), (0,1,3), (0,2,3), (1,2,3)
    // Tetrahedron: (0,1,2,3)
    SimplicialComplex complex;
    complex.addSimplex(Simplex({0}));
    complex.addSimplex(Simplex({1}));
    complex.addSimplex(Simplex({2}));
    complex.addSimplex(Simplex({3}));
    complex.addSimplex(Simplex({0, 1}));
    complex.addSimplex(Simplex({0, 2}));
    complex.addSimplex(Simplex({0, 3}));
    complex.addSimplex(Simplex({1, 2}));
    complex.addSimplex(Simplex({1, 3}));
    complex.addSimplex(Simplex({2, 3}));
    complex.addSimplex(Simplex({0, 1, 2}));
    complex.addSimplex(Simplex({0, 1, 3}));
    complex.addSimplex(Simplex({0, 2, 3}));
    complex.addSimplex(Simplex({1, 2, 3}));
    complex.addSimplex(Simplex({0, 1, 2, 3}));
    return complex;
}

bool check_hypha_dim3_reduction()
{
    auto complex = make_tetrahedron_complex();
    BoundaryMatrix bm(complex, 3);
    HyphaReducer hr;
    auto pairs = hr.compute(bm);
    if (pairs.empty())
    {
        std::cerr << "dim-3 hypha reduction produced no pairs\n";
        return false;
    }
    // Tetrahedron has H3 homology: one essential class (dim-3 cycle with no boundary)
    // Plus dim-0 and dim-1/2 pairs from the boundary matrix
    for (const auto &p : pairs)
    {
        if (!p.isInfinite() && !(p.birth <= p.death + 1e-12))
        {
            std::cerr << "dim-3: birth<=death violated\n";
            return false;
        }
        if (!p.isInfinite() && p.lifetime() < -1e-12)
        {
            std::cerr << "dim-3: negative persistence\n";
            return false;
        }
        if (p.dimension < 0)
        {
            std::cerr << "dim-3: negative dimension\n";
            return false;
        }
    }
    return true;
}

bool check_hypha_dim3_determinism()
{
    auto complex = make_tetrahedron_complex();
    BoundaryMatrix bm(complex, 3);
    HyphaReducer hr;
    auto run1 = hr.compute(bm);
    auto run2 = hr.compute(bm);
    if (run1.size() != run2.size())
    {
        std::cerr << "dim-3 determinism size mismatch: " << run1.size()
                  << " vs " << run2.size() << "\n";
        return false;
    }
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

    if (!hasCudaAtRuntime())
    {
        std::cerr << "No CUDA device at runtime  --  skipping GPU-dependent hypha tests\n";
        return 0;
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
    if (!check_hypha_dim3_reduction())
    {
        std::cerr << "FAIL: hypha dim-3 reduction\n";
        return 1;
    }
    if (!check_hypha_dim3_determinism())
    {
        std::cerr << "FAIL: hypha dim-3 determinism\n";
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

#ifdef NERVE_HAS_CUDA
#include "nerve/algebra/boundary.hpp"
#include "nerve/algebra/complex.hpp"
#include "nerve/algebra/simplex.hpp"
#include "nerve/core_types.hpp"
#include "nerve/persistence/cuda/detail/cuda_detail.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

namespace
{

using nerve::algebra::BoundaryMatrix;
using nerve::algebra::Simplex;
using nerve::algebra::SimplicialComplex;
using nerve::gpu::persistence::detail::GPUClearingEngine;

bool check_gpu_clearing_construction()
{
    BoundaryMatrix matrix(0, 0);
    std::vector<int> dims;
    std::vector<double> filt;
    GPUClearingEngine::ClearingResult result;
    auto status = GPUClearingEngine::applyClearingOptimization(
        matrix, dims, filt, 0, 1.0, result);
    if (status.isError())
        return false;
    return true;
}

bool check_gpu_clearing_triangle()
{
    // Build a triangle: 3 vertices, 3 edges, 1 triangle
    SimplicialComplex complex;
    complex.addSimplexWithFiltration(Simplex({0}), 0.0);
    complex.addSimplexWithFiltration(Simplex({1}), 0.0);
    complex.addSimplexWithFiltration(Simplex({2}), 0.0);
    complex.addSimplexWithFiltration(Simplex({0, 1}), 1.0);
    complex.addSimplexWithFiltration(Simplex({1, 2}), 2.0);
    complex.addSimplexWithFiltration(Simplex({0, 2}), 3.0);
    complex.addSimplexWithFiltration(Simplex({0, 1, 2}), 4.0);

    BoundaryMatrix bm(complex, 2);
    std::vector<int> dims;
    std::vector<double> filt;
    for (nerve::Size i = 0; i < bm.cols(); ++i)
    {
        dims.push_back(static_cast<int>(bm.getColSimplexDimension(i)));
        filt.push_back(bm.getFiltrationValue(i));
    }

    GPUClearingEngine::ClearingResult result;
    auto status = GPUClearingEngine::applyClearingOptimization(
        bm, dims, filt, 2, 100.0, result);
    if (status.isError())
    {
        std::fprintf(stderr, "clearing triangle failed: %s\n", status.compactSummary().c_str());
        return false;
    }

    // Verify results are populated
    if (result.positive_simplices.size() != static_cast<std::size_t>(bm.cols()))
    {
        std::fprintf(stderr, "positive_simplices size mismatch: %zu vs %zu\n",
                     result.positive_simplices.size(), static_cast<std::size_t>(bm.cols()));
        return false;
    }
    if (result.columns_to_clear.size() != static_cast<std::size_t>(bm.cols()))
    {
        std::fprintf(stderr, "columns_to_clear size mismatch\n");
        return false;
    }

    // Triangle should have some positive simplices and some columns to clear
    bool has_positive = false;
    bool has_clear = false;
    for (std::size_t i = 0; i < result.positive_simplices.size(); ++i)
    {
        if (result.positive_simplices[i]) has_positive = true;
        if (result.columns_to_clear[i]) has_clear = true;
    }
    if (!has_positive)
    {
        std::fprintf(stderr, "expected at least one positive simplex\n");
        return false;
    }
    return true;
}

bool check_gpu_clearing_error_paths()
{
    // Non-finite max_filtration
    {
        BoundaryMatrix matrix(0, 0);
        std::vector<int> dims;
        std::vector<double> filt;
        GPUClearingEngine::ClearingResult result;
        auto status = GPUClearingEngine::applyClearingOptimization(
            matrix, dims, filt, 0, std::numeric_limits<double>::quiet_NaN(), result);
        if (status.isSuccess())
        {
            std::fprintf(stderr, "expected error for NaN max_filtration\n");
            return false;
        }
    }

    // Non-finite filtration values
    {
        BoundaryMatrix matrix(0, 0);
        std::vector<int> dims;
        std::vector<double> filt = {std::numeric_limits<double>::quiet_NaN()};
        GPUClearingEngine::ClearingResult result;
        auto status = GPUClearingEngine::applyClearingOptimization(
            matrix, dims, filt, 0, 1.0, result);
        if (status.isSuccess())
        {
            std::fprintf(stderr, "expected error for NaN filtration value\n");
            return false;
        }
    }

    // Mismatched sizes (filt size != dims size)
    {
        BoundaryMatrix matrix(0, 0);
        std::vector<int> dims = {0, 1};
        std::vector<double> filt = {0.0};
        GPUClearingEngine::ClearingResult result;
        auto status = GPUClearingEngine::applyClearingOptimization(
            matrix, dims, filt, 0, 1.0, result);
        if (status.isSuccess())
        {
            std::fprintf(stderr, "expected error for mismatched sizes\n");
            return false;
        }
    }

    // Zero dimensions + zero filtration (valid empty case)
    {
        BoundaryMatrix matrix(0, 0);
        std::vector<int> dims;
        std::vector<double> filt;
        GPUClearingEngine::ClearingResult result;
        auto status = GPUClearingEngine::applyClearingOptimization(
            matrix, dims, filt, 0, 1.0, result);
        if (status.isError())
        {
            std::fprintf(stderr, "expected success for empty case\n");
            return false;
        }
    }

    return true;
}

bool check_gpu_clearing_estimate_savings()
{
    // Build a triangle: 3 vertices, 3 edges, 1 triangle
    SimplicialComplex complex;
    complex.addSimplex(Simplex({0}));
    complex.addSimplex(Simplex({1}));
    complex.addSimplex(Simplex({2}));
    complex.addSimplex(Simplex({0, 1}));
    complex.addSimplex(Simplex({1, 2}));
    complex.addSimplex(Simplex({0, 2}));
    complex.addSimplex(Simplex({0, 1, 2}));

    BoundaryMatrix bm(complex, 2);
    std::vector<int> dims;
    for (nerve::Size i = 0; i < bm.cols(); ++i)
        dims.push_back(static_cast<int>(bm.getColSimplexDimension(i)));

    std::size_t savings = GPUClearingEngine::estimateSavingsCpu(dims, bm);
    if (savings == 0)
    {
        std::fprintf(stderr, "expected non-zero savings for triangle\n");
        return false;
    }

    return true;
}

bool check_gpu_clearing_square()
{
    // Square with diagonal: 4 vertices, 5 edges, 2 triangles (11 simplices, dim-2)
    SimplicialComplex complex;
    complex.addSimplexWithFiltration(Simplex({0}), 0.0);
    complex.addSimplexWithFiltration(Simplex({1}), 0.0);
    complex.addSimplexWithFiltration(Simplex({2}), 0.0);
    complex.addSimplexWithFiltration(Simplex({3}), 0.0);
    complex.addSimplexWithFiltration(Simplex({0, 1}), 1.0);
    complex.addSimplexWithFiltration(Simplex({0, 2}), 1.0);
    complex.addSimplexWithFiltration(Simplex({1, 3}), 1.0);
    complex.addSimplexWithFiltration(Simplex({2, 3}), 1.0);
    complex.addSimplexWithFiltration(Simplex({0, 3}), 1.414);
    complex.addSimplexWithFiltration(Simplex({0, 1, 3}), 2.0);
    complex.addSimplexWithFiltration(Simplex({0, 2, 3}), 2.0);

    BoundaryMatrix bm(complex, 2);
    std::vector<int> dims;
    std::vector<double> filt;
    for (nerve::Size i = 0; i < bm.cols(); ++i)
    {
        dims.push_back(static_cast<int>(bm.getColSimplexDimension(i)));
        filt.push_back(bm.getFiltrationValue(i));
    }

    if (bm.cols() != 11)
    {
        std::fprintf(stderr, "square: expected 11 columns, got %zu\n",
                     static_cast<std::size_t>(bm.cols()));
        return false;
    }

    GPUClearingEngine::ClearingResult result;
    auto status = GPUClearingEngine::applyClearingOptimization(
        bm, dims, filt, 2, 100.0, result);
    if (status.isError())
    {
        std::fprintf(stderr, "square clearing failed: %s\n", status.compactSummary().c_str());
        return false;
    }

    if (result.positive_simplices.size() != 11 || result.columns_to_clear.size() != 11)
    {
        std::fprintf(stderr, "square: result size mismatch\n");
        return false;
    }

    // Count positive and clearable columns
    int n_positive = 0, n_clear = 0;
    for (std::size_t i = 0; i < result.positive_simplices.size(); ++i)
    {
        if (result.positive_simplices[i]) n_positive++;
        if (result.columns_to_clear[i]) n_clear++;
    }

    if (n_positive == 0)
    {
        std::fprintf(stderr, "square: expected positive simplices, got %d\n", n_positive);
        return false;
    }

    // Operations saved should be non-zero for a non-trivial complex
    if (result.operations_saved == 0)
    {
        std::fprintf(stderr, "square: expected non-zero operations_saved\n");
        return false;
    }

    return true;
}

bool check_gpu_clearing_octahedron()
{
    // Octahedron: 6 vertices, 12 edges, 8 triangles (26 simplices, dim-2)
    SimplicialComplex complex;
    // Vertices
    complex.addSimplexWithFiltration(Simplex({0}), 0.0);
    complex.addSimplexWithFiltration(Simplex({1}), 0.0);
    complex.addSimplexWithFiltration(Simplex({2}), 0.0);
    complex.addSimplexWithFiltration(Simplex({3}), 0.0);
    complex.addSimplexWithFiltration(Simplex({4}), 0.0);
    complex.addSimplexWithFiltration(Simplex({5}), 0.0);
    // Edges from north pole (0) to equator (1,2,3,4)
    complex.addSimplexWithFiltration(Simplex({0, 1}), 1.0);
    complex.addSimplexWithFiltration(Simplex({0, 2}), 1.0);
    complex.addSimplexWithFiltration(Simplex({0, 3}), 1.0);
    complex.addSimplexWithFiltration(Simplex({0, 4}), 1.0);
    // Edges from south pole (5) to equator
    complex.addSimplexWithFiltration(Simplex({1, 5}), 1.0);
    complex.addSimplexWithFiltration(Simplex({2, 5}), 1.0);
    complex.addSimplexWithFiltration(Simplex({3, 5}), 1.0);
    complex.addSimplexWithFiltration(Simplex({4, 5}), 1.0);
    // Equatorial edges
    complex.addSimplexWithFiltration(Simplex({1, 2}), 1.0);
    complex.addSimplexWithFiltration(Simplex({2, 3}), 1.0);
    complex.addSimplexWithFiltration(Simplex({3, 4}), 1.0);
    complex.addSimplexWithFiltration(Simplex({4, 1}), 1.0);
    // Upper triangles
    complex.addSimplexWithFiltration(Simplex({0, 1, 2}), 2.0);
    complex.addSimplexWithFiltration(Simplex({0, 2, 3}), 2.0);
    complex.addSimplexWithFiltration(Simplex({0, 3, 4}), 2.0);
    complex.addSimplexWithFiltration(Simplex({0, 4, 1}), 2.0);
    // Lower triangles
    complex.addSimplexWithFiltration(Simplex({1, 2, 5}), 2.0);
    complex.addSimplexWithFiltration(Simplex({2, 3, 5}), 2.0);
    complex.addSimplexWithFiltration(Simplex({3, 4, 5}), 2.0);
    complex.addSimplexWithFiltration(Simplex({4, 1, 5}), 2.0);

    BoundaryMatrix bm(complex, 2);
    std::vector<int> dims;
    for (nerve::Size i = 0; i < bm.cols(); ++i)
        dims.push_back(static_cast<int>(bm.getColSimplexDimension(i)));

    std::vector<double> filt;
    for (nerve::Size i = 0; i < bm.cols(); ++i)
        filt.push_back(bm.getFiltrationValue(i));

    if (bm.cols() != 26)
    {
        std::fprintf(stderr, "octahedron: expected 26 columns, got %zu\n",
                     static_cast<std::size_t>(bm.cols()));
        return false;
    }

    GPUClearingEngine::ClearingResult result;
    auto status = GPUClearingEngine::applyClearingOptimization(
        bm, dims, filt, 2, 100.0, result);
    if (status.isError())
    {
        std::fprintf(stderr, "octahedron clearing failed: %s\n", status.compactSummary().c_str());
        return false;
    }

    if (result.positive_simplices.size() != 26 || result.columns_to_clear.size() != 26)
    {
        std::fprintf(stderr, "octahedron: result size mismatch\n");
        return false;
    }

    // All 6 vertices should be positive
    int n_positive = 0, n_positive_dim0 = 0;
    for (std::size_t i = 0; i < result.positive_simplices.size(); ++i)
    {
        if (result.positive_simplices[i])
        {
            n_positive++;
            if (dims[i] == 0) n_positive_dim0++;
        }
    }

    if (n_positive_dim0 != 6)
    {
        std::fprintf(stderr, "octahedron: expected 6 positive dim-0 simplices, got %d\n",
                     n_positive_dim0);
        return false;
    }
    if (n_positive == 0)
    {
        std::fprintf(stderr, "octahedron: no positive simplices\n");
        return false;
    }

    int n_clear = 0;
    for (std::size_t i = 0; i < result.columns_to_clear.size(); ++i)
        if (result.columns_to_clear[i]) n_clear++;

    if (n_clear == 0)
    {
        std::fprintf(stderr, "octahedron: expected some clearable columns\n");
        return false;
    }

    if (result.operations_saved == 0)
    {
        std::fprintf(stderr, "octahedron: expected non-zero operations_saved\n");
        return false;
    }

    return true;
}

bool check_gpu_clearing_tetrahedron_dim3()
{
    // Tetrahedron: 4 vertices, 6 edges, 4 triangles, 1 tetrahedron (15 simplices, dim-3 target)
    SimplicialComplex complex;
    complex.addSimplexWithFiltration(Simplex({0}), 0.0);
    complex.addSimplexWithFiltration(Simplex({1}), 0.0);
    complex.addSimplexWithFiltration(Simplex({2}), 0.0);
    complex.addSimplexWithFiltration(Simplex({3}), 0.0);
    complex.addSimplexWithFiltration(Simplex({0, 1}), 1.0);
    complex.addSimplexWithFiltration(Simplex({0, 2}), 1.0);
    complex.addSimplexWithFiltration(Simplex({0, 3}), 1.0);
    complex.addSimplexWithFiltration(Simplex({1, 2}), 1.0);
    complex.addSimplexWithFiltration(Simplex({1, 3}), 1.0);
    complex.addSimplexWithFiltration(Simplex({2, 3}), 1.0);
    complex.addSimplexWithFiltration(Simplex({0, 1, 2}), 2.0);
    complex.addSimplexWithFiltration(Simplex({0, 1, 3}), 2.0);
    complex.addSimplexWithFiltration(Simplex({0, 2, 3}), 2.0);
    complex.addSimplexWithFiltration(Simplex({1, 2, 3}), 2.0);
    complex.addSimplexWithFiltration(Simplex({0, 1, 2, 3}), 3.0);

    BoundaryMatrix bm(complex, 3);
    std::vector<int> dims;
    std::vector<double> filt;
    for (nerve::Size i = 0; i < bm.cols(); ++i)
    {
        dims.push_back(static_cast<int>(bm.getColSimplexDimension(i)));
        filt.push_back(bm.getFiltrationValue(i));
    }

    if (bm.cols() != 15)
    {
        std::fprintf(stderr, "tetrahedron dim-3: expected 15 columns, got %zu\n",
                     static_cast<std::size_t>(bm.cols()));
        return false;
    }

    // Test with target_dimension=3 (clearing at tetrahedron level)
    GPUClearingEngine::ClearingResult result;
    auto status = GPUClearingEngine::applyClearingOptimization(
        bm, dims, filt, 3, 100.0, result);
    if (status.isError())
    {
        std::fprintf(stderr, "tetrahedron dim-3 clearing failed: %s\n",
                     status.compactSummary().c_str());
        return false;
    }

    if (result.positive_simplices.size() != 15 || result.columns_to_clear.size() != 15)
    {
        std::fprintf(stderr, "tetrahedron dim-3: result size mismatch\n");
        return false;
    }

    // All 4 vertices should be positive
    int n_positive_dim0 = 0, n_positive_dim3 = 0;
    for (std::size_t i = 0; i < result.positive_simplices.size(); ++i)
    {
        if (result.positive_simplices[i] && dims[i] == 0) n_positive_dim0++;
        if (result.positive_simplices[i] && dims[i] == 3) n_positive_dim3++;
    }

    if (n_positive_dim0 != 4)
    {
        std::fprintf(stderr, "tetrahedron dim-3: expected 4 positive dim-0, got %d\n",
                     n_positive_dim0);
        return false;
    }

    if (result.operations_saved == 0)
    {
        std::fprintf(stderr, "tetrahedron dim-3: expected non-zero operations_saved\n");
        return false;
    }

    return true;
}

bool check_gpu_clearing_cubical_grid()
{
    // Cubical grid: 8 vertices, 12 edges, 12 triangles (32 simplices total, dim-2)
    // Faces are quadrilateral, triangulated: each face split into 2 triangles
    SimplicialComplex complex;
    // Vertices
    for (int i = 0; i < 8; ++i)
        complex.addSimplexWithFiltration(Simplex({i}), 0.0);
    // Bottom face edges
    complex.addSimplexWithFiltration(Simplex({0, 1}), 1.0);
    complex.addSimplexWithFiltration(Simplex({1, 3}), 1.0);
    complex.addSimplexWithFiltration(Simplex({3, 2}), 1.0);
    complex.addSimplexWithFiltration(Simplex({2, 0}), 1.0);
    // Top face edges
    complex.addSimplexWithFiltration(Simplex({4, 5}), 1.0);
    complex.addSimplexWithFiltration(Simplex({5, 7}), 1.0);
    complex.addSimplexWithFiltration(Simplex({7, 6}), 1.0);
    complex.addSimplexWithFiltration(Simplex({6, 4}), 1.0);
    // Vertical edges
    complex.addSimplexWithFiltration(Simplex({0, 4}), 1.0);
    complex.addSimplexWithFiltration(Simplex({1, 5}), 1.0);
    complex.addSimplexWithFiltration(Simplex({2, 6}), 1.0);
    complex.addSimplexWithFiltration(Simplex({3, 7}), 1.0);
    // Bottom face triangles (split)
    complex.addSimplexWithFiltration(Simplex({0, 1, 3}), 2.0);
    complex.addSimplexWithFiltration(Simplex({0, 2, 3}), 2.0);
    // Top face triangles
    complex.addSimplexWithFiltration(Simplex({4, 5, 7}), 2.0);
    complex.addSimplexWithFiltration(Simplex({4, 6, 7}), 2.0);
    // Side face triangles
    complex.addSimplexWithFiltration(Simplex({0, 1, 5}), 2.0);
    complex.addSimplexWithFiltration(Simplex({0, 4, 5}), 2.0);
    complex.addSimplexWithFiltration(Simplex({1, 3, 7}), 2.0);
    complex.addSimplexWithFiltration(Simplex({1, 5, 7}), 2.0);
    complex.addSimplexWithFiltration(Simplex({2, 3, 7}), 2.0);
    complex.addSimplexWithFiltration(Simplex({2, 6, 7}), 2.0);
    complex.addSimplexWithFiltration(Simplex({0, 2, 6}), 2.0);
    complex.addSimplexWithFiltration(Simplex({0, 4, 6}), 2.0);

    BoundaryMatrix bm(complex, 2);
    std::vector<int> dims;
    std::vector<double> filt;
    for (nerve::Size i = 0; i < bm.cols(); ++i)
    {
        dims.push_back(static_cast<int>(bm.getColSimplexDimension(i)));
        filt.push_back(bm.getFiltrationValue(i));
    }

    const std::size_t expected = 8 + 12 + 12; // 8 vertices + 12 edges + 12 triangles
    if (static_cast<std::size_t>(bm.cols()) != expected)
    {
        std::fprintf(stderr, "cubical grid: expected %zu columns, got %zu\n",
                     expected, static_cast<std::size_t>(bm.cols()));
        return false;
    }

    GPUClearingEngine::ClearingResult result;
    auto status = GPUClearingEngine::applyClearingOptimization(
        bm, dims, filt, 2, 100.0, result);
    if (status.isError())
    {
        std::fprintf(stderr, "cubical grid clearing failed: %s\n", status.compactSummary().c_str());
        return false;
    }

    if (result.positive_simplices.size() != expected ||
        result.columns_to_clear.size() != expected)
    {
        std::fprintf(stderr, "cubical grid: result size mismatch\n");
        return false;
    }

    // All 8 vertices should be positive
    int n_positive_dim0 = 0;
    for (std::size_t i = 0; i < result.positive_simplices.size(); ++i)
        if (result.positive_simplices[i] && dims[i] == 0) n_positive_dim0++;

    if (n_positive_dim0 != 8)
    {
        std::fprintf(stderr, "cubical grid: expected 8 positive dim-0, got %d\n", n_positive_dim0);
        return false;
    }

    if (result.operations_saved == 0)
    {
        std::fprintf(stderr, "cubical grid: expected non-zero operations_saved\n");
        return false;
    }

    return true;
}

bool check_gpu_clearing_ratio_validation()
{
    // Construct several complexes and verify clearing ratio patterns.
    // The clearing ratio = columns_to_clear / total_columns should be
    // non-zero for any complex with structure, and should increase with
    // larger complexes.

    struct ComplexCase
    {
        std::string name;
        int n_columns;
        int n_clearable;
        std::size_t ops_saved;
    };
    std::vector<ComplexCase> cases;

    // Case 1: Empty (no simplices)
    {
        SimplicialComplex complex;
        BoundaryMatrix bm(complex, 0);
        std::vector<int> dims;
        std::vector<double> filt;
        GPUClearingEngine::ClearingResult result;
        auto status = GPUClearingEngine::applyClearingOptimization(
            bm, dims, filt, 0, 1.0, result);
        if (status.isError())
        {
            std::fprintf(stderr, "ratio empty: expected success\n");
            return false;
        }
        cases.push_back({"empty", 0, 0, 0});
    }

    // Case 2: Single point (dim-0 only, no boundary)
    {
        SimplicialComplex complex;
        complex.addSimplexWithFiltration(Simplex({0}), 0.0);
        BoundaryMatrix bm(complex, 0);
        std::vector<int> dims;
        std::vector<double> filt;
        for (nerve::Size i = 0; i < bm.cols(); ++i)
        {
            dims.push_back(static_cast<int>(bm.getColSimplexDimension(i)));
            filt.push_back(bm.getFiltrationValue(i));
        }
        GPUClearingEngine::ClearingResult result;
        auto status = GPUClearingEngine::applyClearingOptimization(
            bm, dims, filt, 0, 1.0, result);
        if (status.isError())
        {
            std::fprintf(stderr, "ratio single point: expected success\n");
            return false;
        }
        int n_clear = 0;
        for (std::size_t i = 0; i < result.columns_to_clear.size(); ++i)
            if (result.columns_to_clear[i]) n_clear++;
        cases.push_back({"single_point", static_cast<int>(bm.cols()), n_clear,
                         result.operations_saved});
    }

    // Case 3: Triangle (7 simplices)
    {
        SimplicialComplex complex;
        complex.addSimplexWithFiltration(Simplex({0}), 0.0);
        complex.addSimplexWithFiltration(Simplex({1}), 0.0);
        complex.addSimplexWithFiltration(Simplex({2}), 0.0);
        complex.addSimplexWithFiltration(Simplex({0, 1}), 1.0);
        complex.addSimplexWithFiltration(Simplex({1, 2}), 2.0);
        complex.addSimplexWithFiltration(Simplex({0, 2}), 3.0);
        complex.addSimplexWithFiltration(Simplex({0, 1, 2}), 4.0);

        BoundaryMatrix bm(complex, 2);
        std::vector<int> dims;
        std::vector<double> filt;
        for (nerve::Size i = 0; i < bm.cols(); ++i)
        {
            dims.push_back(static_cast<int>(bm.getColSimplexDimension(i)));
            filt.push_back(bm.getFiltrationValue(i));
        }
        GPUClearingEngine::ClearingResult result;
        auto status = GPUClearingEngine::applyClearingOptimization(
            bm, dims, filt, 2, 100.0, result);
        if (status.isError())
        {
            std::fprintf(stderr, "ratio triangle: expected success\n");
            return false;
        }
        int n_clear = 0;
        for (std::size_t i = 0; i < result.columns_to_clear.size(); ++i)
            if (result.columns_to_clear[i]) n_clear++;
        cases.push_back({"triangle", static_cast<int>(bm.cols()), n_clear,
                         result.operations_saved});
    }

    // Case 4: Octahedron (26 simplices)
    {
        SimplicialComplex complex;
        for (int i = 0; i < 6; ++i)
            complex.addSimplexWithFiltration(Simplex({i}), 0.0);
        complex.addSimplexWithFiltration(Simplex({0, 1}), 1.0);
        complex.addSimplexWithFiltration(Simplex({0, 2}), 1.0);
        complex.addSimplexWithFiltration(Simplex({0, 3}), 1.0);
        complex.addSimplexWithFiltration(Simplex({0, 4}), 1.0);
        complex.addSimplexWithFiltration(Simplex({1, 5}), 1.0);
        complex.addSimplexWithFiltration(Simplex({2, 5}), 1.0);
        complex.addSimplexWithFiltration(Simplex({3, 5}), 1.0);
        complex.addSimplexWithFiltration(Simplex({4, 5}), 1.0);
        complex.addSimplexWithFiltration(Simplex({1, 2}), 1.0);
        complex.addSimplexWithFiltration(Simplex({2, 3}), 1.0);
        complex.addSimplexWithFiltration(Simplex({3, 4}), 1.0);
        complex.addSimplexWithFiltration(Simplex({4, 1}), 1.0);
        complex.addSimplexWithFiltration(Simplex({0, 1, 2}), 2.0);
        complex.addSimplexWithFiltration(Simplex({0, 2, 3}), 2.0);
        complex.addSimplexWithFiltration(Simplex({0, 3, 4}), 2.0);
        complex.addSimplexWithFiltration(Simplex({0, 4, 1}), 2.0);
        complex.addSimplexWithFiltration(Simplex({1, 2, 5}), 2.0);
        complex.addSimplexWithFiltration(Simplex({2, 3, 5}), 2.0);
        complex.addSimplexWithFiltration(Simplex({3, 4, 5}), 2.0);
        complex.addSimplexWithFiltration(Simplex({4, 1, 5}), 2.0);

        BoundaryMatrix bm(complex, 2);
        std::vector<int> dims;
        std::vector<double> filt;
        for (nerve::Size i = 0; i < bm.cols(); ++i)
        {
            dims.push_back(static_cast<int>(bm.getColSimplexDimension(i)));
            filt.push_back(bm.getFiltrationValue(i));
        }
        GPUClearingEngine::ClearingResult result;
        auto status = GPUClearingEngine::applyClearingOptimization(
            bm, dims, filt, 2, 100.0, result);
        if (status.isError())
        {
            std::fprintf(stderr, "ratio octahedron: expected success\n");
            return false;
        }
        int n_clear = 0;
        for (std::size_t i = 0; i < result.columns_to_clear.size(); ++i)
            if (result.columns_to_clear[i]) n_clear++;
        cases.push_back({"octahedron", static_cast<int>(bm.cols()), n_clear,
                         result.operations_saved});
    }

    // Validate patterns across all cases
    for (const auto &c : cases)
    {
        // Empty case: no columns, no clearing
        if (c.n_columns == 0)
        {
            if (c.n_clearable != 0 || c.ops_saved != 0)
            {
                std::fprintf(stderr, "ratio %s: empty should have 0 clearable/ops\n",
                             c.name.c_str());
                return false;
            }
            continue;
        }

        // Non-empty case: clearing ratio should be >= 0 and <= 1
        double ratio = static_cast<double>(c.n_clearable) / static_cast<double>(c.n_columns);
        if (ratio < 0.0 || ratio > 1.0)
        {
            std::fprintf(stderr, "ratio %s: invalid ratio %f\n", c.name.c_str(), ratio);
            return false;
        }

        // Triangle and octahedron should have at least some clearable columns
        if (c.n_columns >= 7 && c.n_clearable == 0)
        {
            std::fprintf(stderr, "ratio %s: expected some clearable columns\n", c.name.c_str());
            return false;
        }
    }

    // Larger complex should have more operations saved (strictly)
    // Octahedron has more columns than triangle, so ops_saved should be >= triangle
    auto &tri_case = cases[2]; // triangle is third
    auto &oct_case = cases[3]; // octahedron is fourth
    if (oct_case.ops_saved < tri_case.ops_saved)
    {
        std::fprintf(stderr, "ratio: octahedron ops_saved=%zu < triangle ops_saved=%zu\n",
                     oct_case.ops_saved, tri_case.ops_saved);
        return false;
    }

    return true;
}

bool check_gpu_clearing_no_filtration()
{
    // Verify clearing works without filtration values
    SimplicialComplex complex;
    complex.addSimplex(Simplex({0}));
    complex.addSimplex(Simplex({1}));
    complex.addSimplex(Simplex({2}));
    complex.addSimplex(Simplex({0, 1}));
    complex.addSimplex(Simplex({1, 2}));
    complex.addSimplex(Simplex({0, 2}));
    complex.addSimplex(Simplex({0, 1, 2}));

    BoundaryMatrix bm(complex, 2);
    std::vector<int> dims;
    for (nerve::Size i = 0; i < bm.cols(); ++i)
        dims.push_back(static_cast<int>(bm.getColSimplexDimension(i)));

    // Pass empty filtration vector
    std::vector<double> empty_filt;
    GPUClearingEngine::ClearingResult result;
    auto status = GPUClearingEngine::applyClearingOptimization(
        bm, dims, empty_filt, 2, 100.0, result);
    if (status.isError())
    {
        std::fprintf(stderr, "no filtration: %s\n", status.compactSummary().c_str());
        return false;
    }

    if (result.positive_simplices.size() != 7 || result.columns_to_clear.size() != 7)
    {
        std::fprintf(stderr, "no filtration: size mismatch\n");
        return false;
    }

    int n_positive = 0, n_clear = 0;
    for (std::size_t i = 0; i < result.positive_simplices.size(); ++i)
    {
        if (result.positive_simplices[i]) n_positive++;
        if (result.columns_to_clear[i]) n_clear++;
    }

    if (n_positive < 3)
    {
        std::fprintf(stderr, "no filtration: expected >=3 positive, got %d\n", n_positive);
        return false;
    }

    return true;
}

bool check_gpu_clearing_target_dimensions()
{
    // Verify clearing behaves differently at different target dimensions
    SimplicialComplex complex;
    complex.addSimplexWithFiltration(Simplex({0}), 0.0);
    complex.addSimplexWithFiltration(Simplex({1}), 0.0);
    complex.addSimplexWithFiltration(Simplex({2}), 0.0);
    complex.addSimplexWithFiltration(Simplex({0, 1}), 1.0);
    complex.addSimplexWithFiltration(Simplex({1, 2}), 2.0);
    complex.addSimplexWithFiltration(Simplex({0, 2}), 3.0);
    complex.addSimplexWithFiltration(Simplex({0, 1, 2}), 4.0);

    BoundaryMatrix bm(complex, 2);
    std::vector<int> dims;
    std::vector<double> filt;
    for (nerve::Size i = 0; i < bm.cols(); ++i)
    {
        dims.push_back(static_cast<int>(bm.getColSimplexDimension(i)));
        filt.push_back(bm.getFiltrationValue(i));
    }

    // Run at target_dimension=1 (edge-level clearing)
    GPUClearingEngine::ClearingResult result_dim1;
    auto status1 = GPUClearingEngine::applyClearingOptimization(
        bm, dims, filt, 1, 100.0, result_dim1);
    if (status1.isError())
    {
        std::fprintf(stderr, "target dim=1: %s\n", status1.compactSummary().c_str());
        return false;
    }

    // Run at target_dimension=2 (triangle-level clearing)
    GPUClearingEngine::ClearingResult result_dim2;
    auto status2 = GPUClearingEngine::applyClearingOptimization(
        bm, dims, filt, 2, 100.0, result_dim2);
    if (status2.isError())
    {
        std::fprintf(stderr, "target dim=2: %s\n", status2.compactSummary().c_str());
        return false;
    }

    if (result_dim1.positive_simplices.size() != result_dim2.positive_simplices.size())
    {
        std::fprintf(stderr, "target dim: size mismatch\n");
        return false;
    }

    // Target dimension should affect clearing decisions
    // Different target dims may produce different clearing results
    bool different_clearing = false;
    for (std::size_t i = 0; i < result_dim1.columns_to_clear.size(); ++i)
    {
        if (result_dim1.columns_to_clear[i] != result_dim2.columns_to_clear[i])
        {
            different_clearing = true;
            break;
        }
    }

    if (!different_clearing)
    {
        std::fprintf(stderr, "target dim: expected different clearing for dim=1 vs dim=2\n");
        return false;
    }

    return true;
}

bool check_gpu_cohomology_configures()
{
    BoundaryMatrix matrix(0, 0);
    std::vector<nerve::Index> pivots;
    std::vector<bool> processed;
    std::vector<std::vector<nerve::Size>> coboundary;
    auto status = nerve::gpu::detail::GPUCohomologyEngine::performCohomologyReduction(
        matrix, pivots, processed, coboundary);
    if (status.isError())
        return false;
    return true;
}

bool check_gpu_reduction_configures()
{
    BoundaryMatrix matrix(0, 0);
    std::vector<nerve::Index> pivots;
    std::vector<std::pair<nerve::Size, nerve::Size>> pairs;
    auto status = nerve::gpu::persistence::ReductionEngine::computeReduction(matrix, pivots, pairs);
    if (status.isError())
        return false;
    return true;
}

bool check_multi_gpu_configuration()
{
    auto info = nerve::gpu::multi::detectMultiGpuConfiguration();
    if (info.num_gpus < 1)
        return false;
    auto indices = nerve::gpu::multi::distributeIndices(100, std::max(1, info.num_gpus));
    if (indices.empty())
        return false;
    return true;
}

bool check_error_mapping()
{
    namespace cu = nerve::persistence::accelerated::cuda_utils;
    if (cu::is_cuda_available())
    {
        auto props = cu::getDeviceProperties(0);
        if (props.isError())
            return false;
    }
    nerve::Size block = cu::getOptimalBlockSize(0, 0);
    if (block == 0)
        return false;
    nerve::Size grid = cu::getOptimalGridSize(1000, 256);
    if (grid == 0)
        return false;
    return true;
}

} // namespace

int main()
{
    if (!check_gpu_clearing_construction())
        return 1;
    if (!check_gpu_clearing_triangle())
        return 1;
    if (!check_gpu_clearing_error_paths())
        return 1;
    if (!check_gpu_clearing_estimate_savings())
        return 1;
    if (!check_gpu_clearing_square())
        return 1;
    if (!check_gpu_clearing_octahedron())
        return 1;
    if (!check_gpu_clearing_tetrahedron_dim3())
        return 1;
    if (!check_gpu_clearing_cubical_grid())
        return 1;
    if (!check_gpu_clearing_ratio_validation())
        return 1;
    if (!check_gpu_clearing_no_filtration())
        return 1;
    if (!check_gpu_clearing_target_dimensions())
        return 1;
    if (!check_gpu_cohomology_configures())
        return 1;
    if (!check_gpu_reduction_configures())
        return 1;
    if (!check_multi_gpu_configuration())
        return 1;
    if (!check_error_mapping())
        return 1;
    return 0;
}
#else
int main()
{
    return 0;
}
#endif

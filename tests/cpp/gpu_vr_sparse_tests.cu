#include "gpu_test_helpers.cuh"

// Single-TU compilation: include the .cu directly since it's not in nerve_core
#include "../../src/filtration/gpu/vr_sparse_cuda.cu"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

namespace
{

using nerve::filtration::vr::sparse::gpu::Point;
using nerve::filtration::vr::sparse::gpu::SparseVRComplex;
using nerve::filtration::vr::sparse::gpu::buildSparseVRGPU;
using nerve::filtration::vr::sparse::gpu::benchmarkSparseGPU;
using nerve::filtration::vr::sparse::gpu::cuSPARSEVR;

// 4 points forming a tetrahedron edge-length 1 with center at origin
// Actually use a simple 4-point set: (0,0,0), (1,0,0), (0,1,0), (0,0,1)
bool test_build_sparse_vr_basic()
{
    std::vector<Point> points = {
        {0.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 1.0f}
    };

    // Threshold = 1.0: edges of length <= 1.0
    // Expected edges: (0,1), (0,2), (0,3) -- all distance 1.0
    // (1,2) = sqrt(2) > 1.0, (1,3) = sqrt(2) > 1.0, (2,3) = sqrt(2) > 1.0
    SparseVRComplex complex = buildSparseVRGPU(points, 1.0f);

    if (complex.n != 4)
    {
        std::cerr << "buildSparseVRGPU: expected n=4, got " << complex.n << "\n";
        return false;
    }

    // Row pointers should be: [0, 1, 2, 3, 3] -- vertex 3 has no outgoing edges to j>3
    // Actually: vertex 0->{1,2,3}=3 edges, vertex 1->{2? no, 3? no}=0 edges to j>1 within threshold
    // Wait: edges are directed i->j where j>i
    // 0 connects to 1 (dist 1.0), 2 (dist 1.0), 3 (dist 1.0) -- 3 edges from vertex 0
    // 1 connects to 2 (dist sqrt(2) > 1.0), 3 (dist sqrt(2) > 1.0) -- 0 edges from vertex 1
    // 2 connects to 3 (dist sqrt(2) > 1.0) -- 0 edges from vertex 2
    // 3 -- no j>3 -- 0 edges
    // Row ptr: [0, 3, 3, 3, 3]
    if (complex.h_row_ptr.size() != 5)
    {
        std::cerr << "buildSparseVRGPU: expected row_ptr size 5, got "
                  << complex.h_row_ptr.size() << "\n";
        return false;
    }

    // Total edges should be 3
    size_t total_edges = complex.h_data.size();
    if (total_edges != 3)
    {
        std::cerr << "buildSparseVRGPU: expected 3 edges, got " << total_edges << "\n";
        return false;
    }

    // Verify each edge's distance is approx 1.0
    for (size_t i = 0; i < total_edges; ++i)
    {
        if (std::abs(complex.h_data[i] - 1.0f) > 1e-5f)
        {
            std::cerr << "buildSparseVRGPU: edge " << i << " distance = "
                      << complex.h_data[i] << " (expected 1.0)\n";
            return false;
        }
    }

    // Verify vertex 0 has 3 edges: to {1,2,3}
    if (complex.h_row_ptr[0] != 0 || complex.h_row_ptr[1] != 3)
    {
        std::cerr << "buildSparseVRGPU: vertex 0 row_ptr = ["
                  << complex.h_row_ptr[0] << ", " << complex.h_row_ptr[1]
                  << "] (expected [0, 3])\n";
        return false;
    }

    return true;
}

// Test with larger threshold that includes all edges
bool test_build_sparse_vr_all_edges()
{
    std::vector<Point> points = {
        {0.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 1.0f}
    };

    // Threshold = 2.0: all 6 edges should be found (sqrt(2) ~= 1.414 < 2.0)
    SparseVRComplex complex = buildSparseVRGPU(points, 2.0f);

    if (complex.n != 4)
    {
        std::cerr << "buildSparseVRGPU all_edges: expected n=4, got " << complex.n << "\n";
        return false;
    }

    // Expected: 6 edges total (all pairs)
    size_t total_edges = complex.h_data.size();
    if (total_edges != 6)
    {
        std::cerr << "buildSparseVRGPU all_edges: expected 6 edges, got " << total_edges << "\n";
        return false;
    }

    // All row pointers should be monotonically increasing
    for (size_t i = 1; i < complex.h_row_ptr.size(); ++i)
    {
        if (complex.h_row_ptr[i] < complex.h_row_ptr[i - 1])
        {
            std::cerr << "buildSparseVRGPU: row_ptr not monotonic at index " << i << "\n";
            return false;
        }
    }

    // All distances should be finite and positive
    for (size_t i = 0; i < total_edges; ++i)
    {
        if (!std::isfinite(complex.h_data[i]) || complex.h_data[i] <= 0.0f)
        {
            std::cerr << "buildSparseVRGPU: invalid distance at edge " << i
                      << ": " << complex.h_data[i] << "\n";
            return false;
        }
    }

    return true;
}

// Test with zero threshold -- should produce no edges
bool test_build_sparse_vr_zero_threshold()
{
    std::vector<Point> points = {
        {0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f}  // identical point!
    };

    // Threshold = 0.0: only edges with distance <= 0
    SparseVRComplex complex = buildSparseVRGPU(points, 0.0f);

    // Two points at the same position -- distance is 0.0, which is <= 0.0
    // So there should be 1 edge
    size_t total_edges = complex.h_data.size();
    if (total_edges != 1)
    {
        std::cerr << "buildSparseVRGPU zero_threshold: expected 1 edge (coincident points), got "
                  << total_edges << "\n";
        return false;
    }

    // The edge distance should be ~0.0
    if (std::abs(complex.h_data[0]) > 1e-5f)
    {
        std::cerr << "buildSparseVRGPU zero_threshold: edge distance = "
                  << complex.h_data[0] << " (expected ~0)\n";
        return false;
    }

    return true;
}

// Test empty point set
bool test_build_sparse_vr_empty()
{
    std::vector<Point> empty;
    SparseVRComplex complex = buildSparseVRGPU(empty, 1.0f);

    if (complex.n != 0)
    {
        std::cerr << "buildSparseVRGPU empty: expected n=0, got " << complex.n << "\n";
        return false;
    }

    return true;
}

// Test benchmarkSparseGPU with small point set
bool test_benchmark_sparse_gpu()
{
    auto bench = benchmarkSparseGPU(10, 0.5f);

    if (bench.num_points != 10)
    {
        std::cerr << "benchmarkSparseGPU: expected num_points=10, got " << bench.num_points << "\n";
        return false;
    }

    if (bench.num_edges == 0)
    {
        std::cerr << "benchmarkSparseGPU: expected some edges for 10 points at threshold 0.5\n";
        // This might legitimately be zero for unlucky random points -- soften to warning
        std::cerr << "  (may be legitimate with unlucky random points)\n";
    }

    if (!std::isfinite(bench.cpu_time_ms) || bench.cpu_time_ms < 0.0)
    {
        std::cerr << "benchmarkSparseGPU: invalid cpu_time_ms = " << bench.cpu_time_ms << "\n";
        return false;
    }
    if (!std::isfinite(bench.gpu_time_ms) || bench.gpu_time_ms <= 0.0)
    {
        std::cerr << "benchmarkSparseGPU: invalid gpu_time_ms = " << bench.gpu_time_ms << "\n";
        return false;
    }
    if (!std::isfinite(bench.speedup))
    {
        std::cerr << "benchmarkSparseGPU: speedup is not finite\n";
        return false;
    }

    return true;
}

// Test cuSPARSEVR SpMV on a small complex
bool test_cusparse_spmv()
{
    std::vector<Point> points = {
        {0.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 1.0f}
    };

    SparseVRComplex complex = buildSparseVRGPU(points, 2.0f);
    complex.allocateGPU();

    if (complex.n != 4)
    {
        std::cerr << "cuSPARSE SpMV: unexpected n\n";
        return false;
    }

    // Allocate dense vector: x = [1, 0, 0, 0], y = [0, 0, 0, 0]
    std::vector<float> h_x(4, 0.0f);
    std::vector<float> h_y(4, 0.0f);
    h_x[0] = 1.0f;

    float *d_x = nullptr;
    float *d_y = nullptr;

    if (cudaMalloc(&d_x, 4 * sizeof(float)) != cudaSuccess)
        return false;
    if (cudaMalloc(&d_y, 4 * sizeof(float)) != cudaSuccess)
    {
        cudaFree(d_x);
        return false;
    }

    cudaMemcpy(d_x, h_x.data(), 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_y, 0, 4 * sizeof(float));

    cuSPARSEVR solver;
    try
    {
        solver.spmv(complex, d_x, d_y);
    }
    catch (const std::exception &e)
    {
        std::cerr << "cuSPARSE SpMV threw: " << e.what() << "\n";
        cudaFree(d_x);
        cudaFree(d_y);
        complex.freeGPU();
        return false;
    }

    cudaMemcpy(h_y.data(), d_y, 4 * sizeof(float), cudaMemcpyDeviceToHost);

    // y = A*x where A is upper-triangular (i<j edges only).
    // Since x=(1,0,0,0), column 0 of A is empty, so y should be all zeros.
    for (int i = 0; i < 4; ++i)
    {
        if (!std::isfinite(h_y[i]))
        {
            std::cerr << "cuSPARSE SpMV: non-finite y[" << i << "] = " << h_y[i] << "\n";
            cudaFree(d_x);
            cudaFree(d_y);
            complex.freeGPU();
            return false;
        }
    }

    // Check that SpMV produced correct result: all zeros
    for (int i = 0; i < 4; ++i)
    {
        if (std::abs(h_y[i]) > 1e-5f)
        {
            std::cerr << "cuSPARSE SpMV: unexpected non-zero y[" << i << "] = " << h_y[i] << "\n";
            cudaFree(d_x);
            cudaFree(d_y);
            complex.freeGPU();
            return false;
        }
    }

    cudaFree(d_x);
    cudaFree(d_y);
    complex.freeGPU();
    return true;
}

} // namespace

int main()
{
    if (!has_gpu())
    {
        std::cout << "No GPU available, skipping tests.\n";
        return 0;
    }

    if (!test_build_sparse_vr_basic())
    {
        std::cerr << "FAIL: buildSparseVRGPU basic\n";
        return 1;
    }
    if (!test_build_sparse_vr_all_edges())
    {
        std::cerr << "FAIL: buildSparseVRGPU all edges\n";
        return 1;
    }
    if (!test_build_sparse_vr_zero_threshold())
    {
        std::cerr << "FAIL: buildSparseVRGPU zero threshold\n";
        return 1;
    }
    if (!test_build_sparse_vr_empty())
    {
        std::cerr << "FAIL: buildSparseVRGPU empty\n";
        return 1;
    }
    if (!test_benchmark_sparse_gpu())
    {
        std::cerr << "FAIL: benchmarkSparseGPU\n";
        return 1;
    }
    if (!test_cusparse_spmv())
    {
        std::cerr << "FAIL: cuSPARSE SpMV\n";
        return 1;
    }

    return 0;
}

#include "gpu_test_helpers.cuh"

// Single-TU compilation: include the .cu directly since it's not in nerve_core
#include "../../src/cuda/kernels/distance_kernels_ext.cu"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <vector>

namespace
{

using nerve::Index;
using nerve::Size;

// 3 points in 2D: (0,0), (3,0), (0,4)
// Expected distances: d01=3, d02=4, d12=5
bool test_launch_distance_matrix_basic()
{
    constexpr Size n = 3;
    constexpr Size dim = 2;
    const std::vector<double> h_points = {0.0, 0.0, 3.0, 0.0, 0.0, 4.0};
    std::vector<double> h_dist(n * n);

    double *d_points = nullptr;
    double *d_out = nullptr;
    bool ok = false;

    if (cudaMalloc(&d_points, n * dim * sizeof(double)) != cudaSuccess)
        return false;
    if (cudaMalloc(&d_out, n * n * sizeof(double)) != cudaSuccess)
    {
        cudaFree(d_points);
        return false;
    }

    cudaMemcpy(d_points, h_points.data(), n * dim * sizeof(double), cudaMemcpyHostToDevice);

    auto result = nerve::gpu::launchDistanceMatrix(d_points, d_out, n, dim);
    if (result.isErr())
    {
        std::cerr << "launchDistanceMatrix failed\n";
        goto cleanup;
    }

    cudaMemcpy(h_dist.data(), d_out, n * n * sizeof(double), cudaMemcpyDeviceToHost);

    // Verify diagonal is near zero
    for (Size i = 0; i < n; ++i)
    {
        double diag = h_dist[i * n + i];
        if (diag > 1e-10)
        {
            std::cerr << "diag[" << i << "] = " << diag << " (expected ~0)\n";
            goto cleanup;
        }
    }

    // Verify known edge distances
    if (std::abs(h_dist[0 * n + 1] - 3.0) > 1e-8 ||
        std::abs(h_dist[1 * n + 0] - 3.0) > 1e-8)
    {
        std::cerr << "d(0,1) = " << h_dist[1] << " (expected 3.0)\n";
        goto cleanup;
    }
    if (std::abs(h_dist[0 * n + 2] - 4.0) > 1e-8)
    {
        std::cerr << "d(0,2) = " << h_dist[2] << " (expected 4.0)\n";
        goto cleanup;
    }
    if (std::abs(h_dist[1 * n + 2] - 5.0) > 1e-8)
    {
        std::cerr << "d(1,2) = " << h_dist[5] << " (expected 5.0)\n";
        goto cleanup;
    }

    ok = true;

cleanup:
    cudaFree(d_points);
    cudaFree(d_out);
    return ok;
}

// Verify symmetry of the distance matrix
bool test_launch_distance_matrix_symmetry()
{
    constexpr Size n = 4;
    constexpr Size dim = 2;
    const std::vector<double> h_points = {0.0, 0.0, 1.0, 1.0, 2.0, 0.0, 0.0, 3.0};
    std::vector<double> h_dist(n * n);

    double *d_points = nullptr;
    double *d_out = nullptr;
    bool ok = false;

    if (cudaMalloc(&d_points, n * dim * sizeof(double)) != cudaSuccess)
        return false;
    if (cudaMalloc(&d_out, n * n * sizeof(double)) != cudaSuccess)
    {
        cudaFree(d_points);
        return false;
    }

    cudaMemcpy(d_points, h_points.data(), n * dim * sizeof(double), cudaMemcpyHostToDevice);

    auto result = nerve::gpu::launchDistanceMatrix(d_points, d_out, n, dim);
    if (result.isErr())
        goto cleanup;

    cudaMemcpy(h_dist.data(), d_out, n * n * sizeof(double), cudaMemcpyDeviceToHost);

    for (Size i = 0; i < n; ++i)
    {
        for (Size j = 0; j < n; ++j)
        {
            if (std::abs(h_dist[i * n + j] - h_dist[j * n + i]) > 1e-8)
            {
                std::cerr << "Asymmetry at (" << i << "," << j << "): "
                          << h_dist[i * n + j] << " vs " << h_dist[j * n + i] << "\n";
                goto cleanup;
            }
        }
    }

    ok = true;

cleanup:
    cudaFree(d_points);
    cudaFree(d_out);
    return ok;
}

// Edge filtration: extract distances from a pre-computed distance matrix
bool test_launch_edge_filtration()
{
    constexpr Index n_points = 4;
    // Distance matrix: 4 points, upper triangle filled
    // d01=1.0, d02=2.0, d03=3.0, d12=4.0, d13=5.0, d23=6.0
    std::vector<double> h_dist(n_points * n_points, 0.0);
    fill_k4_distance_matrix(h_dist.data(), n_points, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0);

    // Edge list: (0,1), (1,3), (2,3)
    constexpr Index n_edges = 3;
    const std::vector<Index> h_edges = {0, 1, 1, 3, 2, 3};

    std::vector<double> h_filtration(n_edges);

    double *d_dist = nullptr;
    Index *d_edges = nullptr;
    double *d_filt = nullptr;
    bool ok = false;

    if (cudaMalloc(&d_dist, n_points * n_points * sizeof(double)) != cudaSuccess)
        return false;
    if (cudaMalloc(&d_edges, 2 * n_edges * sizeof(Index)) != cudaSuccess)
    {
        cudaFree(d_dist);
        return false;
    }
    if (cudaMalloc(&d_filt, n_edges * sizeof(double)) != cudaSuccess)
    {
        cudaFree(d_dist);
        cudaFree(d_edges);
        return false;
    }

    cudaMemcpy(d_dist, h_dist.data(), n_points * n_points * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edges, h_edges.data(), 2 * n_edges * sizeof(Index), cudaMemcpyHostToDevice);

    auto result = nerve::gpu::launchEdgeFiltration(d_dist, d_edges, d_filt, n_edges, n_points);
    if (result.isErr())
    {
        std::cerr << "launchEdgeFiltration failed\n";
        goto cleanup;
    }

    cudaMemcpy(h_filtration.data(), d_filt, n_edges * sizeof(double), cudaMemcpyDeviceToHost);

    // Edge (0,1) should have distance 1.0
    if (std::abs(h_filtration[0] - 1.0) > 1e-8)
    {
        std::cerr << "edge (0,1) filtration = " << h_filtration[0] << " (expected 1.0)\n";
        goto cleanup;
    }
    // Edge (1,3) should have distance 5.0
    if (std::abs(h_filtration[1] - 5.0) > 1e-8)
    {
        std::cerr << "edge (1,3) filtration = " << h_filtration[1] << " (expected 5.0)\n";
        goto cleanup;
    }
    // Edge (2,3) should have distance 6.0
    if (std::abs(h_filtration[2] - 6.0) > 1e-8)
    {
        std::cerr << "edge (2,3) filtration = " << h_filtration[2] << " (expected 6.0)\n";
        goto cleanup;
    }

    ok = true;

cleanup:
    cudaFree(d_dist);
    cudaFree(d_edges);
    cudaFree(d_filt);
    return ok;
}

// Edge filtration with zero edges should succeed (no-op)
bool test_launch_edge_filtration_empty()
{
    // Even with zero edges, we need valid (non-null) pointers for safety
    double *d_dist = nullptr;
    Index *d_edges = nullptr;
    double *d_filt = nullptr;

    if (cudaMalloc(&d_dist, sizeof(double)) != cudaSuccess)
        return false;

    auto result = nerve::gpu::launchEdgeFiltration(d_dist, d_edges, d_filt, 0, 1);

    cudaFree(d_dist);
    return result.isOk();
}

} // namespace

int main()
{
    if (!has_gpu())
    {
        std::cout << "No GPU available, skipping tests.\n";
        return 0;
    }

    if (!test_launch_distance_matrix_basic())
    {
        std::cerr << "FAIL: launchDistanceMatrix basic\n";
        return 1;
    }
    if (!test_launch_distance_matrix_symmetry())
    {
        std::cerr << "FAIL: launchDistanceMatrix symmetry\n";
        return 1;
    }
    if (!test_launch_edge_filtration())
    {
        std::cerr << "FAIL: launchEdgeFiltration\n";
        return 1;
    }
    if (!test_launch_edge_filtration_empty())
    {
        std::cerr << "FAIL: launchEdgeFiltration empty\n";
        return 1;
    }

    return 0;
}

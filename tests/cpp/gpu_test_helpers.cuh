#pragma once

#include "nerve/persistence/cuda/cuda_distance_matrix.hpp"
#include "nerve/persistence/cuda/cuda_edge_extraction.hpp"
#include "nerve/persistence/cuda/thread_block_cluster.hpp"

#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

namespace
{

using namespace nerve::persistence::accelerated;

inline bool check_cuda(cudaError_t code, const char *expression)
{
    if (code == cudaSuccess)
        return true;
    std::cerr << expression << " failed: " << cudaGetErrorString(code) << '\n';
    return false;
}

inline bool has_gpu()
{
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return err == cudaSuccess && device_count > 0;
}

inline bool approx_equal(double a, double b, double eps = 1e-8)
{
    return std::abs(a - b) <= eps * std::max(1.0, std::max(std::abs(a), std::abs(b)));
}

// 3 points forming a right triangle: (0,0), (3,0), (0,4)
// Distances: d(0,1)=3, d(0,2)=4, d(1,2)=5
const double kTrianglePoints[] = {0.0, 0.0, 3.0, 0.0, 0.0, 4.0};
constexpr int kTriangleN = 3;
constexpr int kTriangleDim = 2;

// 4 points in 2D: (0,0), (1,1), (2,0), (0,3)
const double kQuadPoints[] = {0.0, 0.0, 1.0, 1.0, 2.0, 0.0, 0.0, 3.0};
constexpr int kQuadN = 4;
constexpr int kQuadDim = 2;

inline bool run_distance_matrix_config(const double *h_points, int n, int dim, double max_radius,
                                       const CUDADistanceMatrixConfig &config,
                                       std::vector<double> &out_distances)
{
    out_distances.resize(static_cast<size_t>(n) * static_cast<size_t>(n));

    double *d_points = nullptr;
    double *d_distances = nullptr;
    size_t points_bytes = static_cast<size_t>(n) * static_cast<size_t>(dim) * sizeof(double);
    size_t dist_bytes = static_cast<size_t>(n) * static_cast<size_t>(n) * sizeof(double);

    if (cudaMalloc(&d_points, points_bytes) != cudaSuccess)
        return false;
    if (cudaMalloc(&d_distances, dist_bytes) != cudaSuccess)
    {
        cudaFree(d_points);
        return false;
    }
    cudaMemcpy(d_points, h_points, points_bytes, cudaMemcpyHostToDevice);

    auto result = cuda_host::launchDistanceMatrixKernel(d_points, d_distances,
                                                        static_cast<Size>(n),
                                                        static_cast<Size>(dim),
                                                        max_radius, config, 0, 0);
    if (result.isError())
    {
        cudaFree(d_points);
        cudaFree(d_distances);
        return false;
    }

    cudaMemcpy(out_distances.data(), d_distances, dist_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_points);
    cudaFree(d_distances);
    return true;
}

// Fill a 4-point symmetric distance matrix with 6 edge weights.
// Edges are (0,1), (0,2), (0,3), (1,2), (1,3), (2,3).
inline void fill_k4_distance_matrix(double *h_dist, int n,
                                    double w01, double w02, double w03,
                                    double w12, double w13, double w23)
{
    h_dist[0 * n + 1] = h_dist[1 * n + 0] = w01;
    h_dist[0 * n + 2] = h_dist[2 * n + 0] = w02;
    h_dist[0 * n + 3] = h_dist[3 * n + 0] = w03;
    h_dist[1 * n + 2] = h_dist[2 * n + 1] = w12;
    h_dist[1 * n + 3] = h_dist[3 * n + 1] = w13;
    h_dist[2 * n + 3] = h_dist[3 * n + 2] = w23;
}

// Fill a 5-point symmetric distance matrix with 10 edge weights.
inline void fill_k5_distance_matrix(double *h_dist, int n,
                                    double w01, double w02, double w03, double w04,
                                    double w12, double w13, double w14,
                                    double w23, double w24, double w34)
{
    h_dist[0 * n + 1] = h_dist[1 * n + 0] = w01;
    h_dist[0 * n + 2] = h_dist[2 * n + 0] = w02;
    h_dist[0 * n + 3] = h_dist[3 * n + 0] = w03;
    h_dist[0 * n + 4] = h_dist[4 * n + 0] = w04;
    h_dist[1 * n + 2] = h_dist[2 * n + 1] = w12;
    h_dist[1 * n + 3] = h_dist[3 * n + 1] = w13;
    h_dist[1 * n + 4] = h_dist[4 * n + 1] = w14;
    h_dist[2 * n + 3] = h_dist[3 * n + 2] = w23;
    h_dist[2 * n + 4] = h_dist[4 * n + 2] = w24;
    h_dist[3 * n + 4] = h_dist[4 * n + 3] = w34;
}

// Fill a 3-point symmetric distance matrix with 3 edge weights.
// Edges are (0,1), (0,2), (1,2).
inline void fill_k3_distance_matrix(double *h_dist, int n,
                                    double w01, double w02, double w12)
{
    h_dist[0 * n + 1] = h_dist[1 * n + 0] = w01;
    h_dist[0 * n + 2] = h_dist[2 * n + 0] = w02;
    h_dist[1 * n + 2] = h_dist[2 * n + 1] = w12;
}

// Fill a 6-point symmetric distance matrix with 15 edge weights.
// Edges are (0,1..5), (1,2..5), (2,3..5), (3,4..5), (4,5).
inline void fill_k6_distance_matrix(double *h_dist, int n,
                                    double w01, double w02, double w03, double w04, double w05,
                                    double w12, double w13, double w14, double w15,
                                    double w23, double w24, double w25,
                                    double w34, double w35,
                                    double w45)
{
    h_dist[0 * n + 1] = h_dist[1 * n + 0] = w01;
    h_dist[0 * n + 2] = h_dist[2 * n + 0] = w02;
    h_dist[0 * n + 3] = h_dist[3 * n + 0] = w03;
    h_dist[0 * n + 4] = h_dist[4 * n + 0] = w04;
    h_dist[0 * n + 5] = h_dist[5 * n + 0] = w05;
    h_dist[1 * n + 2] = h_dist[2 * n + 1] = w12;
    h_dist[1 * n + 3] = h_dist[3 * n + 1] = w13;
    h_dist[1 * n + 4] = h_dist[4 * n + 1] = w14;
    h_dist[1 * n + 5] = h_dist[5 * n + 1] = w15;
    h_dist[2 * n + 3] = h_dist[3 * n + 2] = w23;
    h_dist[2 * n + 4] = h_dist[4 * n + 2] = w24;
    h_dist[2 * n + 5] = h_dist[5 * n + 2] = w25;
    h_dist[3 * n + 4] = h_dist[4 * n + 3] = w34;
    h_dist[3 * n + 5] = h_dist[5 * n + 3] = w35;
    h_dist[4 * n + 5] = h_dist[5 * n + 4] = w45;
}

} // namespace

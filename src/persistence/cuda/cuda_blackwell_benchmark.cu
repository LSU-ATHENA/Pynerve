// Benchmark helpers for Hopper/Blackwell-oriented distance launchers.

#include "nerve/persistence/cuda/cuda_blackwell_tma.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <limits>
#include <vector>

namespace nerve
{
namespace gpu
{
namespace blackwell
{

namespace
{

BlackwellBenchmark emptyBenchmark()
{
    BlackwellBenchmark result{};
    result.ampere_time_ms = 0.0;
    result.blackwell_time_ms = 0.0;
    result.speedup = 1.0;
    return result;
}

bool validBenchmarkShape(const std::vector<double> &points, size_t point_dim, size_t num_points,
                         double max_radius)
{
    if (points.empty() || point_dim == 0 || num_points == 0 || !std::isfinite(max_radius) ||
        max_radius <= 0.0)
    {
        return false;
    }
    return num_points <= (std::numeric_limits<size_t>::max() / point_dim) &&
           points.size() >= num_points * point_dim &&
           num_points <= (std::numeric_limits<size_t>::max() / num_points);
}

} // namespace

BlackwellBenchmark benchmarkBlackwell(const std::vector<double> &points, size_t point_dim,
                                      size_t num_points, double max_radius)
{
    BlackwellBenchmark result = emptyBenchmark();
    if (!validBenchmarkShape(points, point_dim, num_points, max_radius))
    {
        return result;
    }

    double *d_points = nullptr;
    float *d_distances = nullptr;
    const size_t points_bytes = num_points * point_dim * sizeof(double);
    const size_t dist_bytes = num_points * num_points * sizeof(float);
    if (cudaMalloc(&d_points, points_bytes) != cudaSuccess ||
        cudaMalloc(&d_distances, dist_bytes) != cudaSuccess)
    {
        cudaFree(d_points);
        cudaFree(d_distances);
        return result;
    }

    if (cudaMemcpy(d_points, points.data(), points_bytes, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        cudaFree(d_points);
        cudaFree(d_distances);
        return result;
    }

    cudaEvent_t start{};
    cudaEvent_t stop{};
    if (cudaEventCreate(&start) != cudaSuccess || cudaEventCreate(&stop) != cudaSuccess)
    {
        if (start != nullptr)
        {
            cudaEventDestroy(start);
        }
        if (stop != nullptr)
        {
            cudaEventDestroy(stop);
        }
        cudaFree(d_points);
        cudaFree(d_distances);
        return result;
    }

    const auto launch_and_time = [&](auto launch_fn, double *out_ms) -> bool {
        if (cudaEventRecord(start) != cudaSuccess)
        {
            return false;
        }
        launch_fn();
        if (cudaEventRecord(stop) != cudaSuccess || cudaEventSynchronize(stop) != cudaSuccess)
        {
            return false;
        }
        float elapsed = 0.0f;
        if (cudaEventElapsedTime(&elapsed, start, stop) != cudaSuccess)
        {
            return false;
        }
        *out_ms = static_cast<double>(elapsed);
        return true;
    };

    const int n_points_i = static_cast<int>(num_points);
    const int point_dim_i = static_cast<int>(point_dim);
    const bool baseline_ok = launch_and_time(
        [&]() {
            launchPersistentDistanceMatrix(d_points, d_distances, n_points_i, point_dim_i,
                                           max_radius, 0);
        },
        &result.ampere_time_ms);
    const bool blackwell_ok =
        baseline_ok && launch_and_time(
                           [&]() {
                               launchBlackwellDistanceMatrix(d_points, d_distances, n_points_i,
                                                             point_dim_i, max_radius, 0);
                           },
                           &result.blackwell_time_ms);

    if (blackwell_ok && std::isfinite(result.blackwell_time_ms) &&
        std::isfinite(result.ampere_time_ms) && result.blackwell_time_ms > 0.0 &&
        result.ampere_time_ms > 0.0)
    {
        result.speedup = result.ampere_time_ms / result.blackwell_time_ms;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_points);
    cudaFree(d_distances);
    return result;
}

} // namespace blackwell
} // namespace gpu
} // namespace nerve

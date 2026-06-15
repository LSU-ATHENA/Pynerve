#include "nerve/persistence/cuda/cuda_tensor_core.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace nerve
{
namespace gpu
{
namespace tensorcore
{

__global__ void standardDistanceMatrixKernel(const double *points, float *distances, int n_points,
                                             int point_dim, double max_radius_sq);

namespace
{

constexpr int kBenchmarkBlockSize = 256;
constexpr float kNoEdge = std::numeric_limits<float>::infinity();

bool checkedSizeProduct(size_t lhs, size_t rhs, size_t &out)
{
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs)
    {
        return false;
    }
    out = lhs * rhs;
    return true;
}

bool checkedIntSquare(int value, int &out)
{
    if (value <= 0 || value > std::numeric_limits<int>::max() / value)
    {
        return false;
    }
    out = value * value;
    return true;
}

int ceilDivInt(int value, int divisor)
{
    return (value + divisor - 1) / divisor;
}

void freeDeviceBuffers(double *d_points, float *d_standard, float *d_tensor)
{
    if (d_points != nullptr)
    {
        cudaFree(d_points);
    }
    if (d_standard != nullptr)
    {
        cudaFree(d_standard);
    }
    if (d_tensor != nullptr)
    {
        cudaFree(d_tensor);
    }
}

void destroyEvents(cudaEvent_t start, cudaEvent_t stop)
{
    if (start != nullptr)
    {
        cudaEventDestroy(start);
    }
    if (stop != nullptr)
    {
        cudaEventDestroy(stop);
    }
}

bool recordElapsed(cudaEvent_t start, cudaEvent_t stop, double &elapsed_ms)
{
    if (cudaEventRecord(stop) != cudaSuccess || cudaEventSynchronize(stop) != cudaSuccess)
    {
        return false;
    }
    float elapsed = 0.0f;
    if (cudaEventElapsedTime(&elapsed, start, stop) != cudaSuccess)
    {
        return false;
    }
    elapsed_ms = static_cast<double>(elapsed);
    return elapsed_ms > 0.0;
}

bool measureAccuracy(const std::vector<float> &reference, const std::vector<float> &candidate,
                     TensorCoreBenchmark &result)
{
    double max_error = 0.0;
    double sum_error = 0.0;
    size_t compared = 0;
    for (size_t i = 0; i < reference.size(); ++i)
    {
        const float ref = reference[i];
        const float cand = candidate[i];
        if (std::isinf(ref) && std::isinf(cand))
        {
            continue;
        }
        if (!std::isfinite(ref) || !std::isfinite(cand))
        {
            result.max_relative_error = 1.0;
            result.mean_relative_error = 1.0;
            return true;
        }
        const double denominator = std::max(std::abs(static_cast<double>(ref)), 1.0e-9);
        const double relative = std::abs(static_cast<double>(cand) - ref) / denominator;
        max_error = std::max(max_error, relative);
        sum_error += relative;
        ++compared;
    }
    if (compared == 0)
    {
        return false;
    }
    result.max_relative_error = max_error;
    result.mean_relative_error = sum_error / static_cast<double>(compared);
    return true;
}

} // namespace

TensorCoreBenchmark benchmarkTensorCore(const std::vector<double> &points, size_t point_dim,
                                        size_t num_points, double max_radius)
{
    TensorCoreBenchmark result;
    result.num_points = num_points;
    result.point_dim = point_dim;

    if (point_dim == 0 || num_points == 0 || !std::isfinite(max_radius) || max_radius <= 0.0)
    {
        return result;
    }
    const double max_radius_sq = max_radius * max_radius;
    if (!std::isfinite(max_radius_sq))
    {
        return result;
    }
    if (num_points > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        point_dim > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        return result;
    }

    size_t point_elements = 0;
    size_t distance_elements = 0;
    size_t point_bytes = 0;
    size_t distance_bytes = 0;
    if (!checkedSizeProduct(num_points, point_dim, point_elements) ||
        points.size() != point_elements ||
        !checkedSizeProduct(num_points, num_points, distance_elements) ||
        !checkedSizeProduct(point_elements, sizeof(double), point_bytes) ||
        !checkedSizeProduct(distance_elements, sizeof(float), distance_bytes))
    {
        return result;
    }

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0)
    {
        return result;
    }

    double *d_points = nullptr;
    float *d_standard = nullptr;
    float *d_tensor = nullptr;
    if (cudaMalloc(&d_points, point_bytes) != cudaSuccess ||
        cudaMalloc(&d_standard, distance_bytes) != cudaSuccess ||
        cudaMalloc(&d_tensor, distance_bytes) != cudaSuccess)
    {
        freeDeviceBuffers(d_points, d_standard, d_tensor);
        return result;
    }
    if (cudaMemcpy(d_points, points.data(), point_bytes, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        freeDeviceBuffers(d_points, d_standard, d_tensor);
        return result;
    }

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    if (cudaEventCreate(&start) != cudaSuccess || cudaEventCreate(&stop) != cudaSuccess)
    {
        destroyEvents(start, stop);
        freeDeviceBuffers(d_points, d_standard, d_tensor);
        return result;
    }

    const int n_points_i = static_cast<int>(num_points);
    const int point_dim_i = static_cast<int>(point_dim);
    int total_i = 0;
    if (!checkedIntSquare(n_points_i, total_i))
    {
        destroyEvents(start, stop);
        freeDeviceBuffers(d_points, d_standard, d_tensor);
        return result;
    }

    const int standard_blocks = ceilDivInt(total_i, kBenchmarkBlockSize);
    if (cudaEventRecord(start) != cudaSuccess)
    {
        destroyEvents(start, stop);
        freeDeviceBuffers(d_points, d_standard, d_tensor);
        return result;
    }
    standardDistanceMatrixKernel<<<standard_blocks, kBenchmarkBlockSize>>>(
        d_points, d_standard, n_points_i, point_dim_i, max_radius_sq);
    if (cudaGetLastError() != cudaSuccess || !recordElapsed(start, stop, result.standard_time_ms))
    {
        destroyEvents(start, stop);
        freeDeviceBuffers(d_points, d_standard, d_tensor);
        return result;
    }

    if (cudaEventRecord(start) != cudaSuccess)
    {
        destroyEvents(start, stop);
        freeDeviceBuffers(d_points, d_standard, d_tensor);
        return result;
    }
    launchTensorCoreDistanceMatrix(d_points, d_tensor, n_points_i, point_dim_i, max_radius, 0);
    if (cudaGetLastError() != cudaSuccess ||
        !recordElapsed(start, stop, result.tensor_core_time_ms))
    {
        destroyEvents(start, stop);
        freeDeviceBuffers(d_points, d_standard, d_tensor);
        return result;
    }

    if (std::isfinite(result.standard_time_ms) && std::isfinite(result.tensor_core_time_ms) &&
        result.standard_time_ms >= 0.0 && result.tensor_core_time_ms > 0.0)
    {
        result.speedup = result.standard_time_ms / result.tensor_core_time_ms;
    }
    else
    {
        result.speedup = 1.0;
    }

    std::vector<float> host_standard(distance_elements, kNoEdge);
    std::vector<float> host_tensor(distance_elements, kNoEdge);
    if (cudaMemcpy(host_standard.data(), d_standard, distance_bytes, cudaMemcpyDeviceToHost) ==
            cudaSuccess &&
        cudaMemcpy(host_tensor.data(), d_tensor, distance_bytes, cudaMemcpyDeviceToHost) ==
            cudaSuccess)
    {
        measureAccuracy(host_standard, host_tensor, result);
    }

    destroyEvents(start, stop);
    freeDeviceBuffers(d_points, d_standard, d_tensor);
    return result;
}

} // namespace tensorcore
} // namespace gpu
} // namespace nerve

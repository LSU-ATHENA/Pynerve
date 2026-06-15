#include "nerve/gpu/cuda_error_check.hpp"
#include "nerve/gpu/device_array.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace nerve::gpu::persistence_image
{

namespace
{

inline constexpr int kBlockSize = 256;
inline constexpr int kTileSize = 16;

template <typename T>
__device__ __forceinline__ T device_exp(T x)
{
    return static_cast<T>(expf(static_cast<float>(x)));
}

template <typename T>
__device__ __forceinline__ T device_max(T a, T b)
{
    return a > b ? a : b;
}

template <typename T>
__device__ __forceinline__ T device_min(T a, T b)
{
    return a < b ? a : b;
}

template <typename T>
__global__ __launch_bounds__(256)
    persistence_image_pixel_kernel(const T *__restrict__ births, const T *__restrict__ deaths,
                                   int n_pairs, int resolution, T sigma, T *__restrict__ image,
                                   T x_min, T x_max, T y_min, T y_max)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= resolution || py >= resolution)
        return;

    T x_range = x_max - x_min;
    T y_range = y_max - y_min;
    if (x_range <= T{0})
        x_range = T{1};
    if (y_range <= T{0})
        y_range = T{1};

    T pixel_x = x_min + (static_cast<T>(px) + T{0.5}) * x_range / static_cast<T>(resolution);
    T pixel_y = y_min + (static_cast<T>(py) + T{0.5}) * y_range / static_cast<T>(resolution);

    T sigma_sq2 = T{2} * sigma * sigma;
    T neg_inv_sigma_sq2 = T{-1} / sigma_sq2;

    T weight_sum = T{0};
    for (int i = 0; i < n_pairs; ++i)
    {
        T birth = births[i];
        T death = deaths[i];
        T persistence = death - birth;

        T dx = pixel_x - birth;
        T dy = pixel_y - persistence;
        T dist2 = dx * dx + dy * dy;

        weight_sum += device_exp(neg_inv_sigma_sq2 * dist2);
    }

    image[py * resolution + px] += weight_sum;
}

template <typename T>
__global__ __launch_bounds__(256)
    persistence_image_pair_kernel(const T *__restrict__ births, const T *__restrict__ deaths,
                                  int n_pairs, int resolution, T sigma, T *__restrict__ image,
                                  T x_min, T x_max, T y_min, T y_max)
{
    __shared__ T shared_image[1024];

    int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_idx >= n_pairs)
        return;

    T birth = births[pair_idx];
    T death = deaths[pair_idx];
    T persistence = death - birth;

    T x_range = x_max - x_min;
    T y_range = y_max - y_min;
    if (x_range <= T{0})
        x_range = T{1};
    if (y_range <= T{0})
        y_range = T{1};

    T bx = (birth - x_min) / x_range * static_cast<T>(resolution - 1);
    T py = (persistence - y_min) / y_range * static_cast<T>(resolution - 1);

    T sigma_sq2 = T{2} * sigma * sigma;
    T neg_inv_sigma_sq2 = T{-1} / sigma_sq2;

    int x0 = max(0, static_cast<int>(bx - T{3} * sigma * resolution / x_range));
    int x1 = min(resolution - 1, static_cast<int>(bx + T{3} * sigma * resolution / x_range));
    int y0 = max(0, static_cast<int>(py - T{3} * sigma * resolution / y_range));
    int y1 = min(resolution - 1, static_cast<int>(py + T{3} * sigma * resolution / y_range));

    for (int x = x0; x <= x1; ++x)
    {
        for (int y = y0; y <= y1; ++y)
        {
            T pixel_x = x_min + (static_cast<T>(x) + T{0.5}) * x_range / static_cast<T>(resolution);
            T pixel_y = y_min + (static_cast<T>(y) + T{0.5}) * y_range / static_cast<T>(resolution);

            T dx = pixel_x - birth;
            T dy = pixel_y - persistence;
            T dist2 = dx * dx + dy * dy;
            T weight = device_exp(neg_inv_sigma_sq2 * dist2);

            atomicAdd(&image[y * resolution + x], weight);
        }
    }
}

template <typename T>
void launch_persistence_image_gpu_impl(const T *births, const T *deaths, int n_pairs,
                                       int resolution, T sigma, T *d_image, T x_min, T x_max,
                                       T y_min, T y_max, cudaStream_t stream)
{
    int total_pixels = resolution * resolution;

    if (n_pairs > total_pixels)
    {
        dim3 block(kTileSize, kTileSize);
        dim3 grid((resolution + kTileSize - 1) / kTileSize,
                  (resolution + kTileSize - 1) / kTileSize);
        persistence_image_pixel_kernel<T><<<grid, block, 0, stream>>>(
            births, deaths, n_pairs, resolution, sigma, d_image, x_min, x_max, y_min, y_max);
        cudaError_t launch_err = cudaGetLastError();
        if (launch_err != cudaSuccess)
        {
            throw std::runtime_error("Kernel launch failed: " +
                                     std::string(cudaGetErrorString(launch_err)));
        }
    }
    else
    {
        int block_size = kBlockSize;
        int grid_size = (n_pairs + block_size - 1) / block_size;
        persistence_image_pair_kernel<T><<<grid_size, block_size, 0, stream>>>(
            births, deaths, n_pairs, resolution, sigma, d_image, x_min, x_max, y_min, y_max);
        cudaError_t launch_err = cudaGetLastError();
        if (launch_err != cudaSuccess)
        {
            throw std::runtime_error("Kernel launch failed: " +
                                     std::string(cudaGetErrorString(launch_err)));
        }
    }
}

} // namespace

void compute_persistence_image_gpu(
    const std::vector<float> &births, const std::vector<float> &deaths, int resolution, float sigma,
    std::function<void(const std::vector<std::vector<double>> &)> callback)
{
    const int n_pairs = static_cast<int>(std::min(births.size(), deaths.size()));
    if (n_pairs == 0)
    {
        auto empty_img =
            std::vector<std::vector<double>>(resolution, std::vector<double>(resolution, 0.0));
        callback(empty_img);
        return;
    }

    float b_min = std::numeric_limits<float>::max();
    float b_max = std::numeric_limits<float>::lowest();
    float p_min = std::numeric_limits<float>::max();
    float p_max = std::numeric_limits<float>::lowest();

    for (int i = 0; i < n_pairs; ++i)
    {
        float b = births[i];
        float d = deaths[i];
        if (std::isfinite(d) && d > b)
        {
            float pers = d - b;
            b_min = std::min(b_min, b);
            b_max = std::max(b_max, b);
            p_min = std::min(p_min, pers);
            p_max = std::max(p_max, pers);
        }
    }

    if (b_max <= b_min)
    {
        b_max = b_min + 1.0f;
    }
    if (p_max <= p_min)
    {
        p_max = p_min + 1.0f;
    }

    float x_range = b_max - b_min;
    float y_range = p_max - p_min;
    float x_min = b_min - 0.1f * x_range;
    float x_max = b_max + 0.1f * x_range;
    float y_min = p_min - 0.1f * y_range;
    float y_max = p_max + 0.1f * y_range;

    DeviceArray<float> d_births(n_pairs);
    DeviceArray<float> d_deaths(n_pairs);
    int total_pixels = resolution * resolution;
    DeviceArray<float> d_image(total_pixels);

    std::vector<float> h_image_zero(total_pixels, 0.0f);
    d_image.copyFromHost(h_image_zero.data(), total_pixels);
    d_births.copyFromHost(births.data(), n_pairs);
    d_deaths.copyFromHost(deaths.data(), n_pairs);

    launch_persistence_image_gpu_impl<float>(d_births.get(), d_deaths.get(), n_pairs, resolution,
                                             sigma, d_image.get(), x_min, x_max, y_min, y_max, 0);
    cudaDeviceSynchronize();

    std::vector<float> h_image(total_pixels);
    d_image.copyToHost(h_image.data(), total_pixels);
    cudaDeviceSynchronize();

    std::vector<std::vector<double>> result(resolution, std::vector<double>(resolution, 0.0));
    for (int y = 0; y < resolution; ++y)
    {
        for (int x = 0; x < resolution; ++x)
        {
            result[y][x] = static_cast<double>(h_image[y * resolution + x]);
        }
    }

    callback(result);
}

} // namespace nerve::gpu::persistence_image

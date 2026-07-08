#include "nerve/gpu/cuda_error_check.hpp"
#include "nerve/gpu/device_array.hpp"
#include "nerve/gpu/gpu_ptx_ops.cuh"
#include "nerve/math/persistence_metrics/point2d.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace nerve::gpu::bottleneck
{
using namespace nerve::gpu::ptx;

namespace
{

inline constexpr int kBlockSize = 256;

template <typename T>
__device__ __forceinline__ T device_inf();

template <>
__device__ __forceinline__ float device_inf<float>()
{
    return __int_as_float(0x7f800000);
}

template <>
__device__ __forceinline__ double device_inf<double>()
{
    return __longlong_as_double(0x7ff0000000000000ULL);
}

template <typename T>
__device__ __forceinline__ T device_abs(T val)
{
    return val < T{0} ? -val : val;
}

template <typename T>
__device__ __forceinline__ T device_max(T a, T b)
{
    return a > b ? a : b;
}

template <typename T>
__device__ __forceinline__ T linf_distance(T x1, T y1, T x2, T y2)
{
    if constexpr (std::is_same_v<T, float>)
        return ptx::hwmax_f32(ptx::hwmax_f32(x1 - x2, x2 - x1), ptx::hwmax_f32(y1 - y2, y2 - y1));
    else
        return ptx::hwmax_f64(ptx::hwmax_f64(x1 - x2, x2 - x1), ptx::hwmax_f64(y1 - y2, y2 - y1));
}

template <typename T>
__global__ __launch_bounds__(256)
    void build_candidate_epsilon_kernel(const T *__restrict__ d1_x, const T *__restrict__ d1_y, int n1,
                                        const T *__restrict__ d2_x, const T *__restrict__ d2_y, int n2,
                                        T *__restrict__ candidates, int *__restrict__ candidate_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pairs = n1 * n2;
    int total_candidates = total_pairs + n1 + n2;

    if (idx >= total_candidates)
        return;

    T candidate = T{0};

    if (idx < total_pairs)
    {
        int i = idx / n2;
        int j = idx % n2;
        candidate = linf_distance(d1_x[i], d1_y[i], d2_x[j], d2_y[j]);
    }
    else if (idx < total_pairs + n1)
    {
        int i = idx - total_pairs;
        candidate = device_abs(d1_y[i] - d1_x[i]) / T{2};
    }
    else
    {
        int j = idx - total_pairs - n1;
        candidate = device_abs(d2_y[j] - d2_x[j]) / T{2};
    }

    candidates[idx] = candidate;
    atomicMax(candidate_count, 0);
    __threadfence();
}

template <typename T>
__global__ __launch_bounds__(256)
    void build_adjacency_kernel(const T *__restrict__ d1_x, const T *__restrict__ d1_y, int n1,
                                const T *__restrict__ d2_x, const T *__restrict__ d2_y, int n2,
                                T epsilon, int *__restrict__ adj_offsets, int *__restrict__ adj_data,
                                int max_degree)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n1)
        return;

    int out_pos = 0;
    for (int j = 0; j < n2 && out_pos < max_degree; ++j)
    {
        if (linf_distance(d1_x[i], d1_y[i], d2_x[j], d2_y[j]) <= epsilon)
        {
            adj_data[i * max_degree + out_pos] = j;
            ++out_pos;
        }
    }
    T diagonal_dist = device_abs(d1_y[i] - d1_x[i]) / T{2};
    if (diagonal_dist <= epsilon && out_pos < max_degree)
    {
        adj_data[i * max_degree + out_pos] = n2 + i;
        ++out_pos;
    }

    for (int j = 0; j < out_pos; ++j)
    {
        adj_data[i * max_degree + j] = adj_data[i * max_degree + j];
    }
    adj_offsets[i] = out_pos;
}

template <typename T>
__global__ __launch_bounds__(256)
    void greedy_matching_kernel(const T *__restrict__ d1_x, const T *__restrict__ d1_y, int n1,
                                const T *__restrict__ d2_x, const T *__restrict__ d2_y, int n2,
                                T epsilon, int *__restrict__ match_l, int *__restrict__ match_r,
                                int *__restrict__ matched_count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n1)
        return;

    T best_dist = device_inf<T>();
    int best_j = -1;

    for (int j = 0; j < n2; ++j)
    {
        if (atomicCAS(&match_r[j], -1, i) == -1)
        {
            T dist = linf_distance(d1_x[i], d1_y[i], d2_x[j], d2_y[j]);
            if (dist <= epsilon)
            {
                best_j = j;
                break;
            }
            match_r[j] = -1;
        }
    }

    if (best_j >= 0)
    {
        match_l[i] = best_j;
        match_r[best_j] = i;
        atomicAdd(matched_count, 2);
    }
    else
    {
        T diagonal_dist = device_abs(d1_y[i] - d1_x[i]) / T{2};
        if (diagonal_dist <= epsilon)
        {
            match_l[i] = n2 + i;
            atomicAdd(matched_count, 1);
        }
    }
}

template <typename T>
__global__ __launch_bounds__(256)
    void greedy_matching_kernel_d2(const T *__restrict__ d2_x, const T *__restrict__ d2_y, int n2,
                                   T epsilon, int *__restrict__ match_r, int *__restrict__ matched_count)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n2)
        return;

    if (match_r[j] == -1)
    {
        T diagonal_dist = device_abs(d2_y[j] - d2_x[j]) / T{2};
        if (diagonal_dist <= epsilon)
        {
            atomicAdd(matched_count, 1);
            match_r[j] = -2;
        }
    }
}

template <typename T>
__global__ __launch_bounds__(256)
    void check_convergence_kernel(int *__restrict__ matched_count, int *__restrict__ converged,
                                  int target_count)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        *converged = (*matched_count >= target_count) ? 1 : 0;
    }
}

} // namespace

void compute_bottleneck_distance_gpu(const std::vector<nerve::math::Point2D> &d1,
                                     const std::vector<nerve::math::Point2D> &d2,
                                     std::function<void(double)> callback)
{
    if (d1.empty() && d2.empty())
    {
        callback(0.0);
        return;
    }

    const int n1 = static_cast<int>(d1.size());
    const int n2 = static_cast<int>(d2.size());
    const int total_pairs = n1 * n2;
    const int total_candidates = total_pairs + n1 + n2;

    std::vector<double> h1_x(n1), h1_y(n1);
    std::vector<double> h2_x(n2), h2_y(n2);
    for (int i = 0; i < n1; ++i)
    {
        h1_x[i] = d1[i].x;
        h1_y[i] = d1[i].y;
    }
    for (int j = 0; j < n2; ++j)
    {
        h2_x[j] = d2[j].x;
        h2_y[j] = d2[j].y;
    }

    DeviceArray<double> dd1_x(n1), dd1_y(n1);
    DeviceArray<double> dd2_x(n2), dd2_y(n2);
    DeviceArray<double> d_candidates(total_candidates);
    DeviceArray<int> d_candidate_count(1);
    DeviceArray<int> d_match_l(n1);
    DeviceArray<int> d_match_r(n2);
    DeviceArray<int> d_matched_count(1);
    DeviceArray<int> d_converged(1);

    dd1_x.copyFromHost(h1_x.data(), n1);
    dd1_y.copyFromHost(h1_y.data(), n1);
    dd2_x.copyFromHost(h2_x.data(), n2);
    dd2_y.copyFromHost(h2_y.data(), n2);

    int h_zero = 0;
    cudaMemcpy(d_candidate_count.get(), &h_zero, sizeof(int), cudaMemcpyHostToDevice);

    int block = kBlockSize;
    int grid_cand = (total_candidates + block - 1) / block;
    build_candidate_epsilon_kernel<double>
        <<<grid_cand, block>>>(dd1_x.get(), dd1_y.get(), n1, dd2_x.get(), dd2_y.get(), n2,
                               d_candidates.get(), d_candidate_count.get());
    cudaDeviceSynchronize();
    {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            throw std::runtime_error("Kernel launch failed: " +
                                     std::string(cudaGetErrorString(err)));
        }
    }

    std::vector<double> h_candidates(total_candidates);
    d_candidates.copyToHost(h_candidates.data(), total_candidates);
    cudaDeviceSynchronize();

    std::sort(h_candidates.begin(), h_candidates.end());
    h_candidates.erase(std::unique(h_candidates.begin(), h_candidates.end(),
                                   [](double a, double b) { return std::abs(a - b) < 1e-15; }),
                       h_candidates.end());

    const int target_matched = n1 + n2;
    double answer = h_candidates.back();

    int lo = 0;
    int hi = static_cast<int>(h_candidates.size()) - 1;
    const int max_iters = 200;
    int iteration = 0;

    std::vector<double> sorted_candidates = h_candidates;

    if (!sorted_candidates.empty())
    {
        answer = sorted_candidates.back();
        lo = 0;
        hi = static_cast<int>(sorted_candidates.size()) - 1;

        while (lo <= hi && iteration < max_iters)
        {
            int mid = lo + (hi - lo) / 2;
            double epsilon = sorted_candidates[mid];

            int h_init_m1 = 0;
            cudaMemcpy(d_matched_count.get(), &h_init_m1, sizeof(int), cudaMemcpyHostToDevice);

            std::vector<int> h_match_l_init(n1, -1);
            std::vector<int> h_match_r_init(n2, -1);
            d_match_l.copyFromHost(h_match_l_init.data(), n1);
            d_match_r.copyFromHost(h_match_r_init.data(), n2);
            cudaDeviceSynchronize();

            int grid_n1 = (n1 + block - 1) / block;
            if (grid_n1 > 0)
            {
                greedy_matching_kernel<double><<<grid_n1, block>>>(
                    dd1_x.get(), dd1_y.get(), n1, dd2_x.get(), dd2_y.get(), n2, epsilon,
                    d_match_l.get(), d_match_r.get(), d_matched_count.get());
                cudaDeviceSynchronize();
                {
                    cudaError_t err = cudaGetLastError();
                    if (err != cudaSuccess)
                    {
                        throw std::runtime_error("Kernel launch failed: " +
                                                 std::string(cudaGetErrorString(err)));
                    }
                }
            }

            int grid_n2 = (n2 + block - 1) / block;
            if (grid_n2 > 0)
            {
                greedy_matching_kernel_d2<double><<<grid_n2, block>>>(
                    dd2_x.get(), dd2_y.get(), n2, epsilon, d_match_r.get(), d_matched_count.get());
                cudaDeviceSynchronize();
                {
                    cudaError_t err = cudaGetLastError();
                    if (err != cudaSuccess)
                    {
                        throw std::runtime_error("Kernel launch failed: " +
                                                 std::string(cudaGetErrorString(err)));
                    }
                }
            }

            check_convergence_kernel<double>
                <<<1, 1>>>(d_matched_count.get(), d_converged.get(), target_matched);
            cudaDeviceSynchronize();
            {
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess)
                {
                    throw std::runtime_error("Kernel launch failed: " +
                                             std::string(cudaGetErrorString(err)));
                }
            }

            int h_converged = 0;
            d_converged.copyToHost(&h_converged, 1);
            cudaDeviceSynchronize();

            if (h_converged)
            {
                answer = epsilon;
                hi = mid - 1;
            }
            else
            {
                lo = mid + 1;
            }
            ++iteration;
        }
    }

    callback(answer);
}

} // namespace nerve::gpu::bottleneck

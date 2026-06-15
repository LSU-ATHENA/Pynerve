#include "nerve/gpu/cuda_error_check.hpp"
#include "nerve/gpu/device_array.hpp"
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

namespace nerve::gpu::wasserstein
{

namespace
{

inline constexpr int kBlockSize = 256;
inline constexpr int kSinkhornBlock = 256;

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
__device__ __forceinline__ T device_abs(T a)
{
    return a < T{0} ? -a : a;
}

template <typename T>
__device__ __forceinline__ T device_max(T a, T b)
{
    return a > b ? a : b;
}

template <typename T>
__device__ __forceinline__ T device_pow(T base, T exp)
{
    if (base <= T{0})
        return T{0};
    return static_cast<T>(powf(static_cast<float>(base), static_cast<float>(exp)));
}

template <typename T>
__device__ __forceinline__ T device_exp(T x)
{
    return static_cast<T>(expf(static_cast<float>(x)));
}

template <typename T>
__device__ __forceinline__ T linf_distance(T x1, T y1, T x2, T y2)
{
    return device_max(device_abs(x1 - x2), device_abs(y1 - y2));
}

template <typename T>
__global__ __launch_bounds__(256)
    build_cost_matrix_kernel(const T *__restrict__ d1_x, const T *__restrict__ d1_y, int n1,
                             const T *__restrict__ d2_x, const T *__restrict__ d2_y, int n2, T p,
                             T *__restrict__ cost, int cost_ld)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n1 || col >= n2)
        return;

    T dist = linf_distance(d1_x[row], d1_y[row], d2_x[col], d2_y[col]);
    cost[row * cost_ld + col] = device_pow(dist, p);
}

template <typename T>
__global__ __launch_bounds__(256)
    build_sinkhorn_kernel_matrix(const T *__restrict__ d1_x, const T *__restrict__ d1_y, int n1,
                                 const T *__restrict__ d2_x, const T *__restrict__ d2_y, int n2,
                                 T lambda, T *__restrict__ K, int ld)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n1 || col >= n2)
        return;

    T dist = linf_distance(d1_x[row], d1_y[row], d2_x[col], d2_y[col]);
    K[row * ld + col] = device_exp(-lambda * dist);
}

template <typename T>
__global__ __launch_bounds__(256)
    sinkhorn_row_scale_kernel(T *__restrict__ K, T *__restrict__ u, const T *__restrict__ v,
                              int n_rows, int n_cols, int ld, T lambda)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows)
        return;

    T sum = T{0};
    for (int j = 0; j < n_cols; ++j)
    {
        sum += K[row * ld + j] * v[j];
    }

    if (sum > T{0})
    {
        T inv_sum = T{1} / sum;
        u[row] = inv_sum;
        for (int j = 0; j < n_cols; ++j)
        {
            K[row * ld + j] *= inv_sum;
        }
    }
}

template <typename T>
__global__ __launch_bounds__(256)
    sinkhorn_col_scale_kernel(T *__restrict__ K, const T *__restrict__ u, T *__restrict__ v,
                              int n_rows, int n_cols, int ld, T lambda,
                              const T *__restrict__ target_marginals)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n_cols)
        return;

    T sum = T{0};
    for (int i = 0; i < n_rows; ++i)
    {
        sum += K[i * ld + col] * u[i];
    }

    T target = target_marginals ? target_marginals[col] : (T{1} / static_cast<T>(n_cols));
    if (sum > T{0})
    {
        T scale = target / sum;
        v[col] *= scale;
        for (int i = 0; i < n_rows; ++i)
        {
            K[i * ld + col] *= scale;
        }
    }
}

template <typename T>
__global__ __launch_bounds__(256)
    sinkhorn_cost_kernel(const T *__restrict__ K, const T *__restrict__ u, const T *__restrict__ v,
                         const T *__restrict__ cost, T *__restrict__ total_cost, int n_rows,
                         int n_cols, int cost_ld)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows)
        return;

    __shared__ T shared_sum;

    if (threadIdx.x == 0)
        shared_sum = T{0};
    __syncthreads();

    T local_sum = T{0};
    for (int j = 0; j < n_cols; ++j)
    {
        T transport = K[row * cost_ld + j] * u[row] * v[j];
        local_sum += transport * cost[row * cost_ld + j];
    }

    atomicAdd(&shared_sum, local_sum);
    __syncthreads();

    if (threadIdx.x == 0)
    {
        atomicAdd(total_cost, shared_sum);
    }
}

template <typename T>
__global__ __launch_bounds__(256)
    auction_init_bids_kernel(const T *__restrict__ d1_x, const T *__restrict__ d1_y, int n1,
                             const T *__restrict__ d2_x, const T *__restrict__ d2_y, int n2, T p,
                             T *__restrict__ cost, int cost_ld)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n1 || col >= n2)
        return;

    T dist = linf_distance(d1_x[row], d1_y[row], d2_x[col], d2_y[col]);
    cost[row * cost_ld + col] = device_pow(dist, p);
}

template <typename T>
__global__ __launch_bounds__(256)
    auction_bidding_kernel(const T *__restrict__ cost, int n_rows, int n_cols, int cost_ld,
                           T *__restrict__ prices, int *__restrict__ assignments,
                           int *__restrict__ converged, T epsilon)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rows)
        return;

    T best_val = device_inf<T>();
    T second_best = device_inf<T>();
    int best_j = -1;

    for (int j = 0; j < n_cols; ++j)
    {
        T net_val = -(cost[i * cost_ld + j] - prices[j]);
        if (net_val > best_val)
        {
            second_best = best_val;
            best_val = net_val;
            best_j = j;
        }
        else if (net_val > second_best)
        {
            second_best = net_val;
        }
    }

    if (best_j >= 0)
    {
        T bid = cost[i * cost_ld + best_j] + epsilon + second_best - best_val;
        atomicMin((int *)&prices[best_j], __float_as_int(static_cast<float>(bid)));
        *converged = 0;
    }
}

template <typename T>
__global__ __launch_bounds__(256)
    auction_assign_kernel(int *__restrict__ assignments, const T *__restrict__ prices,
                          const T *__restrict__ cost, int n_rows, int n_cols, int cost_ld,
                          T *__restrict__ final_assignments)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rows)
        return;

    T best_cost = device_inf<T>();
    int best_j = 0;

    for (int j = 0; j < n_cols; ++j)
    {
        T delta = device_abs(cost[i * cost_ld + j] - prices[j]);
        if (delta < best_cost)
        {
            best_cost = delta;
            best_j = j;
        }
    }

    final_assignments[i] = static_cast<T>(best_j);
}

} // namespace

void compute_sinkhorn_distance_gpu(const std::vector<nerve::math::Point2D> &d1,
                                   const std::vector<nerve::math::Point2D> &d2, double p,
                                   double epsilon_reg, int max_iterations,
                                   std::function<void(double)> callback)
{
    if (d1.empty() || d2.empty())
    {
        callback(0.0);
        return;
    }

    const int n1 = static_cast<int>(d1.size());
    const int n2 = static_cast<int>(d2.size());
    const int ld = n2;

    std::vector<float> h1_x(n1), h1_y(n1);
    std::vector<float> h2_x(n2), h2_y(n2);
    for (int i = 0; i < n1; ++i)
    {
        h1_x[i] = static_cast<float>(d1[i].x);
        h1_y[i] = static_cast<float>(d1[i].y);
    }
    for (int j = 0; j < n2; ++j)
    {
        h2_x[j] = static_cast<float>(d2[j].x);
        h2_y[j] = static_cast<float>(d2[j].y);
    }

    DeviceArray<float> dd1_x(n1), dd1_y(n1);
    DeviceArray<float> dd2_x(n2), dd2_y(n2);
    DeviceArray<float> d_K(n1 * n2);
    DeviceArray<float> d_cost(n1 * n2);
    DeviceArray<float> d_u(n1);
    DeviceArray<float> d_v(n2);
    DeviceArray<float> d_marginals(n2);
    DeviceArray<float> d_total_cost(1);

    dd1_x.copyFromHost(h1_x.data(), n1);
    dd1_y.copyFromHost(h1_y.data(), n1);
    dd2_x.copyFromHost(h2_x.data(), n2);
    dd2_y.copyFromHost(h2_y.data(), n2);

    std::vector<float> h_u_init(n1, 1.0f);
    std::vector<float> h_v_init(n2, 1.0f);
    d_u.copyFromHost(h_u_init.data(), n1);
    d_v.copyFromHost(h_v_init.data(), n2);

    float f_zero = 0.0f;
    cudaMemcpy(d_total_cost.get(), &f_zero, sizeof(float), cudaMemcpyHostToDevice);

    dim3 block_tile(16, 16);
    dim3 grid_cost((n2 + 15) / 16, (n1 + 15) / 16);

    build_cost_matrix_kernel<float>
        <<<grid_cost, block_tile>>>(dd1_x.get(), dd1_y.get(), n1, dd2_x.get(), dd2_y.get(), n2,
                                    static_cast<float>(p), d_cost.get(), ld);
    cudaDeviceSynchronize();
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess)
    {
        throw std::runtime_error("Kernel launch failed: " +
                                 std::string(cudaGetErrorString(launch_err)));
    }

    float lambda = 1.0f / static_cast<float>(epsilon_reg);
    build_sinkhorn_kernel_matrix<float><<<grid_cost, block_tile>>>(
        dd1_x.get(), dd1_y.get(), n1, dd2_x.get(), dd2_y.get(), n2, lambda, d_K.get(), ld);
    cudaDeviceSynchronize();
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess)
    {
        throw std::runtime_error("Kernel launch failed: " +
                                 std::string(cudaGetErrorString(launch_err)));
    }

    std::vector<float> h_marginals(n2, 1.0f / static_cast<float>(n2));
    d_marginals.copyFromHost(h_marginals.data(), n2);

    int grid_rows = (n1 + kSinkhornBlock - 1) / kSinkhornBlock;
    int grid_cols = (n2 + kSinkhornBlock - 1) / kSinkhornBlock;

    for (int iter = 0; iter < max_iterations; ++iter)
    {
        sinkhorn_row_scale_kernel<float>
            <<<grid_rows, kSinkhornBlock>>>(d_K.get(), d_u.get(), d_v.get(), n1, n2, ld, lambda);
        cudaDeviceSynchronize();
        cudaError_t launch_err = cudaGetLastError();
        if (launch_err != cudaSuccess)
        {
            throw std::runtime_error("Kernel launch failed: " +
                                     std::string(cudaGetErrorString(launch_err)));
        }

        sinkhorn_col_scale_kernel<float><<<grid_cols, kSinkhornBlock>>>(
            d_K.get(), d_u.get(), d_v.get(), n1, n2, ld, lambda, d_marginals.get());
        cudaDeviceSynchronize();
        cudaError_t launch_err = cudaGetLastError();
        if (launch_err != cudaSuccess)
        {
            throw std::runtime_error("Kernel launch failed: " +
                                     std::string(cudaGetErrorString(launch_err)));
        }
    }

    sinkhorn_cost_kernel<float><<<grid_rows, kSinkhornBlock>>>(
        d_K.get(), d_u.get(), d_v.get(), d_cost.get(), d_total_cost.get(), n1, n2, ld);
    cudaDeviceSynchronize();
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess)
    {
        throw std::runtime_error("Kernel launch failed: " +
                                 std::string(cudaGetErrorString(launch_err)));
    }

    float h_total_cost = 0.0f;
    d_total_cost.copyToHost(&h_total_cost, 1);
    cudaDeviceSynchronize();

    double dist = std::pow(std::max(0.0, static_cast<double>(h_total_cost)), 1.0 / p);
    callback(dist);
}

void compute_auction_distance_gpu(const std::vector<nerve::math::Point2D> &d1,
                                  const std::vector<nerve::math::Point2D> &d2, double p,
                                  std::function<void(double)> callback)
{
    if (d1.empty() || d2.empty())
    {
        callback(0.0);
        return;
    }

    const int n1 = static_cast<int>(d1.size());
    const int n2 = static_cast<int>(d2.size());
    const int cost_ld = n2;
    const int max_n = std::max(n1, n2);

    std::vector<float> h1_x(n1), h1_y(n1);
    std::vector<float> h2_x(n2), h2_y(n2);
    for (int i = 0; i < n1; ++i)
    {
        h1_x[i] = static_cast<float>(d1[i].x);
        h1_y[i] = static_cast<float>(d1[i].y);
    }
    for (int j = 0; j < n2; ++j)
    {
        h2_x[j] = static_cast<float>(d2[j].x);
        h2_y[j] = static_cast<float>(d2[j].y);
    }

    DeviceArray<float> dd1_x(n1), dd1_y(n1);
    DeviceArray<float> dd2_x(n2), dd2_y(n2);
    DeviceArray<float> d_cost(n1 * n2);
    DeviceArray<float> d_prices(n2);
    DeviceArray<int> d_assignments(n1);
    DeviceArray<int> d_converged(1);

    dd1_x.copyFromHost(h1_x.data(), n1);
    dd1_y.copyFromHost(h1_y.data(), n1);
    dd2_x.copyFromHost(h2_x.data(), n2);
    dd2_y.copyFromHost(h2_y.data(), n2);

    std::vector<float> h_prices_init(n2, 0.0f);
    std::vector<int> h_assignments_init(n1, -1);
    d_prices.copyFromHost(h_prices_init.data(), n2);
    d_assignments.copyFromHost(h_assignments_init.data(), n1);

    dim3 block_tile(16, 16);
    dim3 grid_cost((n2 + 15) / 16, (n1 + 15) / 16);

    auction_init_bids_kernel<float>
        <<<grid_cost, block_tile>>>(dd1_x.get(), dd1_y.get(), n1, dd2_x.get(), dd2_y.get(), n2,
                                    static_cast<float>(p), d_cost.get(), cost_ld);
    cudaDeviceSynchronize();
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess)
    {
        throw std::runtime_error("Kernel launch failed: " +
                                 std::string(cudaGetErrorString(launch_err)));
    }

    int grid_n1 = (n1 + kBlockSize - 1) / kBlockSize;
    float epsilon_auction = 0.1f;
    const int max_iter = 500;

    for (int iter = 0; iter < max_iter; ++iter)
    {
        int h_one = 1;
        cudaMemcpy(d_converged.get(), &h_one, sizeof(int), cudaMemcpyHostToDevice);

        auction_bidding_kernel<float><<<grid_n1, kBlockSize>>>(d_cost.get(), n1, n2, cost_ld,
                                                               d_prices.get(), d_assignments.get(),
                                                               d_converged.get(), epsilon_auction);
        cudaDeviceSynchronize();
        cudaError_t launch_err = cudaGetLastError();
        if (launch_err != cudaSuccess)
        {
            throw std::runtime_error("Kernel launch failed: " +
                                     std::string(cudaGetErrorString(launch_err)));
        }

        int h_converged = 0;
        d_converged.copyToHost(&h_converged, 1);
        cudaDeviceSynchronize();

        if (h_converged)
            break;

        epsilon_auction *= 0.5f;
    }

    std::vector<int> h_assignments(n1);
    d_assignments.copyToHost(h_assignments.data(), n1);
    cudaDeviceSynchronize();

    double total_cost = 0.0;
    for (int i = 0; i < n1; ++i)
    {
        int j = h_assignments[i];
        if (j >= 0 && j < n2)
        {
            double dist = d1[i].distanceTo(d2[j]);
            total_cost += std::pow(dist, p);
        }
        else
        {
            double diag = d1[i].diagonalDistance();
            total_cost += std::pow(diag, p);
        }
    }

    double distance = std::pow(total_cost, 1.0 / p);
    callback(distance);
}

} // namespace nerve::gpu::wasserstein

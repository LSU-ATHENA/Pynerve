#include "nerve/gpu/cuda_error_check.hpp"
#include "nerve/gpu/device_array.hpp"

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

namespace nerve::gpu::mapper
{

namespace
{

inline constexpr int kBlockSize = 256;
inline constexpr int kMaxCentroids = 256;
inline constexpr int kMaxClusters = 1024;

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
__device__ __forceinline__ T device_sqrt(T x)
{
    return static_cast<T>(sqrtf(static_cast<float>(x)));
}

template <typename T>
__device__ __forceinline__ T device_pow(T base, int exp)
{
    T result = T{1};
    for (int i = 0; i < exp; ++i)
    {
        result *= base;
    }
    return result;
}

template <typename T>
__global__ void __launch_bounds__(256)
    compute_density_filter_kernel(const T *__restrict__ points, int n_points, int dim,
                                  int k_neighbors, T *__restrict__ densities)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_points)
        return;

    T total_dist = T{0};
    int count = 0;

    for (int j = 0; j < n_points; ++j)
    {
        if (i == j)
            continue;
        T dist_sq = T{0};
        for (int d = 0; d < dim; ++d)
        {
            T diff = points[i * dim + d] - points[j * dim + d];
            dist_sq += diff * diff;
        }
        total_dist += device_sqrt(dist_sq);
        ++count;
        if (count >= k_neighbors * 2)
            break;
    }

    if (count > 0)
    {
        densities[i] = static_cast<T>(count) / (total_dist + T{1e-9});
    }
    else
    {
        densities[i] = T{0};
    }
}

template <typename T>
__global__ void __launch_bounds__(256)
    compute_eccentricity_filter_kernel(const T *__restrict__ points, int n_points, int dim,
                                       T *__restrict__ eccentricities)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_points)
        return;

    T max_dist_sq = T{0};
    for (int j = 0; j < n_points; ++j)
    {
        if (i == j)
            continue;
        T dist_sq = T{0};
        for (int d = 0; d < dim; ++d)
        {
            T diff = points[i * dim + d] - points[j * dim + d];
            dist_sq += diff * diff;
        }
        if (dist_sq > max_dist_sq)
        {
            max_dist_sq = dist_sq;
        }
    }
    eccentricities[i] = device_sqrt(max_dist_sq);
}

template <typename T>
__global__ void __launch_bounds__(1)
    kmeans_plusplus_init_kernel(const T *__restrict__ points, int n_points, int dim,
                                T *__restrict__ centroids, int k, T *__restrict__ min_distances,
                                uint64_t seed)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;

    uint64_t state = seed + 1;
    auto xorshift = [&state]() -> uint64_t {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return state;
    };

    int first_idx = static_cast<int>(xorshift() % static_cast<uint64_t>(n_points));
    for (int d = 0; d < dim; ++d)
    {
        centroids[d] = points[first_idx * dim + d];
    }

    for (int i = 0; i < n_points; ++i)
    {
        T dist_sq = T{0};
        for (int d = 0; d < dim; ++d)
        {
            T diff = points[i * dim + d] - centroids[d];
            dist_sq += diff * diff;
        }
        min_distances[i] = dist_sq;
    }

    for (int c = 1; c < k; ++c)
    {
        T total_weight = T{0};
        for (int i = 0; i < n_points; ++i)
        {
            total_weight += min_distances[i];
        }

        if (total_weight <= T{0})
            break;

        T r = static_cast<T>(xorshift() % 1000000ULL) / T{1000000} * total_weight;
        T running_sum = T{0};
        int selected = 0;
        for (int i = 0; i < n_points; ++i)
        {
            running_sum += min_distances[i];
            if (running_sum >= r)
            {
                selected = i;
                break;
            }
        }

        for (int d = 0; d < dim; ++d)
        {
            centroids[c * dim + d] = points[selected * dim + d];
        }

        for (int i = 0; i < n_points; ++i)
        {
            T dist_sq = T{0};
            for (int d = 0; d < dim; ++d)
            {
                T diff = points[i * dim + d] - centroids[c * dim + d];
                dist_sq += diff * diff;
            }
            if (dist_sq < min_distances[i])
            {
                min_distances[i] = dist_sq;
            }
        }
    }
}

template <typename T>
__global__ void __launch_bounds__(256)
    kmeans_assign_kernel(const T *__restrict__ points, int n_points, int dim,
                         const T *__restrict__ centroids, int k, int *__restrict__ labels,
                         T *__restrict__ cluster_sums, int *__restrict__ cluster_counts)
{
    __shared__ T shared_centroids[4096];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_points)
        return;

    for (int t = threadIdx.x; t < k * dim; t += blockDim.x)
    {
        shared_centroids[t] = centroids[t];
    }
    __syncthreads();

    int best_cluster = 0;
    T best_dist_sq = std::numeric_limits<T>::max();

    for (int c = 0; c < k; ++c)
    {
        T dist_sq = T{0};
        for (int d = 0; d < dim; ++d)
        {
            T diff = points[i * dim + d] - shared_centroids[c * dim + d];
            dist_sq += diff * diff;
        }
        if (dist_sq < best_dist_sq)
        {
            best_dist_sq = dist_sq;
            best_cluster = c;
        }
    }

    labels[i] = best_cluster;
    atomicAdd(&cluster_counts[best_cluster], 1);

    for (int d = 0; d < dim; ++d)
    {
        atomicAdd(&cluster_sums[best_cluster * dim + d], points[i * dim + d]);
    }
}

template <typename T>
__global__ void __launch_bounds__(256)
    kmeans_update_kernel(T *__restrict__ centroids, const T *__restrict__ cluster_sums,
                         const int *__restrict__ cluster_counts, int k, int dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = k * dim;
    if (idx >= total)
        return;

    int c = idx / dim;
    int count = cluster_counts[c];
    if (count > 0)
    {
        centroids[idx] = cluster_sums[idx] / static_cast<T>(count);
    }
}

template <typename T>
__global__ void __launch_bounds__(256)
    build_cover_kernel(const T *__restrict__ filter_values, int n_points, int n_filter_dims,
                       int resolution, T overlap, int *__restrict__ cover_sizes,
                       int *__restrict__ cover_indices, int max_cover_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_points)
        return;

    T interval = (T{1} + T{2} * overlap) / static_cast<T>(resolution);
    int n_intervals = resolution;

    int write_pos = 0;
    int total_sets = static_cast<int>(device_pow(static_cast<T>(n_intervals), n_filter_dims));

    for (int s = 0; s < total_sets && write_pos < max_cover_size; ++s)
    {
        bool in_cover = true;
        int remaining = s;
        for (int d = 0; d < n_filter_dims; ++d)
        {
            int bucket = remaining % n_intervals;
            remaining /= n_intervals;

            T center = (static_cast<T>(bucket) + T{0.5}) * interval - overlap;
            T half_width = interval / T{2};

            T f_val = filter_values[i * n_filter_dims + d];
            if (f_val < center - half_width || f_val > center + half_width)
            {
                in_cover = false;
                break;
            }
        }
        if (in_cover)
        {
            int pos = i * max_cover_size + write_pos;
            if (pos < n_points * max_cover_size)
            {
                cover_indices[pos] = s;
            }
            ++write_pos;
        }
    }
    cover_sizes[i] = write_pos;
}

template <typename T>
__global__ void __launch_bounds__(256)
    compute_nerve_edges_kernel(const int *__restrict__ node_cover_sets,
                               int *__restrict__ node_cover_starts,
                               const int *__restrict__ node_cover_sizes, int n_nodes,
                               int *__restrict__ edge_src, int *__restrict__ edge_dst,
                               int *__restrict__ edge_count, int max_edges)
{
    int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_pairs = n_nodes * (n_nodes - 1) / 2;

    if (pair_idx >= n_pairs)
        return;

    int i = 0;
    int j = 1;
    int remaining = pair_idx;
    while (remaining >= (n_nodes - 1 - i))
    {
        remaining -= (n_nodes - 1 - i);
        ++i;
    }
    j = i + 1 + remaining;

    int start_i = node_cover_starts[i];
    int size_i = node_cover_sizes[i];
    int start_j = node_cover_starts[j];
    int size_j = node_cover_sizes[j];

    for (int si = 0; si < size_i; ++si)
    {
        for (int sj = 0; sj < size_j; ++sj)
        {
            if (node_cover_sets[start_i + si] == node_cover_sets[start_j + sj])
            {
                int pos = atomicAdd(edge_count, 1);
                if (pos < max_edges)
                {
                    edge_src[pos] = i;
                    edge_dst[pos] = j;
                }
                return;
            }
        }
    }
}

} // namespace

void compute_density_filter_gpu(const std::vector<float> &points, int n_points, int dim,
                                int k_neighbors, std::function<void(std::vector<float>)> callback)
{
    if (n_points == 0)
    {
        callback({});
        return;
    }

    DeviceArray<float> d_points(points.size());
    DeviceArray<float> d_densities(n_points);

    d_points.copyFromHost(points.data(), points.size());

    int grid = (n_points + kBlockSize - 1) / kBlockSize;
    compute_density_filter_kernel<float>
        <<<grid, kBlockSize>>>(d_points.get(), n_points, dim, k_neighbors, d_densities.get());
    cudaDeviceSynchronize();
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess)
    {
        throw std::runtime_error("Kernel launch failed: " +
                                 std::string(cudaGetErrorString(launch_err)));
    }

    std::vector<float> h_densities(n_points);
    d_densities.copyToHost(h_densities.data(), n_points);
    cudaDeviceSynchronize();

    callback(h_densities);
}

void compute_eccentricity_filter_gpu(const std::vector<float> &points, int n_points, int dim,
                                     std::function<void(std::vector<float>)> callback)
{
    if (n_points == 0)
    {
        callback({});
        return;
    }

    DeviceArray<float> d_points(points.size());
    DeviceArray<float> d_eccentricities(n_points);

    d_points.copyFromHost(points.data(), points.size());

    int grid = (n_points + kBlockSize - 1) / kBlockSize;
    compute_eccentricity_filter_kernel<float>
        <<<grid, kBlockSize>>>(d_points.get(), n_points, dim, d_eccentricities.get());
    cudaDeviceSynchronize();
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess)
    {
        throw std::runtime_error("Kernel launch failed: " +
                                 std::string(cudaGetErrorString(launch_err)));
    }

    std::vector<float> h_eccentricities(n_points);
    d_eccentricities.copyToHost(h_eccentricities.data(), n_points);
    cudaDeviceSynchronize();

    callback(h_eccentricities);
}

void compute_kmeans_clustering_gpu(const std::vector<float> &points, int n_points, int dim, int k,
                                   int max_iterations,
                                   std::function<void(std::vector<int>)> callback)
{
    if (n_points == 0)
    {
        callback({});
        return;
    }

    DeviceArray<float> d_points(points.size());
    DeviceArray<float> d_centroids(k * dim);
    DeviceArray<float> d_min_distances(n_points);
    DeviceArray<int> d_labels(n_points);
    DeviceArray<float> d_cluster_sums(k * dim);
    DeviceArray<int> d_cluster_counts(k);

    d_points.copyFromHost(points.data(), points.size());

    std::vector<float> h_zero_sums(k * dim, 0.0f);
    std::vector<int> h_zero_counts(k, 0);

    kmeans_plusplus_init_kernel<float><<<1, 1>>>(d_points.get(), n_points, dim, d_centroids.get(),
                                                 k, d_min_distances.get(), 123456789ULL);
    cudaDeviceSynchronize();
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess)
    {
        throw std::runtime_error("Kernel launch failed: " +
                                 std::string(cudaGetErrorString(launch_err)));
    }

    int grid = (n_points + kBlockSize - 1) / kBlockSize;
    int grid_cent = (k * dim + kBlockSize - 1) / kBlockSize;

    for (int iter = 0; iter < max_iterations; ++iter)
    {
        d_cluster_sums.copyFromHost(h_zero_sums.data(), k * dim);
        d_cluster_counts.copyFromHost(h_zero_counts.data(), k);
        cudaDeviceSynchronize();

        kmeans_assign_kernel<float>
            <<<grid, kBlockSize>>>(d_points.get(), n_points, dim, d_centroids.get(), k,
                                   d_labels.get(), d_cluster_sums.get(), d_cluster_counts.get());
        cudaDeviceSynchronize();
        cudaError_t launch_err = cudaGetLastError();
        if (launch_err != cudaSuccess)
        {
            throw std::runtime_error("Kernel launch failed: " +
                                     std::string(cudaGetErrorString(launch_err)));
        }

        kmeans_update_kernel<float><<<grid_cent, kBlockSize>>>(
            d_centroids.get(), d_cluster_sums.get(), d_cluster_counts.get(), k, dim);
        cudaDeviceSynchronize();
        cudaError_t launch_err = cudaGetLastError();
        if (launch_err != cudaSuccess)
        {
            throw std::runtime_error("Kernel launch failed: " +
                                     std::string(cudaGetErrorString(launch_err)));
        }
    }

    d_cluster_sums.copyFromHost(h_zero_sums.data(), k * dim);
    d_cluster_counts.copyFromHost(h_zero_counts.data(), k);
    cudaDeviceSynchronize();

    kmeans_assign_kernel<float><<<grid, kBlockSize>>>(d_points.get(), n_points, dim,
                                                      d_centroids.get(), k, d_labels.get(),
                                                      d_cluster_sums.get(), d_cluster_counts.get());
    cudaDeviceSynchronize();
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess)
    {
        throw std::runtime_error("Kernel launch failed: " +
                                 std::string(cudaGetErrorString(launch_err)));
    }

    std::vector<int> h_labels(n_points);
    d_labels.copyToHost(h_labels.data(), n_points);
    cudaDeviceSynchronize();

    callback(h_labels);
}

void compute_nerve_graph_gpu(const std::vector<std::vector<int>> &nodes_cover_sets,
                             std::function<void(std::vector<std::pair<int, int>>)> callback)
{
    int n_nodes = static_cast<int>(nodes_cover_sets.size());
    if (n_nodes == 0)
    {
        callback({});
        return;
    }

    std::vector<int> h_node_cover_sizes(n_nodes);
    std::vector<int> h_node_cover_starts(n_nodes);
    std::vector<int> h_node_cover_sets;

    int offset = 0;
    for (int i = 0; i < n_nodes; ++i)
    {
        h_node_cover_starts[i] = offset;
        h_node_cover_sizes[i] = static_cast<int>(nodes_cover_sets[i].size());
        for (int cover_val : nodes_cover_sets[i])
        {
            h_node_cover_sets.push_back(cover_val);
        }
        offset += static_cast<int>(nodes_cover_sets[i].size());
    }

    DeviceArray<int> d_cover_sets(h_node_cover_sets.size());
    DeviceArray<int> d_cover_starts(n_nodes);
    DeviceArray<int> d_cover_sizes(n_nodes);

    d_cover_sets.copyFromHost(h_node_cover_sets.data(), h_node_cover_sets.size());
    d_cover_starts.copyFromHost(h_node_cover_starts.data(), n_nodes);
    d_cover_sizes.copyFromHost(h_node_cover_sizes.data(), n_nodes);

    int n_pairs = n_nodes * (n_nodes - 1) / 2;
    int max_edges = n_pairs;
    DeviceArray<int> d_edge_src(max_edges);
    DeviceArray<int> d_edge_dst(max_edges);
    DeviceArray<int> d_edge_count(1);

    int h_zero = 0;
    cudaMemcpy(d_edge_count.get(), &h_zero, sizeof(int), cudaMemcpyHostToDevice);

    int grid = (n_pairs + kBlockSize - 1) / kBlockSize;
    compute_nerve_edges_kernel<float><<<grid, kBlockSize>>>(
        d_cover_sets.get(), d_cover_starts.get(), d_cover_sizes.get(), n_nodes, d_edge_src.get(),
        d_edge_dst.get(), d_edge_count.get(), max_edges);
    cudaDeviceSynchronize();
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess)
    {
        throw std::runtime_error("Kernel launch failed: " +
                                 std::string(cudaGetErrorString(launch_err)));
    }

    int h_edge_count = 0;
    d_edge_count.copyToHost(&h_edge_count, 1);
    cudaDeviceSynchronize();

    std::vector<int> h_edge_src(h_edge_count);
    std::vector<int> h_edge_dst(h_edge_count);
    d_edge_src.copyToHost(h_edge_src.data(), h_edge_count);
    d_edge_dst.copyToHost(h_edge_dst.data(), h_edge_count);
    cudaDeviceSynchronize();

    std::vector<std::pair<int, int>> edges;
    edges.reserve(h_edge_count);
    for (int e = 0; e < h_edge_count; ++e)
    {
        edges.emplace_back(h_edge_src[e], h_edge_dst[e]);
    }

    callback(edges);
}

} // namespace nerve::gpu::mapper

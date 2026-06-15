#include "cuda/cuda_error.hpp"
#include "nerve/config.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <queue>
#include <vector>

#if defined(NERVE_HAS_CUDA)
#include <cuda_runtime.h>
#endif

#include "nerve/graphs/graph.hpp"

namespace nerve::graphs
{

#if defined(NERVE_HAS_CUDA)

struct MultiGpuGraphContext
{
    int num_gpus;
    std::vector<int> device_ids;
    std::vector<cudaStream_t> streams;
    std::vector<int *> d_adjacency_ptrs;
    std::vector<float *> d_weights_ptrs;
};

static cudaError_t setupMultiGpu(MultiGpuGraphContext &ctx)
{
    cudaGetDeviceCount(&ctx.num_gpus);
    if (ctx.num_gpus < 1)
        return cudaErrorNoDevice;
    ctx.device_ids.resize(ctx.num_gpus);
    ctx.streams.resize(ctx.num_gpus);
    ctx.d_adjacency_ptrs.resize(ctx.num_gpus, nullptr);
    ctx.d_weights_ptrs.resize(ctx.num_gpus, nullptr);
    for (int g = 0; g < ctx.num_gpus; ++g)
    {
        ctx.device_ids[g] = g;
        cudaSetDevice(g);
        cudaStreamCreate(&ctx.streams[g]);
    }
    for (int g = 0; g < ctx.num_gpus; ++g)
    {
        cudaSetDevice(g);
        for (int p = g + 1; p < ctx.num_gpus; ++p)
        {
            int can = 0;
            cudaDeviceCanAccessPeer(&can, g, p);
            if (can)
            {
                cudaSetDevice(g);
                cudaDeviceEnablePeerAccess(p, 0);
                cudaSetDevice(p);
                cudaDeviceEnablePeerAccess(g, 0);
            }
        }
    }
    return cudaSuccess;
}

cudaError_t scatterGraphData(int *h_adj, Size total_edges, int num_gpus, std::vector<int *> &d_adj)
{
    MultiGpuGraphContext ctx;
    auto err = setupMultiGpu(ctx);
    if (err != cudaSuccess)
        return err;

    Size edges_per_gpu = total_edges / num_gpus;
    Size extra = total_edges % num_gpus;

    d_adj.resize(num_gpus);
    Size offset = 0;
    for (int g = 0; g < num_gpus; ++g)
    {
        Size count = edges_per_gpu + (g < static_cast<int>(extra) ? 1 : 0);
        cudaSetDevice(g);
        cudaMalloc(&d_adj[g], count * sizeof(int));
        cudaMemcpyAsync(d_adj[g], h_adj + offset, count * sizeof(int), cudaMemcpyHostToDevice,
                        ctx.streams[g]);
        offset += count;
    }

    for (int g = 0; g < num_gpus; ++g)
    {
        cudaSetDevice(g);
        cudaStreamSynchronize(ctx.streams[g]);
        cudaStreamDestroy(ctx.streams[g]);
    }

    return cudaSuccess;
}

cudaError_t multiGpuBfs(int *d_adj, Size n_vertices, Size n_edges, int source, int num_gpus,
                        int *d_distances)
{
    if (n_vertices == 0 || n_edges == 0)
        return cudaSuccess;
    if (num_gpus <= 1)
        return cudaErrorInvalidConfiguration;

    // Copy CSR adjacency from device to host
    cudaSetDevice(0);
    std::vector<int> h_row_ptr(n_vertices + 1);
    std::vector<int> h_col_idx(n_edges);
    cudaError_t err =
        cudaMemcpy(h_row_ptr.data(), d_adj, (n_vertices + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
        return err;
    err = cudaMemcpy(h_col_idx.data(), d_adj + n_vertices + 1, n_edges * sizeof(int),
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
        return err;

    // Host-side BFS
    std::vector<int> h_distances(n_vertices, -1);
    std::queue<int> frontier;
    if (source >= 0 && source < static_cast<int>(n_vertices))
    {
        h_distances[source] = 0;
        frontier.push(source);
    }

    while (!frontier.empty())
    {
        int u = frontier.front();
        frontier.pop();
        int next_dist = h_distances[u] + 1;
        for (Size i = static_cast<Size>(h_row_ptr[u]); i < static_cast<Size>(h_row_ptr[u + 1]); ++i)
        {
            int v = h_col_idx[i];
            if (h_distances[v] == -1)
            {
                h_distances[v] = next_dist;
                frontier.push(v);
            }
        }
    }

    // Copy distances back to device
    err = cudaMemcpy(d_distances, h_distances.data(), n_vertices * sizeof(int),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
        return err;

    // Sync remaining GPUs: copy distances to each GPU partition
    Size chunk = (n_vertices + num_gpus - 1) / num_gpus;
    for (int g = 1; g < num_gpus; ++g)
    {
        cudaSetDevice(g);
        Size start = g * chunk;
        Size count = std::min(chunk, n_vertices - start);
        cudaMemcpyPeer(d_distances + start, g, d_distances + start, 0, count * sizeof(int));
    }
    cudaSetDevice(0);

    return cudaSuccess;
}

cudaError_t multiGpuPageRank(const float *d_adj, Size n, float alpha, int num_gpus, float *d_result)
{
    if (n == 0)
        return cudaSuccess;
    if (num_gpus <= 1)
        return cudaErrorInvalidConfiguration;

    // Copy dense row-major matrix from device to host
    cudaSetDevice(0);
    std::vector<float> h_matrix(n * n);
    cudaError_t err =
        cudaMemcpy(h_matrix.data(), d_adj, n * n * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
        return err;

    // Power iteration
    constexpr int max_iterations = 50;
    const float damping = (1.0f - alpha) / static_cast<float>(n);
    std::vector<float> h_rank(n, 1.0f / static_cast<float>(n));
    std::vector<float> h_rank_new(n, 0.0f);

    for (int iter = 0; iter < max_iterations; ++iter)
    {
        // r_new = alpha * A^T * r + (1-alpha)/n
        for (Size i = 0; i < n; ++i)
        {
            float sum = 0.0f;
            for (Size j = 0; j < n; ++j)
            {
                sum += h_matrix[j * n + i] * h_rank[j];
            }
            h_rank_new[i] = alpha * sum + damping;
        }
        h_rank.swap(h_rank_new);
    }

    // Copy result to device
    err = cudaMemcpy(d_result, h_rank.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
        return err;

    // Sync remaining GPUs
    Size chunk = (n + num_gpus - 1) / num_gpus;
    for (int g = 1; g < num_gpus; ++g)
    {
        cudaSetDevice(g);
        Size start = g * chunk;
        Size count = std::min(chunk, n - start);
        cudaMemcpyPeer(d_result + start, g, d_result + start, 0, count * sizeof(float));
    }
    cudaSetDevice(0);

    return cudaSuccess;
}

#else

cudaError_t scatterGraphData(int *, Size, int, std::vector<int *> &)
{
    return cudaErrorNotSupported;
}

cudaError_t multiGpuBfs(int *, Size, Size, int, int, int *)
{
    return cudaErrorNotSupported;
}

cudaError_t multiGpuPageRank(const float *, Size, float, int, float *)
{
    return cudaErrorNotSupported;
}

#endif

} // namespace nerve::graphs

#ifdef __CUDACC__

#include "detail/vr_gpu_cuda_launchers.inl"
#include "detail/vr_gpu_helpers.inl"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace nerve::gpu::algebra::detail
{

static void expandCliquesGpu(const std::vector<VRSimplex> &edges,
                             const std::vector<int> &adjacencyMatrix, int n_points,
                             int max_dimension, double max_radius,
                             const std::vector<std::vector<double>> &points,
                             std::vector<VRSimplex> &out_simplices)
{
    int *d_adjacency;
    std::size_t adjacency_entries = 0;
    std::size_t adj_size = 0;
    if (!checkedProduct(static_cast<std::size_t>(n_points), static_cast<std::size_t>(n_points),
                        adjacency_entries) ||
        !checkedByteCount(adjacency_entries, sizeof(int), adj_size))
    {
        throw std::length_error("VR GPU clique adjacency allocation exceeds host limits");
    }
    cudaMalloc(&d_adjacency, adj_size);
    cudaMemcpy(d_adjacency, adjacencyMatrix.data(), adj_size, cudaMemcpyHostToDevice);

    int *d_edges;
    int n_edges = 0;
    if (!checkedIntSize(edges.size(), n_edges))
    {
        cudaFree(d_adjacency);
        throw std::length_error("VR GPU clique edge count exceeds int range");
    }
    std::size_t edge_values = 0;
    std::size_t edge_bytes = 0;
    if (!checkedProduct(edges.size(), static_cast<std::size_t>(2), edge_values) ||
        !checkedByteCount(edge_values, sizeof(int), edge_bytes))
    {
        cudaFree(d_adjacency);
        throw std::length_error("VR GPU clique edge allocation exceeds host limits");
    }
    cudaMalloc(&d_edges, edge_bytes);

    std::vector<int> edge_data;
    edge_data.reserve(edge_values);
    for (const auto &edge : edges)
    {
        edge_data.push_back(edge.vertices[0]);
        edge_data.push_back(edge.vertices[1]);
    }
    cudaMemcpy(d_edges, edge_data.data(), edge_bytes, cudaMemcpyHostToDevice);

    int *d_out_simplices;
    int *d_out_count;
    size_t max_output = 0;
    for (int d = 2; d <= max_dimension; ++d)
    {
        size_t dim_simplices = 1;
        for (int k = 1; k <= d + 1; ++k)
        {
            dim_simplices = dim_simplices * (n_points - k + 1) / k;
        }
        max_output += dim_simplices;
    }
    max_output = std::min(max_output, size_t(10000000));

    std::size_t output_width = static_cast<std::size_t>(max_dimension) + 1;
    std::size_t output_values = 0;
    std::size_t output_bytes = 0;
    if (!checkedProduct(max_output, output_width, output_values) ||
        !checkedByteCount(output_values, sizeof(int), output_bytes))
    {
        cudaFree(d_adjacency);
        cudaFree(d_edges);
        throw std::length_error("VR GPU clique output allocation exceeds host limits");
    }
    cudaMalloc(&d_out_simplices, output_bytes);
    cudaMalloc(&d_out_count, sizeof(int));
    cudaMemset(d_out_count, 0, sizeof(int));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    launchCliqueExpansion(d_adjacency, d_edges, n_edges, d_out_simplices, d_out_count,
                          max_dimension, n_points, stream);

    cudaStreamSynchronize(stream);

    int output_count;
    cudaMemcpy(&output_count, d_out_count, sizeof(int), cudaMemcpyDeviceToHost);

    if (output_count < 0)
    {
        cudaFree(d_adjacency);
        cudaFree(d_edges);
        cudaFree(d_out_simplices);
        cudaFree(d_out_count);
        cudaStreamDestroy(stream);
        throw std::length_error("VR GPU clique output count is negative");
    }
    std::size_t output_count_values = 0;
    std::size_t output_count_bytes = 0;
    if (!checkedProduct(static_cast<std::size_t>(output_count), output_width,
                        output_count_values) ||
        !checkedByteCount(output_count_values, sizeof(int), output_count_bytes))
    {
        cudaFree(d_adjacency);
        cudaFree(d_edges);
        cudaFree(d_out_simplices);
        cudaFree(d_out_count);
        cudaStreamDestroy(stream);
        throw std::length_error("VR GPU clique copy size exceeds host limits");
    }
    std::vector<int> output_data(output_count_values);
    cudaMemcpy(output_data.data(), d_out_simplices, output_count_bytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < output_count; ++i)
    {
        VRSimplex simplex;
        std::size_t offset = static_cast<std::size_t>(i) * output_width;
        int dim = 0;

        for (int j = 0; j <= max_dimension; ++j)
        {
            int v = output_data[offset + static_cast<std::size_t>(j)];
            if (v < 0)
                break;
            simplex.vertices.push_back(v);
            dim++;
        }

        if (dim >= 2)
        {
            simplex.dimension = dim - 1;

            double max_edge = 0.0;
            for (size_t a = 0; a < simplex.vertices.size(); ++a)
            {
                for (size_t b = a + 1; b < simplex.vertices.size(); ++b)
                {
                    int v1 = simplex.vertices[a];
                    int v2 = simplex.vertices[b];
                    max_edge = std::max(max_edge, checkedPointDistance(points, v1, v2));
                }
            }
            simplex.filtration_value = max_edge;

            if (simplex.filtration_value <= max_radius)
            {
                out_simplices.push_back(simplex);
            }
        }
    }

    cudaFree(d_adjacency);
    cudaFree(d_edges);
    cudaFree(d_out_simplices);
    cudaFree(d_out_count);
    cudaStreamDestroy(stream);
}

} // namespace nerve::gpu::algebra::detail

#endif // __CUDACC__

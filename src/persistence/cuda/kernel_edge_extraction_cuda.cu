#include "nerve/gpu/gpu_error.hpp"
#include "nerve/persistence/cuda/cuda_edge_extraction.hpp"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace nerve::persistence::accelerated
{
namespace
{

constexpr Size kEdgeBlockSize = 256;
constexpr Size kMaxGridX = 2147483647ULL;

__device__ Size reserveEdge(Size *edge_count)
{
    return static_cast<Size>(atomicAdd(reinterpret_cast<unsigned long long *>(edge_count), 1ULL));
}

__device__ bool keepDistance(double distance, double max_radius, double min_edge_weight = 0.0)
{
    return isfinite(distance) && distance > 0.0 && distance <= max_radius &&
           distance >= min_edge_weight;
}

__device__ void writeEdgeIfRoom(Edge *edges, Size *edge_count, Size max_edges, Size row, Size col,
                                double distance)
{
    const Size edge_idx = reserveEdge(edge_count);
    if (edge_idx < max_edges)
    {
        edges[edge_idx] = Edge(static_cast<int>(row), static_cast<int>(col), distance);
    }
}

__global__ void __launch_bounds__(256)
    edgeExtractionBasicKernel(const double *__restrict__ distances, Edge *__restrict__ edges,
                              Size *__restrict__ edge_count, Size n_points, Size total_elements,
                              double max_radius, Size max_edges)
{
    const Size start = static_cast<Size>(blockIdx.x) * blockDim.x + threadIdx.x;
    const Size stride = static_cast<Size>(gridDim.x) * blockDim.x;
    for (Size index = start; index < total_elements; index += stride)
    {
        const Size row = index / n_points;
        const Size col = index % n_points;
        if (row >= col)
        {
            continue;
        }
        const double distance = distances[index];
        if (keepDistance(distance, max_radius))
        {
            writeEdgeIfRoom(edges, edge_count, max_edges, row, col, distance);
        }
    }
}

__global__ void __launch_bounds__(256)
    edgeExtractionEarlyKernel(const double *__restrict__ distances, Edge *__restrict__ edges,
                              Size *__restrict__ edge_count, Size n_points, Size total_elements,
                              double max_radius, Size max_edges, bool *__restrict__ stop)
{
    const Size start = static_cast<Size>(blockIdx.x) * blockDim.x + threadIdx.x;
    const Size stride = static_cast<Size>(gridDim.x) * blockDim.x;
    for (Size index = start; index < total_elements && !*stop; index += stride)
    {
        const Size row = index / n_points;
        const Size col = index % n_points;
        if (row >= col)
        {
            continue;
        }
        const double distance = distances[index];
        if (!keepDistance(distance, max_radius))
        {
            continue;
        }
        const Size edge_idx = reserveEdge(edge_count);
        if (edge_idx >= max_edges)
        {
            *stop = true;
            return;
        }
        edges[edge_idx] = Edge(static_cast<int>(row), static_cast<int>(col), distance);
    }
}

__global__ void __launch_bounds__(256)
    edgeExtractionSharedKernelImpl(const double *__restrict__ distances, Edge *__restrict__ edges,
                                   Size *__restrict__ edge_count, Size n_points,
                                   Size total_elements, double max_radius, Size max_edges)
{
    extern __shared__ double tile[];
    const Size start = static_cast<Size>(blockIdx.x) * blockDim.x + threadIdx.x;
    const Size stride = static_cast<Size>(gridDim.x) * blockDim.x;
    for (Size index = start; index < total_elements; index += stride)
    {
        const Size row = index / n_points;
        const Size col = index % n_points;
        const bool candidate = row < col;
        tile[threadIdx.x] = candidate ? distances[index] : 0.0;
        __syncthreads();
        const double distance = tile[threadIdx.x];
        __syncthreads();
        if (candidate && keepDistance(distance, max_radius))
        {
            writeEdgeIfRoom(edges, edge_count, max_edges, row, col, distance);
        }
    }
}

__global__ void __launch_bounds__(256)
    edgeExtractionFilteredKernelImpl(const double *__restrict__ distances, Edge *__restrict__ edges,
                                     Size *__restrict__ edge_count, Size n_points,
                                     Size total_elements, double max_radius, Size max_edges,
                                     double min_edge_weight, Size max_degree,
                                     Size *__restrict__ degree_count)
{
    const Size start = static_cast<Size>(blockIdx.x) * blockDim.x + threadIdx.x;
    const Size stride = static_cast<Size>(gridDim.x) * blockDim.x;
    for (Size index = start; index < total_elements; index += stride)
    {
        const Size row = index / n_points;
        const Size col = index % n_points;
        if (row >= col)
        {
            continue;
        }
        const double distance = distances[index];
        if (!keepDistance(distance, max_radius, min_edge_weight))
        {
            continue;
        }
        const Size row_degree = reserveEdge(&degree_count[row]);
        const Size col_degree = reserveEdge(&degree_count[col]);
        if (row_degree < max_degree && col_degree < max_degree)
        {
            writeEdgeIfRoom(edges, edge_count, max_edges, row, col, distance);
        }
    }
}

struct EdgeWeightLess
{
    __host__ __device__ bool operator()(const Edge &lhs, const Edge &rhs) const
    {
        if (lhs.w != rhs.w)
        {
            return lhs.w < rhs.w;
        }
        if (lhs.u != rhs.u)
        {
            return lhs.u < rhs.u;
        }
        return lhs.v < rhs.v;
    }
};

bool validEdgeLaunch(const double *distances, Edge *edges, Size *edge_count, Size n_points,
                     double max_radius, Size max_edges, Size &total_elements)
{
    if (distances == nullptr || edges == nullptr || edge_count == nullptr || n_points == 0 ||
        max_edges == 0 || n_points > static_cast<Size>(std::numeric_limits<int>::max()) ||
        !std::isfinite(max_radius) || !(max_radius > 0.0))
    {
        return false;
    }
    return detail::checkedSizeProduct(n_points, n_points, total_elements);
}

unsigned int gridFor(Size total_elements)
{
    const Size blocks = std::max<Size>(1, detail::ceilDiv(total_elements, kEdgeBlockSize));
    return static_cast<unsigned int>(std::min(blocks, kMaxGridX));
}

void sortExtractedEdges(Edge *edges, Size *edge_count, Size max_edges)
{
    Size host_count = 0;
    if (cudaMemcpy(&host_count, edge_count, sizeof(Size), cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        return;
    }
    const Size sort_count = std::min(host_count, max_edges);
    if (sort_count <= 1)
    {
        return;
    }
    thrust::device_ptr<Edge> begin(edges);
    thrust::sort(thrust::device, begin, begin + sort_count, EdgeWeightLess{});
}

} // namespace

namespace cuda_kernels
{

void extractEdgesKernel(const double *__restrict__ distances, Edge *__restrict__ edges,
                        Size *__restrict__ edge_count, Size n_points, double max_radius,
                        Size max_edges)
{
    Size total_elements = 0;
    if (!validEdgeLaunch(distances, edges, edge_count, n_points, max_radius, max_edges,
                         total_elements))
    {
        return;
    }
    edgeExtractionBasicKernel<<<gridFor(total_elements), kEdgeBlockSize>>>(
        distances, edges, edge_count, n_points, total_elements, max_radius, max_edges);
    GPU_CHECK(cudaPeekAtLastError());
}

void extractEdgesEarlyTerminationKernel(const double *__restrict__ distances,
                                        Edge *__restrict__ edges, Size *__restrict__ edge_count,
                                        Size n_points, double max_radius, Size max_edges,
                                        bool *__restrict__ early_termination_flag)
{
    Size total_elements = 0;
    if (early_termination_flag == nullptr ||
        !validEdgeLaunch(distances, edges, edge_count, n_points, max_radius, max_edges,
                         total_elements))
    {
        return;
    }
    edgeExtractionEarlyKernel<<<gridFor(total_elements), kEdgeBlockSize>>>(
        distances, edges, edge_count, n_points, total_elements, max_radius, max_edges,
        early_termination_flag);
    GPU_CHECK(cudaPeekAtLastError());
}

void extractEdgesSharedKernel(const double *__restrict__ distances, Edge *__restrict__ edges,
                              Size *__restrict__ edge_count, Size n_points, double max_radius,
                              Size max_edges)
{
    Size total_elements = 0;
    if (!validEdgeLaunch(distances, edges, edge_count, n_points, max_radius, max_edges,
                         total_elements))
    {
        return;
    }
    edgeExtractionSharedKernelImpl<<<gridFor(total_elements), kEdgeBlockSize,
                                     kEdgeBlockSize * sizeof(double)>>>(
        distances, edges, edge_count, n_points, total_elements, max_radius, max_edges);
    GPU_CHECK(cudaPeekAtLastError());
}

void extractEdgesFilteredKernel(const double *__restrict__ distances, Edge *__restrict__ edges,
                                Size *__restrict__ edge_count, Size n_points, double max_radius,
                                Size max_edges, double min_edge_weight, Size max_degree,
                                Size *__restrict__ degree_count)
{
    Size total_elements = 0;
    if (degree_count == nullptr || max_degree == 0 || !std::isfinite(min_edge_weight) ||
        min_edge_weight < 0.0 ||
        !validEdgeLaunch(distances, edges, edge_count, n_points, max_radius, max_edges,
                         total_elements))
    {
        return;
    }
    edgeExtractionFilteredKernelImpl<<<gridFor(total_elements), kEdgeBlockSize>>>(
        distances, edges, edge_count, n_points, total_elements, max_radius, max_edges,
        min_edge_weight, max_degree, degree_count);
    GPU_CHECK(cudaPeekAtLastError());
}

void extractEdgesSortedKernel(const double *__restrict__ distances, Edge *__restrict__ edges,
                              Size *__restrict__ edge_count, Size n_points, double max_radius,
                              Size max_edges, bool sort_by_weight)
{
    extractEdgesKernel(distances, edges, edge_count, n_points, max_radius, max_edges);
    if (sort_by_weight)
    {
        sortExtractedEdges(edges, edge_count, max_edges);
    }
}

} // namespace cuda_kernels

} // namespace nerve::persistence::accelerated

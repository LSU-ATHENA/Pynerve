#include "graph_algorithms_detail.hpp"
#include "nerve/graphs/gpu_graphs.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace nerve::graphs::gpu
{
namespace
{

constexpr int kBlockSize = 256;
constexpr int kFloydTileDim = 16;
constexpr int kInfDistance = detail::kInfDistance;

void checkCuda(cudaError_t status, const char *operation)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(std::string("Nerve CUDA graph operation failed during ") +
                                 operation + ": " + cudaGetErrorString(status));
    }
}

int ceilDiv(int value, int divisor)
{
    if (value < 0 || divisor <= 0)
    {
        throw std::invalid_argument("invalid CUDA graph grid dimensions");
    }
    const auto numerator = static_cast<std::size_t>(value) + static_cast<std::size_t>(divisor) - 1;
    const auto result = numerator / static_cast<std::size_t>(divisor);
    if (result > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::length_error("CUDA graph grid dimension exceeds int range");
    }
    return static_cast<int>(result);
}

template <typename T>
std::size_t checkedElementBytes(std::size_t count, const char *operation)
{
    if (count != 0 && count > std::numeric_limits<std::size_t>::max() / sizeof(T))
    {
        throw std::length_error(std::string(operation) + " byte count exceeds size_t");
    }
    return count * sizeof(T);
}

template <typename T>
std::size_t vectorBytes(const std::vector<T> &values, const char *operation)
{
    return checkedElementBytes<T>(values.size(), operation);
}

template <typename T>
class DeviceBuffer
{
public:
    explicit DeviceBuffer(std::size_t count)
        : count_(count)
    {
        if (count_ > 0)
        {
            checkCuda(cudaMalloc(&ptr_, checkedElementBytes<T>(count_, "cudaMalloc")),
                      "cudaMalloc");
        }
    }

    ~DeviceBuffer()
    {
        if (ptr_ != nullptr)
        {
            cudaFree(ptr_);
        }
    }

    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    T *data() { return ptr_; }
    const T *data() const { return ptr_; }
    std::size_t size() const { return count_; }

    void swap(DeviceBuffer &other) noexcept
    {
        std::swap(ptr_, other.ptr_);
        std::swap(count_, other.count_);
    }

private:
    T *ptr_ = nullptr;
    std::size_t count_ = 0;
};

__global__ void __launch_bounds__(256)
    bfsFrontierKernel(const int *__restrict__ row_ptr, const int *__restrict__ col_idx,
                      int *__restrict__ distances, int *__restrict__ visited,
                      const int *__restrict__ current_queue, int current_queue_size,
                      int *__restrict__ next_queue, int *__restrict__ next_queue_size,
                      int *__restrict__ overflow, int max_queue_size, int num_vertices)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= current_queue_size)
    {
        return;
    }

    const int vertex = current_queue[idx];
    const int current_distance = distances[vertex];
    if (current_distance == kInfDistance)
    {
        return;
    }

    for (int edge = row_ptr[vertex]; edge < row_ptr[vertex + 1]; ++edge)
    {
        const int neighbor = col_idx[edge];
        if (neighbor < 0 || neighbor >= num_vertices)
        {
            continue;
        }
        if (atomicExch(&visited[neighbor], 1) == 0)
        {
            atomicMin(&distances[neighbor], current_distance + 1);
            const int queue_pos = atomicAdd(next_queue_size, 1);
            if (queue_pos < max_queue_size)
            {
                next_queue[queue_pos] = neighbor;
            }
            else
            {
                *overflow = 1;
            }
        }
    }
}

__global__ void __launch_bounds__(256)
    connectedComponentsKernel(const int *__restrict__ row_ptr, const int *__restrict__ col_idx,
                              int *__restrict__ labels, int *__restrict__ changed, int num_vertices)
{
    const int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex >= num_vertices)
    {
        return;
    }

    const int current = labels[vertex];
    int minimum = current;
    for (int edge = row_ptr[vertex]; edge < row_ptr[vertex + 1]; ++edge)
    {
        const int neighbor = col_idx[edge];
        if (neighbor >= 0 && neighbor < num_vertices)
        {
            minimum = min(minimum, labels[neighbor]);
        }
    }

    if (minimum < current)
    {
        atomicMin(&labels[vertex], minimum);
        *changed = 1;
    }
}

__global__ void __launch_bounds__(256) floydWarshallKernel(float *__restrict__ dist, int n, int k)
{
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || j >= n)
    {
        return;
    }

    const float through_k = dist[i * n + k] + dist[k * n + j];
    if (through_k < dist[i * n + j])
    {
        dist[i * n + j] = through_k;
    }
}

} // namespace

class GPUGraph::Impl
{
public:
    Impl(int num_vertices, int edge_capacity)
        : num_vertices_(num_vertices)
        , edge_capacity_(edge_capacity)
    {
        if (num_vertices_ < 0 || edge_capacity_ < 0)
        {
            throw std::invalid_argument("graph dimensions must be non-negative");
        }
        allocateDeviceStorage();
    }

    ~Impl() { releaseDeviceStorage(); }

    void setCSR(const std::vector<int> &row_ptr, const std::vector<int> &col_idx,
                const std::vector<float> &weights)
    {
        detail::validateCsr(num_vertices_, edge_capacity_, row_ptr, col_idx, weights);
        row_ptr_ = row_ptr;
        col_idx_ = col_idx;
        weights_ = weights.empty() ? std::vector<float>(col_idx.size(), 1.0f) : weights;

        if (!row_ptr_.empty())
        {
            checkCuda(cudaMemcpy(d_row_ptr_, row_ptr_.data(),
                                 vectorBytes(row_ptr_, "copy CSR row pointer"),
                                 cudaMemcpyHostToDevice),
                      "copy CSR row pointer");
        }
        if (!col_idx_.empty())
        {
            checkCuda(cudaMemcpy(d_col_idx_, col_idx_.data(),
                                 vectorBytes(col_idx_, "copy CSR columns"), cudaMemcpyHostToDevice),
                      "copy CSR columns");
            checkCuda(cudaMemcpy(d_weights_, weights_.data(),
                                 vectorBytes(weights_, "copy CSR weights"), cudaMemcpyHostToDevice),
                      "copy CSR weights");
        }
    }

    std::vector<int> bfs(int source)
    {
        validateReady(source);
        std::vector<int> distances(num_vertices_, kInfDistance);
        distances[source] = 0;

        DeviceBuffer<int> queue(num_vertices_);
        DeviceBuffer<int> next_queue(num_vertices_);
        DeviceBuffer<int> queue_size(1);
        DeviceBuffer<int> next_queue_size(1);
        DeviceBuffer<int> overflow(1);

        checkCuda(cudaMemcpy(d_distances_, distances.data(),
                             vectorBytes(distances, "initialize BFS distances"),
                             cudaMemcpyHostToDevice),
                  "initialize BFS distances");
        checkCuda(cudaMemset(d_visited_, 0,
                             checkedElementBytes<int>(static_cast<std::size_t>(num_vertices_),
                                                      "initialize BFS visited")),
                  "initialize BFS visited");
        checkCuda(cudaMemcpy(queue.data(), &source, sizeof(int), cudaMemcpyHostToDevice),
                  "initialize BFS queue");
        int one = 1;
        checkCuda(cudaMemcpy(queue_size.data(), &one, sizeof(int), cudaMemcpyHostToDevice),
                  "initialize BFS queue size");
        checkCuda(cudaMemcpy(d_visited_ + source, &one, sizeof(int), cudaMemcpyHostToDevice),
                  "mark BFS source");

        for (int depth = 0; depth < num_vertices_; ++depth)
        {
            int current_size = 0;
            checkCuda(
                cudaMemcpy(&current_size, queue_size.data(), sizeof(int), cudaMemcpyDeviceToHost),
                "read BFS frontier size");
            if (current_size == 0)
            {
                break;
            }

            int zero = 0;
            checkCuda(
                cudaMemcpy(next_queue_size.data(), &zero, sizeof(int), cudaMemcpyHostToDevice),
                "reset next BFS queue size");
            checkCuda(cudaMemcpy(overflow.data(), &zero, sizeof(int), cudaMemcpyHostToDevice),
                      "reset BFS overflow flag");

            bfsFrontierKernel<<<ceilDiv(current_size, kBlockSize), kBlockSize>>>(
                d_row_ptr_, d_col_idx_, d_distances_, d_visited_, queue.data(), current_size,
                next_queue.data(), next_queue_size.data(), overflow.data(), num_vertices_,
                num_vertices_);
            checkCuda(cudaGetLastError(), "launch BFS frontier kernel");
            checkCuda(cudaDeviceSynchronize(), "synchronize BFS frontier kernel");

            int overflow_flag = 0;
            checkCuda(
                cudaMemcpy(&overflow_flag, overflow.data(), sizeof(int), cudaMemcpyDeviceToHost),
                "read BFS overflow flag");
            if (overflow_flag != 0)
            {
                throw std::runtime_error("BFS frontier exceeded the graph vertex capacity");
            }

            queue.swap(next_queue);
            queue_size.swap(next_queue_size);
        }

        checkCuda(cudaMemcpy(distances.data(), d_distances_,
                             vectorBytes(distances, "copy BFS distances"), cudaMemcpyDeviceToHost),
                  "copy BFS distances");
        return distances;
    }

    std::vector<int> connectedComponents()
    {
        ensureReady();
        std::vector<int> labels(num_vertices_);
        std::iota(labels.begin(), labels.end(), 0);
        if (labels.empty())
        {
            return labels;
        }

        DeviceBuffer<int> changed(1);
        checkCuda(cudaMemcpy(d_labels_, labels.data(),
                             vectorBytes(labels, "initialize component labels"),
                             cudaMemcpyHostToDevice),
                  "initialize component labels");

        for (int iter = 0; iter < num_vertices_; ++iter)
        {
            int zero = 0;
            checkCuda(cudaMemcpy(changed.data(), &zero, sizeof(int), cudaMemcpyHostToDevice),
                      "reset component convergence flag");
            connectedComponentsKernel<<<ceilDiv(num_vertices_, kBlockSize), kBlockSize>>>(
                d_row_ptr_, d_col_idx_, d_labels_, changed.data(), num_vertices_);
            checkCuda(cudaGetLastError(), "launch connected-components kernel");
            checkCuda(cudaDeviceSynchronize(), "synchronize connected-components kernel");

            int changed_host = 0;
            checkCuda(
                cudaMemcpy(&changed_host, changed.data(), sizeof(int), cudaMemcpyDeviceToHost),
                "read component convergence flag");
            if (changed_host == 0)
            {
                break;
            }
        }

        checkCuda(cudaMemcpy(labels.data(), d_labels_, vectorBytes(labels, "copy component labels"),
                             cudaMemcpyDeviceToHost),
                  "copy component labels");
        return labels;
    }

    std::vector<float> allPairsShortestPaths()
    {
        ensureReady();
        std::vector<float> dist =
            detail::buildDistanceMatrix(num_vertices_, row_ptr_, col_idx_, weights_);
        if (dist.empty())
        {
            return dist;
        }

        DeviceBuffer<float> d_dist(dist.size());
        checkCuda(cudaMemcpy(d_dist.data(), dist.data(),
                             vectorBytes(dist, "initialize APSP matrix"), cudaMemcpyHostToDevice),
                  "initialize APSP matrix");

        const dim3 block(kFloydTileDim, kFloydTileDim);
        const dim3 grid(ceilDiv(num_vertices_, kFloydTileDim),
                        ceilDiv(num_vertices_, kFloydTileDim));
        for (int k = 0; k < num_vertices_; ++k)
        {
            floydWarshallKernel<<<grid, block>>>(d_dist.data(), num_vertices_, k);
            checkCuda(cudaGetLastError(), "launch Floyd-Warshall kernel");
        }
        checkCuda(cudaDeviceSynchronize(), "synchronize Floyd-Warshall kernels");
        checkCuda(cudaMemcpy(dist.data(), d_dist.data(), vectorBytes(dist, "copy APSP matrix"),
                             cudaMemcpyDeviceToHost),
                  "copy APSP matrix");
        return dist;
    }

private:
    void allocateDeviceStorage()
    {
        try
        {
            if (num_vertices_ > 0)
            {
                checkCuda(cudaMalloc(&d_row_ptr_, checkedElementBytes<int>(
                                                      static_cast<std::size_t>(num_vertices_) + 1,
                                                      "allocate CSR row pointer")),
                          "allocate CSR row pointer");
                checkCuda(cudaMalloc(&d_distances_, checkedElementBytes<int>(
                                                        static_cast<std::size_t>(num_vertices_),
                                                        "allocate BFS distances")),
                          "allocate BFS distances");
                checkCuda(cudaMalloc(&d_visited_, checkedElementBytes<int>(
                                                      static_cast<std::size_t>(num_vertices_),
                                                      "allocate BFS visited flags")),
                          "allocate BFS visited flags");
                checkCuda(cudaMalloc(&d_labels_, checkedElementBytes<int>(
                                                     static_cast<std::size_t>(num_vertices_),
                                                     "allocate component labels")),
                          "allocate component labels");
            }
            if (edge_capacity_ > 0)
            {
                checkCuda(cudaMalloc(&d_col_idx_, checkedElementBytes<int>(
                                                      static_cast<std::size_t>(edge_capacity_),
                                                      "allocate CSR columns")),
                          "allocate CSR columns");
                checkCuda(cudaMalloc(&d_weights_, checkedElementBytes<float>(
                                                      static_cast<std::size_t>(edge_capacity_),
                                                      "allocate CSR weights")),
                          "allocate CSR weights");
            }
        }
        catch (...)
        {
            releaseDeviceStorage();
            throw;
        }
    }

    void releaseDeviceStorage() noexcept
    {
        cudaFree(d_row_ptr_);
        cudaFree(d_col_idx_);
        cudaFree(d_weights_);
        cudaFree(d_distances_);
        cudaFree(d_visited_);
        cudaFree(d_labels_);
        d_row_ptr_ = nullptr;
        d_col_idx_ = nullptr;
        d_weights_ = nullptr;
        d_distances_ = nullptr;
        d_visited_ = nullptr;
        d_labels_ = nullptr;
    }

    void ensureReady() const
    {
        if (row_ptr_.empty() && num_vertices_ > 0)
        {
            throw std::logic_error("CSR graph data has not been uploaded");
        }
    }

    void validateReady(int source) const
    {
        ensureReady();
        detail::validateVertex(source, num_vertices_, "source vertex");
    }

    int num_vertices_ = 0;
    int edge_capacity_ = 0;
    std::vector<int> row_ptr_;
    std::vector<int> col_idx_;
    std::vector<float> weights_;
    int *d_row_ptr_ = nullptr;
    int *d_col_idx_ = nullptr;
    float *d_weights_ = nullptr;
    int *d_distances_ = nullptr;
    int *d_visited_ = nullptr;
    int *d_labels_ = nullptr;
};

GPUGraph::GPUGraph(int num_vertices, int num_edges)
    : impl_(std::make_unique<Impl>(num_vertices, num_edges))
{}

GPUGraph::~GPUGraph() = default;

void GPUGraph::setCSR(const std::vector<int> &row_ptr, const std::vector<int> &col_idx,
                      const std::vector<float> &weights)
{
    impl_->setCSR(row_ptr, col_idx, weights);
}

std::vector<int> GPUGraph::bfs(int source)
{
    return impl_->bfs(source);
}

std::vector<int> GPUGraph::connectedComponents()
{
    return impl_->connectedComponents();
}

std::vector<float> GPUGraph::allPairsShortestPaths()
{
    return impl_->allPairsShortestPaths();
}

} // namespace nerve::graphs::gpu

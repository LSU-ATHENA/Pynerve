#include "graph_algorithms_detail.hpp"
#include "nerve/graphs/gpu_graphs.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nerve::graphs
{
namespace
{

std::vector<int> saturatingFloatDistances(const std::vector<float> &distances)
{
    std::vector<int> converted;
    converted.reserve(distances.size());
    for (float value : distances)
    {
        if (!std::isfinite(value))
        {
            converted.push_back(std::numeric_limits<int>::max());
        }
        else if (value >= static_cast<float>(std::numeric_limits<int>::max()))
        {
            converted.push_back(std::numeric_limits<int>::max());
        }
        else
        {
            converted.push_back(static_cast<int>(std::lround(value)));
        }
    }
    return converted;
}

} // namespace

class GraphEngine::Impl
{
public:
    explicit Impl(const GraphConfig &config)
        : config_(config)
    {
        if (config_.num_vertices < 0 || config_.num_edges < 0)
        {
            throw std::invalid_argument("graph configuration dimensions must be non-negative");
        }
        row_ptr_.assign(static_cast<std::size_t>(config_.num_vertices) + 1, 0);
    }

    void setGraph(const std::vector<std::pair<int, int>> &edges)
    {
        std::vector<float> unit_weights(edges.size(), 1.0f);
        setWeightedGraph(edges, unit_weights);
        weighted_ = false;
    }

    void setWeightedGraph(const std::vector<std::pair<int, int>> &edges,
                          const std::vector<float> &weights)
    {
        if (weights.size() != edges.size())
        {
            throw std::invalid_argument("weighted graph requires one weight per edge");
        }
        std::vector<std::tuple<int, int, float>> adjacency;
        if (edges.size() > std::numeric_limits<std::size_t>::max() / 2)
        {
            throw std::length_error("graph adjacency reserve exceeds size_t");
        }
        adjacency.reserve(edges.size() * 2);
        for (std::size_t i = 0; i < edges.size(); ++i)
        {
            const auto [u, v] = edges[i];
            gpu::detail::validateVertex(u, config_.num_vertices, "edge source");
            gpu::detail::validateVertex(v, config_.num_vertices, "edge target");
            if (!std::isfinite(weights[i]) || weights[i] < 0.0f)
            {
                throw std::invalid_argument("graph weights must be finite and non-negative");
            }
            adjacency.emplace_back(u, v, weights[i]);
            if (u != v)
            {
                adjacency.emplace_back(v, u, weights[i]);
            }
        }

        row_ptr_.assign(static_cast<std::size_t>(config_.num_vertices) + 1, 0);
        for (const auto &[u, v, weight] : adjacency)
        {
            (void)v;
            (void)weight;
            ++row_ptr_[static_cast<std::size_t>(u) + 1];
        }
        for (int i = 0; i < config_.num_vertices; ++i)
        {
            row_ptr_[static_cast<std::size_t>(i) + 1] += row_ptr_[static_cast<std::size_t>(i)];
        }

        std::vector<int> cursor = row_ptr_;
        col_idx_.assign(adjacency.size(), 0);
        weights_.assign(adjacency.size(), 1.0f);
        for (const auto &[u, v, weight] : adjacency)
        {
            const int offset = cursor[u]++;
            col_idx_[static_cast<std::size_t>(offset)] = v;
            weights_[static_cast<std::size_t>(offset)] = weight;
        }
        weighted_ = true;
    }

    GraphTraversalResult bfs(int source)
    {
        gpu::detail::validateVertex(source, config_.num_vertices, "source vertex");
        ensureGraphSet();
        const auto start = std::chrono::steady_clock::now();

        std::vector<int> distances;
        if (config_.use_gpu && gpuAvailable())
        {
            const int edge_count = gpu::detail::checkedIntSize(
                col_idx_.size(), "GPU graph edge count exceeds int range");
            gpu::GPUGraph graph(config_.num_vertices, edge_count);
            graph.setCSR(row_ptr_, col_idx_, {});
            distances = graph.bfs(source);
        }
        else
        {
            distances = gpu::detail::cpuBfs(row_ptr_, col_idx_, source);
        }

        GraphTraversalResult result;
        result.distances = std::move(distances);
        result.predecessors = derivePredecessors(result.distances, source);
        result.computation_time_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                .count();
        return result;
    }

    GraphTraversalResult dijkstra(int source)
    {
        gpu::detail::validateVertex(source, config_.num_vertices, "source vertex");
        ensureGraphSet();
        const auto start = std::chrono::steady_clock::now();

        using QueueEntry = std::pair<float, int>;
        std::priority_queue<QueueEntry, std::vector<QueueEntry>, std::greater<QueueEntry>> queue;
        std::vector<float> distances(config_.num_vertices, std::numeric_limits<float>::infinity());
        std::vector<int> predecessors(config_.num_vertices, -1);
        distances[source] = 0.0f;
        queue.emplace(0.0f, source);

        while (!queue.empty())
        {
            const auto [distance, u] = queue.top();
            queue.pop();
            if (distance != distances[u])
            {
                continue;
            }
            for (int edge = row_ptr_[u]; edge < row_ptr_[u + 1]; ++edge)
            {
                const int v = col_idx_[edge];
                const float candidate = distance + weights_[edge];
                if (candidate < distances[v])
                {
                    distances[v] = candidate;
                    predecessors[v] = u;
                    queue.emplace(candidate, v);
                }
            }
        }

        GraphTraversalResult result;
        result.distances = saturatingFloatDistances(distances);
        result.predecessors = std::move(predecessors);
        result.computation_time_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                .count();
        return result;
    }

    std::vector<int> connectedComponents()
    {
        ensureGraphSet();
        if (config_.use_gpu && gpuAvailable())
        {
            const int edge_count = gpu::detail::checkedIntSize(
                col_idx_.size(), "GPU graph edge count exceeds int range");
            gpu::GPUGraph graph(config_.num_vertices, edge_count);
            graph.setCSR(row_ptr_, col_idx_, {});
            return graph.connectedComponents();
        }
        return gpu::detail::cpuConnectedComponents(row_ptr_, col_idx_);
    }

    std::vector<float> allPairsShortestPaths()
    {
        ensureGraphSet();
        if (config_.use_gpu && gpuAvailable())
        {
            const int edge_count = gpu::detail::checkedIntSize(
                col_idx_.size(), "GPU graph edge count exceeds int range");
            gpu::GPUGraph graph(config_.num_vertices, edge_count);
            graph.setCSR(row_ptr_, col_idx_, weights_);
            return graph.allPairsShortestPaths();
        }

        std::vector<float> dist =
            gpu::detail::buildDistanceMatrix(config_.num_vertices, row_ptr_, col_idx_, weights_);
        for (int k = 0; k < config_.num_vertices; ++k)
        {
            for (int i = 0; i < config_.num_vertices; ++i)
            {
                for (int j = 0; j < config_.num_vertices; ++j)
                {
                    const float candidate =
                        dist[static_cast<std::size_t>(i) *
                                 static_cast<std::size_t>(config_.num_vertices) +
                             static_cast<std::size_t>(k)] +
                        dist[static_cast<std::size_t>(k) *
                                 static_cast<std::size_t>(config_.num_vertices) +
                             static_cast<std::size_t>(j)];
                    float &target = dist[static_cast<std::size_t>(i) *
                                             static_cast<std::size_t>(config_.num_vertices) +
                                         static_cast<std::size_t>(j)];
                    if (candidate < target)
                    {
                        target = candidate;
                    }
                }
            }
        }
        return dist;
    }

    std::vector<std::pair<float, float>> computeGraphPersistence()
    {
        ensureGraphSet();
        struct Edge
        {
            int u;
            int v;
            float weight;
        };
        std::unordered_map<std::uint64_t, float> edge_weights;
        for (int u = 0; u < config_.num_vertices; ++u)
        {
            for (int edge = row_ptr_[u]; edge < row_ptr_[u + 1]; ++edge)
            {
                const int v = col_idx_[edge];
                const int a = std::min(u, v);
                const int b = std::max(u, v);
                const auto encoded =
                    (static_cast<std::uint64_t>(static_cast<std::uint32_t>(a)) << 32) |
                    static_cast<std::uint32_t>(b);
                auto [it, inserted] = edge_weights.emplace(encoded, weights_[edge]);
                if (!inserted)
                {
                    it->second = std::min(it->second, weights_[edge]);
                }
            }
        }

        std::vector<Edge> edges;
        edges.reserve(edge_weights.size());
        for (const auto &[encoded, weight] : edge_weights)
        {
            edges.push_back(
                {static_cast<int>(encoded >> 32), static_cast<int>(encoded & 0xffffffffu), weight});
        }
        std::sort(edges.begin(), edges.end(), [](const Edge &lhs, const Edge &rhs) {
            return std::tie(lhs.weight, lhs.u, lhs.v) < std::tie(rhs.weight, rhs.u, rhs.v);
        });

        std::vector<int> parent(config_.num_vertices);
        std::vector<int> rank(config_.num_vertices, 0);
        std::iota(parent.begin(), parent.end(), 0);
        auto find_root = [&](int x) {
            while (parent[x] != x)
            {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            return x;
        };

        std::vector<std::pair<float, float>> intervals;
        intervals.reserve(static_cast<std::size_t>(config_.num_vertices));
        int components = config_.num_vertices;
        for (const Edge &edge : edges)
        {
            int a = find_root(edge.u);
            int b = find_root(edge.v);
            if (a == b)
            {
                continue;
            }
            if (rank[a] < rank[b])
            {
                std::swap(a, b);
            }
            parent[b] = a;
            if (rank[a] == rank[b])
            {
                ++rank[a];
            }
            --components;
            intervals.emplace_back(0.0f, edge.weight);
        }
        for (int i = 0; i < components; ++i)
        {
            intervals.emplace_back(0.0f, std::numeric_limits<float>::infinity());
        }
        return intervals;
    }

private:
    void ensureGraphSet() const
    {
        if (row_ptr_.empty() ||
            row_ptr_.size() != static_cast<std::size_t>(config_.num_vertices) + 1)
        {
            throw std::logic_error("graph data has not been set");
        }
    }

    static bool gpuAvailable()
    {
        int device_count = 0;
        return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
    }

    std::vector<int> derivePredecessors(const std::vector<int> &distances, int source) const
    {
        std::vector<int> predecessors(config_.num_vertices, -1);
        if (source >= 0 && source < config_.num_vertices)
        {
            predecessors[source] = source;
        }
        for (int u = 0; u < config_.num_vertices; ++u)
        {
            if (distances[u] == std::numeric_limits<int>::max())
            {
                continue;
            }
            for (int edge = row_ptr_[u]; edge < row_ptr_[u + 1]; ++edge)
            {
                const int v = col_idx_[edge];
                if (distances[v] == distances[u] + 1 && predecessors[v] == -1)
                {
                    predecessors[v] = u;
                }
            }
        }
        return predecessors;
    }

    GraphConfig config_;
    std::vector<int> row_ptr_;
    std::vector<int> col_idx_;
    std::vector<float> weights_;
    bool weighted_ = false;
};

GraphEngine::GraphEngine(const GraphConfig &config)
    : impl_(std::make_unique<Impl>(config))
{}

GraphEngine::~GraphEngine() = default;

void GraphEngine::setGraph(const std::vector<std::pair<int, int>> &edges)
{
    impl_->setGraph(edges);
}

void GraphEngine::setWeightedGraph(const std::vector<std::pair<int, int>> &edges,
                                   const std::vector<float> &weights)
{
    impl_->setWeightedGraph(edges, weights);
}

GraphTraversalResult GraphEngine::bfs(int source)
{
    return impl_->bfs(source);
}

GraphTraversalResult GraphEngine::dijkstra(int source)
{
    return impl_->dijkstra(source);
}

std::vector<int> GraphEngine::connectedComponents()
{
    return impl_->connectedComponents();
}

std::vector<float> GraphEngine::allPairsShortestPaths()
{
    return impl_->allPairsShortestPaths();
}

std::vector<std::pair<float, float>> GraphEngine::computeGraphPersistence()
{
    return impl_->computeGraphPersistence();
}

GraphHardwareInfo detectGraphHardware()
{
    GraphHardwareInfo info{};
    int device_count = 0;
    const cudaError_t count_status = cudaGetDeviceCount(&device_count);
    if (count_status != cudaSuccess || device_count <= 0)
    {
        cudaGetLastError();
        return info;
    }

    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess)
    {
        cudaGetLastError();
        return info;
    }

    info.has_gpu = true;
    info.has_cugraph = false;
    info.gpu_memory = prop.totalGlobalMem;
    info.max_vertices = static_cast<int>(
        std::min<std::size_t>(std::numeric_limits<int>::max(),
                              std::max<std::size_t>(1, prop.totalGlobalMem / (64 * sizeof(int)))));
    info.max_edges = static_cast<int>(std::min<std::size_t>(
        std::numeric_limits<int>::max(),
        std::max<std::size_t>(1, prop.totalGlobalMem / (8 * (sizeof(int) + sizeof(float))))));
    return info;
}

} // namespace nerve::graphs

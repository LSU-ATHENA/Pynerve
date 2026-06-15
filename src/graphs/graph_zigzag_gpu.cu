#include "graph_algorithms_detail.hpp"
#include "nerve/graphs/gpu_graphs.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace nerve::graphs::gpu
{
namespace
{

std::size_t checkedDoubledSize(std::size_t value, const char *context)
{
    if (value > std::numeric_limits<std::size_t>::max() / 2)
    {
        throw std::length_error(context);
    }
    return value * 2;
}

} // namespace

class GPUZigzagGraphPersistence::Impl
{
public:
    explicit Impl(int max_vertices)
        : max_vertices_(max_vertices)
        , open_component_births_(static_cast<std::size_t>(std::max(0, max_vertices)), 0.0f)
    {
        if (max_vertices_ < 0)
        {
            throw std::invalid_argument("max_vertices must be non-negative");
        }
    }

    void updateGraph(const std::vector<std::pair<int, int>> &added_edges,
                     const std::vector<std::pair<int, int>> &removed_edges, float time)
    {
        if (!std::isfinite(time))
        {
            throw std::invalid_argument("graph update time must be finite");
        }
        for (const auto &[u, v] : added_edges)
        {
            addEdge(u, v, time);
        }
        for (const auto &[u, v] : removed_edges)
        {
            removeEdge(u, v, time);
        }
    }

    std::vector<Barcode> getBarcodes() const { return barcodes_; }

private:
    static std::uint64_t encodeEdge(int u, int v)
    {
        if (u > v)
        {
            std::swap(u, v);
        }
        return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(u)) << 32) |
               static_cast<std::uint32_t>(v);
    }

    bool validEdge(int u, int v) const
    {
        return u >= 0 && v >= 0 && u < max_vertices_ && v < max_vertices_ && u != v;
    }

    int componentCount() const
    {
        std::vector<int> row_ptr(static_cast<std::size_t>(max_vertices_) + 1, 0);
        std::vector<int> col_idx;
        const std::size_t edge_entries =
            checkedDoubledSize(active_edges_.size(), "zigzag graph edge list size overflows");
        col_idx.reserve(edge_entries);
        for (std::uint64_t encoded : active_edges_)
        {
            const int u = static_cast<int>(encoded >> 32);
            const int v = static_cast<int>(encoded & 0xffffffffu);
            ++row_ptr[static_cast<std::size_t>(u) + 1];
            ++row_ptr[static_cast<std::size_t>(v) + 1];
        }
        for (int i = 0; i < max_vertices_; ++i)
        {
            row_ptr[static_cast<std::size_t>(i) + 1] += row_ptr[static_cast<std::size_t>(i)];
        }
        std::vector<int> cursor = row_ptr;
        col_idx.resize(edge_entries);
        for (std::uint64_t encoded : active_edges_)
        {
            const int u = static_cast<int>(encoded >> 32);
            const int v = static_cast<int>(encoded & 0xffffffffu);
            col_idx[static_cast<std::size_t>(cursor[u]++)] = v;
            col_idx[static_cast<std::size_t>(cursor[v]++)] = u;
        }
        const auto labels = detail::cpuConnectedComponents(row_ptr, col_idx);
        const auto components = std::unordered_set<int>(labels.begin(), labels.end()).size();
        return detail::checkedIntSize(components, "zigzag component count exceeds int range");
    }

    void addEdge(int u, int v, float time)
    {
        if (!validEdge(u, v))
        {
            throw std::out_of_range("dynamic graph edge endpoint is outside the vertex range");
        }
        const int before = componentCount();
        if (!active_edges_.insert(encodeEdge(u, v)).second)
        {
            return;
        }
        const int merged = std::max(0, before - componentCount());
        for (int i = 0; i < merged && !open_component_births_.empty(); ++i)
        {
            const float birth = open_component_births_.back();
            open_component_births_.pop_back();
            barcodes_.push_back({birth, time, 0});
        }
    }

    void removeEdge(int u, int v, float time)
    {
        if (!validEdge(u, v))
        {
            throw std::out_of_range("dynamic graph edge endpoint is outside the vertex range");
        }
        const int before = componentCount();
        if (active_edges_.erase(encodeEdge(u, v)) == 0)
        {
            return;
        }
        const int split = std::max(0, componentCount() - before);
        for (int i = 0; i < split; ++i)
        {
            open_component_births_.push_back(time);
        }
    }

    int max_vertices_ = 0;
    std::vector<Barcode> barcodes_;
    std::vector<float> open_component_births_;
    std::unordered_set<std::uint64_t> active_edges_;
};

GPUZigzagGraphPersistence::GPUZigzagGraphPersistence(int max_vertices)
    : impl_(std::make_unique<Impl>(max_vertices))
{}

GPUZigzagGraphPersistence::~GPUZigzagGraphPersistence() = default;

void GPUZigzagGraphPersistence::updateGraph(const std::vector<std::pair<int, int>> &added_edges,
                                            const std::vector<std::pair<int, int>> &removed_edges,
                                            float time)
{
    impl_->updateGraph(added_edges, removed_edges, time);
}

std::vector<GPUZigzagGraphPersistence::Barcode> GPUZigzagGraphPersistence::getBarcodes() const
{
    return impl_->getBarcodes();
}

GraphGPUBenchmark benchmarkGraphGPU(int num_vertices, int num_edges)
{
    if (num_vertices <= 0 || num_edges < 0)
    {
        throw std::invalid_argument("benchmark graph dimensions must be positive");
    }

    std::vector<std::pair<int, int>> edges;
    edges.reserve(static_cast<std::size_t>(num_edges));
    for (int edge = 0; edge < num_edges; ++edge)
    {
        const int u = edge % num_vertices;
        const int stride = 1 + (edge / num_vertices) % std::max(1, num_vertices - 1);
        edges.emplace_back(u, (u + stride) % num_vertices);
    }

    std::vector<int> row_ptr(static_cast<std::size_t>(num_vertices) + 1, 0);
    for (const auto &[u, v] : edges)
    {
        (void)v;
        ++row_ptr[static_cast<std::size_t>(u) + 1];
    }
    for (int i = 0; i < num_vertices; ++i)
    {
        row_ptr[static_cast<std::size_t>(i) + 1] += row_ptr[static_cast<std::size_t>(i)];
    }
    std::vector<int> cursor = row_ptr;
    std::vector<int> col_idx(edges.size());
    for (const auto &[u, v] : edges)
    {
        col_idx[static_cast<std::size_t>(cursor[u]++)] = v;
    }

    GraphGPUBenchmark bench{};
    bench.num_vertices = num_vertices;
    bench.num_edges =
        detail::checkedIntSize(col_idx.size(), "graph benchmark edge count exceeds int range");

    const auto cpu_bfs_start = std::chrono::steady_clock::now();
    const auto cpu_dist = detail::cpuBfs(row_ptr, col_idx, 0);
    bench.cpu_bfs_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cpu_bfs_start)
            .count();

    const auto cpu_cc_start = std::chrono::steady_clock::now();
    const auto cpu_labels = detail::cpuConnectedComponents(row_ptr, col_idx);
    bench.cpu_cc_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cpu_cc_start)
            .count();

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0)
    {
        bench.gpu_bfs_ms = 0.0;
        bench.gpu_cc_ms = 0.0;
        bench.speedup_bfs = 0.0;
        bench.speedup_cc = 0.0;
        return bench;
    }

    GPUGraph graph(
        num_vertices,
        detail::checkedIntSize(col_idx.size(), "graph benchmark edge count exceeds int range"));
    graph.setCSR(row_ptr, col_idx, {});

    const auto gpu_bfs_start = std::chrono::steady_clock::now();
    const auto gpu_dist = graph.bfs(0);
    bench.gpu_bfs_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - gpu_bfs_start)
            .count();

    const auto gpu_cc_start = std::chrono::steady_clock::now();
    const auto gpu_labels = graph.connectedComponents();
    bench.gpu_cc_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - gpu_cc_start)
            .count();

    if (gpu_dist.size() != cpu_dist.size() || gpu_labels.size() != cpu_labels.size())
    {
        throw std::runtime_error("GPU graph benchmark returned inconsistent result sizes");
    }
    bench.speedup_bfs = bench.gpu_bfs_ms > 0.0 ? bench.cpu_bfs_ms / bench.gpu_bfs_ms : 0.0;
    bench.speedup_cc = bench.gpu_cc_ms > 0.0 ? bench.cpu_cc_ms / bench.gpu_cc_ms : 0.0;
    return bench;
}

} // namespace nerve::graphs::gpu

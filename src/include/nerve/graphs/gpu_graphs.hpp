
#pragma once

/**
 * @file gpu_graphs.hpp
 * @brief CUDA-backed graph traversal and graph-filtration helpers.
 *
 * Provides CUDA implementations for CSR graph traversal, connected
 * components, and Floyd-Warshall all-pairs shortest paths when CUDA is
 * enabled. Dynamic graph persistence currently tracks exact H0 connectivity
 * intervals for edge insertions and removals.
 *
 * Throughput varies with graph density, batch size, and device architecture.
 */

#include "nerve/config.hpp"
#include "nerve/core.hpp"

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace nerve::graphs
{

// GPU Graph Algorithms

namespace gpu
{

#if HAS_CUDA
/**
 * @brief CUDA graph using CSR adjacency storage.
 */
class GPUGraph
{
public:
    GPUGraph(int num_vertices, int num_edges);
    ~GPUGraph();

    void setCSR(const std::vector<int> &row_ptr, const std::vector<int> &col_idx,
                const std::vector<float> &weights);

    std::vector<int> bfs(int source);
    std::vector<int> connectedComponents();
    std::vector<float> allPairsShortestPaths();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief Exact H0 dynamic graph persistence for CUDA graph workflows.
 */
class GPUZigzagGraphPersistence
{
public:
    GPUZigzagGraphPersistence(int max_vertices);
    ~GPUZigzagGraphPersistence();

    void updateGraph(const std::vector<std::pair<int, int>> &added_edges,
                     const std::vector<std::pair<int, int>> &removed_edges, float time);

    struct Barcode
    {
        float birth;
        float death;
        int dimension;
    };

    std::vector<Barcode> getBarcodes() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief Benchmark CPU baselines against CUDA graph kernels when a device is available.
 */
struct GraphGPUBenchmark
{
    double cpu_bfs_ms;
    double gpu_bfs_ms;
    double cpu_cc_ms;
    double gpu_cc_ms;
    double speedup_bfs;
    double speedup_cc;
    int num_vertices;
    int num_edges;
};

GraphGPUBenchmark benchmarkGraphGPU(int num_vertices, int num_edges);
#endif

} // namespace gpu

// Parallel Graph Filtration

namespace parallel
{

/**
 * @brief Parallel graph filtration computation
 */
class ParallelGraphFiltration
{
public:
    static std::vector<float> computeFiltration(const std::vector<std::vector<int>> &adjacency_list,
                                                int num_threads = 0)
    {
#ifdef _OPENMP
        if (num_threads > 0)
        {
            omp_set_num_threads(num_threads);
        }
#else
        (void)num_threads;
#endif

        std::vector<float> filtration(adjacency_list.size(), 0.0f);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) if (adjacency_list.size() > 512)
#endif
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(adjacency_list.size()); ++i)
        {
            filtration[static_cast<std::size_t>(i)] =
                static_cast<float>(adjacency_list[static_cast<std::size_t>(i)].size());
        }
        return filtration;
    }
};

} // namespace parallel

// Unified Interface

#if HAS_CUDA
/**
 * @brief Graph configuration
 */
struct GraphConfig
{
    int num_vertices;
    int num_edges;
    bool use_gpu = true;
    bool use_parallel = true;
    bool weighted = false;
};

/**
 * @brief Graph traversal result
 */
struct GraphTraversalResult
{
    std::vector<int> distances;
    std::vector<int> predecessors;
    double computation_time_ms;
};

/**
 * @brief Unified graph engine
 */
class GraphEngine
{
public:
    explicit GraphEngine(const GraphConfig &config);
    ~GraphEngine();

    void setGraph(const std::vector<std::pair<int, int>> &edges);
    void setWeightedGraph(const std::vector<std::pair<int, int>> &edges,
                          const std::vector<float> &weights);

    GraphTraversalResult bfs(int source);
    GraphTraversalResult dijkstra(int source);
    std::vector<int> connectedComponents();
    std::vector<float> allPairsShortestPaths();

    std::vector<std::pair<float, float>> computeGraphPersistence();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief Hardware capability detection
 */
struct GraphHardwareInfo
{
    bool has_gpu;
    bool has_cugraph;
    size_t gpu_memory;
    int max_vertices;
    int max_edges;
};

GraphHardwareInfo detectGraphHardware();
#endif

} // namespace nerve::graphs

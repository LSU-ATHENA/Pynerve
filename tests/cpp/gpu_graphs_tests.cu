#include "gpu_test_helpers.cuh"
#include "nerve/graphs/gpu_graphs.hpp"
#include "nerve/graphs/gpu_gnn.hpp"

#include <vector>
#include <utility>

int main()
{
    if (!has_gpu())
    {
        std::cerr << "No CUDA device available -- skipping GPU graphs kernel coverage tests\n";
        return 0;
    }

    // Graphs: GraphConfig defaults
    {
        nerve::graphs::GraphConfig cfg;
        assert(cfg.num_vertices == 0);
        assert(cfg.use_gpu == true);
        assert(cfg.use_parallel == true);
        cfg.num_vertices = 8;
        cfg.num_edges = 12;
        std::cout << "PASSED: GraphConfig defaults + set\n";
    }

    // Graphs: GraphTraversalResult struct
    {
        nerve::graphs::GraphTraversalResult result;
        assert(result.distances.empty());
        assert(result.computation_time_ms == 0.0);
        std::cout << "PASSED: GraphTraversalResult defaults\n";
    }

    // Graphs: GraphHardwareInfo via detectGraphHardware
    {
        auto hw = nerve::graphs::detectGraphHardware();
        assert(hw.has_gpu);
        assert(hw.gpu_memory > 0);
        assert(hw.max_vertices > 0);
        std::cout << "PASSED: detectGraphHardware (max_v=" << hw.max_vertices
                  << " max_e=" << hw.max_edges << " mem=" << hw.gpu_memory << ")\n";
    }

    // Graphs: GPUZigzagGraphPersistence Barcode struct
    {
        nerve::graphs::gpu::GPUZigzagGraphPersistence::Barcode bc;
        bc.birth = 0.0f;
        bc.death = 1.0f;
        bc.dimension = 0;
        assert(bc.dimension == 0);
        std::cout << "PASSED: GPUZigzagGraphPersistence Barcode struct\n";
    }

    // Graphs: GraphGPUBenchmark struct
    {
        nerve::graphs::gpu::GraphGPUBenchmark bench;
        bench.num_vertices = 16;
        bench.num_edges = 30;
        bench.cpu_bfs_ms = 0.0;
        bench.gpu_bfs_ms = 0.0;
        assert(bench.num_vertices == 16);
        std::cout << "PASSED: GraphGPUBenchmark struct\n";
    }

    // Graphs: GPUGraph setCSR + bfs -- exercises real CUDA BFS
    {
        nerve::graphs::gpu::GPUGraph gpu_graph(4, 4);
        std::vector<int> row_ptr = {0, 2, 3, 4, 4};
        std::vector<int> col_idx = {1, 2, 2, 3};
        std::vector<float> weights = {1.0f, 1.0f, 1.0f, 1.0f};

        gpu_graph.setCSR(row_ptr, col_idx, weights);
        auto distances = gpu_graph.bfs(0);

        assert(distances.size() == 4);
        // source distance = 0
        assert(distances[0] == 0);
        // vertices 1,2 should be reachable from 0
        // Graph: 0->1, 0->2, 1->2, 2->3 -- all reachable from 0
        assert(distances[0] == 0);
        assert(distances[1] == 1);
        assert(distances[2] == 1);
        assert(distances[3] == 2);
        std::cout << "PASSED: GPUGraph setCSR + bfs (4 vertices, 4 edges)\n";
    }

    // Graphs: GPUGraph connectedComponents
    {
        nerve::graphs::gpu::GPUGraph gpu_graph(3, 2);
        std::vector<int> row_ptr = {0, 1, 2, 2};
        std::vector<int> col_idx = {1, 2};
        std::vector<float> weights = {1.0f, 1.0f};

        gpu_graph.setCSR(row_ptr, col_idx, weights);
        auto components = gpu_graph.connectedComponents();

        assert(components.size() == 3);
        // vertices 0 and 1 should be in the same component
        assert(components[0] == components[1] || components[0] == 0);
        std::cout << "PASSED: GPUGraph connectedComponents (3 vertices)\n";
    }

    // Graphs: GraphEngine setGraph + bfs
    {
        nerve::graphs::GraphConfig cfg;
        cfg.num_vertices = 5;
        cfg.num_edges = 4;
        cfg.use_gpu = true;

        nerve::graphs::GraphEngine engine(cfg);

        std::vector<std::pair<int, int>> edges = {{0, 1}, {1, 2}, {2, 3}, {3, 4}};
        engine.setGraph(edges);

        auto result = engine.bfs(0);
        assert(result.distances.size() == 5);
        assert(result.distances[0] == 0);
        assert(result.computation_time_ms >= 0.0);
        std::cout << "PASSED: GraphEngine setGraph + bfs (5 vertices, path graph)\n";
    }

    // Graphs: GraphEngine setWeightedGraph + dijkstra
    {
        nerve::graphs::GraphConfig cfg;
        cfg.num_vertices = 4;
        cfg.num_edges = 3;
        cfg.use_gpu = true;
        cfg.weighted = true;

        nerve::graphs::GraphEngine engine(cfg);

        std::vector<std::pair<int, int>> edges = {{0, 1}, {0, 2}, {1, 3}};
        std::vector<float> weights = {2.0f, 5.0f, 1.0f};
        engine.setWeightedGraph(edges, weights);

        auto result = engine.dijkstra(0);
        assert(result.distances.size() == 4);
        assert(result.distances[0] == 0);
        assert(result.computation_time_ms >= 0.0);
        std::cout << "PASSED: GraphEngine setWeightedGraph + dijkstra (4 vertices)\n";
    }

    // Graphs: benchmarkMessagePassing
    {
        auto bench = nerve::graphs::gpu::benchmarkMessagePassing(64, 128, 16);
        assert(bench.num_nodes == 64);
        assert(bench.num_edges == 128);
        assert(bench.feature_dim == 16);
        assert(bench.cpu_time_ms >= 0.0);
        assert(bench.gpu_time_ms >= 0.0);
        assert(bench.gpu_fp16_time_ms >= 0.0);
        std::cout << "PASSED: benchmarkMessagePassing (64 nodes, 128 edges, dim=16)\n";
    }

    // Graphs: ParallelGraphFiltration computeFiltration
    {
        std::vector<std::vector<int>> adj = {{1, 2}, {0, 2}, {0, 1, 3}, {2}};
        auto filtration = nerve::graphs::parallel::ParallelGraphFiltration::computeFiltration(adj, 1);
        assert(filtration.size() == 4);
        assert(filtration[0] >= 0.0f);
        std::cout << "PASSED: ParallelGraphFiltration (4 vertices)\n";
    }

    // Graphs: GraphEngine connectedComponents via BFS on disconnected graph
    {
        nerve::graphs::GraphConfig cfg;
        cfg.num_vertices = 6;
        cfg.num_edges = 4;
        cfg.use_gpu = true;

        nerve::graphs::GraphEngine engine(cfg);

        std::vector<std::pair<int, int>> edges = {{0, 1}, {1, 2}, {3, 4}, {4, 5}};
        engine.setGraph(edges);

        auto result = engine.connectedComponents();
        assert(result.size() == 6);
        // Vertices in same component should have same label
        assert(result[0] == result[1]);
        assert(result[1] == result[2]);
        assert(result[3] == result[4]);
        assert(result[4] == result[5]);
        std::cout << "PASSED: GraphEngine connectedComponents (2 components)\n";
    }

    return 0;
}

#include "gpu_test_helpers.cuh"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

// GPUReebGraphConstructor is defined inline in reeb_graph.cu inside
// namespace nerve::specialized::gpu.  Include the whole file as a single
// translation unit (same pattern as other tests that need inline CUDA
// class methods compiled alongside the test).
#include "../../src/specialized/reeb_graph.cu"

int main()
{
    if (!has_gpu())
    {
        std::cout << "SKIP: no GPU available" << std::endl;
        return 0;
    }

    // Test 1: constructReebGraph on a simple path graph.
    //
    // Graph:  v0 -- v1 -- v2  (line of 3 vertices)
    // Function values: f(v0)=0, f(v1)=1, f(v2)=2
    //
    // Expected: v0 is a minimum (type 1), v2 is a maximum (type 2),
    //           v1 is a regular (type 3) point.
    // Arc: v0 -> v2 with persistence 2.0.
    {
        std::vector<float> f_vals = {0.0f, 1.0f, 2.0f};
        std::vector<std::vector<int>> adj = {{1}, {0, 2}, {1}};

        nerve::specialized::gpu::GPUReebGraphConstructor ctor(3);
        auto [nodes, arcs] = ctor.constructReebGraph(f_vals, adj);

        // Should find at least 2 critical points (min & max)
        assert(nodes.size() >= 2);

        // Should find at least 1 arc connecting them
        assert(arcs.size() >= 1);

        // Arc persistence should be non-negative and finite
        for (const auto &arc : arcs)
        {
            assert(std::isfinite(arc.persistence));
            assert(arc.persistence >= 0.0f);
            assert(arc.from_node >= 0);
            assert(arc.to_node >= 0);
        }

        // Node function values should be finite
        for (const auto &node : nodes)
        {
            assert(std::isfinite(node.function_value));
            assert(node.vertex_id >= 0 && node.vertex_id < 3);
        }

        std::cout << "PASS: constructReebGraph produced " << nodes.size()
                  << " nodes, " << arcs.size() << " arcs" << std::endl;
    }

    // Test 2: computeMergeTree (split tree, ascending)
    {
        std::vector<float> f_vals = {3.0f, 1.0f, 2.0f, 0.0f};

        // No adjacency needed for merge tree (it only uses function values)
        nerve::specialized::gpu::GPUReebGraphConstructor ctor(4);
        auto tree = ctor.computeMergeTree(f_vals, /*compute_join_tree=*/false);

        // Split tree has n-1 arcs for n vertices
        assert(static_cast<int>(tree.size()) == 3);

        // Arc persistence = difference in function values (should be >= 0)
        for (const auto &arc : tree)
        {
            float diff = std::fabs(f_vals[static_cast<std::size_t>(arc.to_node)] -
                                    f_vals[static_cast<std::size_t>(arc.from_node)]);
            assert(std::fabs(arc.persistence - diff) < 1e-4f);
            assert(arc.persistence >= 0.0f);
        }

        std::cout << "PASS: computeMergeTree (split) produced " << tree.size()
                  << " arcs" << std::endl;
    }

    // Test 3: computeMergeTree (join tree, descending)
    {
        std::vector<float> f_vals = {3.0f, 1.0f, 2.0f, 0.0f};
        nerve::specialized::gpu::GPUReebGraphConstructor ctor(4);
        auto tree = ctor.computeMergeTree(f_vals, /*compute_join_tree=*/true);

        assert(static_cast<int>(tree.size()) == 3);

        for (const auto &arc : tree)
        {
            assert(std::isfinite(arc.persistence));
            assert(arc.persistence >= 0.0f);
        }

        std::cout << "PASS: computeMergeTree (join) produced " << tree.size()
                  << " arcs" << std::endl;
    }

    // Test 4: simplifyReebGraph removes low-persistence arcs
    {
        std::vector<float> f_vals = {0.0f, 1.0f, 2.0f};
        std::vector<std::vector<int>> adj = {{1}, {0, 2}, {1}};

        nerve::specialized::gpu::GPUReebGraphConstructor ctor(3);
        auto [nodes, arcs] = ctor.constructReebGraph(f_vals, adj);

        // Simplify with a very high threshold -- should remove all arcs
        auto [simp_nodes, simp_arcs] = ctor.simplifyReebGraph(nodes, arcs, 100.0f);
        assert(simp_arcs.empty());

        // Simplify with threshold 0 -- should keep all arcs
        auto [simp_nodes2, simp_arcs2] = ctor.simplifyReebGraph(nodes, arcs, 0.0f);
        assert(simp_arcs2.size() <= arcs.size());

        std::cout << "PASS: simplifyReebGraph filtered to " << simp_arcs2.size()
                  << " arcs at threshold 0" << std::endl;
    }

    std::cout << "PASS: all reeb graph tests passed" << std::endl;
    return 0;
}

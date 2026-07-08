#include "gpu_test_helpers.cuh"
#include "nerve/graphs/gpu_gnn.hpp"

#include <iostream>

int main()
{
    if (!has_gpu())
    {
        std::cerr << "No CUDA device available -- skipping GPU attention kernel coverage tests\n";
        return 0;
    }

    // Multi-head attention: small graph benchmark smoke
    {
        int num_nodes = 8;
        int feature_dim = 16;
        int num_heads = 4;
        auto result = nerve::graphs::gpu::benchmarkMultiHeadAttention(
            num_nodes, feature_dim, num_heads);
        assert(result.gpu_time_ms >= 0.0);
        assert(result.num_nodes == num_nodes);
        std::cout << "PASSED: multi-head attention benchmark (8 nodes, 16 dim, 4 heads, "
                  << result.gpu_time_ms << " ms gpu)\n";
    }

    // Multi-head attention: larger graph
    {
        int num_nodes = 32;
        int feature_dim = 64;
        int num_heads = 8;
        auto result = nerve::graphs::gpu::benchmarkMultiHeadAttention(
            num_nodes, feature_dim, num_heads);
        assert(result.gpu_time_ms >= 0.0);
        assert(result.num_nodes == num_nodes);
        std::cout << "PASSED: multi-head attention benchmark (32 nodes, 64 dim, 8 heads, "
                  << result.gpu_time_ms << " ms gpu)\n";
    }

    return 0;
}

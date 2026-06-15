
#pragma once

#include <vector>

namespace nerve::graphs
{
namespace gpu
{

struct MessagePassingBenchmark
{
    double cpu_time_ms;
    double gpu_time_ms;
    double gpu_fp16_time_ms;
    double speedup;
    int num_nodes;
    int num_edges;
    int feature_dim;
};

MessagePassingBenchmark benchmarkMessagePassing(int num_nodes, int num_edges, int feature_dim);

struct AttentionBenchmark
{
    double cpu_time_ms;
    double gpu_time_ms;
    double tensor_core_time_ms;
    double speedup;
    int num_nodes;
    int feature_dim;
    int num_heads;
};

AttentionBenchmark benchmarkMultiHeadAttention(int num_nodes, int feature_dim, int num_heads);

} // namespace gpu
} // namespace nerve::graphs

#pragma once

#include "nerve/core_types.hpp"
#include "nerve/graphs/graph.hpp"

#include <vector>

namespace nerve::graphs
{

class PersistentHomology
{
public:
    explicit PersistentHomology(const WeightedGraph &graph);
    std::vector<std::pair<int, double>> compute();
};

struct TopologyResult
{
    std::vector<std::vector<int>> components;
    int num_cycles;
};

TopologyResult computeTopology(const WeightedGraph &graph);

class NeuralLayer
{
public:
    NeuralLayer(size_t in_features, size_t out_features);
    std::vector<double> forward(const std::vector<double> &input);
};

} // namespace nerve::graphs

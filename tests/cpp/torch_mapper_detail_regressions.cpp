#include "nerve/torch/mapper.hpp"

#include <cstddef>
#include <iostream>
#include <random>
#include <vector>

#ifdef NERVE_HAS_TORCH
#include <torch/torch.h>
#endif

namespace
{

#ifdef NERVE_HAS_TORCH

using at::Tensor;
using nerve::torch::ClustererType;
using nerve::torch::CoverType;
using nerve::torch::Mapper;
using nerve::torch::MapperConfig;
using nerve::torch::MapperEdge;
using nerve::torch::MapperGraph;
using nerve::torch::MapperNode;

bool check_mapper_construction()
{
    MapperConfig config;
    config.cover_resolution = 5;
    config.cover_overlap = 0.3;
    config.clusterer = ClustererType::SINGLE_LINKAGE;
    Mapper mapper(config);
    return true;
}

bool check_mapper_graph_connectivity()
{
    MapperConfig config;
    config.cover_resolution = 4;
    config.cover_overlap = 0.4;
    config.clusterer = ClustererType::CONNECTED;
    Mapper mapper(config);
    Tensor points = at::randn({20, 3});
    MapperGraph graph = mapper.fit_transform(points);
    if (graph.nodes.empty() && graph.edges.empty())
    {
        std::cerr << "graph should have some nodes or edges\n";
        return false;
    }
    return true;
}

bool check_mapper_node_pullback()
{
    MapperConfig config;
    config.cover_resolution = 4;
    config.cover_overlap = 0.3;
    config.clusterer = ClustererType::SINGLE_LINKAGE;
    Mapper mapper(config);
    Tensor points = at::randn({15, 2});
    MapperGraph graph = mapper.fit_transform(points);
    for (const auto &node : graph.nodes)
    {
        if (node.point_indices.empty())
        {
            std::cerr << "node should have points\n";
            return false;
        }
        if (node.centroid.numel() == 0)
        {
            std::cerr << "node should have centroid\n";
            return false;
        }
    }
    for (const auto &edge : graph.edges)
    {
        if (edge.weight < 0.0)
        {
            std::cerr << "edge weight should be non-negative\n";
            return false;
        }
    }
    return true;
}

bool check_mapper_default_config()
{
    Mapper mapper;
    Tensor points = at::randn({10, 2});
    MapperGraph graph = mapper.fit_transform(points);
    (void)graph;
    return true;
}

#endif

} // namespace

int main()
{
#ifndef NERVE_HAS_TORCH
    std::cerr << "SKIP: NERVE_HAS_TORCH not defined\n";
    return 0;
#else
    if (!check_mapper_construction())
    {
        std::cerr << "FAIL: mapper_construction\n";
        return 1;
    }
    if (!check_mapper_graph_connectivity())
    {
        std::cerr << "FAIL: mapper_graph_connectivity\n";
        return 1;
    }
    if (!check_mapper_node_pullback())
    {
        std::cerr << "FAIL: mapper_node_pullback\n";
        return 1;
    }
    if (!check_mapper_default_config())
    {
        std::cerr << "FAIL: mapper_default_config\n";
        return 1;
    }
    return 0;
#endif
}

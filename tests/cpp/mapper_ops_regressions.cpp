#include "nerve/algorithms/mapper.hpp"
#include "nerve/common/accelerated_types.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core_types.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <random>
#include <span>
#include <tuple>
#include <vector>

namespace
{

using nerve::Dimension;
using nerve::Field;
using nerve::Size;
using nerve::core::BufferView;

constexpr double kTol = 1e-10;

bool check_mapper_node_construction()
{
    nerve::algorithms::MapperNode<float> node;
    node.id = 0;
    node.point_indices = {0, 1, 2};
    node.centroid = {0.5f, 0.5f};

    if (node.size() != 3)
    {
        std::cerr << "expected node size 3, got " << node.size() << "\n";
        return false;
    }
    if (node.centroid.size() != 2)
    {
        std::cerr << "expected centroid dim 2\n";
        return false;
    }

    return true;
}

bool check_mapper_edge_construction()
{
    nerve::algorithms::MapperEdge<float> edge;
    edge.source = 0;
    edge.target = 1;
    edge.weight = 0.5f;
    edge.overlap_size = 3;

    if (edge.weight <= 0.0f)
    {
        std::cerr << "edge weight should be positive\n";
        return false;
    }
    if (edge.source >= edge.target && edge.source != edge.target)
    {
        std::cerr << "expected source < target\n";
        return false;
    }

    return true;
}

bool check_mapper_graph_basic()
{
    nerve::algorithms::MapperGraph<float> graph;

    nerve::algorithms::MapperNode<float> n0, n1, n2;
    n0.id = 0;
    n0.point_indices = {0, 1};
    n1.id = 1;
    n1.point_indices = {2, 3};
    n2.id = 2;
    n2.point_indices = {4, 5};

    graph.nodes = {n0, n1, n2};

    graph.edges = {{0, 1, 0.5f, 1}, {1, 2, 0.3f, 1}};

    graph.build_adjacency();

    if (graph.n_nodes() != 3)
    {
        std::cerr << "expected 3 nodes\n";
        return false;
    }
    if (graph.n_edges() != 2)
    {
        std::cerr << "expected 2 edges\n";
        return false;
    }
    if (graph.adjacency_list.size() != 3)
    {
        std::cerr << "adjacency list size wrong\n";
        return false;
    }

    return true;
}

bool check_filter_function()
{
    auto filter = std::make_shared<nerve::algorithms::EccentricityFilter<float>>();

    std::vector<float> points = {0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f};
    auto values = filter->apply(points, 3, 2);

    if (values.size() != 3)
    {
        std::cerr << "expected 3 filter values, got " << values.size() << "\n";
        return false;
    }
    if (filter->name().empty())
    {
        std::cerr << "filter name should not be empty\n";
        return false;
    }

    for (auto v : values)
    {
        if (v < 0.0f)
        {
            std::cerr << "eccentricity should be >= 0\n";
            return false;
        }
    }

    return true;
}

bool check_pca_filter()
{
    auto filter = std::make_shared<nerve::algorithms::PCAFilter<float>>(2);

    std::vector<float> points = {0.0f, 0.0f, 1.0f, 0.0f, 2.0f, 0.0f, 3.0f, 0.0f};
    auto values = filter->apply(points, 4, 2);

    if (values.empty())
    {
        std::cerr << "PCA filter should produce values\n";
        return false;
    }

    if (filter->name().find("pca") == std::string::npos)
    {
        std::cerr << "PCA filter name should contain 'pca'\n";
        return false;
    }

    return true;
}

bool check_dbscan_clustering()
{
    nerve::algorithms::DBSCANClustering<float>::Config cfg;
    cfg.eps = 0.5f;
    cfg.min_samples = 2;

    nerve::algorithms::DBSCANClustering<float> clusterer(cfg);

    std::vector<float> points = {0.0f, 0.0f, 0.1f, 0.1f, 5.0f, 5.0f, 5.1f, 5.1f};
    auto labels = clusterer.cluster(points, 4, 2);

    if (labels.size() != 4)
    {
        std::cerr << "expected 4 labels, got " << labels.size() << "\n";
        return false;
    }

    if (clusterer.name() != "dbscan")
    {
        std::cerr << "expected dbscan name\n";
        return false;
    }

    return true;
}

bool check_mapper_algorithm_config()
{
    nerve::algorithms::MapperAlgorithm<float>::Config cfg;
    cfg.filter = std::make_shared<nerve::algorithms::EccentricityFilter<float>>();
    cfg.cover_resolution = 10;
    cfg.cover_overlap = 0.25f;
    cfg.clusterer = std::make_shared<nerve::algorithms::ConnectedComponentsClustering<float>>();
    cfg.return_graph = true;

    if (!cfg.filter)
    {
        std::cerr << "filter should be set\n";
        return false;
    }
    if (!cfg.clusterer)
    {
        std::cerr << "clusterer should be set\n";
        return false;
    }
    if (cfg.cover_resolution <= 0)
    {
        std::cerr << "resolution should be > 0\n";
        return false;
    }
    if (cfg.cover_overlap <= 0.0f || cfg.cover_overlap >= 1.0f)
    {
        std::cerr << "overlap should be in (0,1)\n";
        return false;
    }

    return true;
}

} // namespace

int main()
{
    if (!check_mapper_node_construction())
    {
        std::cerr << "FAIL: mapper node construction\n";
        return 1;
    }
    if (!check_mapper_edge_construction())
    {
        std::cerr << "FAIL: mapper edge construction\n";
        return 1;
    }
    if (!check_mapper_graph_basic())
    {
        std::cerr << "FAIL: mapper graph basic\n";
        return 1;
    }
    if (!check_filter_function())
    {
        std::cerr << "FAIL: filter function\n";
        return 1;
    }
    if (!check_pca_filter())
    {
        std::cerr << "FAIL: pca filter\n";
        return 1;
    }
    if (!check_dbscan_clustering())
    {
        std::cerr << "FAIL: dbscan clustering\n";
        return 1;
    }
    if (!check_mapper_algorithm_config())
    {
        std::cerr << "FAIL: mapper algorithm config\n";
        return 1;
    }
    return 0;
}

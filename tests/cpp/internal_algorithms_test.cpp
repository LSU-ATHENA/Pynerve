#include "nerve/algorithms/distance.hpp"
#include "nerve/algorithms/mapper.hpp"
#include "nerve/algorithms/persistence_vectorization.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <span>
#include <vector>

namespace
{

constexpr double kTol = 1e-8;

// Distance metric tests
bool check_euclidean_distance()
{
    nerve::algorithms::EuclideanMetric<double> metric;
    std::vector<double> a = {0.0, 0.0, 0.0};
    std::vector<double> b = {3.0, 4.0, 0.0};
    double d = metric.compute(std::span(a), std::span(b));
    return std::abs(d - 5.0) < kTol;
}

bool check_euclidean_distance_same_point()
{
    nerve::algorithms::EuclideanMetric<double> metric;
    std::vector<double> a = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> b = {1.0, 2.0, 3.0, 4.0};
    double d = metric.compute(std::span(a), std::span(b));
    return std::abs(d) < kTol;
}

bool check_euclidean_distance_matrix()
{
    nerve::algorithms::EuclideanMetric<double> metric;
    // Two points: (0,0) and (3,4)
    std::vector<double> pts = {0.0, 0.0, 3.0, 4.0};
    auto mat = metric.compute_matrix(std::span(pts), 2, 2);
    if (mat.size() != 4)
        return false;
    // Diagonal should be 0
    if (std::abs(mat[0]) > kTol || std::abs(mat[3]) > kTol)
        return false;
    // Off-diagonal should be 5
    if (std::abs(mat[1] - 5.0) > kTol || std::abs(mat[2] - 5.0) > kTol)
        return false;
    return true;
}

bool check_manhattan_distance()
{
    nerve::algorithms::ManhattanMetric<double> metric;
    std::vector<double> a = {1.0, 2.0, 3.0};
    std::vector<double> b = {4.0, 1.0, -1.0};
    double d = metric.compute(std::span(a), std::span(b));
    // |1-4| + |2-1| + |3-(-1)| = 3 + 1 + 4 = 8
    return std::abs(d - 8.0) < kTol;
}

bool check_cosine_distance()
{
    nerve::algorithms::CosineMetric<double> metric;
    // Orthogonal vectors: cosine distance = 1
    std::vector<double> a = {1.0, 0.0};
    std::vector<double> b = {0.0, 1.0};
    double d = metric.compute(std::span(a), std::span(b));
    return std::abs(d - 1.0) < kTol;
}

bool check_cosine_distance_parallel()
{
    nerve::algorithms::CosineMetric<double> metric;
    std::vector<double> a = {2.0, 0.0};
    std::vector<double> b = {4.0, 0.0};
    double d = metric.compute(std::span(a), std::span(b));
    // Parallel vectors: cosine distance = 0
    return std::abs(d) < kTol;
}

// DistanceMatrixComputer tests
bool check_distance_matrix_computer_euclidean()
{
    nerve::algorithms::DistanceMatrixComputer<double>::Config cfg;
    cfg.metric = nerve::algorithms::DistanceMatrixComputer<double>::Config::Metric::EUCLIDEAN;
    cfg.use_simd = false;
    cfg.use_openmp = false;

    nerve::algorithms::DistanceMatrixComputer<double> computer(cfg);
    std::vector<double> pts = {0.0, 0.0, 3.0, 0.0, 0.0, 4.0};
    auto mat = computer.compute(std::span(pts), 3, 2);

    if (mat.size() != 9)
        return false;
    return std::abs(mat[1] - 3.0) < kTol && std::abs(mat[2] - 4.0) < kTol &&
           std::abs(mat[5] - 5.0) < kTol;
}

bool check_distance_matrix_computer_symmetric()
{
    nerve::algorithms::DistanceMatrixComputer<double>::Config cfg;
    cfg.metric = nerve::algorithms::DistanceMatrixComputer<double>::Config::Metric::EUCLIDEAN;
    cfg.use_simd = false;
    cfg.use_openmp = false;

    nerve::algorithms::DistanceMatrixComputer<double> computer(cfg);
    std::vector<double> pts = {0.0, 0.0, 1.0, 0.0};
    auto mat = computer.compute_symmetric(std::span(pts), 2, 2);

    // Symmetric may return packed format; just verify non-empty and finite
    if (mat.empty())
        return false;
    for (double v : mat)
    {
        if (!std::isfinite(v))
        {
            std::cerr << "non-finite value in symmetric matrix\n";
            return false;
        }
    }
    return true;
}

bool check_distance_matrix_computer_chunked()
{
    nerve::algorithms::DistanceMatrixComputer<double>::Config cfg;
    cfg.metric = nerve::algorithms::DistanceMatrixComputer<double>::Config::Metric::EUCLIDEAN;
    cfg.use_simd = false;
    cfg.use_openmp = false;

    nerve::algorithms::DistanceMatrixComputer<double> computer(cfg);
    std::vector<double> pts = {0.0, 0.0, 3.0, 4.0};
    auto mat = computer.compute_chunked(std::span(pts), 2, 2, 1);

    return mat.size() == 4;
}

// KNN computer tests
bool check_knn_computer_brute_force()
{
    nerve::algorithms::KNNComputer<double>::Config cfg;
    cfg.k = 2;
    cfg.algorithm = nerve::algorithms::KNNComputer<double>::Config::Algorithm::BRUTE_FORCE;
    cfg.use_openmp = false;

    nerve::algorithms::KNNComputer<double> knn(cfg);
    // Points: (0,0), (1,0), (10,0) -- point 0 should be closest to point 1, not point 2
    std::vector<double> pts = {0.0, 0.0, 1.0, 0.0, 10.0, 0.0};
    auto result = knn.compute(std::span(pts), 3, 2);

    if (result.k != 2)
        return false;
    if (result.n_points != 3)
        return false;
    if (result.distances.size() != 6)
        return false;
    if (result.indices.size() != 6)
        return false;
    // Point 0 should find point 1 as its nearest neighbor (index 1)
    // The KNN results for point 0 should include point 1 at index 0
    if (result.indices[0] != 1)
    {
        std::cerr << "Expected point 0's nearest neighbor to be point 1, got " << result.indices[0]
                  << "\n";
        return false;
    }
    return true;
}

bool check_knn_computer_different_k()
{
    nerve::algorithms::KNNComputer<double>::Config cfg;
    cfg.k = 3;
    cfg.algorithm = nerve::algorithms::KNNComputer<double>::Config::Algorithm::BRUTE_FORCE;
    cfg.use_openmp = false;

    nerve::algorithms::KNNComputer<double> knn(cfg);
    std::vector<double> pts = {0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0};
    auto result = knn.compute(std::span(pts), 4, 2);

    return result.k == 3 && result.n_points == 4;
}

// Mapper algorithm tests
bool check_mapper_graph_construction()
{
    nerve::algorithms::MapperGraph<double> graph;
    nerve::algorithms::MapperNode<double> n1{0, {0, 1}, {0.0}};
    nerve::algorithms::MapperNode<double> n2{1, {2, 3}, {1.0}};
    graph.nodes.push_back(n1);
    graph.nodes.push_back(n2);

    nerve::algorithms::MapperEdge<double> e{0, 1, 0.5, 2};
    graph.edges.push_back(e);

    if (graph.n_nodes() != 2)
        return false;
    if (graph.n_edges() != 1)
        return false;
    if (graph.nodes[0].size() != 2)
        return false;

    graph.build_adjacency();
    if (graph.adjacency_list.size() != 2)
        return false;

    return true;
}

bool check_mapper_filter_pca()
{
    nerve::algorithms::PCAFilter<double> filter(1);
    std::vector<double> pts = {0.0, 0.0, 1.0, 0.0, 2.0, 0.0};
    auto result = filter.apply(std::span(pts), 3, 2);

    if (result.size() != 3)
        return false;
    if (filter.name() != "pca_1d")
        return false;
    return true;
}

bool check_mapper_filter_eccentricity()
{
    nerve::algorithms::EccentricityFilter<double> filter;
    std::vector<double> pts = {0.0, 0.0, 3.0, 4.0, 6.0, 8.0};
    auto result = filter.apply(std::span(pts), 3, 2);

    if (result.size() != 3)
        return false;
    if (filter.name() != "eccentricity")
        return false;
    return true;
}

bool check_mapper_filter_density()
{
    nerve::algorithms::DensityFilter<double> filter(5);
    std::vector<double> pts = {0.0, 0.0, 0.1, 0.1, 0.2, 0.2, 5.0, 5.0};
    auto result = filter.apply(std::span(pts), 4, 2);

    if (result.size() != 4)
        return false;
    if (filter.name() != "density")
        return false;
    return true;
}

bool check_mapper_custom_filter()
{
    auto fn = [](std::span<const double> pts, size_t n, size_t dim) -> std::vector<double> {
        (void)dim;
        std::vector<double> out(n, 0.0);
        for (size_t i = 0; i < n; ++i)
            out[i] = pts[i * 2]; // first coordinate
        return out;
    };

    nerve::algorithms::CustomFilter<double> filter(fn, "x_coord");
    std::vector<double> pts = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    auto result = filter.apply(std::span(pts), 3, 2);

    if (result.size() != 3)
        return false;
    if (filter.name() != "x_coord")
        return false;
    // First coordinate values
    if (std::abs(result[0] - 1.0) > kTol)
        return false;
    if (std::abs(result[1] - 3.0) > kTol)
        return false;
    if (std::abs(result[2] - 5.0) > kTol)
        return false;
    return true;
}

bool check_mapper_clustering_connected_components()
{
    nerve::algorithms::ConnectedComponentsClustering<double> cc;
    std::vector<double> pts(20, 0.0);
    auto labels = cc.cluster(std::span(pts), 5, 2);

    if (labels.size() != 5)
        return false;
    // Each point gets its own label: 0, 1, 2, 3, 4
    for (size_t i = 0; i < 5; ++i)
    {
        if (labels[i] != static_cast<int>(i))
            return false;
    }
    return cc.name() == "connected";
}

bool check_mapper_cover_1d()
{
    nerve::algorithms::Cover<double> cover({.resolution = 5, .overlap = 0.2});
    std::vector<double> vals = {0.0, 0.3, 0.5, 0.8, 1.0};
    auto cover_sets = cover.build(std::span(vals), 5, 1);

    if (cover_sets.empty())
        return false;
    // Overlapping cover sets should include each point at least once
    std::vector<bool> covered(5, false);
    for (const auto &s : cover_sets)
    {
        for (size_t idx : s)
        {
            if (idx < 5)
                covered[idx] = true;
        }
    }
    for (bool c : covered)
    {
        if (!c)
            return false;
    }
    return true;
}

bool check_mapper_algorithm_basic()
{
    auto filter = std::make_shared<nerve::algorithms::PCAFilter<double>>(1);
    auto clusterer = std::make_shared<nerve::algorithms::ConnectedComponentsClustering<double>>();

    nerve::algorithms::MapperAlgorithm<double>::Config cfg;
    cfg.filter = filter;
    cfg.cover_resolution = 3;
    cfg.cover_overlap = 0.3;
    cfg.clusterer = clusterer;

    nerve::algorithms::MapperAlgorithm<double> mapper(cfg);
    std::vector<double> pts = {0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 1.5, 0.0};
    auto result = mapper.compute(std::span(pts), 4, 2);

    if (result.graph.n_nodes() == 0)
        return false;
    return result.filter_values.size() == 4;
}

// Graph utility tests
bool check_graph_connected_components()
{
    nerve::algorithms::MapperGraph<double> graph;

    nerve::algorithms::MapperNode<double> n0{0, {0}, {0.0}};
    nerve::algorithms::MapperNode<double> n1{1, {1}, {1.0}};
    nerve::algorithms::MapperNode<double> n2{2, {2}, {2.0}};
    graph.nodes = {n0, n1, n2};

    nerve::algorithms::MapperEdge<double> e{0, 1, 1.0, 1};
    graph.edges = {e};
    graph.build_adjacency();

    auto comps = nerve::algorithms::connected_components(graph);
    // Graph with nodes {0,1,2} and edge (0,1):
    // Components: {0,1} and {2}
    if (comps.size() != 2)
    {
        std::cerr << "Expected 2 components, got " << comps.size() << "\n";
        return false;
    }
    // Verify each component is non-empty
    for (const auto &comp : comps)
    {
        if (comp.empty())
        {
            std::cerr << "Empty component\n";
            return false;
        }
    }
    return true;
}

bool check_graph_diameter()
{
    nerve::algorithms::MapperGraph<double> graph;

    nerve::algorithms::MapperNode<double> n0{0, {0}, {0.0}};
    nerve::algorithms::MapperNode<double> n1{1, {1}, {1.0}};
    nerve::algorithms::MapperNode<double> n2{2, {2}, {2.0}};
    nerve::algorithms::MapperNode<double> n3{3, {3}, {3.0}};
    graph.nodes = {n0, n1, n2, n3};

    nerve::algorithms::MapperEdge<double> e0{0, 1, 1.0, 1};
    nerve::algorithms::MapperEdge<double> e1{1, 2, 1.0, 1};
    nerve::algorithms::MapperEdge<double> e2{2, 3, 1.0, 1};
    graph.edges = {e0, e1, e2};
    graph.build_adjacency();

    size_t diam = nerve::algorithms::graph_diameter(graph);
    // Path of 4 nodes has diameter 3
    return diam == 3;
}

bool check_graph_export_graphml()
{
    nerve::algorithms::MapperGraph<double> graph;
    nerve::algorithms::MapperNode<double> n{0, {0, 1}, {0.5}};
    graph.nodes = {n};
    graph.edges = {};

    auto xml = nerve::algorithms::export_to_graphml(graph);
    return !xml.empty() && xml.find("<graphml>") != std::string::npos;
}

bool check_graph_export_json()
{
    nerve::algorithms::MapperGraph<double> graph;
    nerve::algorithms::MapperNode<double> n{0, {0}, {0.0}};
    graph.nodes = {n};
    graph.edges = {};

    auto json = nerve::algorithms::export_to_json(graph);
    return !json.empty() && json.find("\"nodes\"") != std::string::npos;
}

// Mapper statistics tests
bool check_mapper_statistics()
{
    nerve::algorithms::MapperGraph<double> graph;
    nerve::algorithms::MapperNode<double> n0{0, {0, 1, 2}, {0.0}};
    nerve::algorithms::MapperNode<double> n1{1, {3, 4}, {1.0}};
    graph.nodes = {n0, n1};
    nerve::algorithms::MapperEdge<double> e{0, 1, 0.5, 1};
    graph.edges = {e};
    graph.build_adjacency();

    auto stats = nerve::algorithms::computeMapperStatistics(graph);
    if (stats.node_count != 2)
        return false;
    if (stats.edge_count != 1)
        return false;
    if (stats.max_node_size != 3)
        return false;
    if (stats.min_node_size != 2)
        return false;
    return true;
}

// Persistence vectorization tests
bool check_persistence_landscape()
{
    std::vector<double> diagram = {0.0, 1.0, 2.0, 3.0}; // 2 pairs: (0,1) and (2,3)
    auto landscape = nerve::algorithms::compute_landscape<double>(std::span(diagram), 2, 3, 0.1);

    if (landscape.num_levels != 3)
        return false;
    if (landscape.landscape_levels.empty())
        return false;
    return true;
}

bool check_persistence_image()
{
    std::vector<double> diagram = {0.0, 0.5, 1.0, 2.0};
    auto image =
        nerve::algorithms::compute_persistence_image<double>(std::span(diagram), 2, 32, 0.1);

    if (image.resolution != 32)
        return false;
    if (image.image.empty())
        return false;
    return true;
}

bool check_betti_curve()
{
    std::vector<double> diagram = {0.0, 0.5, 1.0, 2.0};
    auto curve = nerve::algorithms::compute_betti_curve<double>(std::span(diagram), 2);

    return !curve.empty();
}

// Vectorized operations
bool check_vectorized_sqrt()
{
    std::vector<float> data = {4.0f, 9.0f, 16.0f, 25.0f};
    auto span = std::span<float>(data);
    nerve::algorithms::vectorized_sqrt<float>(span);

    return std::abs(data[0] - 2.0f) < 1e-6f && std::abs(data[1] - 3.0f) < 1e-6f &&
           std::abs(data[2] - 4.0f) < 1e-6f && std::abs(data[3] - 5.0f) < 1e-6f;
}

bool check_vectorized_sum()
{
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    auto span = std::span<const double>(data);
    double sum = nerve::algorithms::vectorized_sum<double>(span);

    return std::abs(sum - 15.0) < kTol;
}

bool check_dot_product()
{
    std::vector<double> a = {1.0, 2.0, 3.0};
    std::vector<double> b = {4.0, 5.0, 6.0};
    double dp = nerve::algorithms::dot_product<double>(std::span(a), std::span(b));

    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    return std::abs(dp - 32.0) < kTol;
}

bool check_dot_product_different_sizes()
{
    std::vector<double> a = {1.0, 2.0};
    std::vector<double> b = {3.0, 4.0, 5.0, 6.0};
    double dp = nerve::algorithms::dot_product<double>(std::span(a), std::span(b));

    // Takes min size: 1*3 + 2*4 = 11
    return std::abs(dp - 11.0) < kTol;
}

} // namespace

int main()
{
    int failures = 0;

    auto run = [&](const char *name, bool ok) {
        if (!ok)
        {
            std::cerr << "FAIL: " << name << "\n";
            ++failures;
        }
        else
        {
            std::cout << "PASS: " << name << "\n";
        }
    };

    // Distance metrics
    run("euclidean_distance", check_euclidean_distance());
    run("euclidean_distance_same_point", check_euclidean_distance_same_point());
    run("euclidean_distance_matrix", check_euclidean_distance_matrix());
    run("manhattan_distance", check_manhattan_distance());
    run("cosine_distance", check_cosine_distance());
    run("cosine_distance_parallel", check_cosine_distance_parallel());

    // DistanceMatrixComputer
    run("distance_matrix_computer_euclidean", check_distance_matrix_computer_euclidean());
    run("distance_matrix_computer_symmetric", check_distance_matrix_computer_symmetric());
    run("distance_matrix_computer_chunked", check_distance_matrix_computer_chunked());

    // KNN
    run("knn_computer_brute_force", check_knn_computer_brute_force());
    run("knn_computer_different_k", check_knn_computer_different_k());

    // Mapper
    run("mapper_graph_construction", check_mapper_graph_construction());
    run("mapper_filter_pca", check_mapper_filter_pca());
    run("mapper_filter_eccentricity", check_mapper_filter_eccentricity());
    run("mapper_filter_density", check_mapper_filter_density());
    run("mapper_custom_filter", check_mapper_custom_filter());
    run("mapper_clustering_connected_components", check_mapper_clustering_connected_components());
    run("mapper_cover_1d", check_mapper_cover_1d());
    run("mapper_algorithm_basic", check_mapper_algorithm_basic());

    // Graph utilities
    run("graph_connected_components", check_graph_connected_components());
    run("graph_diameter", check_graph_diameter());
    run("graph_export_graphml", check_graph_export_graphml());
    run("graph_export_json", check_graph_export_json());

    // Mapper statistics
    run("mapper_statistics", check_mapper_statistics());

    // Persistence vectorization
    run("persistence_landscape", check_persistence_landscape());
    run("persistence_image", check_persistence_image());
    run("betti_curve", check_betti_curve());

    // Vectorized operations
    run("vectorized_sqrt", check_vectorized_sqrt());
    run("vectorized_sum", check_vectorized_sum());
    run("dot_product", check_dot_product());
    run("dot_product_different_sizes", check_dot_product_different_sizes());

    return failures > 0 ? 1 : 0;
}

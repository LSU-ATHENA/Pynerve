#pragma once

#include "nerve/core_types.hpp"

#include <array>
#include <functional>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace nerve::algorithms
{

template <typename T = float>
struct MapperNode
{
    size_t id;
    std::vector<size_t> point_indices;
    std::vector<T> centroid;
    size_t size() const { return point_indices.size(); }
};

template <typename T = float>
struct MapperEdge
{
    size_t source;
    size_t target;
    T weight;
    size_t overlap_size;
};

template <typename T = float>
struct MapperGraph
{
    std::vector<MapperNode<T>> nodes;
    std::vector<MapperEdge<T>> edges;
    std::vector<std::vector<size_t>> adjacency_list;

    [[nodiscard]] size_t n_nodes() const { return nodes.size(); }
    [[nodiscard]] size_t n_edges() const { return edges.size(); }

    void build_adjacency();
};

template <typename T = float>
class FilterFunction
{
public:
    virtual ~FilterFunction() = default;

    [[nodiscard]] virtual std::vector<T> apply(std::span<const T> points, size_t n_points,
                                               size_t dim) const = 0;

    [[nodiscard]] virtual std::string name() const = 0;
};

template <typename T = float>
class PCAFilter : public FilterFunction<T>
{
public:
    explicit PCAFilter(int n_components = 2)
        : n_components_(n_components)
    {}

    [[nodiscard]] std::vector<T> apply(std::span<const T> points, size_t n_points,
                                       size_t dim) const override;

    [[nodiscard]] std::string name() const override
    {
        return "pca_" + std::to_string(n_components_) + "d";
    }

private:
    int n_components_;
};

template <typename T = float>
class EccentricityFilter : public FilterFunction<T>
{
public:
    [[nodiscard]] std::vector<T> apply(std::span<const T> points, size_t n_points,
                                       size_t dim) const override;

    [[nodiscard]] std::string name() const override { return "eccentricity"; }
};

template <typename T = float>
class DensityFilter : public FilterFunction<T>
{
public:
    explicit DensityFilter(int k_neighbors = 10)
        : k_neighbors_(k_neighbors)
    {}

    [[nodiscard]] std::vector<T> apply(std::span<const T> points, size_t n_points,
                                       size_t dim) const override;

    [[nodiscard]] std::string name() const override { return "density"; }

private:
    int k_neighbors_;
};

template <typename T = float>
class CustomFilter : public FilterFunction<T>
{
public:
    using FunctionType = std::function<std::vector<T>(std::span<const T>, size_t, size_t)>;

    CustomFilter(FunctionType func, std::string name)
        : func_(std::move(func))
        , name_(std::move(name))
    {}

    [[nodiscard]] std::vector<T> apply(std::span<const T> points, size_t n_points,
                                       size_t dim) const override
    {
        return func_(points, n_points, dim);
    }

    [[nodiscard]] std::string name() const override { return name_; }

private:
    FunctionType func_;
    std::string name_;
};

template <typename T = float>
class ClusteringAlgorithm
{
public:
    virtual ~ClusteringAlgorithm() = default;

    [[nodiscard]] virtual std::vector<int> cluster(std::span<const T> points, size_t n_points,
                                                   size_t dim) const = 0;

    [[nodiscard]] virtual std::string name() const = 0;
};

template <typename T = float>
class DBSCANClustering : public ClusteringAlgorithm<T>
{
public:
    struct Config
    {
        T eps = T(0.5);
        int min_samples = 5;
    };

    explicit DBSCANClustering(Config config)
        : config_(config)
    {}

    [[nodiscard]] std::vector<int> cluster(std::span<const T> points, size_t n_points,
                                           size_t dim) const override;

    [[nodiscard]] std::string name() const override { return "dbscan"; }

private:
    Config config_;

    void region_query(std::span<const T> points, size_t n_points, size_t dim, size_t point_idx,
                      std::vector<size_t> &neighbors) const;
};

template <typename T = float>
class SingleLinkageClustering : public ClusteringAlgorithm<T>
{
public:
    struct Config
    {
        T linkage_distance = T(0.5);
    };

    explicit SingleLinkageClustering(Config config)
        : config_(config)
    {}

    [[nodiscard]] std::vector<int> cluster(std::span<const T> points, size_t n_points,
                                           size_t dim) const override;

    [[nodiscard]] std::string name() const override { return "single_linkage"; }

private:
    Config config_;
};

template <typename T = float>
class ConnectedComponentsClustering : public ClusteringAlgorithm<T>
{
public:
    [[nodiscard]] std::vector<int> cluster(std::span<const T>, size_t n_points,
                                           size_t) const override
    {
        if (n_points > static_cast<size_t>(std::numeric_limits<int>::max()))
        {
            return {};
        }
        std::vector<int> labels(n_points);
        for (size_t i = 0; i < n_points; ++i)
        {
            labels[i] = static_cast<int>(i);
        }
        return labels;
    }

    [[nodiscard]] std::string name() const override { return "connected"; }
};

template <typename T = float>
class Cover
{
public:
    struct Config
    {
        int resolution = 10;
        T overlap = T(0.25);
    };

    explicit Cover(Config config)
        : config_(config)
    {}

    [[nodiscard]] std::vector<std::vector<size_t>> build(std::span<const T> filter_values,
                                                         size_t n_points, int n_filter_dims) const;

private:
    Config config_;

    [[nodiscard]] std::vector<std::vector<size_t>> build_1d(std::span<const T> filter_values,
                                                            size_t n_points) const;

    [[nodiscard]] std::vector<std::vector<size_t>>
    build_nd(std::span<const T> filter_values, size_t n_points, int n_filter_dims) const;
};

template <typename T = float>
class MapperAlgorithm
{
public:
    struct Config
    {
        std::shared_ptr<FilterFunction<T>> filter;
        int cover_resolution = 10;
        T cover_overlap = T(0.25);
        std::shared_ptr<ClusteringAlgorithm<T>> clusterer;
        bool return_graph = true;
        bool compute_meta = true;
    };

    struct Result
    {
        MapperGraph<T> graph;
        std::vector<T> filter_values;
        std::vector<std::vector<size_t>> cover_sets;
        std::unordered_map<std::string, T> metadata;
    };

    explicit MapperAlgorithm(Config config)
        : config_(config)
    {}

    [[nodiscard]] Result compute(std::span<const T> point_cloud, size_t n_points, size_t dim) const;

private:
    Config config_;

    [[nodiscard]] std::vector<T> apply_filter(std::span<const T> points, size_t n_points,
                                              size_t dim) const;

    [[nodiscard]] std::vector<std::vector<size_t>>
    build_cover(std::span<const T> filter_values, size_t n_points, int n_filter_dims) const;

    [[nodiscard]] std::vector<MapperNode<T>>
    cluster_cover_sets(std::span<const T> points, size_t dim,
                       const std::vector<std::vector<size_t>> &cover_sets) const;

    [[nodiscard]] std::vector<MapperEdge<T>>
    build_graph(const std::vector<MapperNode<T>> &nodes) const;

    [[nodiscard]] std::vector<T> compute_centroid(std::span<const T> points, size_t dim,
                                                  std::span<const size_t> indices) const;
};

template <typename T = float>
[[nodiscard]] std::vector<std::vector<size_t>> connected_components(const MapperGraph<T> &graph);

template <typename T = float>
[[nodiscard]] size_t graph_diameter(const MapperGraph<T> &graph);

template <typename T = float>
[[nodiscard]] std::string export_to_graphml(const MapperGraph<T> &graph);

template <typename T = float>
[[nodiscard]] std::string export_to_json(const MapperGraph<T> &graph);

struct MapperStatistics
{
    Size node_count = 0;
    Size edge_count = 0;
    std::vector<Size> node_sizes;
    std::vector<double> edge_weights;
    double avg_node_size = 0.0;
    Size max_node_size = 0;
    Size min_node_size = 0;
    Size median_node_size = 0;
    double edge_density = 0.0;
    double avg_overlap = 0.0;
    double median_edge_weight = 0.0;
    double max_edge_weight = 0.0;
    double min_edge_weight = 0.0;
    Size connected_components = 0;
};

template <typename T = float>
[[nodiscard]] MapperStatistics computeMapperStatistics(const MapperGraph<T> &graph);

template <typename T = float>
[[nodiscard]] std::string mapperGraphToDot(const MapperGraph<T> &graph);

template <typename T = float>
[[nodiscard]] std::string mapperGraphToJson(const MapperGraph<T> &graph);

template <typename T = float>
void computeMapperLayout(const MapperGraph<T> &graph, std::vector<std::array<T, 2>> &positions,
                         Size iterations = 50);

template <typename T = float>
[[nodiscard]] double computeMapperGlobalEfficiency(const MapperGraph<T> &graph);

template <typename T = float>
[[nodiscard]] double computeMapperModularity(const MapperGraph<T> &graph);

template <typename T = float>
[[nodiscard]] double computeMapperCoverage(const MapperGraph<T> &graph, Size total_points);

template <typename T = float>
[[nodiscard]] double computeMapperNodeSizeStdDev(const MapperGraph<T> &graph);

template <typename T = float>
[[nodiscard]] double computeMapperClusteringCoefficient(const MapperGraph<T> &graph);

} // namespace nerve::algorithms

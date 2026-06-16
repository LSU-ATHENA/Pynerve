#pragma once

#if __has_include(<torch/torch.h>)
#include <torch/torch.h>
#endif

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nerve::torch
{

enum class CoverType
{
    GRID,
    BALL,
    INTERVAL
};

enum class ClustererType
{
    DBSCAN,
    SINGLE_LINKAGE,
    KMEANS,
    CONNECTED
};

struct MapperConfig
{
    std::string filter_function = "pca_2d";
    int64_t pca_components = 2;

    CoverType cover_type = CoverType::GRID;
    int64_t cover_resolution = 10;
    double cover_overlap = 0.25;

    ClustererType clusterer = ClustererType::DBSCAN;
    double dbscan_eps = 0.5;
    int64_t dbscan_min_samples = 5;
    int64_t kmeans_k = 5;

    bool return_graph = true;
    bool return_node_info = false;
};

at::Tensor filter_pca(const at::Tensor &point_cloud, int64_t n_components);

at::Tensor filter_eccentricity(const at::Tensor &point_cloud);

at::Tensor filter_density(const at::Tensor &point_cloud, double bandwidth = 0.5);

typedef std::function<at::Tensor(const at::Tensor &)> FilterFunction;

std::vector<std::pair<at::Tensor, at::Tensor>> cover_grid(const at::Tensor &filter_values,
                                                          int64_t resolution, double overlap);

std::vector<std::pair<at::Tensor, double>> cover_ball(const at::Tensor &filter_values,
                                                      int64_t n_balls, double overlap);

std::vector<std::vector<int64_t>> cluster_dbscan(const at::Tensor &points, double eps,
                                                 int64_t min_samples);

std::vector<std::vector<int64_t>> cluster_single_linkage(const at::Tensor &points,
                                                         double threshold);

std::vector<std::vector<int64_t>> cluster_connected(const at::Tensor &points, double threshold);

struct MapperNode
{
    int64_t id;
    std::vector<int64_t> point_indices;
    at::Tensor centroid;
    at::Tensor filter_centroid;
    int64_t cover_index;
};

struct MapperEdge
{
    int64_t source;
    int64_t target;
    double weight;
};

struct MapperGraph
{
    std::vector<MapperNode> nodes;
    std::vector<MapperEdge> edges;

    std::vector<std::pair<int64_t, int64_t>> to_edge_list() const;

    at::Tensor to_adjacency_matrix() const;
};

class Mapper
{
public:
    Mapper(const MapperConfig &config = MapperConfig{});

    MapperGraph fit_transform(const at::Tensor &point_cloud);

    void set_filter_function(FilterFunction func);

    at::Tensor get_last_filter_values() const;

private:
    MapperConfig config_;
    FilterFunction custom_filter_;
    at::Tensor last_filter_values_;

    at::Tensor apply_filter(const at::Tensor &point_cloud);
    std::vector<std::vector<int64_t>>
    pull_back_and_cluster(const at::Tensor &point_cloud, const at::Tensor &filter_values,
                          const std::vector<std::pair<at::Tensor, at::Tensor>> &cover_sets);
    MapperGraph build_graph(const std::vector<std::vector<int64_t>> &clusters,
                            const at::Tensor &point_cloud, const at::Tensor &filter_values);
};

MapperGraph quick_mapper(const at::Tensor &point_cloud, const std::string &filter = "pca_2d",
                         int64_t resolution = 10, double overlap = 0.25);

} // namespace nerve::torch

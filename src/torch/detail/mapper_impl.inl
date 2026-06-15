#include "nerve/torch/mapper.hpp"

#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nerve::torch {
namespace {

constexpr int64_t kMaxCoverCells = 1'000'000;
constexpr int kMaxKMeansIterations = 32;

at::Tensor require_point_matrix(const at::Tensor& tensor, const char* name) {
    if (!tensor.defined()) {
        throw std::invalid_argument(std::string(name) + " must be defined");
    }
    if (tensor.dim() != 2) {
        throw std::invalid_argument(std::string(name) + " must be a rank-2 tensor");
    }
    if (tensor.size(0) <= 0) {
        throw std::invalid_argument(std::string(name) + " must be non-empty");
    }
    if (tensor.size(1) <= 0) {
        throw std::invalid_argument(std::string(name) + " must have at least one feature");
    }
    if (!tensor.is_floating_point()) {
        throw std::invalid_argument(std::string(name) + " must have a floating-point dtype");
    }
    if (!at::isfinite(tensor).all().item<bool>()) {
        throw std::invalid_argument(std::string(name) + " must contain only finite values");
    }
    return tensor;
}

at::Tensor require_filter_matrix(const at::Tensor& tensor) {
    if (!tensor.defined()) {
        throw std::invalid_argument("filter_values must be defined");
    }
    if (!tensor.is_floating_point()) {
        throw std::invalid_argument("filter_values must have a floating-point dtype");
    }
    if (!at::isfinite(tensor).all().item<bool>()) {
        throw std::invalid_argument("filter_values must contain only finite values");
    }
    if (tensor.dim() == 1) {
        if (tensor.size(0) <= 0) {
            throw std::invalid_argument("filter_values must be non-empty");
        }
        return tensor.unsqueeze(1);
    }
    if (tensor.dim() != 2) {
        throw std::invalid_argument("filter_values must be a rank-1 or rank-2 tensor");
    }
    if (tensor.size(0) <= 0) {
        throw std::invalid_argument("filter_values must be non-empty");
    }
    if (tensor.size(1) <= 0) {
        throw std::invalid_argument("filter_values must have at least one feature");
    }
    return tensor;
}

at::Tensor as_cpu_double_matrix(const at::Tensor& tensor, const char* name) {
    return require_point_matrix(tensor, name)
        .to(at::kCPU)
        .to(at::kDouble)
        .contiguous();
}

double require_finite_nonnegative(double value, const char* name) {
    if (!std::isfinite(value) || value < 0.0) {
        throw std::invalid_argument(std::string(name) + " must be finite and non-negative");
    }
    return value;
}

double require_finite_positive(double value, const char* name) {
    if (!std::isfinite(value) || value <= 0.0) {
        throw std::invalid_argument(std::string(name) + " must be finite and positive");
    }
    return value;
}

double normalized_overlap(double overlap) {
    if (!std::isfinite(overlap) || overlap < 0.0 || overlap >= 1.0) {
        throw std::invalid_argument("cover_overlap must be finite and in [0, 1)");
    }
    return overlap;
}

int64_t checked_resolution(int64_t resolution, const char* name) {
    if (resolution <= 0) {
        throw std::invalid_argument(std::string(name) + " must be positive");
    }
    return resolution;
}

int64_t checked_cell_count(int64_t resolution, int64_t dim) {
    int64_t cells = 1;
    for (int64_t i = 0; i < dim; ++i) {
        if (cells > kMaxCoverCells / resolution) {
            throw std::invalid_argument("grid cover would allocate too many cells");
        }
        cells *= resolution;
    }
    return cells;
}

void validate_mapper_config(const MapperConfig& config) {
    if (config.pca_components <= 0) {
        throw std::invalid_argument("pca_components must be positive");
    }
    checked_resolution(config.cover_resolution, "cover_resolution");
    normalized_overlap(config.cover_overlap);
    require_finite_positive(config.dbscan_eps, "dbscan_eps");
    if (config.dbscan_min_samples <= 0) {
        throw std::invalid_argument("dbscan_min_samples must be positive");
    }
    if (config.kmeans_k <= 0) {
        throw std::invalid_argument("kmeans_k must be positive");
    }
}

void require_matching_rows(const at::Tensor& point_cloud, const at::Tensor& filter_values) {
    if (point_cloud.size(0) != filter_values.size(0)) {
        throw std::invalid_argument("filter_values must have one row per point");
    }
}

at::Tensor make_index_tensor(const std::vector<int64_t>& indices, const at::Device& device) {
    return at::tensor(indices, at::TensorOptions().dtype(at::kLong).device(device));
}

std::vector<std::vector<int64_t>> cluster_kmeans(const at::Tensor& points, int64_t requested_k) {
    auto cpu_points = as_cpu_double_matrix(points, "points");
    const int64_t n = cpu_points.size(0);
    const int64_t dim = cpu_points.size(1);
    if (n == 0) {
        return {};
    }
    if (requested_k <= 0) {
        throw std::invalid_argument("kmeans_k must be positive");
    }

    const int64_t k = std::min(requested_k, n);
    auto points_a = cpu_points.accessor<double, 2>();
    std::vector<std::vector<double>> centers(static_cast<size_t>(k),
                                             std::vector<double>(static_cast<size_t>(dim)));
    for (int64_t c = 0; c < k; ++c) {
        for (int64_t d = 0; d < dim; ++d) {
            centers[static_cast<size_t>(c)][static_cast<size_t>(d)] = points_a[c][d];
        }
    }

    std::vector<int64_t> labels(static_cast<size_t>(n), 0);
    for (int iter = 0; iter < kMaxKMeansIterations; ++iter) {
        bool changed = false;
        for (int64_t i = 0; i < n; ++i) {
            double best_distance = std::numeric_limits<double>::infinity();
            int64_t best_cluster = 0;
            for (int64_t c = 0; c < k; ++c) {
                double distance_sq = 0.0;
                for (int64_t d = 0; d < dim; ++d) {
                    const double diff =
                        points_a[i][d] - centers[static_cast<size_t>(c)][static_cast<size_t>(d)];
                    distance_sq += diff * diff;
                }
                if (distance_sq < best_distance) {
                    best_distance = distance_sq;
                    best_cluster = c;
                }
            }
            changed = changed || labels[static_cast<size_t>(i)] != best_cluster;
            labels[static_cast<size_t>(i)] = best_cluster;
        }
        if (!changed && iter > 0) {
            break;
        }

        std::vector<std::vector<double>> next_centers(
            static_cast<size_t>(k), std::vector<double>(static_cast<size_t>(dim), 0.0));
        std::vector<int64_t> counts(static_cast<size_t>(k), 0);
        for (int64_t i = 0; i < n; ++i) {
            const auto cluster = labels[static_cast<size_t>(i)];
            ++counts[static_cast<size_t>(cluster)];
            for (int64_t d = 0; d < dim; ++d) {
                next_centers[static_cast<size_t>(cluster)][static_cast<size_t>(d)] +=
                    points_a[i][d];
            }
        }
        for (int64_t c = 0; c < k; ++c) {
            if (counts[static_cast<size_t>(c)] == 0) {
                continue;
            }
            for (int64_t d = 0; d < dim; ++d) {
                next_centers[static_cast<size_t>(c)][static_cast<size_t>(d)] /=
                    static_cast<double>(counts[static_cast<size_t>(c)]);
            }
            centers[static_cast<size_t>(c)] = std::move(next_centers[static_cast<size_t>(c)]);
        }
    }

    std::vector<std::vector<int64_t>> clusters(static_cast<size_t>(k));
    for (int64_t i = 0; i < n; ++i) {
        clusters[static_cast<size_t>(labels[static_cast<size_t>(i)])].push_back(i);
    }
    clusters.erase(std::remove_if(clusters.begin(), clusters.end(),
                                  [](const auto& cluster) { return cluster.empty(); }),
                   clusters.end());
    return clusters;
}

std::vector<std::vector<int64_t>> cluster_subset(const MapperConfig& config,
                                                 const at::Tensor& subset_points) {
    switch (config.clusterer) {
        case ClustererType::DBSCAN:
            return cluster_dbscan(subset_points, config.dbscan_eps, config.dbscan_min_samples);
        case ClustererType::SINGLE_LINKAGE:
        case ClustererType::CONNECTED:
            return cluster_single_linkage(subset_points, config.dbscan_eps);
        case ClustererType::KMEANS:
            return cluster_kmeans(subset_points, config.kmeans_k);
    }
    throw std::invalid_argument("unknown mapper clusterer");
}

std::vector<std::vector<int64_t>> pull_back_ball_cover(
    const at::Tensor& point_cloud,
    const at::Tensor& filter_values,
    const std::vector<std::pair<at::Tensor, double>>& cover_sets,
    const MapperConfig& config) {
    const auto values = require_filter_matrix(filter_values);
    require_matching_rows(point_cloud, values);
    std::vector<std::vector<int64_t>> all_clusters;

    for (const auto& [center, radius] : cover_sets) {
        const auto distances = (values - center).norm(2, 1);
        const auto indices = at::nonzero(distances <= radius).flatten();
        if (indices.numel() == 0) {
            continue;
        }

        const auto subset_points = point_cloud.index_select(0, indices.to(point_cloud.device()));
        auto clusters = cluster_subset(config, subset_points);
        const auto indices_cpu = indices.to(at::kCPU).to(at::kLong).contiguous();
        const auto indices_a = indices_cpu.accessor<int64_t, 1>();

        for (auto& cluster : clusters) {
            std::vector<int64_t> global_cluster;
            global_cluster.reserve(cluster.size());
            for (int64_t local_idx : cluster) {
                if (local_idx >= 0 && local_idx < indices_cpu.size(0)) {
                    global_cluster.push_back(indices_a[local_idx]);
                }
            }
            if (!global_cluster.empty()) {
                all_clusters.push_back(std::move(global_cluster));
            }
        }
    }

    return all_clusters;
}

std::vector<std::vector<int64_t>> normalized_clusters(
    const std::vector<std::vector<int64_t>>& clusters) {
    std::vector<std::vector<int64_t>> result;
    result.reserve(clusters.size());
    for (auto cluster : clusters) {
        std::ranges::sort(cluster);
        cluster.erase(std::unique(cluster.begin(), cluster.end()), cluster.end());
        if (!cluster.empty()) {
            result.push_back(std::move(cluster));
        }
    }
    return result;
}

}  // namespace

at::Tensor filter_pca(const at::Tensor& point_cloud, int64_t n_components) {
    const auto points = require_point_matrix(point_cloud, "point_cloud");
    if (n_components <= 0) {
        throw std::invalid_argument("n_components must be positive");
    }

    const int64_t n = points.size(0);
    const int64_t dim = points.size(1);
    const int64_t computed_components = std::min(n_components, dim);
    if (n == 0 || n == 1) {
        return points.new_zeros({n, n_components});
    }

    const auto mean = points.mean(0, true);
    const auto centered = points - mean;
    const auto cov = at::mm(centered.t(), centered) / static_cast<double>(n - 1);
    const auto eig = at::linalg_eigh(cov);
    const auto eigenvectors = std::get<1>(eig);
    const auto top_vectors = eigenvectors.slice(1, dim - computed_components, dim);
    auto projected = at::mm(centered, top_vectors);

    if (computed_components < n_components) {
        const auto padding = points.new_zeros({n, n_components - computed_components});
        projected = at::cat({projected, padding}, 1);
    }

    return projected;
}

at::Tensor filter_eccentricity(const at::Tensor& point_cloud) {
    const auto points = require_point_matrix(point_cloud, "point_cloud");
    if (points.size(0) == 0) {
        return points.new_empty({0});
    }
    const auto centered = points - points.mean(0, true);
    return centered.norm(2, 1);
}

at::Tensor filter_density(const at::Tensor& point_cloud, double bandwidth) {
    const auto points = require_point_matrix(point_cloud, "point_cloud");
    require_finite_positive(bandwidth, "bandwidth");

    const int64_t n = points.size(0);
    auto densities = points.new_zeros({n});
    const double denom = 2.0 * bandwidth * bandwidth;
    for (int64_t i = 0; i < n; ++i) {
        const auto distances = (points - points[i]).norm(2, 1);
        densities[i] = at::exp(-distances.pow(2) / denom).sum();
    }

    return densities;
}

std::vector<std::pair<at::Tensor, at::Tensor>> cover_grid(const at::Tensor& filter_values,
                                                          int64_t resolution, double overlap) {
    const auto values = require_filter_matrix(filter_values);
    const int64_t n = values.size(0);
    const int64_t dim = values.size(1);
    if (n == 0) {
        return {};
    }

    resolution = checked_resolution(resolution, "cover resolution");
    checked_cell_count(resolution, dim);
    overlap = normalized_overlap(overlap);

    const auto min_result = values.min(0);
    const auto max_result = values.max(0);
    auto min_vals = std::get<0>(min_result);
    auto max_vals = std::get<0>(max_result);
    auto ranges = max_vals - min_vals;
    const auto padding = ranges * 0.05;
    min_vals = min_vals - padding;
    max_vals = max_vals + padding;
    ranges = max_vals - min_vals;

    const auto step = ranges / static_cast<double>(resolution);
    const auto overlap_size = step * overlap;

    std::vector<std::pair<at::Tensor, at::Tensor>> cover_sets;
    cover_sets.reserve(static_cast<size_t>(checked_cell_count(resolution, dim)));
    std::vector<int64_t> cell(static_cast<size_t>(dim), 0);

    auto emit_cell = [&](auto&& self, int64_t axis) -> void {
        if (axis == dim) {
            auto min_bound = min_vals.clone();
            auto max_bound = max_vals.clone();
            for (int64_t d = 0; d < dim; ++d) {
                min_bound[d] = min_vals[d] + step[d] * static_cast<double>(cell[d]) -
                               overlap_size[d];
                max_bound[d] = min_vals[d] + step[d] * static_cast<double>(cell[d] + 1) +
                               overlap_size[d];
            }
            cover_sets.emplace_back(at::clamp(min_bound, min_vals, max_vals),
                                    at::clamp(max_bound, min_vals, max_vals));
            return;
        }
        for (int64_t i = 0; i < resolution; ++i) {
            cell[static_cast<size_t>(axis)] = i;
            self(self, axis + 1);
        }
    };
    emit_cell(emit_cell, 0);

    return cover_sets;
}

std::vector<std::pair<at::Tensor, double>> cover_ball(const at::Tensor& filter_values,
                                                      int64_t n_balls, double overlap) {
    const auto values = require_filter_matrix(filter_values);
    const int64_t n = values.size(0);
    if (n == 0) {
        return {};
    }

    n_balls = checked_resolution(n_balls, "ball cover count");
    overlap = normalized_overlap(overlap);

    const int64_t center_count = std::min(n_balls, n);
    std::vector<int64_t> center_indices(static_cast<size_t>(center_count));
    for (int64_t i = 0; i < center_count; ++i) {
        center_indices[static_cast<size_t>(i)] =
            center_count == 1 ? 0 : (i * (n - 1)) / (center_count - 1);
    }

    const auto centers =
        values.index_select(0, make_index_tensor(center_indices, values.device()));
    const auto distances = at::cdist(values, centers);
    const auto nearest = std::get<0>(distances.min(1));
    const double radius = nearest.max().item<double>() * (1.0 + overlap);

    std::vector<std::pair<at::Tensor, double>> cover_sets;
    cover_sets.reserve(static_cast<size_t>(center_count));
    for (int64_t i = 0; i < center_count; ++i) {
        cover_sets.emplace_back(centers[i], radius);
    }
    return cover_sets;
}

std::vector<std::vector<int64_t>> cluster_dbscan(const at::Tensor& points, double eps,
                                                 int64_t min_samples) {
    require_finite_positive(eps, "eps");
    if (min_samples <= 0) {
        throw std::invalid_argument("min_samples must be positive");
    }

    const auto cpu_points = as_cpu_double_matrix(points, "points");
    const int64_t n = cpu_points.size(0);
    const int64_t dim = cpu_points.size(1);
    if (n == 0) {
        return {};
    }

    const auto points_a = cpu_points.accessor<double, 2>();
    const double eps_sq = eps * eps;
    std::vector<bool> visited(static_cast<size_t>(n), false);
    std::vector<int64_t> cluster_ids(static_cast<size_t>(n), -1);
    int64_t current_cluster = 0;

    auto region_query = [&](int64_t point_idx) {
        std::vector<int64_t> neighbors;
        for (int64_t i = 0; i < n; ++i) {
            double dist_sq = 0.0;
            for (int64_t d = 0; d < dim; ++d) {
                const double diff = points_a[point_idx][d] - points_a[i][d];
                dist_sq += diff * diff;
            }
            if (dist_sq <= eps_sq) {
                neighbors.push_back(i);
            }
        }
        return neighbors;
    };

    for (int64_t i = 0; i < n; ++i) {
        if (visited[static_cast<size_t>(i)]) {
            continue;
        }
        visited[static_cast<size_t>(i)] = true;

        auto neighbors = region_query(i);
        if (neighbors.size() < static_cast<size_t>(min_samples)) {
            continue;
        }

        std::queue<int64_t> to_process;
        cluster_ids[static_cast<size_t>(i)] = current_cluster;
        for (int64_t neighbor : neighbors) {
            to_process.push(neighbor);
        }

        while (!to_process.empty()) {
            const int64_t candidate = to_process.front();
            to_process.pop();

            if (!visited[static_cast<size_t>(candidate)]) {
                visited[static_cast<size_t>(candidate)] = true;
                auto expanded = region_query(candidate);
                if (expanded.size() >= static_cast<size_t>(min_samples)) {
                    for (int64_t neighbor : expanded) {
                        to_process.push(neighbor);
                    }
                }
            }

            if (cluster_ids[static_cast<size_t>(candidate)] == -1) {
                cluster_ids[static_cast<size_t>(candidate)] = current_cluster;
            }
        }

        ++current_cluster;
    }

    std::vector<std::vector<int64_t>> clusters(static_cast<size_t>(current_cluster));
    for (int64_t i = 0; i < n; ++i) {
        const auto cluster_id = cluster_ids[static_cast<size_t>(i)];
        if (cluster_id >= 0) {
            clusters[static_cast<size_t>(cluster_id)].push_back(i);
        }
    }
    return clusters;
}

std::vector<std::vector<int64_t>> cluster_single_linkage(const at::Tensor& points,
                                                         double threshold) {
    require_finite_nonnegative(threshold, "threshold");
    const auto cpu_points = as_cpu_double_matrix(points, "points");
    const int64_t n = cpu_points.size(0);
    const int64_t dim = cpu_points.size(1);
    if (n == 0) {
        return {};
    }

    std::vector<int64_t> parent(static_cast<size_t>(n));
    std::iota(parent.begin(), parent.end(), 0);

    auto find = [&](int64_t x, auto&& find_ref) -> int64_t {
        while (parent[static_cast<size_t>(x)] != x) {
            parent[static_cast<size_t>(x)] =
                parent[static_cast<size_t>(parent[static_cast<size_t>(x)])];
            x = parent[static_cast<size_t>(x)];
        }
        return x;
    };

    auto unite = [&](int64_t x, int64_t y) {
        const auto px = find(x, find);
        const auto py = find(y, find);
        if (px != py) {
            parent[static_cast<size_t>(py)] = px;
        }
    };

    const auto points_a = cpu_points.accessor<double, 2>();
    const double threshold_sq = threshold * threshold;
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = i + 1; j < n; ++j) {
            double dist_sq = 0.0;
            for (int64_t d = 0; d < dim; ++d) {
                const double diff = points_a[i][d] - points_a[j][d];
                dist_sq += diff * diff;
            }
            if (dist_sq <= threshold_sq) {
                unite(i, j);
            }
        }
    }

    std::unordered_map<int64_t, std::vector<int64_t>> components;
    for (int64_t i = 0; i < n; ++i) {
        components[find(i, find)].push_back(i);
    }

    std::vector<std::vector<int64_t>> clusters;
    clusters.reserve(components.size());
    for (auto& [_, members] : components) {
        clusters.push_back(std::move(members));
    }
    std::ranges::sort(clusters, {}, [](const auto& cluster) { return cluster.front(); });
    return clusters;
}

std::vector<std::vector<int64_t>> cluster_connected(const at::Tensor& points, double threshold) {
    return cluster_single_linkage(points, threshold);
}

Mapper::Mapper(const MapperConfig& config) : config_(config) {
    validate_mapper_config(config_);
}

void Mapper::set_filter_function(FilterFunction func) {
    custom_filter_ = std::move(func);
}

at::Tensor Mapper::apply_filter(const at::Tensor& point_cloud) {
    if (custom_filter_) {
        return require_filter_matrix(custom_filter_(point_cloud));
    }

    if (config_.filter_function == "pca") {
        return filter_pca(point_cloud, config_.pca_components);
    }
    if (config_.filter_function == "pca_2d") {
        return filter_pca(point_cloud, 2);
    }
    if (config_.filter_function == "pca_1d") {
        return filter_pca(point_cloud, 1);
    }
    if (config_.filter_function == "eccentricity") {
        return filter_eccentricity(point_cloud);
    }
    if (config_.filter_function == "density") {
        return filter_density(point_cloud);
    }

    throw std::invalid_argument("unknown mapper filter function: " + config_.filter_function);
}

std::vector<std::vector<int64_t>> Mapper::pull_back_and_cluster(
    const at::Tensor& point_cloud,
    const at::Tensor& filter_values,
    const std::vector<std::pair<at::Tensor, at::Tensor>>& cover_sets) {
    const auto values = require_filter_matrix(filter_values);
    require_matching_rows(point_cloud, values);
    std::vector<std::vector<int64_t>> all_clusters;

    for (const auto& [min_bound, max_bound] : cover_sets) {
        const auto in_cover = ((values >= min_bound) & (values <= max_bound)).all(1);
        const auto indices = at::nonzero(in_cover).flatten();
        if (indices.numel() == 0) {
            continue;
        }

        const auto subset_points = point_cloud.index_select(0, indices.to(point_cloud.device()));
        auto clusters = cluster_subset(config_, subset_points);
        const auto indices_cpu = indices.to(at::kCPU).to(at::kLong).contiguous();
        const auto indices_a = indices_cpu.accessor<int64_t, 1>();

        for (auto& cluster : clusters) {
            std::vector<int64_t> global_cluster;
            global_cluster.reserve(cluster.size());
            for (int64_t local_idx : cluster) {
                if (local_idx >= 0 && local_idx < indices_cpu.size(0)) {
                    global_cluster.push_back(indices_a[local_idx]);
                }
            }
            if (!global_cluster.empty()) {
                all_clusters.push_back(std::move(global_cluster));
            }
        }
    }

    return all_clusters;
}

MapperGraph Mapper::build_graph(const std::vector<std::vector<int64_t>>& clusters,
                                const at::Tensor& point_cloud,
                                const at::Tensor& filter_values) {
    const auto values = require_filter_matrix(filter_values);
    const auto clean_clusters = normalized_clusters(clusters);
    MapperGraph graph;

    graph.nodes.reserve(clean_clusters.size());
    for (size_t i = 0; i < clean_clusters.size(); ++i) {
        MapperNode node;
        node.id = static_cast<int64_t>(i);
        node.point_indices = clean_clusters[i];

        const auto point_indices = make_index_tensor(clean_clusters[i], point_cloud.device());
        const auto filter_indices = make_index_tensor(clean_clusters[i], values.device());
        node.centroid = point_cloud.index_select(0, point_indices).mean(0);
        node.filter_centroid = values.index_select(0, filter_indices).mean(0);
        node.cover_index = -1;
        graph.nodes.push_back(std::move(node));
    }

    for (size_t i = 0; i < clean_clusters.size(); ++i) {
        for (size_t j = i + 1; j < clean_clusters.size(); ++j) {
            std::vector<int64_t> intersection;
            std::set_intersection(clean_clusters[i].begin(), clean_clusters[i].end(),
                                  clean_clusters[j].begin(), clean_clusters[j].end(),
                                  std::back_inserter(intersection));
            if (intersection.empty()) {
                continue;
            }

            MapperEdge edge;
            edge.source = static_cast<int64_t>(i);
            edge.target = static_cast<int64_t>(j);
            edge.weight = static_cast<double>(intersection.size()) /
                          static_cast<double>(
                              std::min(clean_clusters[i].size(), clean_clusters[j].size()));
            graph.edges.push_back(edge);
        }
    }

    return graph;
}

MapperGraph Mapper::fit_transform(const at::Tensor& point_cloud) {
    const auto points = require_point_matrix(point_cloud, "point_cloud");
    const auto filter_values = require_filter_matrix(apply_filter(points));
    require_matching_rows(points, filter_values);
    last_filter_values_ = filter_values;

    std::vector<std::vector<int64_t>> clusters;
    switch (config_.cover_type) {
        case CoverType::GRID:
        case CoverType::INTERVAL:
            clusters = pull_back_and_cluster(
                points, filter_values,
                cover_grid(filter_values, config_.cover_resolution, config_.cover_overlap));
            break;
        case CoverType::BALL:
            clusters = pull_back_ball_cover(
                points, filter_values,
                cover_ball(filter_values, config_.cover_resolution, config_.cover_overlap),
                config_);
            break;
    }

    return build_graph(clusters, points, filter_values);
}

at::Tensor Mapper::get_last_filter_values() const {
    return last_filter_values_;
}

std::vector<std::pair<int64_t, int64_t>> MapperGraph::to_edge_list() const {
    std::vector<std::pair<int64_t, int64_t>> edge_list;
    edge_list.reserve(edges.size());
    for (const auto& edge : edges) {
        edge_list.emplace_back(edge.source, edge.target);
    }
    return edge_list;
}

at::Tensor MapperGraph::to_adjacency_matrix() const {
    const int64_t n = static_cast<int64_t>(nodes.size());
    auto adj = at::zeros({n, n}, at::TensorOptions().dtype(at::kDouble));

    for (const auto& edge : edges) {
        if (edge.source >= 0 && edge.target >= 0 && edge.source < n && edge.target < n) {
            adj[edge.source][edge.target] = edge.weight;
            adj[edge.target][edge.source] = edge.weight;
        }
    }

    return adj;
}

MapperGraph quick_mapper(const at::Tensor& point_cloud,
                         const std::string& filter,
                         int64_t resolution,
                         double overlap) {
    MapperConfig config;
    config.filter_function = filter;
    config.cover_resolution = resolution;
    config.cover_overlap = overlap;

    Mapper mapper(config);
    return mapper.fit_transform(point_cloud);
}

}  // namespace nerve::torch

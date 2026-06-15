#include "detail/mapper_safe_arithmetic.hpp"
#include "nerve/algorithms/mapper.hpp"
#include "nerve/math/constants.hpp"

#include <algorithm>
#include <limits>
#include <numeric>
#include <queue>
#include <sstream>
#include <unordered_set>

#ifdef NERVE_HAS_CUDA
#include "nerve/gpu/mapper_gpu.cuh"

#include <future>
#include <type_traits>
#endif

namespace nerve::algorithms
{

template <typename T>
void MapperGraph<T>::build_adjacency()
{
    adjacency_list.assign(nodes.size(), {});
    for (const auto &edge : edges)
    {
        if (edge.source >= nodes.size() || edge.target >= nodes.size())
        {
            continue;
        }
        adjacency_list[edge.source].push_back(edge.target);
        adjacency_list[edge.target].push_back(edge.source);
    }
}

template <typename T>
typename MapperAlgorithm<T>::Result MapperAlgorithm<T>::compute(std::span<const T> point_cloud,
                                                                size_t n_points, size_t dim) const
{
    Result result;
    if (n_points == 0 || !detail::has_flat_span(point_cloud.size(), n_points, dim))
    {
        if (config_.compute_meta)
        {
            result.metadata["n_points"] = static_cast<T>(n_points);
            result.metadata["n_nodes"] = T{};
            result.metadata["n_edges"] = T{};
        }
        return result;
    }
    const size_t point_value_count =
        detail::checked_product_or_throw(n_points, dim, "mapper point cloud");
    detail::require_finite_prefix(point_cloud, point_value_count, "mapper point cloud");

    result.filter_values = apply_filter(point_cloud, n_points, dim);
    int n_filter_dims = 1;
    if (n_points > 0 && !result.filter_values.empty())
    {
        const size_t inferred_dims = result.filter_values.size() / n_points;
        if (!detail::fits_int(inferred_dims))
        {
            if (config_.compute_meta)
            {
                result.metadata["n_points"] = static_cast<T>(n_points);
                result.metadata["n_nodes"] = T{};
                result.metadata["n_edges"] = T{};
            }
            return result;
        }
        n_filter_dims = static_cast<int>(std::max<size_t>(1, inferred_dims));
    }

    result.cover_sets = build_cover(result.filter_values, n_points, n_filter_dims);
    auto nodes = cluster_cover_sets(point_cloud, dim, result.cover_sets);
    auto edges = build_graph(nodes);

    result.graph.nodes = std::move(nodes);
    result.graph.edges = std::move(edges);
    result.graph.build_adjacency();

    if (config_.compute_meta)
    {
        result.metadata["n_points"] = static_cast<T>(n_points);
        result.metadata["n_nodes"] = static_cast<T>(result.graph.n_nodes());
        result.metadata["n_edges"] = static_cast<T>(result.graph.n_edges());
    }

    return result;
}

template <typename T>
std::vector<T> MapperAlgorithm<T>::apply_filter(std::span<const T> points, size_t n_points,
                                                size_t dim) const
{
    if (config_.filter)
    {
        return config_.filter->apply(points, n_points, dim);
    }
    if (n_points == 0 || !detail::has_flat_span(points.size(), n_points, dim))
    {
        return {};
    }
    const size_t point_value_count = detail::checked_product_or_throw(n_points, dim, "points");
    detail::require_finite_prefix(points, point_value_count, "points");

#ifdef NERVE_HAS_CUDA
    if constexpr (std::is_same_v<T, float>)
    {
        if (!config_.filter && n_points >= 32)
        {
            std::vector<float> points_f32(points.begin(), points.end());
            std::promise<std::vector<float>> promise;
            auto future = promise.get_future();
            nerve::gpu::mapper::compute_density_filter_gpu(
                points_f32, static_cast<int>(n_points), static_cast<int>(dim), 15,
                [&promise](std::vector<float> r) { promise.set_value(std::move(r)); });
            auto gpu_result = future.get();
            if (!gpu_result.empty())
                return gpu_result;
        }
    }
#endif

    std::vector<T> result(n_points);
    for (size_t i = 0; i < n_points; ++i)
    {
        result[i] = points[i * dim];
    }
    return result;
}

template <typename T>
std::vector<std::vector<size_t>> MapperAlgorithm<T>::build_cover(std::span<const T> filter_values,
                                                                 size_t n_points,
                                                                 int n_filter_dims) const
{
    typename Cover<T>::Config cover_config;
    cover_config.resolution = config_.cover_resolution;
    cover_config.overlap = config_.cover_overlap;

    Cover<T> cover(cover_config);
    return cover.build(filter_values, n_points, n_filter_dims);
}

template <typename T>
std::vector<MapperNode<T>>
MapperAlgorithm<T>::cluster_cover_sets(std::span<const T> points, size_t dim,
                                       const std::vector<std::vector<size_t>> &cover_sets) const
{
    std::vector<MapperNode<T>> all_nodes;
    size_t node_id = 0;
    if (dim == 0)
    {
        return all_nodes;
    }
    const size_t available_points = points.size() / dim;

    for (const auto &cover_set : cover_sets)
    {
        if (cover_set.empty())
            continue;
        if (!detail::fits_int(cover_set.size()))
        {
            continue;
        }

        size_t subset_size = 0;
        if (!detail::checked_product(cover_set.size(), dim, subset_size))
        {
            continue;
        }
        bool valid_cover_set = true;
        std::vector<T> subset_points(subset_size);
        for (size_t i = 0; i < cover_set.size(); ++i)
        {
            size_t idx = cover_set[i];
            if (idx >= available_points)
            {
                valid_cover_set = false;
                break;
            }
            for (size_t d = 0; d < dim; ++d)
            {
                subset_points[i * dim + d] = points[idx * dim + d];
            }
        }
        if (!valid_cover_set)
        {
            continue;
        }

        std::vector<int> labels;
        if (this->config_.clusterer)
        {
            labels = this->config_.clusterer->cluster(subset_points, cover_set.size(), dim);
        }
        else
        {
            labels.resize(cover_set.size());
            std::iota(labels.begin(), labels.end(), 0);
        }
        if (labels.size() != cover_set.size())
        {
            labels.resize(cover_set.size());
            std::iota(labels.begin(), labels.end(), 0);
        }
        if (labels.empty())
        {
            continue;
        }

        int max_label = *std::max_element(labels.begin(), labels.end());
        for (int cluster_id = 0; cluster_id <= max_label; ++cluster_id)
        {
            MapperNode<T> node;
            node.id = node_id++;

            for (size_t i = 0; i < labels.size(); ++i)
            {
                if (labels[i] == cluster_id)
                {
                    node.point_indices.push_back(cover_set[i]);
                }
            }

            if (!node.point_indices.empty())
            {
                node.centroid = compute_centroid(points, dim, node.point_indices);
                all_nodes.push_back(std::move(node));
            }
        }
    }

    return all_nodes;
}

template <typename T>
std::vector<MapperEdge<T>>
MapperAlgorithm<T>::build_graph(const std::vector<MapperNode<T>> &nodes) const
{
    std::vector<MapperEdge<T>> edges;

    for (size_t i = 0; i < nodes.size(); ++i)
    {
        const std::unordered_set<size_t> points_i(nodes[i].point_indices.begin(),
                                                  nodes[i].point_indices.end());
        for (size_t j = i + 1; j < nodes.size(); ++j)
        {
            size_t overlap = 0;
            for (size_t idx : nodes[j].point_indices)
            {
                if (points_i.count(idx))
                {
                    overlap++;
                }
            }

            if (overlap > 0)
            {
                MapperEdge<T> edge;
                edge.source = i;
                edge.target = j;
                edge.overlap_size = overlap;
                edge.weight = static_cast<T>(overlap) / std::min(nodes[i].size(), nodes[j].size());
                edges.push_back(edge);
            }
        }
    }

    return edges;
}

template <typename T>
std::vector<T> MapperAlgorithm<T>::compute_centroid(std::span<const T> points, size_t dim,
                                                    std::span<const size_t> indices) const
{
    std::vector<T> centroid(dim, nerve::math::Constants<T>::kZero);
    if (dim == 0 || indices.empty())
    {
        return centroid;
    }
    const size_t available_points = points.size() / dim;

    size_t valid_count = 0;
    for (size_t idx : indices)
    {
        if (idx >= available_points)
        {
            continue;
        }
        for (size_t d = 0; d < dim; ++d)
        {
            centroid[d] += points[idx * dim + d];
        }
        ++valid_count;
    }

    if (valid_count == 0)
    {
        return centroid;
    }
    for (size_t d = 0; d < dim; ++d)
    {
        centroid[d] /= valid_count;
    }

    return centroid;
}

template <typename T>
std::vector<std::vector<size_t>> connected_components(const MapperGraph<T> &graph)
{
    const size_t n_nodes = graph.n_nodes();
    std::vector<std::vector<size_t>> adjacency = graph.adjacency_list;
    if (adjacency.size() != n_nodes)
    {
        adjacency.assign(n_nodes, {});
        for (const auto &edge : graph.edges)
        {
            if (edge.source < n_nodes && edge.target < n_nodes)
            {
                adjacency[edge.source].push_back(edge.target);
                adjacency[edge.target].push_back(edge.source);
            }
        }
    }

    std::vector<bool> visited(n_nodes, false);
    std::vector<std::vector<size_t>> components;

    for (size_t start = 0; start < n_nodes; ++start)
    {
        if (visited[start])
            continue;

        std::vector<size_t> component;
        std::queue<size_t> queue;
        queue.push(start);
        visited[start] = true;

        while (!queue.empty())
        {
            size_t current = queue.front();
            queue.pop();
            component.push_back(current);

            for (size_t neighbor : adjacency[current])
            {
                if (neighbor < n_nodes && !visited[neighbor])
                {
                    visited[neighbor] = true;
                    queue.push(neighbor);
                }
            }
        }

        components.push_back(std::move(component));
    }

    return components;
}

template <typename T>
size_t graph_diameter(const MapperGraph<T> &graph)
{
    const size_t n_nodes = graph.n_nodes();
    if (n_nodes <= 1)
    {
        return 0;
    }

    std::vector<std::vector<size_t>> adjacency = graph.adjacency_list;
    if (adjacency.size() != n_nodes)
    {
        adjacency.assign(n_nodes, {});
        for (const auto &edge : graph.edges)
        {
            if (edge.source < n_nodes && edge.target < n_nodes)
            {
                adjacency[edge.source].push_back(edge.target);
                adjacency[edge.target].push_back(edge.source);
            }
        }
    }

    size_t diameter = 0;
    for (size_t start = 0; start < n_nodes; ++start)
    {
        std::vector<size_t> distance(n_nodes, n_nodes);
        std::queue<size_t> queue;
        distance[start] = 0;
        queue.push(start);

        while (!queue.empty())
        {
            const size_t current = queue.front();
            queue.pop();
            for (size_t neighbor : adjacency[current])
            {
                if (neighbor < n_nodes && distance[neighbor] == n_nodes)
                {
                    distance[neighbor] = distance[current] + 1;
                    diameter = std::max(diameter, distance[neighbor]);
                    queue.push(neighbor);
                }
            }
        }
    }
    return diameter;
}

template <typename T>
std::string export_to_graphml(const MapperGraph<T> &graph)
{
    std::ostringstream out;
    out << "<graphml><graph edgedefault=\"undirected\">";
    for (const auto &node : graph.nodes)
    {
        out << "<node id=\"n" << node.id << "\"/>";
    }
    for (size_t edge_id = 0; edge_id < graph.edges.size(); ++edge_id)
    {
        const auto &edge = graph.edges[edge_id];
        out << "<edge id=\"e" << edge_id << "\" source=\"n" << edge.source << "\" target=\"n"
            << edge.target << "\"><data key=\"weight\">" << edge.weight << "</data></edge>";
    }
    out << "</graph></graphml>";
    return out.str();
}

template <typename T>
std::string export_to_json(const MapperGraph<T> &graph)
{
    std::ostringstream out;
    out << "{\"nodes\":[";
    for (size_t i = 0; i < graph.nodes.size(); ++i)
    {
        if (i > 0)
            out << ",";
        out << "{\"id\":" << graph.nodes[i].id << ",\"size\":" << graph.nodes[i].size() << "}";
    }
    out << "],\"edges\":[";
    for (size_t i = 0; i < graph.edges.size(); ++i)
    {
        if (i > 0)
            out << ",";
        const auto &edge = graph.edges[i];
        out << "{\"source\":" << edge.source << ",\"target\":" << edge.target
            << ",\"weight\":" << edge.weight << ",\"overlap_size\":" << edge.overlap_size << "}";
    }
    out << "]}";
    return out.str();
}

template struct MapperNode<float>;
template struct MapperNode<double>;
template struct MapperEdge<float>;
template struct MapperEdge<double>;
template struct MapperGraph<float>;
template struct MapperGraph<double>;
template class MapperAlgorithm<float>;
template class MapperAlgorithm<double>;

template std::vector<std::vector<size_t>> connected_components<float>(const MapperGraph<float> &);
template std::vector<std::vector<size_t>> connected_components<double>(const MapperGraph<double> &);
template size_t graph_diameter<float>(const MapperGraph<float> &);
template size_t graph_diameter<double>(const MapperGraph<double> &);
template std::string export_to_graphml<float>(const MapperGraph<float> &);
template std::string export_to_graphml<double>(const MapperGraph<double> &);
template std::string export_to_json<float>(const MapperGraph<float> &);
template std::string export_to_json<double>(const MapperGraph<double> &);

} // namespace nerve::algorithms

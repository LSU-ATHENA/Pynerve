#include "nerve/algorithms/mapper.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

namespace nerve::algorithms
{

template <typename T>
std::string mapperGraphToDot(const MapperGraph<T> &graph)
{
    std::ostringstream os;
    os << "graph MapperGraph {\n";
    os << "  node [shape=circle, style=filled, fillcolor=lightblue];\n";
    for (size_t i = 0; i < graph.nodes.size(); ++i)
    {
        os << "  " << i << " [label=\"N" << i << "\\n|V|=" << graph.nodes[i].size() << "\"];\n";
    }
    for (const auto &edge : graph.edges)
    {
        os << "  " << edge.source << " -- " << edge.target << " [label=\""
           << static_cast<int>(edge.overlap_size) << "\", penwidth=" << (1.0 + 3.0 * edge.weight)
           << "];\n";
    }
    os << "}\n";
    return os.str();
}

template <typename T>
std::string mapperGraphToJson(const MapperGraph<T> &graph)
{
    std::ostringstream os;
    os << "{\"nodes\":[";
    for (size_t i = 0; i < graph.nodes.size(); ++i)
    {
        if (i > 0)
            os << ',';
        os << "{\"id\":" << graph.nodes[i].id << ",\"size\":" << graph.nodes[i].size()
           << ",\"centroid\":[";
        for (size_t d = 0; d < graph.nodes[i].centroid.size(); ++d)
        {
            if (d > 0)
                os << ',';
            os << graph.nodes[i].centroid[d];
        }
        os << "]}";
    }
    os << "],\"edges\":[";
    for (size_t e = 0; e < graph.edges.size(); ++e)
    {
        if (e > 0)
            os << ',';
        os << "{\"source\":" << graph.edges[e].source << ",\"target\":" << graph.edges[e].target
           << ",\"weight\":" << graph.edges[e].weight
           << ",\"overlap\":" << graph.edges[e].overlap_size << "}";
    }
    os << "]}\n";
    return os.str();
}

template <typename T>
void computeMapperLayout(const MapperGraph<T> &graph, std::vector<std::array<T, 2>> &positions,
                         Size iterations)
{
    const Size n = graph.nodes.size();
    positions.resize(n);
    std::vector<std::array<T, 2>> forces(n);
    const T area = static_cast<T>(n);
    const T k = std::sqrt(area / static_cast<T>(n * 2));
    const T t_init = static_cast<T>(n) * T{0.1};
    const T t_min = T{0.01};
    const T damping = T{0.9};

    for (Size i = 0; i < n; ++i)
    {
        positions[i][0] = static_cast<T>(i % static_cast<Size>(std::sqrt(n))) * k * T{2.0};
        positions[i][1] = static_cast<T>(i / static_cast<Size>(std::sqrt(n))) * k * T{2.0};
    }

    T temp = t_init;
    for (Size iter = 0; iter < iterations; ++iter)
    {
        for (Size i = 0; i < n; ++i)
            forces[i] = {T{0}, T{0}};

        for (Size i = 0; i < n; ++i)
        {
            for (Size j = i + 1; j < n; ++j)
            {
                T dx = positions[j][0] - positions[i][0];
                T dy = positions[j][1] - positions[i][1];
                T dist = std::sqrt(dx * dx + dy * dy);
                if (dist < T{0.01})
                    dist = T{0.01};
                T force = k * k / dist;
                T fx = force * dx / dist;
                T fy = force * dy / dist;
                forces[i][0] -= fx;
                forces[i][1] -= fy;
                forces[j][0] += fx;
                forces[j][1] += fy;
            }
        }

        for (const auto &edge : graph.edges)
        {
            Size i = edge.source;
            Size j = edge.target;
            if (i >= n || j >= n)
                continue;
            T dx = positions[j][0] - positions[i][0];
            T dy = positions[j][1] - positions[i][1];
            T dist = std::sqrt(dx * dx + dy * dy);
            if (dist < T{0.01})
                dist = T{0.01};
            T force = dist * dist / k * edge.weight;
            T fx = force * dx / dist;
            T fy = force * dy / dist;
            forces[i][0] += fx;
            forces[i][1] += fy;
            forces[j][0] -= fx;
            forces[j][1] -= fy;
        }

        for (Size i = 0; i < n; ++i)
        {
            T fx = forces[i][0];
            T fy = forces[i][1];
            T mag = std::sqrt(fx * fx + fy * fy);
            if (mag < T{0.001})
                continue;
            T capped = std::min(mag, temp);
            T scale = capped / mag;
            positions[i][0] += fx * scale;
            positions[i][1] += fy * scale;
        }

        temp = std::max(temp * damping, t_min);
    }
}

template <typename T>
MapperStatistics computeMapperStatistics(const MapperGraph<T> &graph)
{
    MapperStatistics stats;
    stats.node_count = graph.nodes.size();
    stats.edge_count = graph.edges.size();
    if (stats.node_count == 0)
        return stats;

    Size total_points = 0;
    for (const auto &node : graph.nodes)
    {
        total_points += node.size();
        stats.node_sizes.push_back(node.size());
    }
    stats.avg_node_size = static_cast<double>(total_points) / static_cast<double>(stats.node_count);

    Size max_size = 0, min_size = static_cast<Size>(-1);
    for (Size s : stats.node_sizes)
    {
        if (s > max_size)
            max_size = s;
        if (s < min_size)
            min_size = s;
    }
    stats.max_node_size = max_size;
    stats.min_node_size = stats.node_count > 0 ? min_size : 0;

    std::vector<Size> sorted_sizes = stats.node_sizes;
    std::sort(sorted_sizes.begin(), sorted_sizes.end());
    if (!sorted_sizes.empty())
    {
        stats.median_node_size = sorted_sizes[sorted_sizes.size() / 2];
    }

    stats.edge_density = 0.0;
    if (stats.node_count > 1)
    {
        stats.edge_density = static_cast<double>(stats.edge_count) /
                             (static_cast<double>(stats.node_count) *
                              static_cast<double>(stats.node_count - 1) / 2.0);
    }

    double total_overlap = 0.0;
    for (const auto &edge : graph.edges)
    {
        total_overlap += static_cast<double>(edge.overlap_size);
        stats.edge_weights.push_back(static_cast<double>(edge.weight));
    }
    stats.avg_overlap =
        stats.edge_count > 0 ? total_overlap / static_cast<double>(stats.edge_count) : 0.0;

    if (!stats.edge_weights.empty())
    {
        std::sort(stats.edge_weights.begin(), stats.edge_weights.end());
        stats.median_edge_weight = stats.edge_weights[stats.edge_weights.size() / 2];
        stats.max_edge_weight = stats.edge_weights.back();
        stats.min_edge_weight = stats.edge_weights.front();
    }

    Size components = 0;
    std::vector<bool> visited(stats.node_count, false);
    for (Size i = 0; i < stats.node_count; ++i)
    {
        if (visited[i])
            continue;
        ++components;
        std::vector<Size> stack{i};
        visited[i] = true;
        while (!stack.empty())
        {
            Size v = stack.back();
            stack.pop_back();
            for (Size w : graph.adjacency_list[v])
            {
                if (!visited[w])
                {
                    visited[w] = true;
                    stack.push_back(w);
                }
            }
        }
    }
    stats.connected_components = components;

    return stats;
}

template std::string mapperGraphToDot<float>(const MapperGraph<float> &);
template std::string mapperGraphToDot<double>(const MapperGraph<double> &);
template std::string mapperGraphToJson<float>(const MapperGraph<float> &);
template std::string mapperGraphToJson<double>(const MapperGraph<double> &);
template void computeMapperLayout<float>(const MapperGraph<float> &,
                                         std::vector<std::array<float, 2>> &, Size);
template void computeMapperLayout<double>(const MapperGraph<double> &,
                                          std::vector<std::array<double, 2>> &, Size);
template MapperStatistics computeMapperStatistics<float>(const MapperGraph<float> &);
template MapperStatistics computeMapperStatistics<double>(const MapperGraph<double> &);

} // namespace nerve::algorithms

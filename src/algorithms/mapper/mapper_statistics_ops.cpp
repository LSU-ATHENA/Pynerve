#include "nerve/algorithms/mapper.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_set>
#include <vector>

namespace nerve::algorithms
{

template <typename T>
double computeMapperGlobalEfficiency(const MapperGraph<T> &graph)
{
    const Size n = graph.nodes.size();
    if (n < 2)
        return 0.0;
    std::vector<std::vector<Size>> dist(n, std::vector<Size>(n, static_cast<Size>(-1)));
    for (Size i = 0; i < n; ++i)
    {
        dist[i][i] = 0;
        for (Size j : graph.adjacency_list[i])
        {
            dist[i][j] = 1;
        }
    }
    for (Size k = 0; k < n; ++k)
    {
        for (Size i = 0; i < n; ++i)
        {
            for (Size j = 0; j < n; ++j)
            {
                if (dist[i][k] < static_cast<Size>(-1) && dist[k][j] < static_cast<Size>(-1))
                {
                    dist[i][j] = std::min(dist[i][j], dist[i][k] + dist[k][j]);
                }
            }
        }
    }
    double sum_inv = 0.0;
    Size count = 0;
    for (Size i = 0; i < n; ++i)
    {
        for (Size j = i + 1; j < n; ++j)
        {
            if (dist[i][j] > 0 && dist[i][j] < static_cast<Size>(-1))
            {
                sum_inv += 1.0 / static_cast<double>(dist[i][j]);
                ++count;
            }
        }
    }
    return count > 0 ? sum_inv / static_cast<double>(count) : 0.0;
}

template <typename T>
double computeMapperModularity(const MapperGraph<T> &graph)
{
    const Size n = graph.nodes.size();
    if (n == 0)
        return 0.0;
    double m = static_cast<double>(graph.edges.size());
    if (m < 1.0)
        return 0.0;
    std::vector<Size> community(n);
    for (Size i = 0; i < n; ++i)
        community[i] = i;
    constexpr Size max_passes = 10;
    for (Size pass = 0; pass < max_passes; ++pass)
    {
        bool changed = false;
        for (Size i = 0; i < n; ++i)
        {
            std::unordered_map<Size, double> comm_weights;
            for (Size j : graph.adjacency_list[i])
            {
                if (j < n)
                {
                    comm_weights[community[j]] += 1.0;
                }
            }
            Size best_comm = community[i];
            double best_delta = 0.0;
            double k_i = static_cast<double>(graph.adjacency_list[i].size());
            double k_i_in = 0.0;
            for (Size j : graph.adjacency_list[i])
            {
                if (community[j] == community[i])
                    k_i_in += 1.0;
            }
            for (const auto &[c, w] : comm_weights)
            {
                if (c == community[i])
                    continue;
                double k_c = 0.0;
                for (Size j = 0; j < n; ++j)
                {
                    if (community[j] == c)
                    {
                        k_c += static_cast<double>(graph.adjacency_list[j].size());
                    }
                }
                double delta = (w - k_i_in) / m - k_i * (k_c - k_i) / (2.0 * m * m);
                if (delta > best_delta)
                {
                    best_delta = delta;
                    best_comm = c;
                }
            }
            if (best_comm != community[i])
            {
                community[i] = best_comm;
                changed = true;
            }
        }
        if (!changed)
            break;
    }
    double q = 0.0;
    for (Size i = 0; i < n; ++i)
    {
        for (Size j = 0; j < n; ++j)
        {
            if (community[i] != community[j])
                continue;
            bool adjacent = false;
            for (Size k : graph.adjacency_list[i])
            {
                if (k == j)
                {
                    adjacent = true;
                    break;
                }
            }
            double a_ij = adjacent ? 1.0 : 0.0;
            double k_i = static_cast<double>(graph.adjacency_list[i].size());
            double k_j = static_cast<double>(graph.adjacency_list[j].size());
            q += a_ij - k_i * k_j / (2.0 * m);
        }
    }
    return q / (2.0 * m);
}

template <typename T>
double computeMapperCoverage(const MapperGraph<T> &graph, Size total_points)
{
    if (total_points == 0)
        return 0.0;
    std::unordered_set<Size> covered;
    for (const auto &node : graph.nodes)
    {
        for (Size idx : node.point_indices)
        {
            covered.insert(idx);
        }
    }
    return static_cast<double>(covered.size()) / static_cast<double>(total_points);
}

template <typename T>
double computeMapperNodeSizeStdDev(const MapperGraph<T> &graph)
{
    const Size n = graph.nodes.size();
    if (n < 2)
        return 0.0;
    double mean = 0.0;
    for (const auto &node : graph.nodes)
    {
        mean += static_cast<double>(node.size());
    }
    mean /= static_cast<double>(n);
    double variance = 0.0;
    for (const auto &node : graph.nodes)
    {
        double diff = static_cast<double>(node.size()) - mean;
        variance += diff * diff;
    }
    variance /= static_cast<double>(n - 1);
    return std::sqrt(variance);
}

template <typename T>
double computeMapperClusteringCoefficient(const MapperGraph<T> &graph)
{
    const Size n = graph.nodes.size();
    if (n == 0)
        return 0.0;
    double total_coeff = 0.0;
    Size nodes_with_neighbors = 0;
    for (Size i = 0; i < n; ++i)
    {
        const auto &neighbors = graph.adjacency_list[i];
        Size deg = neighbors.size();
        if (deg < 2)
            continue;
        ++nodes_with_neighbors;
        Size triangles = 0;
        for (Size a = 0; a < deg; ++a)
        {
            for (Size b = a + 1; b < deg; ++b)
            {
                Size u = neighbors[a];
                Size v = neighbors[b];
                for (Size w : graph.adjacency_list[u])
                {
                    if (w == v)
                    {
                        ++triangles;
                        break;
                    }
                }
            }
        }
        total_coeff += static_cast<double>(triangles) /
                       (static_cast<double>(deg) * static_cast<double>(deg - 1) / 2.0);
    }
    return nodes_with_neighbors > 0 ? total_coeff / static_cast<double>(nodes_with_neighbors) : 0.0;
}

template double computeMapperGlobalEfficiency<float>(const MapperGraph<float> &);
template double computeMapperGlobalEfficiency<double>(const MapperGraph<double> &);
template double computeMapperModularity<float>(const MapperGraph<float> &);
template double computeMapperModularity<double>(const MapperGraph<double> &);
template double computeMapperCoverage<float>(const MapperGraph<float> &, Size);
template double computeMapperCoverage<double>(const MapperGraph<double> &, Size);
template double computeMapperNodeSizeStdDev<float>(const MapperGraph<float> &);
template double computeMapperNodeSizeStdDev<double>(const MapperGraph<double> &);
template double computeMapperClusteringCoefficient<float>(const MapperGraph<float> &);
template double computeMapperClusteringCoefficient<double>(const MapperGraph<double> &);

} // namespace nerve::algorithms

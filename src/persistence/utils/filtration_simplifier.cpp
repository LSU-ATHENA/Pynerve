#include "nerve/core.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <numeric>
#include <queue>
#include <ranges>
#include <stdexcept>
#include <vector>

namespace nerve::persistence::simplify
{
namespace
{

constexpr double kBaseVertexReduction = 0.3;
constexpr double kBaseEdgeReduction = 0.5;
constexpr double kMediumVertexReduction = 0.4;
constexpr double kMediumEdgeReduction = 0.6;
constexpr double kHighVertexReduction = 0.5;
constexpr double kHighEdgeReduction = 0.7;
constexpr double kLowDensityThreshold = 0.1;
constexpr double kHighDensityThreshold = 0.3;
constexpr double kMaxReportedReduction = 0.95;

size_t checkedPairCount(size_t n)
{
    if (n > 1 && n > std::numeric_limits<size_t>::max() / (n - 1))
    {
        throw std::overflow_error("pair count overflows size_t");
    }
    return n * (n - 1) / 2;
}

double clampReduction(double reduction)
{
    return std::clamp(reduction, 0.0, kMaxReportedReduction);
}

double speedupFromReduction(double reduction)
{
    return 1.0 / (1.0 - clampReduction(reduction) * 0.5);
}

void validateConfig(const SimplificationConfig &config)
{
    if (!std::isfinite(config.radius) || config.radius < 0.0)
    {
        throw std::invalid_argument("simplification radius must be finite and non-negative");
    }
    if (!std::isfinite(config.edge_threshold) || config.edge_threshold < 0.0)
    {
        throw std::invalid_argument(
            "simplification edge threshold must be finite and non-negative");
    }
    if (config.max_iterations < 0)
    {
        throw std::invalid_argument("simplification iteration cap must be non-negative");
    }
}

size_t validatePointCloud(const std::vector<GeometricPoint> &points)
{
    if (points.empty())
    {
        return 0;
    }

    const size_t point_dim = points.front().coords.size();
    for (const auto &point : points)
    {
        if (point.coords.size() != point_dim)
        {
            throw std::invalid_argument("all simplification points must have the same dimension");
        }
        if (!std::ranges::all_of(point.coords, [](double value) { return std::isfinite(value); }))
        {
            throw std::invalid_argument("simplification points must contain finite coordinates");
        }
    }
    return point_dim;
}

} // namespace

struct Edge
{
    int u, v;
    double weight;
    bool operator>(const Edge &other) const { return weight > other.weight; }
};

struct ContractibleEdge
{
    int vertex_u, vertex_v;
};

class FiltrationSimplifier
{
public:
    std::vector<ContractibleEdge>
    findContractibleEdges(const std::vector<GeometricPoint> &points,
                          const std::vector<std::pair<int, int>> &edges,
                          const std::vector<double> &edge_weights, size_t max_contractions)
    {
        if (edges.size() != edge_weights.size())
        {
            throw std::invalid_argument("edge list and edge weight list sizes differ");
        }
        if (max_contractions == 0 || points.size() < 2 || edges.empty())
        {
            return {};
        }

        std::vector<ContractibleEdge> contractible;

        std::vector<std::vector<std::pair<int, double>>> adj(points.size());
        for (size_t i = 0; i < edges.size(); ++i)
        {
            int u = edges[i].first;
            int v = edges[i].second;
            double w = edge_weights[i];
            if (u < 0 || v < 0 || u == v || static_cast<size_t>(u) >= points.size() ||
                static_cast<size_t>(v) >= points.size() || !std::isfinite(w) || w < 0.0)
            {
                throw std::invalid_argument("invalid contractible edge candidate");
            }
            adj[u].push_back({v, w});
            adj[v].push_back({u, w});
        }

        std::priority_queue<Edge, std::vector<Edge>, std::greater<Edge>> pq;
        for (size_t i = 0; i < edges.size(); ++i)
        {
            pq.push({edges[i].first, edges[i].second, edge_weights[i]});
        }

        std::vector<int> parent(points.size());
        std::iota(parent.begin(), parent.end(), 0);

        auto find = [&](int x, auto &&find_ref) -> int {
            if (parent[x] != x)
            {
                parent[x] = find_ref(parent[x], find_ref);
            }
            return parent[x];
        };

        auto union_sets = [&](int x, int y) {
            int px = find(x, find);
            int py = find(y, find);
            if (px != py)
            {
                parent[px] = py;
            }
        };

        std::vector<bool> contracted(points.size(), false);

        while (!pq.empty() && contractible.size() < max_contractions)
        {
            Edge e = pq.top();
            pq.pop();

            if (contracted[e.u] || contracted[e.v])
            {
                continue;
            }

            int pu = find(e.u, find);
            int pv = find(e.v, find);

            if (pu != pv)
            {
                union_sets(e.u, e.v);
                continue;
            }

            if (adj[e.u].size() <= 3 && adj[e.v].size() <= 3)
            {
                contractible.push_back({e.u, e.v});
                contracted[e.v] = true;

                for (auto [neighbor, weight] : adj[e.v])
                {
                    if (neighbor != e.u && !contracted[neighbor])
                    {
                        adj[e.u].push_back({neighbor, weight});
                    }
                }
            }
        }

        return contractible;
    }

    std::vector<GeometricPoint> contractEdges(const std::vector<GeometricPoint> &points,
                                              const std::vector<ContractibleEdge> &contractions)
    {
        if (contractions.empty())
        {
            return points;
        }

        std::vector<GeometricPoint> result;
        std::vector<bool> removed(points.size(), false);

        for (const auto &c : contractions)
        {
            if (c.vertex_u < 0 || c.vertex_v < 0 ||
                static_cast<size_t>(c.vertex_u) >= points.size() ||
                static_cast<size_t>(c.vertex_v) >= points.size() || c.vertex_u == c.vertex_v)
            {
                throw std::invalid_argument("invalid edge contraction");
            }
            removed[static_cast<size_t>(c.vertex_v)] = true;
        }

        for (size_t i = 0; i < points.size(); ++i)
        {
            if (!removed[i])
            {
                result.push_back(points[i]);
            }
        }

        return result;
    }

    double estimateSimplification(size_t num_points, size_t num_edges)
    {
        const size_t max_edges = checkedPairCount(num_points);
        if (num_points < 2 || num_edges == 0 || max_edges == 0)
        {
            return 0.0;
        }

        double vertex_ratio = kBaseVertexReduction;
        double edge_ratio = kBaseEdgeReduction;
        const double density =
            std::min(1.0, static_cast<double>(num_edges) / static_cast<double>(max_edges));
        if (density > kLowDensityThreshold)
        {
            vertex_ratio = kMediumVertexReduction;
            edge_ratio = kMediumEdgeReduction;
        }
        if (density > kHighDensityThreshold)
        {
            vertex_ratio = kHighVertexReduction;
            edge_ratio = kHighEdgeReduction;
        }

        return (vertex_ratio + edge_ratio) / 2.0;
    }
};

SimplificationResult simplifyPointCloud(const std::vector<GeometricPoint> &points,
                                        const SimplificationConfig &config)
{
    validateConfig(config);

    SimplificationResult result;
    result.original_points = points.size();
    result.simplified_points = points;

    if (!config.use_simplification)
    {
        return result;
    }

    const size_t point_dim = validatePointCloud(points);
    if (points.size() < 2 || config.max_iterations == 0 || point_dim == 0)
    {
        return result;
    }

    std::vector<std::pair<int, int>> edges;
    std::vector<double> edge_weights;

    for (size_t i = 0; i < points.size(); ++i)
    {
        for (size_t j = i + 1; j < points.size(); ++j)
        {
            double dist = 0.0;
            for (size_t d = 0; d < point_dim; ++d)
            {
                const double diff = points[i].coords[d] - points[j].coords[d];
                dist += diff * diff;
            }
            dist = std::sqrt(dist);

            if (dist < config.edge_threshold)
            {
                edges.push_back({static_cast<int>(i), static_cast<int>(j)});
                edge_weights.push_back(dist);
            }
        }
    }

    FiltrationSimplifier simplifier;
    const size_t max_contractions =
        std::min(points.size() / 2, static_cast<size_t>(config.max_iterations));
    auto contractible =
        simplifier.findContractibleEdges(points, edges, edge_weights, max_contractions);

    if (!contractible.empty())
    {
        result.simplified_points = simplifier.contractEdges(points, contractible);
        result.simplification_ratio = 1.0 - static_cast<double>(result.simplified_points.size()) /
                                                static_cast<double>(points.size());
        result.applied = true;
    }

    result.estimated_speedup = speedupFromReduction(result.simplification_ratio);

    return result;
}

SimplificationEstimate estimateSimplificationBenefit(size_t num_points, size_t num_edges)
{
    SimplificationEstimate estimate;

    FiltrationSimplifier simplifier;
    const double reduction =
        clampReduction(simplifier.estimateSimplification(num_points, num_edges));

    estimate.vertex_reduction = reduction;
    estimate.edge_reduction = clampReduction(reduction * 1.5);
    estimate.simplex_reduction = clampReduction(reduction * 2.0);
    estimate.estimated_speedup = speedupFromReduction(reduction);
    estimate.recommended = (num_points > 10000);

    return estimate;
}

} // namespace nerve::persistence::simplify

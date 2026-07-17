#include "nerve/core.hpp"
#include "nerve/metrics/gpu_distances.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <queue>
#include <ranges>
#include <stdexcept>
#include <string>
#include <vector>

namespace nerve::metrics::bottleneck
{

class DiagramKDTree;
bool hasPerfectMatchingKD(const std::vector<std::pair<float, float>> &diagram1,
                          const std::vector<std::pair<float, float>> &diagram2,
                          const DiagramKDTree &tree2, double threshold);
bool tryAugment(size_t left, const std::vector<std::vector<size_t>> &adjacency,
                std::vector<bool> &seen, std::vector<int> &match);

struct KDNode
{
    float birth, death;
    int left = -1, right = -1;
    int point_idx;
    bool is_leaf = false;
};

class DiagramKDTree
{
public:
    DiagramKDTree(const std::vector<std::pair<float, float>> &points)
    {
        if (points.empty())
            return;
        if (points.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
        {
            throw std::length_error("bottleneck diagram is too large for KD-tree indexing");
        }

        points_ = points;
        nodes_.reserve(points.size());

        std::vector<int> indices(points.size());
        std::iota(indices.begin(), indices.end(), 0);
        root_ = buildTree(indices, 0, 0, static_cast<int>(points.size()) - 1);
    }

    double nearestNeighbor(float birth, float death, double max_dist) const
    {
        double best_dist = max_dist;
        nearestNeighborRecursive(root_, birth, death, 0, best_dist);
        return best_dist < max_dist ? best_dist : max_dist;
    }

    std::vector<int> rangeQuery(float birth, float death, double radius) const
    {
        std::vector<int> result;
        rangeQueryRecursive(root_, birth, death, radius, 0, result);
        return result;
    }

private:
    std::vector<std::pair<float, float>> points_;
    std::vector<KDNode> nodes_;
    int root_ = -1;

    int buildTree(std::vector<int> &indices, int depth, int start, int end)
    {
        if (start > end)
            return -1;

        int dim = depth % 2;

        std::ranges::sort(indices.begin() + start, indices.begin() + end + 1, {},
                          [&](int i) { return dim == 0 ? points_[i].first : points_[i].second; });

        int mid = (start + end) / 2;
        int node_idx = static_cast<int>(nodes_.size());
        nodes_.emplace_back();

        KDNode &node = nodes_.back();
        node.birth = points_[indices[mid]].first;
        node.death = points_[indices[mid]].second;
        node.point_idx = indices[mid];

        if (start == end)
        {
            node.is_leaf = true;
            return node_idx;
        }

        node.left = buildTree(indices, depth + 1, start, mid - 1);
        node.right = buildTree(indices, depth + 1, mid + 1, end);

        return node_idx;
    }

    static double infSafeDeathDiff(double node_death, double query_death)
    {
        if (std::isinf(node_death) && std::isinf(query_death))
            return 0.0;
        return std::abs(node_death - query_death);
    }

    void nearestNeighborRecursive(int node_idx, float birth, float death, int depth,
                                  double &best_dist) const
    {
        if (node_idx < 0)
            return;

        const KDNode &node = nodes_[node_idx];

        const double dx = std::abs(static_cast<double>(node.birth) - birth);
        const double dy =
            infSafeDeathDiff(static_cast<double>(node.death), static_cast<double>(death));
        const double dist = std::max(dx, dy);
        if (dist < best_dist)
        {
            best_dist = dist;
        }

        if (node.is_leaf)
            return;

        int dim = depth % 2;
        const double diff = dim == 0 ? static_cast<double>(birth) - node.birth
                                     : static_cast<double>(death) - node.death;

        int first = diff < 0 ? node.left : node.right;
        int second = diff < 0 ? node.right : node.left;

        nearestNeighborRecursive(first, birth, death, depth + 1, best_dist);

        if (std::abs(diff) < best_dist)
        {
            nearestNeighborRecursive(second, birth, death, depth + 1, best_dist);
        }
    }

    void rangeQueryRecursive(int node_idx, float birth, float death, double radius, int depth,
                             std::vector<int> &result) const
    {
        if (node_idx < 0)
            return;

        const KDNode &node = nodes_[node_idx];

        const double dx = std::abs(static_cast<double>(node.birth) - birth);
        const double dy =
            infSafeDeathDiff(static_cast<double>(node.death), static_cast<double>(death));
        const double dist = std::max(dx, dy);
        if (dist <= radius)
        {
            result.push_back(node.point_idx);
        }

        if (node.is_leaf)
            return;

        int dim = depth % 2;
        const double diff = dim == 0 ? static_cast<double>(birth) - node.birth
                                     : static_cast<double>(death) - node.death;

        int first = diff < 0 ? node.left : node.right;
        rangeQueryRecursive(first, birth, death, radius, depth + 1, result);

        if (std::abs(diff) <= radius)
        {
            int second = diff < 0 ? node.right : node.left;
            rangeQueryRecursive(second, birth, death, radius, depth + 1, result);
        }
    }
};

void validateDiagram(const std::vector<std::pair<float, float>> &diagram, const char *name)
{
    if (diagram.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        throw std::length_error(std::string(name) + " diagram is too large for indexing");
    }
    for (const auto &[birth, death] : diagram)
    {
        const bool finite_death = std::isfinite(death);
        const bool infinite_death = std::isinf(death);
        if (!std::isfinite(birth) || (!finite_death && !infinite_death))
        {
            throw std::invalid_argument(std::string(name) +
                                        " diagram contains invalid persistence pairs");
        }
        if (finite_death && death < birth)
        {
            throw std::invalid_argument(std::string(name) +
                                        " diagram contains invalid persistence pairs");
        }
    }
}

void validateMatchingSize(size_t n1, size_t n2)
{
    const size_t max_index = static_cast<size_t>(std::numeric_limits<int>::max());
    if (n1 > max_index || n2 > max_index || n1 > max_index - n2)
    {
        throw std::length_error("bottleneck matching graph is too large for indexing");
    }
}

double diagonalDistance(const std::pair<float, float> &point)
{
    return std::abs(static_cast<double>(point.second) - static_cast<double>(point.first)) * 0.5;
}

size_t candidateCapacity(size_t n1, size_t n2)
{
    constexpr size_t max_size = std::numeric_limits<size_t>::max();
    if (n2 > max_size - 1)
    {
        throw std::length_error("bottleneck candidate set size overflow");
    }
    const size_t available = max_size - n2 - 1;
    if (n1 > available / 2)
    {
        throw std::length_error("bottleneck candidate set size overflow");
    }
    return n1 * 2 + n2 + 1;
}

double maxDiagonalDistance(const std::vector<std::pair<float, float>> &diagram)
{
    double result = 0.0;
    for (const auto &point : diagram)
    {
        result = std::max(result, diagonalDistance(point));
    }
    return result;
}

// Compute bottleneck distance on finite points only (separated from essential classes)
static double bottleneckFinitePoints(const std::vector<std::pair<float, float>> &finite1,
                                     const std::vector<std::pair<float, float>> &finite2)
{
    if (finite1.empty() && finite2.empty())
        return 0.0;
    if (finite1.empty())
        return maxDiagonalDistance(finite2);
    if (finite2.empty())
        return maxDiagonalDistance(finite1);

    DiagramKDTree tree2(finite2);

    std::vector<double> candidates;
    candidates.reserve(candidateCapacity(finite1.size(), finite2.size()));
    candidates.push_back(0.0);

    for (const auto &p1 : finite1)
    {
        double dist =
            tree2.nearestNeighbor(p1.first, p1.second, std::numeric_limits<double>::infinity());
        candidates.push_back(dist);
        candidates.push_back(diagonalDistance(p1));
    }
    for (const auto &p2 : finite2)
    {
        candidates.push_back(diagonalDistance(p2));
    }

    std::ranges::sort(candidates);
    const auto [first, last] = std::ranges::unique(candidates);
    candidates.erase(first, last);

    candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
                                    [](double c) { return !std::isfinite(c); }),
                     candidates.end());

    if (candidates.empty())
    {
        return std::numeric_limits<double>::infinity();
    }

    double low = 0.0, high = candidates.back();
    double result = high;

    constexpr double EPSILON = 1e-9;

    while (low <= high)
    {
        double mid = (low + high) / 2.0;

        if (hasPerfectMatchingKD(finite1, finite2, tree2, mid))
        {
            result = mid;
            high = mid - EPSILON;
        }
        else
        {
            low = mid + EPSILON;
        }
    }

    return result;
}

double adaptiveBottleneckDistance(const std::vector<std::pair<float, float>> &diagram1,
                                  const std::vector<std::pair<float, float>> &diagram2)
{
    validateDiagram(diagram1, "first");
    validateDiagram(diagram2, "second");
    validateMatchingSize(diagram1.size(), diagram2.size());

    if (diagram1.empty() && diagram2.empty())
        return 0.0;

    // Separate essential (infinite death) from finite points.
    // The KD-tree algorithm cannot handle inf-inf=NaN in distance calculations.
    std::vector<std::pair<float, float>> finite1, essential1;
    std::vector<std::pair<float, float>> finite2, essential2;
    for (const auto &p : diagram1)
    {
        if (std::isinf(p.second))
            essential1.push_back(p);
        else
            finite1.push_back(p);
    }
    for (const auto &p : diagram2)
    {
        if (std::isinf(p.second))
            essential2.push_back(p);
        else
            finite2.push_back(p);
    }

    // Match essential classes by birth coordinate.
    // Different numbers of essential classes -> infinite distance.
    double essential_cost = 0.0;
    if (essential1.size() != essential2.size())
    {
        return std::numeric_limits<double>::infinity();
    }
    if (!essential1.empty())
    {
        std::vector<float> births1, births2;
        for (const auto &p : essential1)
            births1.push_back(p.first);
        for (const auto &p : essential2)
            births2.push_back(p.first);
        std::sort(births1.begin(), births1.end());
        std::sort(births2.begin(), births2.end());
        for (size_t i = 0; i < births1.size(); ++i)
        {
            essential_cost =
                std::max(essential_cost, static_cast<double>(std::abs(births1[i] - births2[i])));
        }
    }

    // If only essential classes, return essential cost directly.
    if (finite1.empty() && finite2.empty())
        return essential_cost;

    // Compute bottleneck on finite points only, then combine.
    double finite_cost = bottleneckFinitePoints(finite1, finite2);

    return std::max(essential_cost, finite_cost);
}

bool hasPerfectMatchingKD(const std::vector<std::pair<float, float>> &diagram1,
                          const std::vector<std::pair<float, float>> &diagram2,
                          const DiagramKDTree &tree2, double threshold)
{
    if (!std::isfinite(threshold) || threshold < 0.0)
    {
        throw std::invalid_argument("bottleneck threshold must be finite and non-negative");
    }

    size_t n1 = diagram1.size();
    size_t n2 = diagram2.size();
    size_t n = n1 + n2;

    std::vector<std::vector<size_t>> adjacency(n);
    constexpr double EPSILON = 1e-12;

    for (size_t i = 0; i < n1; ++i)
    {
        auto matches = tree2.rangeQuery(diagram1[i].first, diagram1[i].second, threshold + EPSILON);
        for (int j : matches)
        {
            adjacency[i].push_back(j);
        }

        const double diag_dist = diagonalDistance(diagram1[i]);
        if (diag_dist <= threshold + EPSILON)
        {
            adjacency[i].push_back(n2 + i);
        }
    }

    for (size_t j = 0; j < n2; ++j)
    {
        const double diag_dist = diagonalDistance(diagram2[j]);
        if (diag_dist <= threshold + EPSILON)
        {
            adjacency[n1 + j].push_back(j);
        }
        for (size_t i = 0; i < n1; ++i)
        {
            adjacency[n1 + j].push_back(n2 + i);
        }
    }

    std::vector<int> match(n, -1);
    size_t matched = 0;

    for (size_t left = 0; left < n; ++left)
    {
        std::vector<bool> seen(n, false);
        if (tryAugment(left, adjacency, seen, match))
        {
            matched++;
        }
    }

    return matched == n;
}

bool tryAugment(size_t left, const std::vector<std::vector<size_t>> &adjacency,
                std::vector<bool> &seen, std::vector<int> &match)
{
    for (size_t right : adjacency[left])
    {
        if (seen[right])
            continue;
        seen[right] = true;

        if (match[right] < 0 || tryAugment(match[right], adjacency, seen, match))
        {
            match[right] = static_cast<int>(left);
            return true;
        }
    }
    return false;
}

std::vector<double>
parallelBottleneckDistances(const std::vector<std::vector<std::pair<float, float>>> &diagrams1,
                            const std::vector<std::vector<std::pair<float, float>>> &diagrams2)
{
    if (diagrams1.size() != diagrams2.size())
    {
        throw std::invalid_argument("diagram batches must have the same size");
    }
    size_t n = diagrams1.size();        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i)
    {
        validateDiagram(diagrams1[i], "first");
        validateDiagram(diagrams2[i], "second");
        validateMatchingSize(diagrams1[i].size(), diagrams2[i].size());
    }

    std::vector<double> results(n);

#pragma omp parallel for schedule(dynamic)        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i)
    {
        results[i] = adaptiveBottleneckDistance(diagrams1[i], diagrams2[i]);
    }

    return results;
}

} // namespace nerve::metrics::bottleneck

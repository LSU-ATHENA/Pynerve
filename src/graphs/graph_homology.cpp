#include "graph_persistent_homology_detail.hpp"

namespace nerve::graphs
{

using detail::ComponentForest;
using detail::Interval;

GraphHomology::GraphHomology(const Graph &graph)
    : graph_(&graph)
{
    buildMatrices();
}

std::vector<int> GraphHomology::computeBettiNumbers() const
{
    if (graph_ == nullptr)
        return {};
    const int b0 = static_cast<int>(graph_->getConnectedComponents().size());
    const int e = static_cast<int>(graph_->numEdges());
    const int v = static_cast<int>(graph_->numVertices());
    const int b1 = std::max(0, e - v + b0);
    return {b0, b1};
}

std::vector<std::vector<int>> GraphHomology::computeHomologyGroups() const
{
    auto betti = computeBettiNumbers();
    std::vector<std::vector<int>> groups;
    groups.reserve(betti.size());
    for (int b : betti)
        groups.push_back(std::vector<int>(std::max(0, b), 1));
    return groups;
}

std::vector<std::vector<double>> GraphHomology::computeCohomologyGroups() const
{
    auto groups = computeHomologyGroups();
    std::vector<std::vector<double>> out;
    out.reserve(groups.size());
    for (const auto &g : groups)
        out.push_back(std::vector<double>(g.size(), 1.0));
    return out;
}

std::vector<std::pair<double, double>> GraphHomology::computePersistentHomology(
    const std::vector<std::pair<double, std::vector<Index>>> &filtration) const
{
    auto ordered = filtration;
    std::stable_sort(ordered.begin(), ordered.end(),
                     [](const auto &left, const auto &right) { return left.first < right.first; });

    ComponentForest components;
    std::map<Index, Size> vertex_to_component;
    std::vector<Interval> pairs;

    auto ensure_vertex = [&](Index vertex, double birth_time) {
        if (vertex < 0)
        {
            throw std::invalid_argument("filtration vertex index must be non-negative");
        }
        const auto [it, inserted] = vertex_to_component.emplace(vertex, 0);
        if (inserted)
        {
            it->second = components.add(birth_time);
        }
        return it->second;
    };

    for (const auto &[time, simplex] : ordered)
    {
        if (!std::isfinite(time))
        {
            throw std::invalid_argument("filtration time must be finite");
        }
        if (simplex.empty())
        {
            continue;
        }
        if (simplex.size() == 1)
        {
            ensure_vertex(simplex.front(), time);
            continue;
        }
        const Size first = ensure_vertex(simplex[0], time);
        for (Size i = 1; i < simplex.size(); ++i)
        {
            components.unite(first, ensure_vertex(simplex[i], time), time, pairs);
        }
    }

    std::vector<bool> emitted(components.parent.size(), false);
    for (const auto &entry : vertex_to_component)
    {
        const Size component = entry.second;
        const Size root = components.find(component);
        if (!emitted[root])
        {
            pairs.emplace_back(components.birth[root], std::numeric_limits<double>::infinity());
            emitted[root] = true;
        }
    }

    std::sort(pairs.begin(), pairs.end(), [](const auto &left, const auto &right) {
        if (left.first != right.first)
            return left.first < right.first;
        return left.second < right.second;
    });
    return pairs;
}

std::vector<Index> GraphHomology::findCriticalVertices() const
{
    std::vector<Index> out;
    if (graph_ == nullptr)
        return out;
    auto degrees = graph_->getDegreeSequence();
    for (Size i = 0; i < degrees.size(); ++i)
        if (degrees[i] != 2.0)
            out.push_back(static_cast<Index>(i));
    return out;
}

std::vector<int> GraphHomology::computeMorseIndex() const
{
    std::vector<int> out;
    if (graph_ == nullptr)
        return out;
    auto deg = graph_->getDegreeSequence();
    out.reserve(deg.size());
    for (double d : deg)
        out.push_back(static_cast<int>(d));
    return out;
}

std::vector<std::vector<double>> GraphHomology::computeMorseFunctions() const
{
    if (graph_ == nullptr)
        return {};
    auto deg = graph_->getDegreeSequence();
    std::vector<std::vector<double>> out(deg.size(), std::vector<double>(1));
    for (Size i = 0; i < deg.size(); ++i)
        out[i][0] = deg[i];
    return out;
}

void GraphHomology::buildMatrices()
{
    if (graph_ == nullptr)
        return;
    laplacian_ = graph_->getLaplacianMatrix();
    boundary_matrix_ = graph_->getAdjacencyMatrix();
    coboundary_matrix_ = boundary_matrix_;
}

std::vector<std::vector<double>> GraphHomology::computeChainComplex() const
{
    return boundary_matrix_;
}

} // namespace nerve::graphs

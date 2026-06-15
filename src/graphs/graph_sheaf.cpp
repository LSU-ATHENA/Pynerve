#include "graph_persistent_homology_detail.hpp"

namespace nerve::graphs
{

using detail::paddedSection;
using detail::sectionDifference;
using detail::SectionMap;
using detail::sectionWidth;
using detail::validateFiniteValues;
using detail::validateVertex;

GraphSheaf::GraphSheaf(const Graph &graph)
    : graph_(&graph)
{}

void GraphSheaf::assignSection(Index vertex, const std::vector<double> &section)
{
    validateVertex(graph_, vertex);
    validateFiniteValues(section, "section");
    sections_[vertex] = section;
}

void GraphSheaf::assignCochain(const std::vector<Index> &vertices,
                               const std::vector<double> &values)
{
    for (Index vertex : vertices)
    {
        validateVertex(graph_, vertex);
    }
    validateFiniteValues(values, "cochain");
    cochains_[vertices] = values;
}

std::vector<std::vector<double>> GraphSheaf::computeSheafCohomology() const
{
    if (graph_ == nullptr)
        return {};
    const Size width = sectionWidth(sections_);
    std::vector<std::vector<double>> representatives;
    for (const auto &component : graph_->getConnectedComponents())
    {
        std::vector<double> average(width, 0.0);
        Size assigned = 0;
        for (Index vertex : component)
        {
            const auto it = sections_.find(vertex);
            if (it == sections_.end())
            {
                continue;
            }
            const Size copy_width = std::min(width, it->second.size());
            for (Size i = 0; i < copy_width; ++i)
            {
                average[i] += it->second[i];
            }
            ++assigned;
        }
        if (assigned != 0)
        {
            for (double &value : average)
            {
                value /= static_cast<double>(assigned);
            }
        }
        representatives.push_back(std::move(average));
    }
    return representatives;
}

std::vector<std::vector<double>> GraphSheaf::computeLocalCohomology(Index vertex) const
{
    validateVertex(graph_, vertex);
    if (graph_ == nullptr)
        return {};
    const Size width = sectionWidth(sections_);
    std::vector<std::vector<double>> local;
    local.push_back(paddedSection(sections_, vertex, width));
    for (Index neighbor : graph_->getNeighbors(vertex))
    {
        local.push_back(sectionDifference(sections_, vertex, neighbor, width));
    }
    return local;
}

std::vector<std::vector<double>> GraphSheaf::computeRestrictionMaps() const
{
    return computeSheafBoundary();
}

std::vector<std::vector<double>>
GraphSheaf::restrictSection(const std::vector<double> &section,
                            const std::vector<Index> &vertices) const
{
    std::vector<std::vector<double>> out;
    for (Index v : vertices)
    {
        validateVertex(graph_, v);
        if (static_cast<Size>(v) < section.size())
            out.push_back({section[static_cast<Size>(v)]});
        else
            out.push_back({0.0});
    }
    return out;
}

std::vector<double> GraphSheaf::getStalk(Index vertex) const
{
    validateVertex(graph_, vertex);
    auto it = sections_.find(vertex);
    return it == sections_.end() ? std::vector<double>{} : it->second;
}

std::vector<std::vector<double>> GraphSheaf::getGerms(Index vertex) const
{
    std::vector<std::vector<double>> out;
    if (graph_ == nullptr)
        return out;
    for (Index n : graph_->getNeighbors(vertex))
        out.push_back(getStalk(n));
    return out;
}

std::vector<std::vector<double>> GraphSheaf::computeSheafCoboundary() const
{
    if (graph_ == nullptr)
        return {};
    const Size width = sectionWidth(sections_);
    std::vector<std::vector<double>> rows;
    for (const auto &[u, v] : graph_->getEdges())
    {
        rows.push_back(sectionDifference(sections_, u, v, width));
    }
    for (const auto &[vertices, values] : cochains_)
    {
        const Size cochain_width = std::max(width, values.size());
        std::vector<double> residual(cochain_width, 0.0);
        for (Size i = 0; i < values.size(); ++i)
        {
            residual[i] = -values[i];
        }
        for (Size i = 0; i < vertices.size(); ++i)
        {
            const auto section = paddedSection(sections_, vertices[i], cochain_width);
            const double sign = (i % 2 == 0) ? 1.0 : -1.0;
            for (Size j = 0; j < cochain_width; ++j)
            {
                residual[j] += sign * section[j];
            }
        }
        rows.push_back(std::move(residual));
    }
    return rows;
}

std::vector<std::vector<double>> GraphSheaf::computeSheafBoundary() const
{
    if (graph_ == nullptr)
        return {};
    std::vector<std::vector<double>> rows;
    rows.reserve(graph_->numEdges());
    for (const auto &[u, v] : graph_->getEdges())
    {
        std::vector<double> row(graph_->numVertices(), 0.0);
        row[static_cast<Size>(u)] = -1.0;
        row[static_cast<Size>(v)] = 1.0;
        rows.push_back(std::move(row));
    }
    return rows;
}

} // namespace nerve::graphs

#include "nerve/graphs/graph.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <ranges>
#include <stdexcept>

namespace nerve::graphs
{
namespace
{
constexpr double kPersistenceDiagramConnectivityTolerance = 1e-9;

WeightedGraph makeWeightedCopy(const Graph &graph)
{
    WeightedGraph weighted(graph.numVertices());
    for (auto [u, v] : graph.getEdges())
    {
        weighted.addEdge(u, v, graph.getEdgeWeight(u, v));
    }
    return weighted;
}
} // namespace

Graph GraphTopology::fromSimplicialComplex(const algebra::SimplicialComplex &complex)
{
    Graph g;
    for (const auto &simplex : complex.getSimplices())
    {
        const auto &verts = simplex.vertices();
        if (verts.empty())
        {
            continue;
        }
        if (std::ranges::any_of(verts, [](Index vertex) { return vertex < 0; }))
        {
            throw std::invalid_argument("Simplicial graph vertices must be non-negative");
        }
        const Index max_v = *std::max_element(verts.begin(), verts.end());
        while (g.numVertices() <= static_cast<Size>(max_v))
            g.addVertex();
        for (Size i = 0; i < verts.size(); ++i)
            for (Size j = i + 1; j < verts.size(); ++j)
                g.addEdge(verts[i], verts[j]);
    }
    return g;
}

Graph GraphTopology::fromCellularComplex(const algebra::CellularComplex &complex)
{
    Graph g(complex.numCells());
    for (Size i = 0; i < complex.numCells(); ++i)
        for (Index b : complex.getCell(static_cast<Index>(i)).boundary())
            if (b >= 0 && static_cast<Size>(b) < complex.numCells())
                g.addEdge(static_cast<Index>(i), b);
    return g;
}

Graph GraphTopology::fromPersistenceDiagram(const std::vector<std::pair<double, double>> &diagram)
{
    Graph g(diagram.size());
    for (const auto &[birth, death] : diagram)
    {
        const bool finite_death = std::isfinite(death);
        if (!std::isfinite(birth) ||
            (!finite_death && death != std::numeric_limits<double>::infinity()) ||
            (finite_death && death < birth))
        {
            throw std::invalid_argument("Persistence diagram contains invalid pairs");
        }
    }
    for (Size i = 0; i + 1 < diagram.size(); ++i)
        if (std::abs(diagram[i].second - diagram[i + 1].first) <
            kPersistenceDiagramConnectivityTolerance)
            g.addEdge(static_cast<Index>(i), static_cast<Index>(i + 1));
    return g;
}

std::vector<double> GraphTopology::computeGraphPersistence(const Graph &graph)
{
    return graph.getDegreeSequence();
}

std::vector<std::vector<double>> GraphTopology::computeGraphFiltration(const Graph &graph)
{
    return graph.getAdjacencyMatrix();
}

std::vector<int> GraphTopology::computeGraphBetti(const Graph &graph)
{
    return GraphHomology(graph).computeBettiNumbers();
}

double GraphTopology::computeGraphDistance(const Graph &graph1, const Graph &graph2)
{
    return computeWassersteinDistance(graph1, graph2);
}

double GraphTopology::computeGromovHausdorffDistance(const Graph &graph1, const Graph &graph2)
{
    const double vertex_gap = std::abs(static_cast<double>(graph1.numVertices()) -
                                       static_cast<double>(graph2.numVertices()));
    const Size scale = std::max(graph1.numVertices(), graph2.numVertices());
    if (scale == 0)
    {
        return 0.0;
    }
    return std::max(vertex_gap,
                    computeWassersteinDistance(graph1, graph2) / static_cast<double>(scale));
}

double GraphTopology::computeWassersteinDistance(const Graph &graph1, const Graph &graph2)
{
    auto d1 = graph1.getDegreeSequence();
    auto d2 = graph2.getDegreeSequence();
    std::ranges::sort(d1);
    std::ranges::sort(d2);
    const Size n = std::min(d1.size(), d2.size());
    double sum = 0.0;
    for (Size i = 0; i < n; ++i)
        sum += std::abs(d1[i] - d2[i]);
    for (Size i = n; i < d1.size(); ++i)
        sum += std::abs(d1[i]);
    for (Size i = n; i < d2.size(); ++i)
        sum += std::abs(d2[i]);
    return sum;
}

std::vector<int> GraphTopology::computeGraphInvariants(const Graph &graph)
{
    return {static_cast<int>(graph.numVertices()), static_cast<int>(graph.numEdges()),
            static_cast<int>(graph.getConnectedComponents().size())};
}

std::vector<double> GraphTopology::computeSpectralInvariants(const Graph &graph)
{
    return makeWeightedCopy(graph).computeEigenvalues();
}

std::vector<std::complex<double>> GraphTopology::computeComplexInvariants(const Graph &graph)
{
    std::vector<std::complex<double>> out;
    for (double v : graph.getDegreeSequence())
        out.emplace_back(v, 0.0);
    return out;
}

} // namespace nerve::graphs

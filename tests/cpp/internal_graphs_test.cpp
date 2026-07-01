
#include "nerve/graphs/detail/graphs_detail.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

namespace
{

using nerve::algebra::Simplex;
using nerve::graphs::Graph;
using nerve::graphs::GraphNeuralLayer;
using nerve::graphs::GraphTopology;
using nerve::graphs::PersistentGraph;
using nerve::graphs::SimplicialGraph;
using nerve::graphs::WeightedGraph;

constexpr double TOL = 1e-10;

bool check_graph_weighted_ops()
{
    WeightedGraph g(4);
    g.addEdge(0, 1, 2.5);
    g.addEdge(1, 2, 1.5);
    g.addEdge(2, 3, 3.0);
    g.addEdge(0, 3, 0.5);

    if (g.numVertices() != 4)
    {
        return false;
    }
    if (g.numEdges() != 4)
    {
        return false;
    }

    if (std::abs(g.getEdgeWeight(0, 1) - 2.5) > TOL)
    {
        return false;
    }
    if (std::abs(g.getEdgeWeight(2, 3) - 3.0) > TOL)
    {
        return false;
    }

    auto neighbors = g.getNeighbors(1);
    bool found_0 = false;
    bool found_2 = false;
    for (auto v : neighbors)
    {
        if (v == 0)
            found_0 = true;
        if (v == 2)
            found_2 = true;
    }
    if (!found_0 || !found_2)
    {
        return false;
    }

    g.setEdgeWeight(0, 1, 5.0);
    if (std::abs(g.getEdgeWeight(0, 1) - 5.0) > TOL)
    {
        return false;
    }

    g.normalizeWeights();
    if (g.getEdgeWeight(0, 1) > 1.0 + TOL)
    {
        return false;
    }

    auto laplacian = g.getWeightedLaplacian();
    if (laplacian.size() != 4)
    {
        return false;
    }

    return true;
}

bool check_graph_persistent_homology()
{
    PersistentGraph pg(3);
    pg.addEdgePersistent(0, 1, 1.0);
    pg.addEdgePersistent(1, 2, 2.0);
    pg.advanceTime(1.5);

    if (std::abs(pg.getCurrentTime() - 1.5) > TOL)
    {
        return false;
    }

    auto events = pg.getPersistenceEvents();
    if (events.size() != 2)
    {
        return false;
    }

    auto diagram = pg.getPersistenceDiagram();
    if (diagram.empty())
    {
        return false;
    }

    pg.resetPersistence();
    if (std::abs(pg.getCurrentTime()) > TOL)
    {
        return false;
    }

    Graph g(3);
    g.addEdge(0, 1);
    g.addEdge(1, 2);
    nerve::graphs::GraphHomology gh(g);
    auto betti = gh.computeBettiNumbers();
    if (betti.size() >= 2)
    {
        if (betti[0] != 1)
        {
            return false;
        }
    }

    auto critical = gh.findCriticalVertices();
    if (critical.empty())
    {
        return false;
    }

    auto morse = gh.computeMorseIndex();
    if (morse.size() != 3)
    {
        return false;
    }

    return true;
}

bool check_graph_topology_components()
{
    Graph g(5);
    g.addEdge(0, 1);
    g.addEdge(1, 2);
    g.addEdge(3, 4);

    auto components = g.getConnectedComponents();
    if (components.size() != 2)
    {
        return false;
    }

    if (!g.isConnected())
    {
        Graph g2(3);
        g2.addEdge(0, 1);
        g2.addEdge(1, 2);
        if (!g2.isConnected())
        {
            return false;
        }
    }

    auto degree_seq = g.getDegreeSequence();
    if (degree_seq.size() != 5)
    {
        return false;
    }

    auto betti = GraphTopology::computeGraphBetti(g);
    if (betti.empty())
    {
        return false;
    }

    auto invariants = GraphTopology::computeGraphInvariants(g);
    if (invariants.size() < 2)
    {
        return false;
    }

    auto spectral = GraphTopology::computeSpectralInvariants(g);
    (void)spectral;

    return true;
}

bool check_graph_neural_layer_config()
{
    Graph g(3);
    g.addEdge(0, 1);
    g.addEdge(1, 2);
    g.addEdge(0, 2);

    GraphNeuralLayer layer(g, 4, 2);

    std::vector<std::vector<double>> input = {
        {1.0, 0.0, 0.0, 0.0}, {0.0, 1.0, 0.0, 0.0}, {0.0, 0.0, 1.0, 0.0}};
    auto output = layer.forward(input);
    if (output.size() != 3)
    {
        return false;
    }
    for (const auto &row : output)
    {
        if (row.size() != 2)
        {
            return false;
        }
    }

    std::vector<std::vector<double>> grad = {{1.0, 0.0}, {0.0, 1.0}, {0.5, 0.5}};
    auto backward = layer.backward(grad);
    if (backward.size() != 3)
    {
        return false;
    }

    std::vector<std::vector<double>> weights = {{1.0, 0.5}, {-0.5, 1.0}, {0.0, 0.0}, {0.0, 0.0}};
    auto conv = layer.graphConvolution(input, weights);
    if (conv.empty())
    {
        return false;
    }

    return true;
}

} // namespace

int main()
{
    int failures = 0;

    auto run = [&](const char *name, bool ok) {
        if (!ok)
        {
            std::cerr << "FAIL: " << name << "\n";
            ++failures;
        }
        else
        {
            std::cout << "PASS: " << name << "\n";
        }
    };

    run("graph_weighted_ops", check_graph_weighted_ops());
    run("graph_persistent_homology", check_graph_persistent_homology());
    run("graph_topology_components", check_graph_topology_components());
    run("graph_neural_layer_config", check_graph_neural_layer_config());

    return failures > 0 ? 1 : 0;
}

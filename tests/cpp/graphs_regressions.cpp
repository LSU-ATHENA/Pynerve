#include "nerve/algebra/complex.hpp"
#include "nerve/algebra/simplex.hpp"
#include "nerve/common/accelerated_types.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core_types.hpp"
#include "nerve/graphs/gpu_graphs.hpp"
#include "nerve/graphs/graph.hpp"
#include "test_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <random>
#include <tuple>
#include <vector>

namespace
{

using nerve::algebra::Simplex;
using nerve::graphs::Graph;
using nerve::graphs::SimplicialGraph;
using nerve::graphs::WeightedGraph;
using namespace nerve::test;


bool check_graph_construction()
{
    Graph g(3);
    if (g.numVertices() != 3)
    {
        std::cerr << "expected 3 vertices, got " << g.numVertices() << "\n";
        return false;
    }
    if (g.numEdges() != 0)
    {
        std::cerr << "expected 0 edges, got " << g.numEdges() << "\n";
        return false;
    }
    return true;
}

bool check_add_edge()
{
    Graph g(3);
    g.addEdge(0, 1);
    g.addEdge(1, 2);

    if (g.numEdges() != 2)
    {
        std::cerr << "expected 2 edges, got " << g.numEdges() << "\n";
        return false;
    }
    return true;
}

bool check_get_neighbors()
{
    Graph g(4);
    g.addEdge(0, 1);
    g.addEdge(0, 2);
    g.addEdge(0, 3);

    auto neigh = g.getNeighbors(0);
    if (neigh.size() != 3)
    {
        std::cerr << "expected 3 neighbors, got " << neigh.size() << "\n";
        return false;
    }
    return true;
}

bool check_is_connected_connected()
{
    Graph g(3);
    g.addEdge(0, 1);
    g.addEdge(1, 2);

    if (!g.isConnected())
    {
        std::cerr << "triangle path should be connected\n";
        return false;
    }
    return true;
}

bool check_is_connected_disconnected()
{
    Graph g(4);
    g.addEdge(0, 1);
    g.addEdge(2, 3);

    if (g.isConnected())
    {
        std::cerr << "two separate edges should not be connected\n";
        return false;
    }
    return true;
}

bool check_connected_components()
{
    Graph g(5);
    g.addEdge(0, 1);
    g.addEdge(2, 3);

    auto comps = g.getConnectedComponents();
    if (comps.empty())
    {
        std::cerr << "should have at least one component\n";
        return false;
    }
    return true;
}

bool check_weighted_graph_eigenvalues()
{
    WeightedGraph wg(3);
    wg.addEdge(0, 1, 1.0);
    wg.addEdge(1, 2, 1.0);

    auto evals = wg.computeEigenvalues();
    if (evals.empty())
    {
        std::cerr << "eigenvalues should not be empty\n";
        return false;
    }
    for (auto e : evals)
    {
        if (!std::isfinite(e))
        {
            std::cerr << "non-finite eigenvalue\n";
            return false;
        }
    }
    return true;
}

bool check_simplicial_graph_1_skeleton()
{
    SimplicialGraph sg;
    sg.addSimplex(Simplex({0, 1}));
    sg.addSimplex(Simplex({1, 2}));
    sg.addSimplex(Simplex({0, 2}));

    Graph skel = sg.get1Skeleton();
    if (skel.numVertices() < 3)
    {
        std::cerr << "1-skeleton should have at least 3 vertices\n";
        return false;
    }
    if (skel.numEdges() < 3)
    {
        std::cerr << "1-skeleton should have at least 3 edges\n";
        return false;
    }
    return true;
}

bool check_weighted_graph_construction()
{
    WeightedGraph wg(5);
    if (wg.numVertices() != 5)
    {
        std::cerr << "expected 5 vertices\n";
        return false;
    }
    return true;
}

bool check_graph_remove_edge()
{
    Graph g(3);
    g.addEdge(0, 1);
    g.addEdge(0, 2);
    g.removeEdge(0, 1);

    if (g.numEdges() != 1)
    {
        std::cerr << "expected 1 edge after removal, got " << g.numEdges() << "\n";
        return false;
    }
    return true;
}

} // namespace

int main()
{
    if (!check_graph_construction())
    {
        std::cerr << "FAIL: Graph construction\n";
        return 1;
    }
    if (!check_add_edge())
    {
        std::cerr << "FAIL: addEdge\n";
        return 1;
    }
    if (!check_get_neighbors())
    {
        std::cerr << "FAIL: getNeighbors\n";
        return 1;
    }
    if (!check_is_connected_connected())
    {
        std::cerr << "FAIL: isConnected (connected)\n";
        return 1;
    }
    if (!check_is_connected_disconnected())
    {
        std::cerr << "FAIL: isConnected (disconnected)\n";
        return 1;
    }
    if (!check_connected_components())
    {
        std::cerr << "FAIL: getConnectedComponents\n";
        return 1;
    }
    if (!check_weighted_graph_eigenvalues())
    {
        std::cerr << "FAIL: WeightedGraph eigenvalues\n";
        return 1;
    }
    if (!check_simplicial_graph_1_skeleton())
    {
        std::cerr << "FAIL: SimplicialGraph get1Skeleton\n";
        return 1;
    }
    if (!check_weighted_graph_construction())
    {
        std::cerr << "FAIL: WeightedGraph construction\n";
        return 1;
    }
    if (!check_graph_remove_edge())
    {
        std::cerr << "FAIL: removeEdge\n";
        return 1;
    }
    return 0;
}


#pragma once
#include "nerve/algebra/cellular.hpp"
#include "nerve/algebra/complex.hpp"
#include "nerve/algebra/simplex.hpp"
#include "nerve/core_types.hpp"

#include <complex>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>
namespace nerve::graphs
{
class Graph
{
public:
    Graph() = default;
    explicit Graph(Size numVertices);
    void addVertex();
    void addVertices(Size count);
    void removeVertex(Index vertex);
    void addEdge(Index u, Index v, double weight = 1.0);
    void removeEdge(Index u, Index v);
    void setEdgeWeight(Index u, Index v, double weight);
    Size numVertices() const;
    Size numEdges() const;
    std::vector<Index> getVertices() const;
    std::vector<std::pair<Index, Index>> getEdges() const;
    std::vector<Index> getNeighbors(Index vertex) const;
    std::vector<Index> getAdjacentVertices(Index vertex) const;
    double getEdgeWeight(Index u, Index v) const;
    std::vector<std::vector<double>> getAdjacencyMatrix() const;
    std::vector<std::vector<double>> getLaplacianMatrix() const;
    std::vector<double> getDegreeSequence() const;
    bool isConnected() const;
    std::vector<std::vector<Index>> getConnectedComponents() const;
    Index getComponentId(Index vertex) const;

private:
    Size num_vertices_ = 0;
    std::vector<std::vector<double>> adjacency_matrix_;
    std::vector<std::vector<Index>> neighbors_;
    mutable std::vector<Index> component_ids_;
    mutable bool components_valid_ = false;
    void invalidateComponents() const;
    void computeConnectedComponents() const;
};
class WeightedGraph : public Graph
{
public:
    WeightedGraph() = default;
    explicit WeightedGraph(Size numVertices);
    void addVertex();
    void addVertices(Size count);
    void removeVertex(Index vertex);
    void addEdge(Index u, Index v, double weight = 1.0);
    void removeEdge(Index u, Index v);
    void setEdgeWeight(Index u, Index v, double weight);
    void setAllWeights(double weight);
    void normalizeWeights();
    void applyDistanceWeights(const std::vector<std::vector<double>> &points);
    std::vector<std::vector<double>> getWeightedLaplacian() const;
    std::vector<double> computeEigenvalues(Size k = 0) const;
    std::vector<std::vector<double>> computeEigenvectors(Size k = 0) const;
    std::vector<int> spectralCluster(Size k) const;
    std::vector<int> normalizedSpectralCluster(Size k) const;

private:
    std::vector<std::vector<double>> weighted_laplacian_;
    bool weighted_laplacian_valid_ = false;
    void computeWeightedLaplacian();
    void invalidateWeightedLaplacian();
};
class SimplicialGraph
{
public:
    SimplicialGraph() = default;
    void addSimplex(const algebra::Simplex &simplex);
    void addSimplices(const std::vector<algebra::Simplex> &simplices);
    void removeSimplex(const algebra::Simplex &simplex);
    Graph get1Skeleton() const;
    WeightedGraph getWeighted1Skeleton() const;
    Graph getCliqueGraph() const;
    Size numSimplices() const;
    int maxDimension() const;
    std::vector<Index> getSimplicesOfDimension(int dimension) const;
    std::vector<Index> getSimplexNeighbors(const algebra::Simplex &simplex) const;
    std::vector<Index> getSimplexStar(const algebra::Simplex &simplex) const;
    std::vector<Index> getSimplexLink(const algebra::Simplex &simplex) const;

private:
    std::vector<algebra::Simplex> simplices_;
    std::unordered_map<algebra::Simplex, Index, algebra::Simplex::Hash> simplex_to_index_;
    mutable Graph skeleton_cache_;
    mutable bool skeleton_cache_valid_ = false;
    void invalidateSkeletonCache();
    void buildSkeletonCache() const;
};
class PersistentGraph : public WeightedGraph
{
public:
    PersistentGraph() = default;
    explicit PersistentGraph(Size numVertices);
    void addVertexPersistent(Index vertex);
    void removeVertexPersistent(Index vertex);
    void addEdgePersistent(Index u, Index v, double weight = 1.0);
    void removeEdgePersistent(Index u, Index v);
    std::vector<std::pair<double, std::string>> getPersistenceEvents() const;
    std::vector<double> getPersistenceDiagram() const;
    double computePersistenceDistance(const PersistentGraph &other) const;
    void advanceTime(double time_step);
    double getCurrentTime() const;
    void resetPersistence();

private:
    double current_time_ = 0.0;
    std::vector<std::pair<double, std::string>> persistence_events_;
    std::map<std::string, double> last_event_times_;
    void recordEvent(const std::string &event_type, double time);
};
class GraphSheaf
{
public:
    GraphSheaf() = default;
    explicit GraphSheaf(const Graph &graph);
    void assignSection(Index vertex, const std::vector<double> &section);
    void assignCochain(const std::vector<Index> &vertices, const std::vector<double> &values);
    std::vector<std::vector<double>> computeSheafCohomology() const;
    std::vector<std::vector<double>> computeLocalCohomology(Index vertex) const;
    std::vector<std::vector<double>> computeRestrictionMaps() const;
    std::vector<std::vector<double>> restrictSection(const std::vector<double> &section,
                                                     const std::vector<Index> &vertices) const;
    std::vector<double> getStalk(Index vertex) const;
    std::vector<std::vector<double>> getGerms(Index vertex) const;

private:
    const Graph *graph_ = nullptr;
    std::map<Index, std::vector<double>> sections_;
    std::map<std::vector<Index>, std::vector<double>> cochains_;
    std::vector<std::vector<double>> computeSheafCoboundary() const;
    std::vector<std::vector<double>> computeSheafBoundary() const;
};
class GraphHomology
{
public:
    GraphHomology() = default;
    explicit GraphHomology(const Graph &graph);
    std::vector<int> computeBettiNumbers() const;
    std::vector<std::vector<int>> computeHomologyGroups() const;
    std::vector<std::vector<double>> computeCohomologyGroups() const;
    std::vector<std::pair<double, double>> computePersistentHomology(
        const std::vector<std::pair<double, std::vector<Index>>> &filtration) const;
    std::vector<Index> findCriticalVertices() const;
    std::vector<int> computeMorseIndex() const;
    std::vector<std::vector<double>> computeMorseFunctions() const;

private:
    const Graph *graph_ = nullptr;
    std::vector<std::vector<double>> laplacian_;
    std::vector<std::vector<double>> boundary_matrix_;
    std::vector<std::vector<double>> coboundary_matrix_;
    void buildMatrices();
    std::vector<std::vector<double>> computeChainComplex() const;
};
class GraphNeuralLayer
{
public:
    GraphNeuralLayer() = default;
    explicit GraphNeuralLayer(const Graph &graph, Size input_dim, Size output_dim);
    std::vector<std::vector<double>> forward(const std::vector<std::vector<double>> &input) const;
    std::vector<std::vector<double>>
    backward(const std::vector<std::vector<double>> &grad_output) const;
    std::vector<std::vector<double>>
    graphConvolution(const std::vector<std::vector<double>> &input,
                     const std::vector<std::vector<double>> &weights) const;
    std::vector<std::vector<double>>
    graphAttention(const std::vector<std::vector<double>> &input,
                   const std::vector<std::vector<double>> &queries,
                   const std::vector<std::vector<double>> &keys,
                   const std::vector<std::vector<double>> &values) const;

private:
    const Graph *graph_ = nullptr;
    Size input_dim_ = 0;
    Size output_dim_ = 0;
    std::vector<std::vector<double>> weights_;
    std::vector<std::vector<double>> adjacency_matrix_;
    void initializeWeights();
    std::vector<std::vector<double>>
    computeMessagePassing(const std::vector<std::vector<double>> &input) const;
};
class GraphTopology
{
public:
    static Graph fromSimplicialComplex(const algebra::SimplicialComplex &complex);
    static Graph fromCellularComplex(const algebra::CellularComplex &complex);
    static Graph fromPersistenceDiagram(const std::vector<std::pair<double, double>> &diagram);
    static std::vector<double> computeGraphPersistence(const Graph &graph);
    static std::vector<std::vector<double>> computeGraphFiltration(const Graph &graph);
    static std::vector<int> computeGraphBetti(const Graph &graph);
    static double computeGraphDistance(const Graph &graph1, const Graph &graph2);
    static double computeGromovHausdorffDistance(const Graph &graph1, const Graph &graph2);
    static double computeWassersteinDistance(const Graph &graph1, const Graph &graph2);
    static std::vector<int> computeGraphInvariants(const Graph &graph);
    static std::vector<double> computeSpectralInvariants(const Graph &graph);
    static std::vector<std::complex<double>> computeComplexInvariants(const Graph &graph);
};
} // namespace nerve::graphs

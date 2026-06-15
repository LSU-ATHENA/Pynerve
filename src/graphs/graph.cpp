
#include "nerve/graphs/graph.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <queue>
#include <ranges>
#include <stdexcept>

namespace nerve::graphs
{
namespace
{

std::vector<std::vector<double>> makeSquareMatrix(Size n, double value = 0.0)
{
    return std::vector<std::vector<double>>(n, std::vector<double>(n, value));
}

} // namespace

Graph::Graph(Size numVertices)
    : num_vertices_(numVertices)
    , adjacency_matrix_(makeSquareMatrix(numVertices))
    , neighbors_(numVertices)
    , components_valid_(false)
{}

void Graph::addVertex()
{
    addVertices(1);
}

void Graph::addVertices(Size count)
{
    if (count == 0)
    {
        return;
    }
    const Size old_n = num_vertices_;
    num_vertices_ += count;
    for (auto &row : adjacency_matrix_)
    {
        row.resize(num_vertices_, 0.0);
    }
    adjacency_matrix_.resize(num_vertices_, std::vector<double>(num_vertices_, 0.0));
    neighbors_.resize(num_vertices_);
    if (old_n != num_vertices_)
    {
        invalidateComponents();
    }
}

void Graph::removeVertex(Index vertex)
{
    if (vertex < 0 || vertex >= static_cast<Index>(num_vertices_))
    {
        throw std::out_of_range("Vertex index out of range");
    }
    adjacency_matrix_.erase(adjacency_matrix_.begin() + vertex);
    for (auto &row : adjacency_matrix_)
    {
        row.erase(row.begin() + vertex);
    }
    neighbors_.erase(neighbors_.begin() + vertex);
    for (auto &adjacent : neighbors_)
    {
        adjacent.erase(std::remove(adjacent.begin(), adjacent.end(), vertex), adjacent.end());
        for (auto &v : adjacent)
        {
            if (v > vertex)
            {
                --v;
            }
        }
    }
    --num_vertices_;
    invalidateComponents();
}

void Graph::addEdge(Index u, Index v, double weight)
{
    if (u >= static_cast<Index>(num_vertices_) || v >= static_cast<Index>(num_vertices_) || u < 0 ||
        v < 0 || u == v)
    {
        throw std::out_of_range("Edge vertex index out of range");
    }
    if (!std::isfinite(weight) || weight <= 0.0)
    {
        throw std::invalid_argument("Edge weight must be finite and positive");
    }
    adjacency_matrix_[u][v] = weight;
    adjacency_matrix_[v][u] = weight;
    if (std::ranges::find(neighbors_[u], v) == neighbors_[u].end())
    {
        neighbors_[u].push_back(v);
    }
    if (std::ranges::find(neighbors_[v], u) == neighbors_[v].end())
    {
        neighbors_[v].push_back(u);
    }
    invalidateComponents();
}

void Graph::removeEdge(Index u, Index v)
{
    if (u >= static_cast<Index>(num_vertices_) || v >= static_cast<Index>(num_vertices_) || u < 0 ||
        v < 0)
    {
        throw std::out_of_range("Edge vertex index out of range");
    }
    adjacency_matrix_[u][v] = 0.0;
    adjacency_matrix_[v][u] = 0.0;
    neighbors_[u].erase(std::remove(neighbors_[u].begin(), neighbors_[u].end(), v),
                        neighbors_[u].end());
    neighbors_[v].erase(std::remove(neighbors_[v].begin(), neighbors_[v].end(), u),
                        neighbors_[v].end());
    invalidateComponents();
}

void Graph::setEdgeWeight(Index u, Index v, double weight)
{
    if (!std::isfinite(weight))
    {
        throw std::invalid_argument("Edge weight must be finite");
    }
    if (weight <= 0.0)
    {
        removeEdge(u, v);
        return;
    }
    addEdge(u, v, weight);
}

Size Graph::numVertices() const
{
    return num_vertices_;
}

Size Graph::numEdges() const
{
    Size edges = 0;
    for (Size i = 0; i < num_vertices_; ++i)
    {
        for (Size j = i + 1; j < num_vertices_; ++j)
        {
            if (adjacency_matrix_[i][j] > 0.0)
            {
                ++edges;
            }
        }
    }
    return edges;
}

std::vector<Index> Graph::getVertices() const
{
    std::vector<Index> vertices(num_vertices_);
    std::iota(vertices.begin(), vertices.end(), 0);
    return vertices;
}

std::vector<std::pair<Index, Index>> Graph::getEdges() const
{
    std::vector<std::pair<Index, Index>> edges;
    edges.reserve(numEdges());
    for (Size i = 0; i < num_vertices_; ++i)
    {
        for (Size j = i + 1; j < num_vertices_; ++j)
        {
            if (adjacency_matrix_[i][j] > 0.0)
            {
                edges.emplace_back(static_cast<Index>(i), static_cast<Index>(j));
            }
        }
    }
    return edges;
}

std::vector<Index> Graph::getNeighbors(Index vertex) const
{
    if (vertex < 0 || vertex >= static_cast<Index>(num_vertices_))
    {
        throw std::out_of_range("Vertex index out of range");
    }
    return neighbors_[vertex];
}

std::vector<Index> Graph::getAdjacentVertices(Index vertex) const
{
    return getNeighbors(vertex);
}

double Graph::getEdgeWeight(Index u, Index v) const
{
    if (u < 0 || v < 0 || u >= static_cast<Index>(num_vertices_) ||
        v >= static_cast<Index>(num_vertices_))
    {
        throw std::out_of_range("Edge vertex index out of range");
    }
    return adjacency_matrix_[u][v];
}

std::vector<std::vector<double>> Graph::getAdjacencyMatrix() const
{
    return adjacency_matrix_;
}

std::vector<std::vector<double>> Graph::getLaplacianMatrix() const
{
    auto laplacian = makeSquareMatrix(num_vertices_);
    for (Size i = 0; i < num_vertices_; ++i)
    {
        double degree = 0.0;
        for (Size j = 0; j < num_vertices_; ++j)
        {
            degree += adjacency_matrix_[i][j];
            if (i != j)
            {
                laplacian[i][j] = -adjacency_matrix_[i][j];
            }
        }
        laplacian[i][i] = degree;
    }
    return laplacian;
}

std::vector<double> Graph::getDegreeSequence() const
{
    std::vector<double> degrees(num_vertices_, 0.0);
    for (Size i = 0; i < num_vertices_; ++i)
    {
        for (double w : adjacency_matrix_[i])
        {
            degrees[i] += w;
        }
    }
    return degrees;
}

bool Graph::isConnected() const
{
    return getConnectedComponents().size() <= 1;
}

std::vector<std::vector<Index>> Graph::getConnectedComponents() const
{
    if (!components_valid_)
    {
        computeConnectedComponents();
    }
    const auto max_component =
        component_ids_.empty() ? -1 : *std::ranges::max_element(component_ids_);
    std::vector<std::vector<Index>> components(static_cast<Size>(max_component + 1));
    for (Size i = 0; i < component_ids_.size(); ++i)
    {
        const Index id = component_ids_[i];
        if (id >= 0)
        {
            components[static_cast<Size>(id)].push_back(static_cast<Index>(i));
        }
    }
    return components;
}

Index Graph::getComponentId(Index vertex) const
{
    if (!components_valid_)
    {
        computeConnectedComponents();
    }
    if (vertex < 0 || vertex >= static_cast<Index>(component_ids_.size()))
    {
        throw std::out_of_range("Vertex index out of range");
    }
    return component_ids_[vertex];
}

void Graph::invalidateComponents() const
{
    components_valid_ = false;
    component_ids_.clear();
}

void Graph::computeConnectedComponents() const
{
    component_ids_.assign(num_vertices_, static_cast<Index>(-1));
    Index current_id = 0;
    for (Index start = 0; start < static_cast<Index>(num_vertices_); ++start)
    {
        if (component_ids_[start] != static_cast<Index>(-1))
        {
            continue;
        }
        std::queue<Index> q;
        q.push(start);
        component_ids_[start] = current_id;
        while (!q.empty())
        {
            const Index u = q.front();
            q.pop();
            for (Index v : neighbors_[u])
            {
                if (component_ids_[v] == static_cast<Index>(-1))
                {
                    component_ids_[v] = current_id;
                    q.push(v);
                }
            }
        }
        ++current_id;
    }
    components_valid_ = true;
}

} // namespace nerve::graphs

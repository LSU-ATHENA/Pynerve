#include "nerve/graphs/graph.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>

namespace nerve::graphs
{

PersistentGraph::PersistentGraph(Size numVertices)
    : WeightedGraph(numVertices)
    , current_time_(0.0)
{}

void PersistentGraph::addVertexPersistent(Index vertex)
{
    if (vertex < 0)
        throw std::out_of_range("Vertex index out of range");
    while (numVertices() <= static_cast<Size>(vertex))
        addVertex();
    recordEvent("addVertex", current_time_);
}

void PersistentGraph::removeVertexPersistent(Index vertex)
{
    removeVertex(vertex);
    recordEvent("removeVertex", current_time_);
}

void PersistentGraph::addEdgePersistent(Index u, Index v, double weight)
{
    addEdge(u, v, weight);
    recordEvent("addEdge", current_time_);
}

void PersistentGraph::removeEdgePersistent(Index u, Index v)
{
    removeEdge(u, v);
    recordEvent("removeEdge", current_time_);
}

std::vector<std::pair<double, std::string>> PersistentGraph::getPersistenceEvents() const
{
    return persistence_events_;
}

std::vector<double> PersistentGraph::getPersistenceDiagram() const
{
    std::vector<double> d;
    d.reserve(persistence_events_.size());
    for (const auto &e : persistence_events_)
        d.push_back(e.first);
    return d;
}

double PersistentGraph::computePersistenceDistance(const PersistentGraph &other) const
{
    auto a = getPersistenceDiagram();
    auto b = other.getPersistenceDiagram();
    const Size n = std::min(a.size(), b.size());
    double sum = 0.0;
    for (Size i = 0; i < n; ++i)
        sum += std::abs(a[i] - b[i]);
    return sum;
}

void PersistentGraph::advanceTime(double time_step)
{
    if (!std::isfinite(time_step) || time_step < 0.0)
        throw std::invalid_argument("Persistence time step must be finite and nonnegative");
    current_time_ += time_step;
}

double PersistentGraph::getCurrentTime() const
{
    return current_time_;
}

void PersistentGraph::resetPersistence()
{
    persistence_events_.clear();
    last_event_times_.clear();
    current_time_ = 0.0;
}

void PersistentGraph::recordEvent(const std::string &event_type, double time)
{
    persistence_events_.emplace_back(time, event_type);
    last_event_times_[event_type] = time;
}

} // namespace nerve::graphs

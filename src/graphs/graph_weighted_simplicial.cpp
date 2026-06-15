#include "nerve/graphs/graph.hpp"

#include <algorithm>
#include <unordered_set>
#include <utility>

namespace nerve::graphs
{

void SimplicialGraph::addSimplex(const algebra::Simplex &simplex)
{
    if (simplex_to_index_.find(simplex) == simplex_to_index_.end())
    {
        simplex_to_index_[simplex] = static_cast<Index>(simplices_.size());
        simplices_.push_back(simplex);
        skeleton_cache_valid_ = false;
    }
}

void SimplicialGraph::addSimplices(const std::vector<algebra::Simplex> &simplices)
{
    for (const auto &s : simplices)
        addSimplex(s);
}

void SimplicialGraph::removeSimplex(const algebra::Simplex &simplex)
{
    auto it = simplex_to_index_.find(simplex);
    if (it != simplex_to_index_.end())
    {
        simplices_.erase(simplices_.begin() + it->second);
        simplex_to_index_.clear();
        for (Size i = 0; i < simplices_.size(); ++i)
            simplex_to_index_[simplices_[i]] = static_cast<Index>(i);
        skeleton_cache_valid_ = false;
    }
}

Graph SimplicialGraph::get1Skeleton() const
{
    if (!skeleton_cache_valid_)
        buildSkeletonCache();
    return skeleton_cache_;
}

WeightedGraph SimplicialGraph::getWeighted1Skeleton() const
{
    Graph g = get1Skeleton();
    WeightedGraph wg(g.numVertices());
    for (auto [u, v] : g.getEdges())
        wg.addEdge(u, v, g.getEdgeWeight(u, v));
    return wg;
}

Graph SimplicialGraph::getCliqueGraph() const
{
    Graph g(simplices_.size());
    for (Size i = 0; i < simplices_.size(); ++i)
        for (Size j = i + 1; j < simplices_.size(); ++j)
        {
            std::unordered_set<Index> vi(simplices_[i].vertices().begin(),
                                         simplices_[i].vertices().end());
            bool intersects = false;
            for (Index v : simplices_[j].vertices())
                if (vi.count(v))
                {
                    intersects = true;
                    break;
                }
            if (intersects)
                g.addEdge(static_cast<Index>(i), static_cast<Index>(j));
        }
    return g;
}

Size SimplicialGraph::numSimplices() const
{
    return simplices_.size();
}

int SimplicialGraph::maxDimension() const
{
    int max_dim = -1;
    for (const auto &s : simplices_)
        max_dim = std::max(max_dim, static_cast<int>(s.dimension()));
    return max_dim;
}

std::vector<Index> SimplicialGraph::getSimplicesOfDimension(int dimension) const
{
    std::vector<Index> out;
    for (Size i = 0; i < simplices_.size(); ++i)
        if (static_cast<int>(simplices_[i].dimension()) == dimension)
            out.push_back(static_cast<Index>(i));
    return out;
}

std::vector<Index> SimplicialGraph::getSimplexNeighbors(const algebra::Simplex &simplex) const
{
    std::vector<Index> out;
    std::unordered_set<Index> base(simplex.vertices().begin(), simplex.vertices().end());
    for (Size i = 0; i < simplices_.size(); ++i)
    {
        bool intersects = false;
        for (Index v : simplices_[i].vertices())
            if (base.count(v))
            {
                intersects = true;
                break;
            }
        if (intersects)
            out.push_back(static_cast<Index>(i));
    }
    return out;
}

std::vector<Index> SimplicialGraph::getSimplexStar(const algebra::Simplex &simplex) const
{
    std::vector<Index> out;
    for (Size i = 0; i < simplices_.size(); ++i)
        if (simplex.isFaceOf(simplices_[i]))
            out.push_back(static_cast<Index>(i));
    return out;
}

std::vector<Index> SimplicialGraph::getSimplexLink(const algebra::Simplex &simplex) const
{
    std::vector<Index> star = getSimplexStar(simplex);
    std::vector<Index> link;
    for (Index idx : star)
    {
        if (!simplices_[idx].isFaceOf(simplex) && !simplex.isFaceOf(simplices_[idx]))
            link.push_back(idx);
    }
    return link;
}

void SimplicialGraph::invalidateSkeletonCache()
{
    skeleton_cache_valid_ = false;
}

void SimplicialGraph::buildSkeletonCache() const
{
    Index max_vertex = -1;
    for (const auto &s : simplices_)
        for (Index v : s.vertices())
            max_vertex = std::max(max_vertex, v);
    Graph g(static_cast<Size>(max_vertex + 1));
    for (const auto &s : simplices_)
    {
        const auto &verts = s.vertices();
        for (Size i = 0; i < verts.size(); ++i)
            for (Size j = i + 1; j < verts.size(); ++j)
                g.addEdge(verts[i], verts[j]);
    }
    skeleton_cache_ = std::move(g);
    skeleton_cache_valid_ = true;
}

} // namespace nerve::graphs

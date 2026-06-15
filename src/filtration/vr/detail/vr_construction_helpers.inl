#pragma once

#include "nerve/algebra/simplex.hpp"
#include "nerve/core_types.hpp"

#include <algorithm>
#include <vector>

namespace nerve::filtration
{
namespace
{

struct EdgeData
{
    Index u, v;
    double dist;
};

using SimplexEntry = std::pair<algebra::Simplex, double>;

void findCommonNeighbors(const std::vector<std::vector<Index>> &adj, Index u, Index v, Index min_w,
                         std::vector<Index> &out)
{
    const auto &small = adj[static_cast<Size>(u)].size() <= adj[static_cast<Size>(v)].size()
                            ? adj[static_cast<Size>(u)]
                            : adj[static_cast<Size>(v)];
    const auto &large = adj[static_cast<Size>(u)].size() <= adj[static_cast<Size>(v)].size()
                            ? adj[static_cast<Size>(v)]
                            : adj[static_cast<Size>(u)];
    for (const Index w : small)
    {
        if (w < min_w)
            continue;
        if (std::binary_search(large.begin(), large.end(), w))
        {
            out.push_back(w);
        }
    }
}

double simplicialRadius(const std::vector<std::vector<double>> &dist,
                        const std::vector<Index> &verts)
{
    double r = 0.0;
    for (Size i = 0; i < verts.size(); ++i)
    {
        for (Size j = i + 1; j < verts.size(); ++j)
        {
            const double d = dist[static_cast<Size>(verts[i])][static_cast<Size>(verts[j])];
            if (d > r)
                r = d;
        }
    }
    return r;
}

} // namespace
} // namespace nerve::filtration

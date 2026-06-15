#include "detail/vr_construction_helpers.inl"
#include "nerve/filtration/vietoris_rips.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace nerve::filtration
{

void VietorisRips::buildSimplices()
{
    const Size n = points_.size();

    for (Size i = 0; i < n; ++i)
    {
        filtration_.emplace_back(algebra::Simplex({static_cast<Index>(i)}), 0.0);
    }
    if (max_dimension_ == 0)
        return;

    std::vector<EdgeData> edges;
    for (Size i = 0; i < n; ++i)
    {
        for (Size j = i + 1; j < n; ++j)
        {
            const double d = distance_matrix_[i][j];
            if (d <= max_radius_)
            {
                edges.push_back({static_cast<Index>(i), static_cast<Index>(j), d});
                filtration_.emplace_back(
                    algebra::Simplex({static_cast<Index>(i), static_cast<Index>(j)}), d);
            }
        }
    }
    if (max_dimension_ == 1 || edges.empty())
        return;

    std::vector<std::vector<Index>> adj(n);
    for (const auto &e : edges)
    {
        adj[static_cast<Size>(e.u)].push_back(e.v);
        adj[static_cast<Size>(e.v)].push_back(e.u);
    }
    for (auto &list : adj)
    {
        std::sort(list.begin(), list.end());
    }

    std::vector<SimplexEntry> prev;

    for (const auto &e : edges)
    {
        const Index min_w = std::max(e.u, e.v) + 1;
        std::vector<Index> common;
        findCommonNeighbors(adj, e.u, e.v, min_w, common);
        for (const Index w : common)
        {
            const double r =
                std::max({e.dist, distance_matrix_[static_cast<Size>(e.u)][static_cast<Size>(w)],
                          distance_matrix_[static_cast<Size>(e.v)][static_cast<Size>(w)]});
            algebra::Simplex tri({e.u, e.v, w});
            filtration_.emplace_back(tri, r);
            prev.emplace_back(std::move(tri), r);
        }
    }

    for (Size dim = 3; dim <= max_dimension_; ++dim)
    {
        if (prev.empty())
            break;
        std::vector<SimplexEntry> curr;
        for (const auto &[simplex, radius] : prev)
        {
            const auto &verts = simplex.vertices();
            const Index last = verts.back();
            const auto &candidates = adj[static_cast<Size>(verts[0])];
            for (const Index w : candidates)
            {
                if (w <= last)
                    continue;
                bool connected = true;
                double new_r = radius;
                for (Size vi = 1; vi < verts.size(); ++vi)
                {
                    const auto &neighbors = adj[static_cast<Size>(verts[vi])];
                    if (!std::binary_search(neighbors.begin(), neighbors.end(), w))
                    {
                        connected = false;
                        break;
                    }
                    const double d =
                        distance_matrix_[static_cast<Size>(verts[vi])][static_cast<Size>(w)];
                    if (d > new_r)
                        new_r = d;
                }
                if (!connected)
                    continue;
                const double d0 =
                    distance_matrix_[static_cast<Size>(verts[0])][static_cast<Size>(w)];
                if (d0 > new_r)
                    new_r = d0;
                std::vector<Index> new_verts = verts;
                new_verts.push_back(w);
                algebra::Simplex new_simplex(new_verts);
                filtration_.emplace_back(new_simplex, new_r);
                curr.emplace_back(std::move(new_simplex), new_r);
            }
        }
        prev = std::move(curr);
    }
}

} // namespace nerve::filtration

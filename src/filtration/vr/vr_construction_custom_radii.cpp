#include "detail/vr_construction_helpers.inl"
#include "nerve/filtration/vietoris_rips.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace nerve::filtration
{

void VietorisRips::buildKSimplicesWithCustomRadii(Size k, const std::vector<double> &custom_radii,
                                                  double scaling_factor)
{
    const Size n = points_.size();
    if (k == 0 || k + 1 > n)
        return;

    std::vector<EdgeData> edges;
    for (Size i = 0; i < n; ++i)
    {
        for (Size j = i + 1; j < n; ++j)
        {
            const double d = distance_matrix_[i][j];
            if (d > max_radius_)
                continue;
            const double max_custom = std::max(custom_radii[i], custom_radii[j]);
            const double v = std::max(d, max_custom * scaling_factor);
            if (v <= max_radius_)
            {
                edges.push_back({static_cast<Index>(i), static_cast<Index>(j), v});
            }
        }
    }
    if (edges.empty())
        return;

    if (k == 1)
    {
        for (const auto &e : edges)
        {
            filtration_.emplace_back(algebra::Simplex({e.u, e.v}), e.dist);
        }
        return;
    }

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

    std::vector<SimplexEntry> curr;

    for (const auto &e : edges)
    {
        const Index min_w = std::max(e.u, e.v) + 1;
        std::vector<Index> common;
        findCommonNeighbors(adj, e.u, e.v, min_w, common);
        for (const Index w : common)
        {
            const double mc = std::max({custom_radii[static_cast<Size>(e.u)],
                                        custom_radii[static_cast<Size>(e.v)],
                                        custom_radii[static_cast<Size>(w)]});
            const double me = simplicialRadius(distance_matrix_, {e.u, e.v, w});
            const double v = std::max(me, mc * scaling_factor);
            if (k == 2)
            {
                filtration_.emplace_back(algebra::Simplex({e.u, e.v, w}), v);
            }
            else
            {
                curr.emplace_back(algebra::Simplex({e.u, e.v, w}), v);
            }
        }
    }
    if (k == 2)
        return;

    for (Size dim = 3; dim <= k; ++dim)
    {
        if (curr.empty())
            return;
        std::vector<SimplexEntry> next;
        for (const auto &entry : curr)
        {
            const auto &simplex = entry.first;
            const auto &verts = simplex.vertices();
            const Index last = verts.back();
            const auto &candidates = adj[static_cast<Size>(verts[0])];
            for (const Index w : candidates)
            {
                if (w <= last)
                    continue;
                bool connected = true;
                for (const Index v : verts)
                {
                    const auto &neighbors = adj[static_cast<Size>(v)];
                    if (!std::binary_search(neighbors.begin(), neighbors.end(), w))
                    {
                        connected = false;
                        break;
                    }
                }
                if (!connected)
                    continue;

                std::vector<Index> new_verts = verts;
                new_verts.push_back(w);

                double max_custom = 0.0;
                for (const Index v : new_verts)
                {
                    if (custom_radii[static_cast<Size>(v)] > max_custom)
                    {
                        max_custom = custom_radii[static_cast<Size>(v)];
                    }
                }
                const double me = simplicialRadius(distance_matrix_, new_verts);
                const double v = std::max(me, max_custom * scaling_factor);

                algebra::Simplex new_simplex(new_verts);
                if (dim == k)
                {
                    filtration_.emplace_back(new_simplex, v);
                }
                else
                {
                    next.emplace_back(std::move(new_simplex), v);
                }
            }
        }
        curr = std::move(next);
    }
}

} // namespace nerve::filtration

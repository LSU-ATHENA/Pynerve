#include "exact_engine_internal.hpp"
#include "nerve/persistence/utils/exact_engine.hpp"

#include <algorithm>
#include <ranges>
#include <unordered_map>
#include <vector>

namespace nerve::persistence
{
namespace
{

using detail::Column;

} // namespace

ExactPersistenceResult
computeExactCohomologyZ2(const algebra::SimplicialComplex &complex, Size max_dim,
                         const std::vector<std::vector<int>> &neighbors,
                         const std::unordered_map<std::uint64_t, double> &edge_weights)
{
    const auto filtered = complex.getFilteredSimplices();
    if (filtered.empty())
        return {};

    using SimplexFiltration = std::pair<algebra::Simplex, double>;
    std::vector<SimplexFiltration> simplices;
    simplices.reserve(filtered.size());
    for (const auto &[simp, val] : filtered)
        simplices.emplace_back(simp, val);

    auto edge_dist = [&](int a, int b) -> double {
        std::uint64_t k =
            (static_cast<std::uint64_t>(static_cast<std::uint32_t>(std::min(a, b))) << 32) |
            static_cast<std::uint64_t>(static_cast<std::uint32_t>(std::max(a, b)));
        auto it = edge_weights.find(k);
        return (it != edge_weights.end()) ? it->second : std::numeric_limits<double>::infinity();
    };

    int max_d = static_cast<int>(max_dim);
    std::unordered_map<algebra::Simplex, double, algebra::Simplex::Hash> cofacet_set;

    for (const auto &[simp, diam] : simplices)
    {
        if (static_cast<int>(simp.dimension()) != max_d)
            continue;
        const auto &verts = simp.vertices();
        if (verts.empty())
            continue;
        int max_v = static_cast<int>(*std::max_element(verts.begin(), verts.end()));

        for (int v : neighbors[static_cast<size_t>(verts[0])])
        {
            if (v <= max_v)
                continue;
            bool connected = true;
            for (size_t vi = 1; vi < verts.size() && connected; ++vi)
            {
                const auto &nbrs = neighbors[static_cast<size_t>(verts[vi])];
                if (std::find(nbrs.begin(), nbrs.end(), v) == nbrs.end())
                    connected = false;
            }
            if (!connected)
                continue;

            double cf_diam = diam;
            for (int vi : verts)
                cf_diam = std::max(cf_diam, edge_dist(vi, v));

            std::vector<Index> cf_v;
            cf_v.reserve(verts.size() + 1);
            for (Index vx : verts)
                cf_v.push_back(vx);
            cf_v.push_back(static_cast<Index>(v));
            std::sort(cf_v.begin(), cf_v.end());
            algebra::Simplex cf_simp(cf_v);

            auto it = cofacet_set.find(cf_simp);
            if (it == cofacet_set.end() || cf_diam < it->second)
                cofacet_set[cf_simp] = cf_diam;
        }
    }

    for (const auto &[simp, diam] : cofacet_set)
        simplices.emplace_back(simp, diam);

    std::sort(simplices.begin(), simplices.end(), detail::simplexFiltrationOrder);

    std::unordered_map<algebra::Simplex, Index, algebra::Simplex::Hash> s2idx;
    for (Index i = 0; i < static_cast<Index>(simplices.size()); ++i)
        s2idx[simplices[i].first] = i;

    std::vector<Column> coboundary(simplices.size());
    for (Index ci = 0; ci < static_cast<Index>(simplices.size()); ++ci)
    {
        const auto &s = simplices[ci].first;
        if (s.dimension() == 0)
            continue;
        auto faces = s.faces({});
        for (const auto &face : faces)
        {
            auto it = s2idx.find(face);
            if (it != s2idx.end())
                coboundary[it->second].push_back(static_cast<Size>(ci));
        }
    }

    for (auto &cb : coboundary)
        std::sort(cb.begin(), cb.end(), std::greater<Index>());

    std::vector<Index> low(simplices.size(), -1);
    std::vector<Index> low_row_to_col;
    std::vector<Column> reduction_columns(simplices.size());
    std::vector<bool> skip_column(simplices.size(), false);

    for (Index col = 0; col < static_cast<Index>(simplices.size()); ++col)
    {
        if (skip_column[col])
            continue;
        if (static_cast<Size>(simplices[col].first.dimension()) > max_dim)
            continue;
        Column column(coboundary[col].begin(), coboundary[col].end());
        if (column.empty())
            continue;
        while (!column.empty())
        {
            Index pivot = static_cast<Index>(column.front());
            if (pivot >= static_cast<Index>(low_row_to_col.size()))
                low_row_to_col.resize(pivot + 1, -1);
            Index killer = low_row_to_col[pivot];
            if (killer < 0)
            {
                low_row_to_col[pivot] = col;
                low[col] = pivot;
                skip_column[pivot] = true;
                break;
            }
            column = detail::symmetricDifferenceSorted(column, reduction_columns[killer]);
        }
        reduction_columns[col] = std::move(column);
    }

    std::vector<Pair> pairs;
    for (Index col = 0; col < static_cast<Index>(low.size()); ++col)
    {
        Index row = low[col];
        if (row < 0 || static_cast<Index>(simplices.size()) <= row)
            continue;
        int dim = static_cast<int>(simplices[col].first.dimension());
        if (dim > max_d)
            continue;
        double birth_val = simplices[col].second;
        double death_val = simplices[row].second;
        if (death_val > birth_val + 1e-12)
            pairs.push_back({birth_val, death_val, dim});
    }

    std::vector<bool> is_death(low.size(), false);
    for (Index b : low)
        if (b >= 0 && static_cast<size_t>(b) < is_death.size())
            is_death[b] = true;
    for (size_t i = 0; i < simplices.size(); ++i)
    {
        if (!is_death[i] && static_cast<Size>(simplices[i].first.dimension()) <= max_dim)
            pairs.push_back({simplices[i].second, std::numeric_limits<double>::infinity(),
                             static_cast<int>(simplices[i].first.dimension())});
    }

    std::vector<Size> betti;
    for (const auto &p : pairs)
    {
        if (static_cast<size_t>(p.dimension) >= betti.size())
            betti.resize(p.dimension + 1, 0);
        if (std::isinf(p.death))
            betti[p.dimension]++;
    }
    return {pairs, betti, 0};
}

} // namespace nerve::persistence

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

ExactPersistenceResult computeExactCohomologyZ2(const algebra::SimplicialComplex &complex,
                                                Size max_dim)
{
    (void)max_dim;
    const auto filtered = complex.getFilteredSimplices();
    if (filtered.empty())
        return {};

    std::unordered_map<algebra::Simplex, double, algebra::Simplex::Hash> min_filtration;
    min_filtration.reserve(filtered.size() * 2);
    for (const auto &entry : filtered)
    {
        auto it = min_filtration.find(entry.first);
        if (it == min_filtration.end() || entry.second < it->second)
        {
            min_filtration[entry.first] = entry.second;
        }
    }

    std::vector<std::pair<algebra::Simplex, double>> simplices;
    simplices.reserve(min_filtration.size());
    for (const auto &item : min_filtration)
    {
        simplices.emplace_back(item.first, item.second);
    }
    std::sort(simplices.begin(), simplices.end(), detail::simplexFiltrationOrder);

    std::unordered_map<algebra::Simplex, Index, algebra::Simplex::Hash> simplex_to_idx;
    for (Index i = 0; i < static_cast<Index>(simplices.size()); ++i)
    {
        simplex_to_idx[simplices[i].first] = i;
    }

    std::vector<Column> coboundary(simplices.size());
    for (Index ci = 0; ci < static_cast<Index>(simplices.size()); ++ci)
    {
        const auto &s = simplices[ci].first;
        if (s.dimension() == 0)
            continue;
        auto faces = s.faces({});
        for (const auto &face : faces)
        {
            auto it = simplex_to_idx.find(face);
            if (it != simplex_to_idx.end())
            {
                coboundary[it->second].push_back(static_cast<Size>(ci));
            }
        }
    }
    for (auto &cb : coboundary)
    {
        std::sort(cb.begin(), cb.end(), std::greater<Index>());
    }

    std::vector<Index> low(simplices.size(), -1);
    std::vector<Index> low_row_to_col;
    std::vector<Column> reduction_columns(simplices.size());
    std::vector<bool> skip_column(simplices.size(), false);

    for (int64_t col_i = static_cast<int64_t>(simplices.size()) - 1; col_i >= 0; --col_i)
    {
        Index col = static_cast<Index>(col_i);
        if (skip_column[col])
            continue;

        Column column(coboundary[col].begin(), coboundary[col].end());
        if (column.empty())
            continue;

        while (!column.empty())
        {
            Index pivot = static_cast<Index>(column.front());
            if (pivot >= static_cast<Index>(low_row_to_col.size()))
            {
                low_row_to_col.resize(pivot + 1, -1);
            }
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
        if (row < 0)
            continue;
        if (static_cast<Index>(simplices.size()) <= row)
            continue;
        double birth_val = simplices[col].second;
        double death_val = simplices[row].second;
        if (death_val > birth_val + 1e-12)
        {
            int dim = static_cast<int>(simplices[col].first.dimension());
            pairs.push_back({birth_val, death_val, dim});
        }
    }

    std::vector<bool> is_death_column(low.size(), false);
    for (Index birth : low)
    {
        if (birth >= 0 && static_cast<size_t>(birth) < is_death_column.size())
        {
            is_death_column[birth] = true;
        }
    }
    for (size_t i = 0; i < simplices.size(); ++i)
    {
        if (!is_death_column[i])
        {
            pairs.push_back({simplices[i].second, std::numeric_limits<double>::infinity(),
                             static_cast<int>(simplices[i].first.dimension())});
        }
    }

    std::vector<Size> betti;
    for (const auto &p : pairs)
    {
        if (static_cast<size_t>(p.dimension) >= betti.size())
        {
            betti.resize(p.dimension + 1, 0);
        }
        if (std::isinf(p.death))
            betti[p.dimension]++;
    }

    return {pairs, betti, 0};
}

} // namespace nerve::persistence

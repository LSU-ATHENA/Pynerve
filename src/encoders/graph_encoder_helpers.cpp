// Graph encoder helper operations  --  graph construction helpers.

#include "nerve/encoders/encoders.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace nerve::encoders
{

namespace
{

Size checkedMul(Size lhs, Size rhs, const char *context)
{
    if (lhs != 0 && rhs > std::numeric_limits<Size>::max() / lhs)
    {
        throw std::length_error(context);
    }
    return lhs * rhs;
}

Size checkedProduct(std::initializer_list<Size> factors, const char *context)
{
    Size total = 1;
    for (const Size factor : factors)
    {
        total = checkedMul(total, factor, context);
    }
    if (total > std::vector<double>().max_size())
    {
        throw std::length_error(context);
    }
    return total;
}

double rowDistance(const std::vector<double> &lhs, const std::vector<double> &rhs)
{
    const Size dim = lhs.size();
    double sum_sq = 0.0;
    for (Size i = 0; i < dim; ++i)
    {
        const double diff = lhs[i] - rhs[i];
        const double contribution = diff * diff;
        const double next = sum_sq + contribution;
        if (!std::isfinite(contribution) || !std::isfinite(next))
        {
            throw std::overflow_error("graph row distance overflow");
        }
        sum_sq = next;
    }
    const double distance = std::sqrt(sum_sq);
    if (!std::isfinite(distance))
    {
        throw std::overflow_error("graph row distance overflow");
    }
    return distance;
}

void validateRows(const std::vector<std::vector<double>> &data)
{
    if (data.empty())
        return;
    const Size dim = data.front().size();
    if (dim == 0)
    {
        throw std::invalid_argument("graph encoder rows must contain coordinates");
    }
    for (const auto &row : data)
    {
        if (row.size() != dim)
        {
            throw std::invalid_argument("graph encoder rows must have consistent dimensions");
        }
        if (!std::ranges::all_of(row, [](double value) {
                const long double safe_abs =
                    std::sqrt(static_cast<long double>(std::numeric_limits<double>::max())) / 4.0L;
                const long double wide = static_cast<long double>(value);
                return std::isfinite(wide) && std::abs(wide) <= safe_abs;
            }))
        {
            throw std::invalid_argument("graph encoder rows must contain finite safe values");
        }
    }
}

void validateDiagramPair(const persistence::Pair &pair)
{
    const auto isFiniteSafe = [](double value) {
        const long double safe_abs =
            std::sqrt(static_cast<long double>(std::numeric_limits<double>::max())) / 4.0L;
        const long double wide = static_cast<long double>(value);
        return std::isfinite(wide) && std::abs(wide) <= safe_abs;
    };
    const bool death_ok =
        isFiniteSafe(pair.death) || pair.death == std::numeric_limits<double>::infinity();
    if (!isFiniteSafe(pair.birth) || !death_ok ||
        (std::isfinite(pair.death) && pair.death < pair.birth))
    {
        throw std::invalid_argument("graph encoder diagram contains an invalid pair");
    }
}

} // namespace

Tensor GraphEncoder::constructGraph(const std::vector<std::vector<double>> &data) const
{
    validateRows(data);
    const Size n = data.size();
    const Size adjacency_count = checkedProduct({n, n}, "graph adjacency size overflow");
    std::vector<double> adjacency(adjacency_count, 0.0);
    for (Size i = 0; i < n; ++i)
    {
        std::vector<std::pair<double, Size>> distances;
        distances.reserve(n > 0 ? n - 1 : 0);
        for (Size j = 0; j < n; ++j)
        {
            if (i != j)
            {
                distances.emplace_back(rowDistance(data[i], data[j]), j);
            }
        }
        std::ranges::sort(distances);
        for (Size rank = 0; rank < distances.size(); ++rank)
        {
            const bool connect = graph_construction_method_ == "knn"
                                     ? rank < k_neighbors_
                                     : distances[rank].first <= distance_threshold_;
            if (connect)
            {
                adjacency[i * n + distances[rank].second] = 1.0;
                adjacency[distances[rank].second * n + i] = 1.0;
            }
        }
    }
    return Tensor(adjacency, {n, n});
}

Tensor GraphEncoder::constructGraphFromComplex(const SimplicialComplex &complex) const
{
    std::unordered_map<Index, Size> vertex_to_row;
    for (const auto &simplex : complex.getSimplices())
    {
        for (Index vertex : simplex.vertices())
        {
            vertex_to_row.try_emplace(vertex, vertex_to_row.size());
        }
    }

    const Size n = vertex_to_row.size();
    const Size adjacency_count = checkedProduct({n, n}, "complex graph adjacency size overflow");
    std::vector<double> adjacency(adjacency_count, 0.0);
    for (const auto &simplex : complex.getSimplices())
    {
        const auto &vertices = simplex.vertices();
        for (Size i = 0; i < vertices.size(); ++i)
        {
            for (Size j = i + 1; j < vertices.size(); ++j)
            {
                const Size row = vertex_to_row.at(vertices[i]);
                const Size col = vertex_to_row.at(vertices[j]);
                adjacency[row * n + col] = 1.0;
                adjacency[col * n + row] = 1.0;
            }
        }
    }
    return Tensor(adjacency, {n, n});
}

Tensor GraphEncoder::constructGraphFromDiagram(const Diagram &diagram) const
{
    const Size n = diagram.count();
    const Size adjacency_count = checkedProduct({n, n}, "diagram graph adjacency size overflow");
    std::vector<double> adjacency(adjacency_count, 0.0);
    const auto &pairs = diagram.getPairs();
    for (const auto &pair : pairs)
    {
        validateDiagramPair(pair);
    }
    for (Size i = 0; i < n; ++i)
    {
        for (Size j = i + 1; j < n; ++j)
        {
            const double db = pairs[i].birth - pairs[j].birth;
            const double di = pairs[i].dimension == pairs[j].dimension ? 0.0 : 1.0;
            const double li = std::isfinite(pairs[i].death) ? pairs[i].death - pairs[i].birth : 0.0;
            const double lj = std::isfinite(pairs[j].death) ? pairs[j].death - pairs[j].birth : 0.0;
            const double dl = li - lj;
            const double db_sq = db * db;
            const double dl_sq = dl * dl;
            const double distance_sq = db_sq + dl_sq + di;
            if (!std::isfinite(db_sq) || !std::isfinite(dl_sq) || !std::isfinite(distance_sq))
            {
                throw std::overflow_error("diagram graph distance overflow");
            }
            if (std::sqrt(distance_sq) <= distance_threshold_)
            {
                adjacency[i * n + j] = 1.0;
                adjacency[j * n + i] = 1.0;
            }
        }
    }
    return Tensor(adjacency, {n, n});
}

} // namespace nerve::encoders

// Complex-construction helpers for incremental persistence.

#include "nerve/algebra/simplex.hpp"
#include "nerve/persistence/utils/incremental_updates_internal.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <unordered_map>

namespace nerve::persistence
{

namespace
{

using FiltrationMap = std::unordered_map<algebra::Simplex, double, algebra::Simplex::Hash>;
using ComplexBuildResult = ::nerve::ErrorResult<algebra::SimplicialComplex>;

ComplexBuildResult complexError(ErrorCode code, const std::string &message)
{
    return ComplexBuildResult::error(code, message);
}

bool checkedProduct(size_t lhs, size_t rhs, size_t *out)
{
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs)
    {
        *out = 0;
        return false;
    }
    *out = lhs * rhs;
    return true;
}

bool checkedPairCount(size_t n, size_t *out)
{
    if (n < 2)
    {
        *out = 0;
        return true;
    }
    size_t lhs = n;
    size_t rhs = n - 1;
    if (lhs % 2 == 0)
    {
        lhs /= 2;
    }
    else
    {
        rhs /= 2;
    }
    return checkedProduct(lhs, rhs, out) && *out <= std::vector<double>().max_size();
}

std::optional<ErrorCode> validatePoints(const PointCloud &points)
{
    if (points.size() > static_cast<size_t>(std::numeric_limits<Index>::max()))
    {
        return ErrorCode::E41_RESOURCE_LIMIT;
    }
    for (const auto &point : points)
    {
        for (const double value : point)
        {
            if (!std::isfinite(value))
            {
                return ErrorCode::E54_PH4_INVALID_INPUT;
            }
        }
    }
    return std::nullopt;
}

std::optional<ErrorCode> euclideanDistance(const std::vector<double> &lhs,
                                           const std::vector<double> &rhs, double *out)
{
    const size_t dims = std::min(lhs.size(), rhs.size());
    double accum = 0.0;
    for (size_t d = 0; d < dims; ++d)
    {
        const double delta = lhs[d] - rhs[d];
        const double contribution = delta * delta;
        const double next = accum + contribution;
        if (!std::isfinite(delta) || !std::isfinite(contribution) || !std::isfinite(next))
        {
            return ErrorCode::E54_PH4_INVALID_INPUT;
        }
        accum = next;
    }
    const double distance = std::sqrt(accum);
    if (!std::isfinite(distance))
    {
        return ErrorCode::E54_PH4_INVALID_INPUT;
    }
    *out = distance;
    return std::nullopt;
}

std::optional<ErrorCode> insertSimplexClosure(const algebra::Simplex &simplex, double filtration,
                                              FiltrationMap *filtrations)
{
    if (filtrations == nullptr)
    {
        return std::nullopt;
    }
    if (!std::isfinite(filtration))
    {
        return ErrorCode::E54_PH4_INVALID_INPUT;
    }
    if (filtrations->size() >= filtrations->max_size())
    {
        return ErrorCode::E41_RESOURCE_LIMIT;
    }
    const auto [it, inserted] = filtrations->insert({simplex, filtration});
    if (!inserted && filtration >= it->second)
    {
        return std::nullopt;
    }
    if (!inserted)
    {
        it->second = filtration;
    }
    if (simplex.dimension() == 0)
    {
        return std::nullopt;
    }
    for (const auto &face : simplex.faces(core::DeterminismContract{}))
    {
        if (auto error = insertSimplexClosure(face, filtration, filtrations))
        {
            return error;
        }
    }
    return std::nullopt;
}

} // namespace

ComplexBuildResult buildIncrementalComplex(const PointCloud &points, size_t max_dimension)
{
    algebra::SimplicialComplex complex;
    if (points.empty())
    {
        return ComplexBuildResult::success(std::move(complex));
    }
    if (auto error = validatePoints(points))
    {
        return complexError(*error, "invalid incremental point cloud");
    }

    const size_t n = points.size();
    size_t dense_count = 0;
    if (!checkedProduct(n, n, &dense_count) || dense_count > std::vector<double>().max_size())
    {
        return complexError(ErrorCode::E41_RESOURCE_LIMIT, "incremental distance matrix too large");
    }
    size_t pair_count = 0;
    if (!checkedPairCount(n, &pair_count))
    {
        return complexError(ErrorCode::E41_RESOURCE_LIMIT, "incremental distance count too large");
    }
    std::vector<std::vector<double>> dist(n, std::vector<double>(n, 0.0));
    std::vector<double> distances;
    distances.reserve(pair_count);
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = i + 1; j < n; ++j)
        {
            double dij = 0.0;
            if (auto error = euclideanDistance(points[i], points[j], &dij))
            {
                return complexError(*error, "invalid incremental point distance");
            }
            dist[i][j] = dij;
            dist[j][i] = dij;
            distances.push_back(dij);
        }
    }
    std::sort(distances.begin(), distances.end());
    const double threshold = distances.empty() ? 0.0 : distances[distances.size() / 2] * 1.5;
    if (!std::isfinite(threshold))
    {
        return complexError(ErrorCode::E54_PH4_INVALID_INPUT, "invalid incremental threshold");
    }

    FiltrationMap filtrations;
    size_t filtration_reserve = 0;
    if (!checkedProduct(n, size_t{8}, &filtration_reserve) ||
        filtration_reserve > filtrations.max_size())
    {
        return complexError(ErrorCode::E41_RESOURCE_LIMIT, "incremental filtration map too large");
    }
    filtrations.reserve(filtration_reserve);
    for (size_t i = 0; i < n; ++i)
    {
        if (auto error =
                insertSimplexClosure(algebra::Simplex({static_cast<Index>(i)}), 0.0, &filtrations))
        {
            return complexError(*error, "incremental vertex insertion failed");
        }
    }

    if (max_dimension >= 1)
    {
        for (size_t i = 0; i < n; ++i)
        {
            for (size_t j = i + 1; j < n; ++j)
            {
                if (dist[i][j] <= threshold)
                {
                    auto error = insertSimplexClosure(
                        algebra::Simplex({static_cast<Index>(i), static_cast<Index>(j)}),
                        dist[i][j], &filtrations);
                    if (error)
                    {
                        return complexError(*error, "incremental edge insertion failed");
                    }
                }
            }
        }
    }

    if (max_dimension >= 2)
    {
        for (size_t i = 0; i < n; ++i)
        {
            for (size_t j = i + 1; j < n; ++j)
            {
                if (dist[i][j] > threshold)
                {
                    continue;
                }
                for (size_t k = j + 1; k < n; ++k)
                {
                    const double f = std::max({dist[i][j], dist[i][k], dist[j][k]});
                    if (f <= threshold)
                    {
                        auto error = insertSimplexClosure(
                            algebra::Simplex({static_cast<Index>(i), static_cast<Index>(j),
                                              static_cast<Index>(k)}),
                            f, &filtrations);
                        if (error)
                        {
                            return complexError(*error, "incremental triangle insertion failed");
                        }
                    }
                }
            }
        }
    }

    if (max_dimension >= 3)
    {
        for (size_t i = 0; i < n; ++i)
        {
            for (size_t j = i + 1; j < n; ++j)
            {
                if (dist[i][j] > threshold)
                {
                    continue;
                }
                for (size_t k = j + 1; k < n; ++k)
                {
                    for (size_t l = k + 1; l < n; ++l)
                    {
                        const double f = std::max({dist[i][j], dist[i][k], dist[i][l], dist[j][k],
                                                   dist[j][l], dist[k][l]});
                        if (f <= threshold)
                        {
                            auto error = insertSimplexClosure(
                                algebra::Simplex({static_cast<Index>(i), static_cast<Index>(j),
                                                  static_cast<Index>(k), static_cast<Index>(l)}),
                                f, &filtrations);
                            if (error)
                            {
                                return complexError(*error,
                                                    "incremental tetrahedron insertion failed");
                            }
                        }
                    }
                }
            }
        }
    }

    for (const auto &[simplex, filtration] : filtrations)
    {
        complex.addSimplexWithFiltration(simplex, filtration);
    }
    return ComplexBuildResult::success(std::move(complex));
}

void overwriteDiagram(const ExactPersistenceResult &exact, ::nerve::Diagram *diagram)
{
    if (diagram == nullptr)
    {
        return;
    }
    diagram->clear();
    for (const auto &pair : exact.pairs)
    {
        if (pair.dimension < 0)
        {
            continue;
        }
        diagram->addPair(pair.birth, pair.death, static_cast<size_t>(pair.dimension));
    }
}

} // namespace nerve::persistence

#pragma once

#include "nerve/algebra/complex.hpp"
#include "nerve/filtration/vietoris_rips.hpp"
#include "nerve/persistence/utils/exact_engine.hpp"
#include "nerve/probabilistic/probabilistic.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>

namespace nerve::probabilistic::detail
{

constexpr double kEpsilon = 1.0e-12;

inline void requireFinite(double value, const char *context)
{
    if (!std::isfinite(value))
    {
        throw std::invalid_argument(std::string(context) + " contains a non-finite value");
    }
}

inline void requireValidPair(const Pair &pair, const char *context)
{
    if (!std::isfinite(pair.birth))
    {
        throw std::invalid_argument(std::string(context) + " birth must be finite");
    }
    if (std::isnan(pair.death) || pair.death == -std::numeric_limits<Field>::infinity())
    {
        throw std::invalid_argument(std::string(context) + " death must be finite or +inf");
    }
    if (!pair.isInfinite() && !std::isfinite(pair.death))
    {
        throw std::invalid_argument(std::string(context) + " death must be finite or +inf");
    }
    if (pair.dimension < 0)
    {
        throw std::invalid_argument(std::string(context) + " dimension must be non-negative");
    }
    if (!pair.isInfinite() && pair.death < pair.birth)
    {
        throw std::invalid_argument(std::string(context) + " death must not precede birth");
    }
}

inline Size pointDimension(const std::vector<std::vector<double>> &points)
{
    if (points.empty())
    {
        return 0;
    }
    const Size dim = points.front().size();
    if (dim == 0)
    {
        throw std::invalid_argument("point dimension must be positive");
    }
    const long double safe_abs =
        std::sqrt(static_cast<long double>(std::numeric_limits<double>::max()) /
                  static_cast<long double>(dim)) /
        4.0L;
    for (const auto &point : points)
    {
        if (point.size() != dim)
        {
            throw std::invalid_argument("point cloud rows must have a consistent dimension");
        }
        for (double value : point)
        {
            requireFinite(value, "point cloud");
            if (std::abs(static_cast<long double>(value)) > safe_abs)
            {
                throw std::overflow_error("point cloud coordinate magnitude is too large");
            }
        }
    }
    return dim;
}

inline std::vector<double> flattenPoints(const std::vector<std::vector<double>> &points)
{
    const Size dim = pointDimension(points);
    std::vector<double> flat;
    if (dim != 0 && points.size() > std::numeric_limits<Size>::max() / dim)
    {
        throw std::length_error("point cloud flattened size overflows");
    }
    const Size value_count = points.size() * dim;
    if (value_count > flat.max_size())
    {
        throw std::length_error("point cloud flattened size exceeds vector capacity");
    }
    flat.reserve(value_count);
    for (const auto &point : points)
    {
        flat.insert(flat.end(), point.begin(), point.end());
    }
    return flat;
}

inline double maxPairwiseDistance(const std::vector<std::vector<double>> &points)
{
    const Size dim = pointDimension(points);
    double max_distance = 0.0;
    for (Size i = 0; i < points.size(); ++i)
    {
        for (Size j = i + 1; j < points.size(); ++j)
        {
            double squared = 0.0;
            for (Size d = 0; d < dim; ++d)
            {
                const double delta = points[i][d] - points[j][d];
                const double contribution = delta * delta;
                if (!std::isfinite(contribution) ||
                    squared > std::numeric_limits<double>::max() - contribution)
                {
                    throw std::overflow_error("pairwise distance overflowed");
                }
                squared += contribution;
            }
            const double distance = std::sqrt(squared);
            if (!std::isfinite(distance))
            {
                throw std::overflow_error("pairwise distance overflowed");
            }
            max_distance = std::max(max_distance, distance);
        }
    }
    return max_distance;
}

inline Diagram diagramFromPairs(const std::vector<persistence::Pair> &pairs)
{
    Diagram diagram;
    for (const auto &pair : pairs)
    {
        diagram.addPair(pair);
    }
    return diagram;
}

inline Diagram diagramFromPoints(const std::vector<std::vector<double>> &points,
                                 Size max_dimension = 2)
{
    const Size dim = pointDimension(points);
    Diagram diagram;
    if (points.empty())
    {
        return diagram;
    }

    const auto flat = flattenPoints(points);
    const auto view = core::ownership_utils::makeView<const double>(flat);
    const double radius = std::max(maxPairwiseDistance(points), kEpsilon);
    const auto filtration =
        ::nerve::filtration::computeVietorisRipsFiltration(view, dim, radius, max_dimension);

    algebra::SimplicialComplex complex;
    for (Size i = 0; i < points.size(); ++i)
    {
        complex.addSimplexWithFiltration(algebra::Simplex({static_cast<Index>(i)}), 0.0);
    }
    for (const auto &[simplex, value] : filtration)
    {
        complex.addSimplexWithFiltration(simplex, value);
    }

    return diagramFromPairs(persistence::computeExactPersistenceZ2(complex, max_dimension).pairs);
}

inline double finiteLifetime(const Pair &pair)
{
    requireValidPair(pair, "persistence pair");
    if (pair.isInfinite())
    {
        return 0.0;
    }
    return pair.death - pair.birth;
}

inline std::vector<double> finiteLifetimes(const Diagram &diagram)
{
    std::vector<double> values;
    for (const auto &pair : diagram.getPairs())
    {
        const double life = finiteLifetime(pair);
        if (life > kEpsilon)
        {
            values.push_back(life);
        }
    }
    return values;
}

inline double totalLifetime(const Diagram &diagram)
{
    const auto values = finiteLifetimes(diagram);
    return std::accumulate(values.begin(), values.end(), 0.0);
}

inline double mean(const std::vector<double> &values)
{
    if (values.empty())
    {
        return 0.0;
    }
    return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

inline double variance(const std::vector<double> &values)
{
    if (values.size() < 2)
    {
        return 0.0;
    }
    const double avg = mean(values);
    double sum = 0.0;
    for (double value : values)
    {
        const double delta = value - avg;
        sum += delta * delta;
    }
    return sum / static_cast<double>(values.size() - 1);
}

inline double quantile(std::vector<double> values, double q)
{
    if (values.empty())
    {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const double clamped = std::clamp(q, 0.0, 1.0);
    const double position = clamped * static_cast<double>(values.size() - 1);
    const Size lower = static_cast<Size>(std::floor(position));
    const Size upper = static_cast<Size>(std::ceil(position));
    const double weight = position - static_cast<double>(lower);
    return values[lower] * (1.0 - weight) + values[upper] * weight;
}

inline double median(std::vector<double> values)
{
    return quantile(std::move(values), 0.5);
}

inline double confidenceZ(double level)
{
    if (!std::isfinite(level) || level <= 0.0 || level >= 1.0)
    {
        throw std::invalid_argument("confidence level must be in (0, 1)");
    }
    if (level >= 0.99)
    {
        return 2.5758293035489004;
    }
    if (level >= 0.95)
    {
        return 1.959963984540054;
    }
    if (level >= 0.90)
    {
        return 1.6448536269514722;
    }
    return 1.0;
}

inline std::mt19937 seededGenerator(const std::vector<std::vector<double>> &points,
                                    std::uint32_t salt)
{
    std::uint32_t seed = 2166136261u ^ salt;
    for (const auto &point : points)
    {
        for (double value : point)
        {
            const auto scaled = static_cast<std::int64_t>(std::llround(value * 1.0e6));
            seed ^= static_cast<std::uint32_t>(scaled);
            seed *= 16777619u;
        }
    }
    return std::mt19937(seed);
}

} // namespace nerve::probabilistic::detail

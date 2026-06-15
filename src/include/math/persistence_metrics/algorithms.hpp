
#pragma once

#include "math/persistence_metrics/hungarian_solver.hpp"
#include "math/persistence_metrics/point2d.hpp"
#include "nerve/error/error_registry.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace nerve::math
{

namespace detail
{

inline bool isFinitePoint(const Point2D &point) noexcept
{
    return std::isfinite(point.x) && std::isfinite(point.y);
}

inline bool pointsAreFinite(const std::vector<Point2D> &points) noexcept
{
    return std::all_of(points.begin(), points.end(), isFinitePoint);
}

inline size_t checkedBottleneckCandidateCount(size_t n1, size_t n2)
{
    if (n2 != 0 && n1 > std::numeric_limits<size_t>::max() / n2)
    {
        throw std::length_error("bottleneck candidate count overflows size_t");
    }
    const size_t pair_count = n1 * n2;
    if (pair_count > std::numeric_limits<size_t>::max() - n1)
    {
        throw std::length_error("bottleneck candidate count overflows size_t");
    }
    size_t count = pair_count + n1;
    if (count > std::numeric_limits<size_t>::max() - n2)
    {
        throw std::length_error("bottleneck candidate count overflows size_t");
    }
    count += n2;
    if (count == std::numeric_limits<size_t>::max())
    {
        throw std::length_error("bottleneck candidate count overflows size_t");
    }
    ++count;
    if (count > std::vector<double>().max_size())
    {
        throw std::length_error("bottleneck candidate count exceeds vector capacity");
    }
    return count;
}

inline void pushFiniteCandidate(std::vector<double> &candidates, double value)
{
    if (!std::isfinite(value))
    {
        throw std::overflow_error("bottleneck candidate is non-finite");
    }
    candidates.push_back(value);
}

} // namespace detail

class OptimalTransport
{
public:
    struct TransportResult
    {
        double total_cost = 0.0;
        size_t source_size = 0;
        size_t target_size = 0;
        std::vector<std::vector<double>> transport_cost_matrix;
    };

    explicit OptimalTransport(double p = 2.0)
        : p_(p)
    {
        if (!std::isfinite(p_) || p_ < 1.0)
        {
            throw std::invalid_argument("Wasserstein power parameter must be >= 1.0");
        }
    }

    error::Result<TransportResult> solve(const std::vector<Point2D> &source_points,
                                         const std::vector<Point2D> &target_points) const
    {
        try
        {
            TransportResult result;
            result.source_size = source_points.size();
            result.target_size = target_points.size();

            const size_t n_source = source_points.size();
            const size_t n_target = target_points.size();
            if (!detail::pointsAreFinite(source_points) || !detail::pointsAreFinite(target_points))
            {
                return error::Result<TransportResult>::err(error::TDAErrorCode::NaNInInput,
                                                           "transport points must be finite");
            }
            if (n_source > std::numeric_limits<size_t>::max() - n_target)
            {
                return error::Result<TransportResult>::err(error::TDAErrorCode::ResourceLimit,
                                                           "transport matrix dimension overflows");
            }
            const size_t n = n_source + n_target;
            if (n == 0)
            {
                return error::Result<TransportResult>::ok(std::move(result));
            }
            if (n > std::vector<std::vector<double>>().max_size() ||
                n > std::vector<double>().max_size())
            {
                return error::Result<TransportResult>::err(
                    error::TDAErrorCode::ResourceLimit,
                    "transport matrix dimension exceeds vector capacity");
            }

            std::vector<std::vector<double>> cost(
                n, std::vector<double>(n, std::numeric_limits<double>::infinity()));

            double max_cost = 1.0;
            for (size_t i = 0; i < n_source; ++i)
            {
                for (size_t j = 0; j < n_target; ++j)
                {
                    const double d = linfDistance(source_points[i], target_points[j]);
                    const double c = std::pow(d, p_);
                    if (!std::isfinite(c))
                    {
                        return error::Result<TransportResult>::err(
                            error::TDAErrorCode::InvalidFieldOperation,
                            "transport cost must be finite");
                    }
                    cost[i][j] = c;
                    max_cost = std::max(max_cost, c);
                }
            }
            for (size_t i = 0; i < n_source; ++i)
            {
                const double c = std::pow(diagonalDistance(source_points[i]), p_);
                if (!std::isfinite(c))
                {
                    return error::Result<TransportResult>::err(
                        error::TDAErrorCode::InvalidFieldOperation,
                        "transport diagonal cost must be finite");
                }
                cost[i][n_target + i] = c;
                max_cost = std::max(max_cost, c);
            }
            for (size_t j = 0; j < n_target; ++j)
            {
                const double c = std::pow(diagonalDistance(target_points[j]), p_);
                if (!std::isfinite(c))
                {
                    return error::Result<TransportResult>::err(
                        error::TDAErrorCode::InvalidFieldOperation,
                        "transport diagonal cost must be finite");
                }
                cost[n_source + j][j] = c;
                max_cost = std::max(max_cost, c);
            }
            for (size_t row = n_source; row < n; ++row)
            {
                for (size_t col = n_target; col < n; ++col)
                {
                    cost[row][col] = 0.0;
                }
            }

            const double large_penalty = max_cost * (static_cast<double>(n) + 1.0) + 1.0;
            if (!std::isfinite(large_penalty))
            {
                return error::Result<TransportResult>::err(
                    error::TDAErrorCode::InvalidFieldOperation, "transport penalty must be finite");
            }
            for (auto &row : cost)
            {
                for (double &value : row)
                {
                    if (!std::isfinite(value))
                    {
                        value = large_penalty;
                    }
                }
            }

            HungarianSolver solver(cost);
            auto assignment_result = solver.solve();
            if (assignment_result.isErr())
            {
                return error::Result<TransportResult>::err(
                    static_cast<error::TDAErrorCode>(assignment_result.error().value()),
                    std::string(assignment_result.detail()));
            }

            result.total_cost = assignment_result.value().total_cost;
            result.transport_cost_matrix = std::move(cost);
            return error::Result<TransportResult>::ok(std::move(result));
        }
        catch (const std::exception &ex)
        {
            return error::Result<TransportResult>::err(error::TDAErrorCode::InvalidFieldOperation,
                                                       std::string("Optimal transport failed: ") +
                                                           ex.what());
        }
    }

private:
    double p_;
};

namespace detail
{

inline bool tryAugment(size_t left, const std::vector<std::vector<size_t>> &adjacency,
                       std::vector<bool> &seenRight, std::vector<int> &rightMatch)
{
    for (size_t right : adjacency[left])
    {
        if (seenRight[right])
        {
            continue;
        }
        seenRight[right] = true;
        if (rightMatch[right] < 0 ||
            tryAugment(static_cast<size_t>(rightMatch[right]), adjacency, seenRight, rightMatch))
        {
            rightMatch[right] = static_cast<int>(left);
            return true;
        }
    }
    return false;
}

inline bool hasPerfectMatching(const std::vector<Point2D> &points1,
                               const std::vector<Point2D> &points2, double threshold)
{
    const size_t n1 = points1.size();
    const size_t n2 = points2.size();
    const size_t n = n1 + n2;
    if (n == 0)
    {
        return true;
    }

    std::vector<std::vector<size_t>> adjacency(n);
    constexpr double kEpsilon = 1e-12;

    for (size_t i = 0; i < n1; ++i)
    {
        for (size_t j = 0; j < n2; ++j)
        {
            if (linfDistance(points1[i], points2[j]) <= threshold + kEpsilon)
            {
                adjacency[i].push_back(j);
            }
        }
        if (diagonalDistance(points1[i]) <= threshold + kEpsilon)
        {
            adjacency[i].push_back(n2 + i);
        }
    }

    for (size_t j = 0; j < n2; ++j)
    {
        if (diagonalDistance(points2[j]) <= threshold + kEpsilon)
        {
            adjacency[n1 + j].push_back(j);
        }
    }

    std::vector<int> rightMatch(n, -1);
    size_t matched = 0;
    for (size_t left = 0; left < n; ++left)
    {
        std::vector<bool> seenRight(n, false);
        if (tryAugment(left, adjacency, seenRight, rightMatch))
        {
            ++matched;
        }
    }
    return matched == n;
}

inline std::vector<double> bottleneckCandidates(const std::vector<Point2D> &points1,
                                                const std::vector<Point2D> &points2)
{
    std::vector<double> candidates;
    candidates.reserve(checkedBottleneckCandidateCount(points1.size(), points2.size()));
    candidates.push_back(0.0);

    for (const auto &point : points1)
    {
        pushFiniteCandidate(candidates, diagonalDistance(point));
    }
    for (const auto &point : points2)
    {
        pushFiniteCandidate(candidates, diagonalDistance(point));
    }
    for (const auto &lhs : points1)
    {
        for (const auto &rhs : points2)
        {
            pushFiniteCandidate(candidates, linfDistance(lhs, rhs));
        }
    }

    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end(),
                                 [](double a, double b) { return std::abs(a - b) <= 1e-12; }),
                     candidates.end());
    return candidates;
}

} // namespace detail

class PersistenceMetrics
{
public:
    static error::Result<double> bottleneckDistance(const std::vector<Point2D> &points1,
                                                    const std::vector<Point2D> &points2)
    {
        try
        {
            if (points1.empty() && points2.empty())
            {
                return error::Result<double>::ok(0.0);
            }
            if (!detail::pointsAreFinite(points1) || !detail::pointsAreFinite(points2))
            {
                return error::Result<double>::err(error::TDAErrorCode::NaNInInput,
                                                  "diagram points must be finite");
            }

            const auto candidates = detail::bottleneckCandidates(points1, points2);
            if (candidates.empty())
            {
                return error::Result<double>::ok(0.0);
            }

            size_t lo = 0;
            size_t hi = candidates.size() - 1;
            double best = candidates[hi];
            while (lo <= hi)
            {
                const size_t mid = lo + (hi - lo) / 2;
                const double threshold = candidates[mid];
                if (detail::hasPerfectMatching(points1, points2, threshold))
                {
                    best = threshold;
                    if (mid == 0)
                    {
                        break;
                    }
                    hi = mid - 1;
                }
                else
                {
                    lo = mid + 1;
                }
            }

            return error::Result<double>::ok(best);
        }
        catch (const std::exception &ex)
        {
            return error::Result<double>::err(
                error::TDAErrorCode::InvalidFieldOperation,
                std::string("Bottleneck distance computation failed: ") + ex.what());
        }
    }

    static error::Result<double> wassersteinDistance(const std::vector<Point2D> &points1,
                                                     const std::vector<Point2D> &points2,
                                                     double p = 2.0)
    {
        try
        {
            if (!std::isfinite(p) || p < 1.0)
            {
                return error::Result<double>::err(error::TDAErrorCode::InvalidFieldOperation,
                                                  "Wasserstein power parameter must be >= 1.0");
            }
            if (!detail::pointsAreFinite(points1) || !detail::pointsAreFinite(points2))
            {
                return error::Result<double>::err(error::TDAErrorCode::NaNInInput,
                                                  "diagram points must be finite");
            }

            if (points1.empty() && points2.empty())
            {
                return error::Result<double>::ok(0.0);
            }

            OptimalTransport solver(p);
            auto transport_result = solver.solve(points1, points2);
            if (transport_result.isErr())
            {
                return error::Result<double>::err(
                    static_cast<error::TDAErrorCode>(transport_result.error().value()),
                    std::string(transport_result.detail()));
            }

            const double distance = std::pow(transport_result.value().total_cost, 1.0 / p);
            if (!std::isfinite(distance))
            {
                return error::Result<double>::err(error::TDAErrorCode::InvalidFieldOperation,
                                                  "Wasserstein distance must be finite");
            }
            return error::Result<double>::ok(distance);
        }
        catch (const std::exception &ex)
        {
            return error::Result<double>::err(
                error::TDAErrorCode::InvalidFieldOperation,
                std::string("Wasserstein distance computation failed: ") + ex.what());
        }
    }

    static std::vector<Point2D>
    extractPointsFromDiagram(const std::vector<std::pair<double, double>> &persistence_pairs,
                             bool include_essential = true)
    {
        std::vector<Point2D> points;
        points.reserve(persistence_pairs.size());
        for (const auto &[birth, death] : persistence_pairs)
        {
            if (!std::isfinite(birth))
            {
                continue;
            }
            if (std::isfinite(death))
            {
                points.emplace_back(birth, death);
            }
            else if (include_essential)
            {
                points.emplace_back(birth, birth);
            }
        }
        return points;
    }
};

} // namespace nerve::math

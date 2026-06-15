#include "nerve/metrics/lazy_distance.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace nerve::metrics::lazy
{
namespace
{

size_t checkedPointValueCount(size_t n_points, size_t point_dim)
{
    if (point_dim != 0 && n_points > std::numeric_limits<size_t>::max() / point_dim)
    {
        throw std::length_error("sparse distance point coordinate count overflows");
    }
    return n_points * point_dim;
}

void validateMetric(const std::string &metric)
{
    if (metric != "euclidean" && metric != "manhattan" && metric != "cosine")
    {
        throw std::invalid_argument("unsupported sparse distance metric");
    }
}

void validatePoints(std::span<const double> points, size_t n_points, size_t point_dim,
                    const std::string &metric)
{
    validateMetric(metric);
    if (n_points != 0 && point_dim == 0)
    {
        throw std::invalid_argument("sparse distance point dimension must be positive");
    }
    const size_t expected_values = checkedPointValueCount(n_points, point_dim);
    if (points.size() < expected_values)
    {
        throw std::invalid_argument("sparse distance point span is smaller than dimensions");
    }
    for (size_t i = 0; i < expected_values; ++i)
    {
        if (!std::isfinite(points[i]))
        {
            throw std::invalid_argument("sparse distance point coordinates must be finite");
        }
    }
}

void validateIndex(size_t index, size_t n_points)
{
    if (index >= n_points)
    {
        throw std::out_of_range("sparse distance point index out of range");
    }
}

double checkedDifference(double lhs, double rhs, std::string_view message)
{
    const double diff = lhs - rhs;
    if (!std::isfinite(diff))
    {
        throw std::overflow_error(std::string(message));
    }
    return diff;
}

double checkedAdd(double total, double contribution, std::string_view message)
{
    if (!std::isfinite(contribution))
    {
        throw std::overflow_error(std::string(message));
    }
    const double result = total + contribution;
    if (!std::isfinite(result))
    {
        throw std::overflow_error(std::string(message));
    }
    return result;
}

double checkedDistanceResult(double value, std::string_view message)
{
    if (!std::isfinite(value))
    {
        throw std::overflow_error(std::string(message));
    }
    return value;
}

} // namespace

SparseDistanceMatrix::SparseDistanceMatrix(std::span<const double> points, size_t n_points,
                                           size_t point_dim, double threshold,
                                           const std::string &metric)
    : n_points_(n_points)
{
    validatePoints(points, n_points, point_dim, metric);
    if (!std::isfinite(threshold) || threshold < 0.0)
    {
        throw std::invalid_argument("sparse distance threshold must be finite and non-negative");
    }

    // Compute and store only edges within threshold
    for (size_t i = 0; i < n_points; ++i)
    {
        for (size_t j = i + 1; j < n_points; ++j)
        {
            double dist = computeDistance(points, i, j, point_dim, metric);

            if (dist <= threshold)
            {
                edges_[{i, j}] = dist;
            }
        }
    }
}

double SparseDistanceMatrix::getDistance(size_t i, size_t j) const
{
    validateIndex(i, n_points_);
    validateIndex(j, n_points_);
    if (i == j)
        return 0.0;
    if (i > j)
        std::swap(i, j);

    auto it = edges_.find({i, j});
    if (it != edges_.end())
    {
        return it->second;
    }

    return std::numeric_limits<double>::infinity();
}

bool SparseDistanceMatrix::isEdge(size_t i, size_t j) const
{
    validateIndex(i, n_points_);
    validateIndex(j, n_points_);
    if (i == j)
        return true;
    if (i > j)
        std::swap(i, j);
    return edges_.count({i, j}) > 0;
}

size_t SparseDistanceMatrix::memoryBytes() const
{
    return edges_.size() * 24;
}

double SparseDistanceMatrix::getSparsity() const
{
    if (n_points_ > 1 && n_points_ > std::numeric_limits<size_t>::max() / (n_points_ - 1))
    {
        throw std::length_error("sparse distance pair count overflows");
    }
    size_t possible = n_points_ * (n_points_ - 1) / 2;
    return possible == 0 ? 1.0
                         : 1.0 - static_cast<double>(edges_.size()) / static_cast<double>(possible);
}

double SparseDistanceMatrix::computeDistance(std::span<const double> points, size_t i, size_t j,
                                             size_t point_dim, const std::string &metric)
{
    const double *p1 = &points[i * point_dim];
    const double *p2 = &points[j * point_dim];

    if (metric == "euclidean")
    {
        double sum = 0.0;
        for (size_t d = 0; d < point_dim; ++d)
        {
            const double diff = checkedDifference(p1[d], p2[d], "euclidean distance overflow");
            sum = checkedAdd(sum, diff * diff, "euclidean distance overflow");
        }
        return checkedDistanceResult(std::sqrt(sum), "euclidean distance overflow");
    }
    else if (metric == "manhattan")
    {
        double sum = 0.0;
        for (size_t d = 0; d < point_dim; ++d)
        {
            const double diff = checkedDifference(p1[d], p2[d], "manhattan distance overflow");
            sum = checkedAdd(sum, std::abs(diff), "manhattan distance overflow");
        }
        return sum;
    }
    else if (metric == "cosine")
    {
        double dot = 0.0, norm1 = 0.0, norm2 = 0.0;
        for (size_t d = 0; d < point_dim; ++d)
        {
            dot = checkedAdd(dot, p1[d] * p2[d], "cosine distance overflow");
            norm1 = checkedAdd(norm1, p1[d] * p1[d], "cosine distance overflow");
            norm2 = checkedAdd(norm2, p2[d] * p2[d], "cosine distance overflow");
        }
        const double denominator = std::sqrt(norm1) * std::sqrt(norm2);
        if (denominator == 0.0)
        {
            throw std::invalid_argument("cosine distance requires non-zero vectors");
        }
        if (!std::isfinite(denominator))
        {
            throw std::overflow_error("cosine distance overflow");
        }
        return checkedDistanceResult(1.0 - dot / denominator, "cosine distance overflow");
    }

    throw std::invalid_argument("unsupported sparse distance metric");
}

} // namespace nerve::metrics::lazy

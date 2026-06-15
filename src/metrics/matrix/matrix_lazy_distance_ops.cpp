#include "nerve/metrics/lazy_distance.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <ranges>
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
        throw std::length_error("lazy distance point coordinate count overflows");
    }
    return n_points * point_dim;
}

void validateMetric(const std::string &metric)
{
    if (metric != "euclidean" && metric != "manhattan" && metric != "cosine")
    {
        throw std::invalid_argument("unsupported lazy distance metric");
    }
}

void validatePoints(std::span<const double> points, size_t n_points, size_t point_dim,
                    const std::string &metric)
{
    validateMetric(metric);
    if (n_points != 0 && point_dim == 0)
    {
        throw std::invalid_argument("lazy distance point dimension must be positive");
    }
    const size_t expected_values = checkedPointValueCount(n_points, point_dim);
    if (points.size() < expected_values)
    {
        throw std::invalid_argument("lazy distance point span is smaller than dimensions");
    }
    for (size_t i = 0; i < expected_values; ++i)
    {
        if (!std::isfinite(points[i]))
        {
            throw std::invalid_argument("lazy distance point coordinates must be finite");
        }
    }
}

void validateIndex(size_t index, size_t n_points)
{
    if (index >= n_points)
    {
        throw std::out_of_range("lazy distance point index out of range");
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

LazyDistanceMatrix::LazyDistanceMatrix(std::span<const double> points, size_t n_points,
                                       size_t point_dim, const std::string &metric,
                                       size_t max_cache_size)
    : points_(points)
    , n_points_(n_points)
    , point_dim_(point_dim)
    , metric_(metric)
    , max_cache_size_(max_cache_size)
{
    validatePoints(points, n_points, point_dim, metric);
    cache_.reserve(max_cache_size);
}

double LazyDistanceMatrix::getDistance(size_t i, size_t j)
{
    validateIndex(i, n_points_);
    validateIndex(j, n_points_);
    if (i == j)
        return 0.0;
    if (i > j)
        std::swap(i, j);

    {
        std::unique_lock<std::shared_mutex> lock(cache_mutex_);
        auto it = cache_.find({i, j});
        if (it != cache_.end())
        {
            cache_hits_++;
            total_lookups_++;
            recordAccess({i, j});
            return it->second;
        }
        total_lookups_++;
    }

    double dist = computeDistance(i, j);

    if (max_cache_size_ > 0)
    {
        std::unique_lock<std::shared_mutex> lock(cache_mutex_);
        if (static_cast<double>(cache_.size()) >=
            static_cast<double>(max_cache_size_) * CACHE_EVICTION_THRESHOLD)
        {
            evictOldest();
        }
        cache_[{i, j}] = dist;
        recordAccess({i, j});
    }

    return dist;
}

double LazyDistanceMatrix::getDistanceNoCache(size_t i, size_t j) const
{
    validateIndex(i, n_points_);
    validateIndex(j, n_points_);
    if (i == j)
        return 0.0;
    return computeDistance(i, j);
}

bool LazyDistanceMatrix::isWithinRadius(size_t i, size_t j, double radius)
{
    if (!std::isfinite(radius) || radius < 0.0)
    {
        throw std::invalid_argument("lazy distance radius must be finite and non-negative");
    }
    if (i == j)
        return true;
    return getDistance(i, j) <= radius;
}

double LazyDistanceMatrix::getMaxDistanceInSubset(std::span<const size_t> indices)
{
    double max_dist = 0.0;

    for (size_t i = 0; i < indices.size(); ++i)
    {
        for (size_t j = i + 1; j < indices.size(); ++j)
        {
            double dist = getDistance(indices[i], indices[j]);
            max_dist = std::max(max_dist, dist);
        }
    }

    return max_dist;
}

std::vector<std::pair<size_t, double>>
LazyDistanceMatrix::getKNearestNeighbors(size_t point_idx, size_t k, double max_radius)
{
    validateIndex(point_idx, n_points_);
    if (std::isnan(max_radius) || max_radius < 0.0)
    {
        throw std::invalid_argument("lazy distance max radius must be non-negative");
    }
    std::vector<std::pair<size_t, double>> neighbors;
    neighbors.reserve(k);

    std::vector<std::pair<size_t, double>> all_dists;
    all_dists.reserve(n_points_);

    for (size_t i = 0; i < n_points_; ++i)
    {
        if (i != point_idx)
        {
            double dist = getDistance(point_idx, i);
            if (dist <= max_radius)
            {
                all_dists.push_back({i, dist});
            }
        }
    }

    const size_t n_return = std::min(k, all_dists.size());
    const auto nth =
        std::next(all_dists.begin(),
                  static_cast<std::vector<std::pair<size_t, double>>::difference_type>(n_return));
    std::partial_sort(all_dists.begin(), nth, all_dists.end(),
                      [](const auto &a, const auto &b) { return a.second < b.second; });

    return std::vector<std::pair<size_t, double>>(all_dists.begin(), nth);
}

// Cache management
size_t LazyDistanceMatrix::getCacheSize() const
{
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return cache_.size();
}

size_t LazyDistanceMatrix::getMaxCacheSize() const
{
    return max_cache_size_;
}

void LazyDistanceMatrix::clearCache()
{
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    cache_.clear();
    access_times_.clear();
}

double LazyDistanceMatrix::getCacheHitRate() const
{
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (total_lookups_ == 0)
        return 0.0;
    return static_cast<double>(cache_hits_) / static_cast<double>(total_lookups_);
}

size_t LazyDistanceMatrix::memoryBytes() const
{
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return cache_.size() * 32;
}

double LazyDistanceMatrix::memoryReduction() const
{
    size_t dense_bytes = n_points_ * n_points_ * sizeof(double);
    size_t lazy_bytes = memoryBytes();
    return lazy_bytes == 0 ? 0.0
                           : static_cast<double>(dense_bytes) / static_cast<double>(lazy_bytes);
}

// Private methods
double LazyDistanceMatrix::computeDistance(size_t i, size_t j) const
{
    const double *p1 = &points_[i * point_dim_];
    const double *p2 = &points_[j * point_dim_];

    if (metric_ == "euclidean")
    {
        double sum = 0.0;
        for (size_t d = 0; d < point_dim_; ++d)
        {
            const double diff = checkedDifference(p1[d], p2[d], "euclidean distance overflow");
            sum = checkedAdd(sum, diff * diff, "euclidean distance overflow");
        }
        return checkedDistanceResult(std::sqrt(sum), "euclidean distance overflow");
    }
    else if (metric_ == "manhattan")
    {
        double sum = 0.0;
        for (size_t d = 0; d < point_dim_; ++d)
        {
            const double diff = checkedDifference(p1[d], p2[d], "manhattan distance overflow");
            sum = checkedAdd(sum, std::abs(diff), "manhattan distance overflow");
        }
        return sum;
    }
    else if (metric_ == "cosine")
    {
        double dot = 0.0, norm1 = 0.0, norm2 = 0.0;
        for (size_t d = 0; d < point_dim_; ++d)
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

    throw std::invalid_argument("unsupported lazy distance metric");
}

void LazyDistanceMatrix::evictOldest()
{
    size_t target_size = max_cache_size_ / 2;

    std::vector<std::pair<std::chrono::steady_clock::time_point, std::pair<size_t, size_t>>>
        entries;
    entries.reserve(access_times_.size());

    for (const auto &[key, time] : access_times_)
    {
        entries.push_back({time, key});
    }

    std::ranges::sort(
        entries, {},
        &std::pair<std::chrono::steady_clock::time_point, std::pair<size_t, size_t>>::first);

    size_t to_remove = cache_.size() - target_size;
    for (size_t i = 0; i < to_remove && i < entries.size(); ++i)
    {
        const auto &key = entries[i].second;
        cache_.erase(key);
        access_times_.erase(key);
    }
}

void LazyDistanceMatrix::recordAccess(const std::pair<size_t, size_t> &key)
{
    access_times_[key] = std::chrono::steady_clock::now();
}

} // namespace nerve::metrics::lazy

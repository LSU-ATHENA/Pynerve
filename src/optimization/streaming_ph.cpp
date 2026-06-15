#include "nerve/optimization/component_optimizations.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <ranges>
#include <tuple>

namespace nerve::optimization
{

namespace
{

bool multiplyWouldOverflow(size_t lhs, size_t rhs)
{
    return lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs;
}

size_t maxPointDimension(const std::vector<std::vector<float>> &window)
{
    size_t point_dim = 0;
    for (const auto &point : window)
    {
        point_dim = std::max(point_dim, point.size());
    }
    return point_dim;
}

} // namespace

AcceleratedStreamingPh::AcceleratedStreamingPh(const StreamingConfig &config)
    : config_(config)
    , window_size_(0)
    , cache_valid_(false)
    , cached_num_points_(0)
    , cached_point_dim_(0)
    , cached_window_hash_(0)
{}

AcceleratedStreamingPh::~AcceleratedStreamingPh() = default;

AcceleratedStreamingPh::StreamingResult
AcceleratedStreamingPh::computeStreamingPh(const std::vector<std::vector<float>> &points)
{
    const auto start = std::chrono::steady_clock::now();
    StreamingResult result;

    if (points.empty())
    {
        result.persistence_diagram.clear();
        result.actual_error = 0.0;
        result.used_approximation = false;
        result.computation_time_ms = 0.0;
        result.error_code = ErrorCode::SUCCESS;
        return result;
    }

    std::vector<std::vector<float>> working_points = points;
    result.used_approximation = false;

    if (config_.enable_coarsening && config_.coarsening_threshold > 0 &&
        working_points.size() > config_.coarsening_threshold)
    {
        landmark_points_ = working_points;
        applyCoarsening();
        if (!landmark_points_.empty())
        {
            working_points = landmark_points_;
            result.used_approximation = true;
        }
    }

    if (config_.enable_incrementality && isCacheValid())
    {
        result.persistence_diagram = computeIncremental(working_points);
    }
    else
    {
        result.persistence_diagram = computeExact(working_points);
    }

    if (result.used_approximation && !points.empty())
    {
        const double retained =
            static_cast<double>(working_points.size()) / static_cast<double>(points.size());
        result.actual_error = std::clamp(1.0 - retained, 0.0, 1.0);
    }
    else
    {
        result.actual_error = 0.0;
    }

    const auto end = std::chrono::steady_clock::now();
    result.computation_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.error_code = ErrorCode::SUCCESS;
    last_update_time_ms_ = result.computation_time_ms;
    landmark_points_ = working_points;
    measureErrorBudget();
    return result;
}

std::vector<std::pair<float, float>>
AcceleratedStreamingPh::compute(const std::vector<std::vector<float>> &window)
{
    if (config_.enable_incrementality && isCacheValid())
    {
        return computeIncremental(window);
    }
    return computeExact(window);
}

std::vector<std::pair<float, float>>
AcceleratedStreamingPh::computeExact(const std::vector<std::vector<float>> &window)
{
    if (!refreshDistanceCache(window))
    {
        return {};
    }
    return computePairsFromDistanceMatrix(persistence_cache_.data(), window_size_);
}

bool AcceleratedStreamingPh::refreshDistanceCache(const std::vector<std::vector<float>> &window)
{
    const size_t n = window.size();
    if (n == 0)
    {
        cache_valid_ = false;
        cached_num_points_ = 0;
        cached_point_dim_ = 0;
        cached_window_hash_ = 0;
        persistence_cache_.clear();
        window_size_ = 0;
        return false;
    }
    if (multiplyWouldOverflow(n, n))
    {
        cache_valid_ = false;
        persistence_cache_.clear();
        window_size_ = 0;
        return false;
    }

    std::vector<float> distances(n * n);
    const size_t point_dim = maxPointDimension(window);

    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = i + 1; j < n; ++j)
        {
            float dist_sq = 0.0f;
            for (size_t d = 0; d < window[i].size() && d < window[j].size(); ++d)
            {
                float diff = window[i][d] - window[j][d];
                dist_sq += diff * diff;
            }
            distances[i * n + j] = std::sqrt(dist_sq);
            distances[j * n + i] = distances[i * n + j];
        }
    }

    persistence_cache_ = std::move(distances);
    window_size_ = n;

    cached_num_points_ = n;
    cached_point_dim_ = point_dim;
    cached_window_hash_ = computeWindowHash(window);
    cache_valid_ = true;
    return true;
}

std::vector<std::pair<float, float>>
AcceleratedStreamingPh::computeIncremental(const std::vector<std::vector<float>> &window)
{
    const size_t n = window.size();
    const size_t point_dim = maxPointDimension(window);
    const uint64_t window_hash = computeWindowHash(window);

    // Reuse cached metric closure only for identical windows; otherwise refresh.
    if (!isCacheValid() || n != cached_num_points_ || point_dim != cached_point_dim_ ||
        window_hash != cached_window_hash_)
    {
        return computeExact(window);
    }

    return computePairsFromDistanceMatrix(persistence_cache_.data(), n);
}

bool AcceleratedStreamingPh::isCacheValid() const
{
    return cache_valid_ && !persistence_cache_.empty() && cached_num_points_ > 0 &&
           window_size_ >= cached_num_points_;
}

uint64_t
AcceleratedStreamingPh::computeWindowHash(const std::vector<std::vector<float>> &window) const
{
    constexpr uint64_t kFNVOffset = 1469598103934665603ULL;
    constexpr uint64_t kFNVPrime = 1099511628211ULL;

    auto mix = [](uint64_t &hash, uint64_t value) {
        hash ^= value;
        hash *= kFNVPrime;
    };

    uint64_t hash = kFNVOffset;
    mix(hash, static_cast<uint64_t>(window.size()));
    for (const auto &point : window)
    {
        mix(hash, static_cast<uint64_t>(point.size()));
        for (float coord : point)
        {
            static_assert(sizeof(float) == sizeof(uint32_t), "Unexpected float width");
            uint32_t bits = 0;
            std::memcpy(&bits, &coord, sizeof(float));
            mix(hash, static_cast<uint64_t>(bits));
        }
    }
    return hash;
}

std::vector<std::pair<float, float>>
AcceleratedStreamingPh::computePairsFromDistanceMatrix(const float *distances, size_t n) const
{
    std::vector<std::pair<float, float>> pairs;
    std::vector<int> parent(n);
    std::iota(parent.begin(), parent.end(), 0);
    std::vector<float> birth_time(n, 0.0f);

    auto find = [&](int x, auto &&find_ref) -> int {
        if (parent[x] != x)
        {
            parent[x] = find_ref(parent[x], find_ref);
        }
        return parent[x];
    };

    std::vector<std::tuple<float, int, int>> sorted_edges;
    sorted_edges.reserve((n * (n - 1)) / 2);
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = i + 1; j < n; ++j)
        {
            const float edge = distances[i * n + j];
            if (edge > 0.0f)
            {
                sorted_edges.push_back({edge, static_cast<int>(i), static_cast<int>(j)});
            }
        }
    }
    std::ranges::sort(sorted_edges);

    for (const auto &[dist, u, v] : sorted_edges)
    {
        const int root_u = find(u, find);
        const int root_v = find(v, find);
        if (root_u != root_v)
        {
            parent[root_u] = root_v;
            pairs.push_back({birth_time[root_u], dist});
            birth_time[root_v] = dist;
        }
    }

    return pairs;
}

void AcceleratedStreamingPh::updateStreamingPh(const std::vector<std::vector<float>> &new_points,
                                               const std::vector<uint32_t> &changed_indices)
{
    const auto start = std::chrono::steady_clock::now();
    landmark_points_ = new_points;
    changed_neighborhoods_ = changed_indices;

    point_changed_.assign(landmark_points_.size(), false);
    for (const uint32_t idx : changed_indices)
    {
        if (idx < point_changed_.size())
        {
            point_changed_[idx] = true;
        }
    }

    point_to_simplices_.resize(landmark_points_.size());
    for (const uint32_t idx : changed_indices)
    {
        if (idx >= point_to_simplices_.size())
        {
            continue;
        }
        point_to_simplices_[idx].push_back(idx);
    }

    if (config_.enable_coarsening)
    {
        applyCoarsening();
    }
    capSimplexGrowth();

    if (!landmark_points_.empty())
    {
        const size_t point_dim = maxPointDimension(landmark_points_);
        const uint64_t window_hash = computeWindowHash(landmark_points_);
        if (!isCacheValid() || landmark_points_.size() != cached_num_points_ ||
            point_dim != cached_point_dim_ || window_hash != cached_window_hash_)
        {
            refreshDistanceCache(landmark_points_);
        }
    }

    const auto end = std::chrono::steady_clock::now();
    last_update_time_ms_ = std::chrono::duration<double, std::milli>(end - start).count();
    measureErrorBudget();
}

void AcceleratedStreamingPh::capSimplexGrowth()
{
    if (config_.max_active_simplex_growth == 0 || point_to_simplices_.empty())
    {
        return;
    }
    std::size_t total = 0;
    for (const auto &simplices : point_to_simplices_)
    {
        total += simplices.size();
    }
    if (total <= config_.max_active_simplex_growth)
    {
        return;
    }

    const std::size_t target_per_point =
        std::max<std::size_t>(1, config_.max_active_simplex_growth / point_to_simplices_.size());
    for (auto &simplices : point_to_simplices_)
    {
        if (simplices.size() > target_per_point)
        {
            simplices.erase(simplices.begin(),
                            simplices.begin() +
                                static_cast<std::ptrdiff_t>(simplices.size() - target_per_point));
        }
    }
}

void AcceleratedStreamingPh::applyCoarsening()
{
    if (!config_.enable_coarsening || config_.coarsening_threshold == 0 ||
        landmark_points_.size() <= config_.coarsening_threshold)
    {
        return;
    }
    const std::size_t stride =
        std::max<std::size_t>(2, (landmark_points_.size() + config_.coarsening_threshold - 1) /
                                     config_.coarsening_threshold);

    std::vector<std::vector<float>> coarsened;
    coarsened.reserve(config_.coarsening_threshold + 1);
    for (std::size_t idx = 0; idx < landmark_points_.size(); idx += stride)
    {
        coarsened.push_back(landmark_points_[idx]);
    }
    if (!landmark_points_.empty())
    {
        const auto &last_point = landmark_points_.back();
        if (coarsened.empty() || coarsened.back() != last_point)
        {
            coarsened.push_back(last_point);
        }
    }
    if (!coarsened.empty())
    {
        landmark_points_.swap(coarsened);
    }
}

AcceleratedStreamingPh::StreamingResult
AcceleratedStreamingPh::computeSummaryOnly(const std::vector<std::vector<float>> &points,
                                           const CallContract &contract)
{
    const auto start = std::chrono::steady_clock::now();
    StreamingResult result = computeStreamingPh(points);

    constexpr std::size_t kSummaryCap = 64;
    const std::size_t original_size = result.persistence_diagram.size();
    if (config_.enable_summary_cap && result.persistence_diagram.size() > kSummaryCap)
    {
        std::ranges::sort(result.persistence_diagram, [](const auto &lhs, const auto &rhs) {
            return (lhs.second - lhs.first) > (rhs.second - rhs.first);
        });
        result.persistence_diagram.resize(kSummaryCap);
        result.used_approximation = true;
        if (original_size > 0)
        {
            const double retained =
                static_cast<double>(kSummaryCap) / static_cast<double>(original_size);
            result.actual_error =
                std::max(result.actual_error, std::clamp(1.0 - retained, 0.0, 1.0));
        }
    }

    const auto end = std::chrono::steady_clock::now();
    result.computation_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    if (contract.strict_time_enforcement && contract.time_budget_ms > 0.0 &&
        result.computation_time_ms > contract.time_budget_ms)
    {
        result.error_code = ErrorCode::PH_TIME_BUDGET_EXCEEDED;
    }
    return result;
}

void AcceleratedStreamingPh::measureUpdateTime()
{
    if (!cache_valid_ || cached_num_points_ == 0)
    {
        last_update_time_ms_ = 0.0;
        return;
    }
    const double operations = static_cast<double>(cached_num_points_) *
                              static_cast<double>(cached_num_points_) *
                              static_cast<double>(std::max<std::size_t>(1, cached_point_dim_));
    last_update_time_ms_ = operations / 1e7;
}

void AcceleratedStreamingPh::measureErrorBudget()
{
    if (cached_num_points_ == 0)
    {
        last_error_budget_observed_ = 0.0;
        return;
    }
    const double changed_ratio = static_cast<double>(changed_neighborhoods_.size()) /
                                 static_cast<double>(cached_num_points_);
    last_error_budget_observed_ = std::clamp(changed_ratio, 0.0, 1.0);
}

} // namespace nerve::optimization

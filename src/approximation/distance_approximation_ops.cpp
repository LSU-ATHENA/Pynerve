#define _USE_MATH_DEFINES

#include "nerve/approximation/distance_approximation.hpp"
#include "nerve/core_types.hpp"
#include "nerve/simd/simd_base.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace nerve::approximation
{

template <typename T>
void jlProjectRandom(const T *points, Size n, Size dim, Size target_dim, T *projected)
{
    static constexpr T kScale = 3.0;
    std::vector<T> proj(target_dim * dim);
    for (Size i = 0; i < target_dim * dim; ++i)
        proj[i] = (static_cast<T>(rand()) / static_cast<T>(RAND_MAX) - T{0.5}) * kScale;
    for (Size i = 0; i < n; ++i)
    {
        for (Size j = 0; j < target_dim; ++j)
        {
            T sum;
            if constexpr (std::is_same_v<T, double>)
                sum = nerve::simd::simd_dot(&points[i * dim], &proj[j * dim], dim);
            else
                sum = nerve::simd::simd_dot_f32(&points[i * dim], &proj[j * dim], dim);
            projected[i * target_dim + j] = sum;
        }
    }
}

template <typename T>
T computeEpsilonNetRadius(const T *density, Size n)
{
    T total = T{0};
    for (Size i = 0; i < n; ++i)
        total += density[i];
    T mean = total / static_cast<T>(n);
    T sq_total = T{0};
    for (Size i = 0; i < n; ++i)
    {
        T d = density[i] - mean;
        sq_total += d * d;
    }
    T stddev = std::sqrt(sq_total / static_cast<T>(n));
    return mean + T{2.0} * stddev;
}

template <typename T>
Size selectLandmarksUniform(const T *points, Size n, Size k, Size *landmarks)
{
    (void)points;
    if (k >= n)
    {
        for (Size i = 0; i < n; ++i)
            landmarks[i] = i;
        return n;
    }
    std::vector<Size> indices(n);
    for (Size i = 0; i < n; ++i)
        indices[i] = i;
    std::mt19937 rng(std::random_device{}());
    std::shuffle(indices.begin(), indices.end(), rng);
    for (Size i = 0; i < k; ++i)
        landmarks[i] = indices[i];
    return k;
}

template <typename T>
Size selectLandmarksMaxmin(const T *points, Size n, Size dim, Size k, Size *landmarks, T *max_dist)
{
    if (n == 0)
        return 0;
    if (k >= n)
    {
        for (Size i = 0; i < n; ++i)
            landmarks[i] = i;
        return n;
    }

    std::vector<T> min_dist(n, std::numeric_limits<T>::max());
    landmarks[0] = 0;
    Size count = 1;
    T max_d = T{0};

    for (Size i = 0; i < n; ++i)
    {
        T d = T{0};
        for (Size d_ = 0; d_ < dim; ++d_)
            d += (points[i * dim + d_] - points[landmarks[0] * dim + d_]) *
                 (points[i * dim + d_] - points[landmarks[0] * dim + d_]);
        min_dist[i] = d;
        if (d > max_d)
        {
            max_d = d;
            landmarks[1] = i;
        }
    }

    while (count < k)
    {
        Size next = 0;
        max_d = T{0};
        for (Size i = 0; i < n; ++i)
        {
            T d = T{0};
            for (Size d_ = 0; d_ < dim; ++d_)
                d += (points[i * dim + d_] - points[landmarks[count] * dim + d_]) *
                     (points[i * dim + d_] - points[landmarks[count] * dim + d_]);
            min_dist[i] = std::min(min_dist[i], d);
            if (min_dist[i] > max_d)
            {
                max_d = min_dist[i];
                next = i;
            }
        }
        landmarks[++count] = next;
    }
    *max_dist = std::sqrt(max_d);
    return count;
}

template void jlProjectRandom<float>(const float *, Size, Size, Size, float *);
template void jlProjectRandom<double>(const double *, Size, Size, Size, double *);
template float computeEpsilonNetRadius<float>(const float *, Size);
template double computeEpsilonNetRadius<double>(const double *, Size);
template Size selectLandmarksUniform<float>(const float *, Size, Size, Size *);
template Size selectLandmarksUniform<double>(const double *, Size, Size, Size *);
template Size selectLandmarksMaxmin<float>(const float *, Size, Size, Size, Size *, float *);
template Size selectLandmarksMaxmin<double>(const double *, Size, Size, Size, Size *, double *);

SlicedWasserstein::SlicedWasserstein(const ApproximationConfig &config)
    : config_(config)
    , rng_(config.random_seed)
{
    stats_ = DistanceStats{};
}

float SlicedWasserstein::computeDistance(const std::vector<DiagramPoint> &diagram1,
                                         const std::vector<DiagramPoint> &diagram2)
{
    if (projections_.empty())
    {
        projections_ = generateProjections(config_.num_projections);
    }
    if (diagram1.empty() || diagram2.empty())
    {
        return 0.0f;
    }
    float sum = 0.0f;
    for (const auto &proj : projections_)
    {
        auto vals1 = projectDiagram(diagram1, proj);
        auto vals2 = projectDiagram(diagram2, proj);
        sum += compute1dWasserstein(vals1, vals2);
    }
    float dist = sum / static_cast<float>(projections_.size());
    updateStats(dist, 0.0f);
    return dist;
}

std::vector<float>
SlicedWasserstein::computeDistances(const std::vector<std::vector<DiagramPoint>> &diagrams)
{
    if (diagrams.size() < 2)
    {
        return {};
    }
    std::vector<float> result;
    result.reserve(diagrams.size() - 1);
    for (size_t i = 1; i < diagrams.size(); ++i)
    {
        result.push_back(computeDistance(diagrams[0], diagrams[i]));
    }
    return result;
}

std::vector<std::vector<float>>
SlicedWasserstein::computeDistanceMatrix(const std::vector<std::vector<DiagramPoint>> &diagrams)
{
    size_t n = diagrams.size();
    std::vector<std::vector<float>> matrix(n, std::vector<float>(n, 0.0f));
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = i + 1; j < n; ++j)
        {
            float d = computeDistance(diagrams[i], diagrams[j]);
            matrix[i][j] = d;
            matrix[j][i] = d;
        }
    }
    return matrix;
}

float SlicedWasserstein::computeAdaptiveDistance(const std::vector<DiagramPoint> &diagram1,
                                                 const std::vector<DiagramPoint> &diagram2,
                                                 float target_error)
{
    if (target_error <= 0.0f)
    {
        target_error = config_.approximation_tolerance;
    }
    return computeDistance(diagram1, diagram2);
}

std::vector<std::vector<float>> SlicedWasserstein::generateProjections(size_t num_projections)
{
    std::vector<std::vector<float>> projs(num_projections, std::vector<float>(2, 0.0f));
    std::uniform_real_distribution<float> angle_dist(0.0f, static_cast<float>(2.0 * M_PI));
    for (size_t i = 0; i < num_projections; ++i)
    {
        float angle = angle_dist(rng_);
        projs[i][0] = std::cos(angle);
        projs[i][1] = std::sin(angle);
    }
    return projs;
}

void SlicedWasserstein::setProjections(const std::vector<std::vector<float>> &projections)
{
    projections_ = projections;
}

SlicedWasserstein::DistanceStats SlicedWasserstein::getStats() const
{
    std::lock_guard lock(stats_mutex_);
    return stats_;
}

void SlicedWasserstein::resetStats()
{
    std::lock_guard lock(stats_mutex_);
    stats_ = DistanceStats{};
}

float SlicedWasserstein::compute1dWasserstein(const std::vector<float> &values1,
                                              const std::vector<float> &values2)
{
    std::vector<float> a = values1;
    std::vector<float> b = values2;
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    float dist = 0.0f;
    size_t max_sz = std::max(a.size(), b.size());
    for (size_t i = 0; i < max_sz; ++i)
    {
        float av = i < a.size() ? a[i] : 0.0f;
        float bv = i < b.size() ? b[i] : 0.0f;
        dist += std::abs(av - bv);
    }
    return dist;
}

std::vector<float> SlicedWasserstein::projectDiagram(const std::vector<DiagramPoint> &diagram,
                                                     const std::vector<float> &projection)
{
    std::vector<float> result;
    result.reserve(diagram.size());
    float px = projection[0];
    float py = projection[1];
    for (const auto &dp : diagram)
    {
        result.push_back(px * dp.birth + py * dp.death);
    }
    std::sort(result.begin(), result.end());
    return result;
}

std::vector<DiagramPoint>
SlicedWasserstein::preprocessDiagram(const std::vector<DiagramPoint> &diagram)
{
    std::vector<DiagramPoint> result;
    for (const auto &dp : diagram)
    {
        if (dp.isValid() && dp.getLifetime() > 0.0f)
        {
            result.push_back(dp);
        }
    }
    return result;
}

void SlicedWasserstein::updateStats(float distance, float computation_time_ms)
{
    std::lock_guard lock(stats_mutex_);
    stats_.num_computations++;
    if (stats_.num_computations == 1)
    {
        stats_.mean_distance = distance;
        stats_.min_distance = distance;
        stats_.max_distance = distance;
    }
    else
    {
        stats_.mean_distance =
            (stats_.mean_distance * static_cast<float>(stats_.num_computations - 1) + distance) /
            static_cast<float>(stats_.num_computations);
        if (distance < stats_.min_distance)
        {
            stats_.min_distance = distance;
        }
        if (distance > stats_.max_distance)
        {
            stats_.max_distance = distance;
        }
    }
}

DiagramLSH::DiagramLSH(const ApproximationConfig &config)
    : config_(config)
{}

void DiagramLSH::buildHashTables(const std::vector<std::vector<DiagramPoint>> &diagrams)
{
    for (size_t i = 0; i < diagrams.size(); ++i)
    {
        addDiagram(i, diagrams[i]);
    }
}

void DiagramLSH::addDiagram(size_t diagram_id, const std::vector<DiagramPoint> &diagram)
{
    diagram_storage_[diagram_id] = diagram;
}

void DiagramLSH::removeDiagram(size_t diagram_id)
{
    diagram_storage_.erase(diagram_id);
}

std::vector<size_t> DiagramLSH::findSimilarDiagrams(const std::vector<DiagramPoint> &query_diagram,
                                                    size_t max_results)
{
    std::vector<std::pair<size_t, float>> ranked =
        findSimilarDiagramsWithDistance(query_diagram, max_results);
    std::vector<size_t> result;
    result.reserve(ranked.size());
    for (const auto &[id, dist] : ranked)
    {
        result.push_back(id);
    }
    return result;
}

std::vector<std::pair<size_t, float>>
DiagramLSH::findSimilarDiagramsWithDistance(const std::vector<DiagramPoint> &query_diagram,
                                            size_t max_results)
{
    SlicedWasserstein sw(config_);
    std::vector<std::pair<size_t, float>> ranked;
    for (const auto &[id, diagram] : diagram_storage_)
    {
        float dist = sw.computeDistance(query_diagram, diagram);
        ranked.emplace_back(id, dist);
    }
    std::sort(ranked.begin(), ranked.end(),
              [](const auto &a, const auto &b) { return a.second < b.second; });
    if (ranked.size() > max_results)
    {
        ranked.resize(max_results);
    }
    return ranked;
}

std::vector<std::vector<size_t>>
DiagramLSH::clusterDiagrams(const std::vector<std::vector<DiagramPoint>> &diagrams,
                            float similarity_threshold)
{
    (void)similarity_threshold;
    std::vector<std::vector<size_t>> clusters;
    for (size_t i = 0; i < diagrams.size(); ++i)
    {
        clusters.push_back({i});
    }
    return clusters;
}

CoarseGrainedMatcher::CoarseGrainedMatcher(const CoarseConfig &config)
    : config_(config)
{
    initializeGrid();
}

std::vector<std::vector<float>>
CoarseGrainedMatcher::discretizeDiagram(const std::vector<DiagramPoint> &diagram)
{
    std::vector<std::vector<float>> grid(config_.grid_resolution,
                                         std::vector<float>(config_.grid_resolution, 0.0f));
    for (const auto &dp : diagram)
    {
        auto coords = computeGridCoordinates(dp.birth, dp.death);
        int bx = std::clamp(static_cast<int>(coords[0]), 0,
                            static_cast<int>(config_.grid_resolution) - 1);
        int by = std::clamp(static_cast<int>(coords[1]), 0,
                            static_cast<int>(config_.grid_resolution) - 1);
        grid[bx][by] += dp.getLifetime();
    }
    return grid;
}

float CoarseGrainedMatcher::computeGridDistance(const std::vector<std::vector<float>> &grid1,
                                                const std::vector<std::vector<float>> &grid2)
{
    float dist = 0.0f;
    for (size_t i = 0; i < grid1.size() && i < grid2.size(); ++i)
    {
        for (size_t j = 0; j < grid1[i].size() && j < grid2[i].size(); ++j)
        {
            dist += std::abs(grid1[i][j] - grid2[i][j]);
        }
    }
    return dist;
}

void CoarseGrainedMatcher::adaptGridResolution(
    const std::vector<std::vector<DiagramPoint>> &diagrams)
{
    (void)diagrams;
    initializeGrid();
}

std::vector<size_t>
CoarseGrainedMatcher::findMatches(const std::vector<DiagramPoint> &query_diagram,
                                  const std::vector<std::vector<DiagramPoint>> &database,
                                  size_t max_results)
{
    std::vector<std::pair<size_t, float>> ranked;
    auto query_grid = discretizeDiagram(query_diagram);
    for (size_t i = 0; i < database.size(); ++i)
    {
        auto db_grid = discretizeDiagram(database[i]);
        float dist = computeGridDistance(query_grid, db_grid);
        ranked.emplace_back(i, dist);
    }
    std::sort(ranked.begin(), ranked.end(),
              [](const auto &a, const auto &b) { return a.second < b.second; });
    if (ranked.size() > max_results)
    {
        ranked.resize(max_results);
    }
    std::vector<size_t> result;
    result.reserve(ranked.size());
    for (const auto &[id, d] : ranked)
    {
        result.push_back(id);
    }
    return result;
}

void CoarseGrainedMatcher::buildIndex(const std::vector<std::vector<DiagramPoint>> &diagrams)
{
    for (size_t i = 0; i < diagrams.size(); ++i)
    {
        addToIndex(i, diagrams[i]);
    }
}

void CoarseGrainedMatcher::addToIndex(size_t diagram_id, const std::vector<DiagramPoint> &diagram)
{
    index_storage_[diagram_id] = discretizeDiagram(diagram);
}

void CoarseGrainedMatcher::removeFromIndex(size_t diagram_id)
{
    index_storage_.erase(diagram_id);
}

void CoarseGrainedMatcher::initializeGrid()
{
    birth_grid_.assign(config_.grid_resolution, std::vector<float>(config_.grid_resolution, 0.0f));
    death_grid_.assign(config_.grid_resolution, std::vector<float>(config_.grid_resolution, 0.0f));
}

std::vector<float> CoarseGrainedMatcher::computeGridCoordinates(float birth, float death)
{
    float bx =
        (birth - config_.birth_range_min) / (config_.birth_range_max - config_.birth_range_min);
    float by =
        (death - config_.death_range_min) / (config_.death_range_max - config_.death_range_min);
    bx = std::clamp(bx, 0.0f, 1.0f);
    by = std::clamp(by, 0.0f, 1.0f);
    return {bx * static_cast<float>(config_.grid_resolution - 1),
            by * static_cast<float>(config_.grid_resolution - 1)};
}

ApproximateBottleneck::ApproximateBottleneck(const BottleneckConfig &config)
    : config_(config)
    , rng_(42)
{}

float ApproximateBottleneck::computeDistance(const std::vector<DiagramPoint> &diagram1,
                                             const std::vector<DiagramPoint> &diagram2)
{
    if (config_.enable_random_sampling && diagram1.size() > config_.num_landmark_points)
    {
        return computeSamplingDistance(diagram1, diagram2);
    }
    return computeLandmarkDistance(diagram1, diagram2);
}

float ApproximateBottleneck::computeLandmarkDistance(const std::vector<DiagramPoint> &diagram1,
                                                     const std::vector<DiagramPoint> &diagram2)
{
    auto lm1 = selectLandmarkPoints(diagram1);
    auto lm2 = selectLandmarkPoints(diagram2);
    return computeExactBottleneck(lm1, lm2);
}

float ApproximateBottleneck::computeSamplingDistance(const std::vector<DiagramPoint> &diagram1,
                                                     const std::vector<DiagramPoint> &diagram2)
{
    auto s1 = sampleDiagram(diagram1, config_.sampling_ratio);
    auto s2 = sampleDiagram(diagram2, config_.sampling_ratio);
    return computeExactBottleneck(s1, s2);
}

float ApproximateBottleneck::computeMultiscaleDistance(const std::vector<DiagramPoint> &diagram1,
                                                       const std::vector<DiagramPoint> &diagram2)
{
    float d1 = computeLandmarkDistance(diagram1, diagram2);
    float d2 = computeSamplingDistance(diagram1, diagram2);
    return std::max(d1, d2);
}

std::vector<DiagramPoint>
ApproximateBottleneck::selectLandmarkPoints(const std::vector<DiagramPoint> &diagram)
{
    if (diagram.size() <= config_.num_landmark_points)
    {
        return diagram;
    }
    std::vector<DiagramPoint> result;
    float step =
        static_cast<float>(diagram.size()) / static_cast<float>(config_.num_landmark_points);
    for (size_t i = 0; i < config_.num_landmark_points; ++i)
    {
        result.push_back(diagram[static_cast<size_t>(static_cast<float>(i) * step)]);
    }
    return result;
}

std::vector<DiagramPoint>
ApproximateBottleneck::sampleDiagram(const std::vector<DiagramPoint> &diagram, float ratio)
{
    if (diagram.empty())
    {
        return {};
    }
    size_t count = static_cast<size_t>(static_cast<float>(diagram.size()) * ratio);
    if (count == 0)
    {
        count = 1;
    }
    std::vector<DiagramPoint> result;
    std::uniform_int_distribution<size_t> dist(0, diagram.size() - 1);
    for (size_t i = 0; i < count; ++i)
    {
        result.push_back(diagram[dist(rng_)]);
    }
    return result;
}

float ApproximateBottleneck::computeExactBottleneck(const std::vector<DiagramPoint> &diagram1,
                                                    const std::vector<DiagramPoint> &diagram2)
{
    if (diagram1.empty() && diagram2.empty())
    {
        return 0.0f;
    }
    float max_dist = 0.0f;
    for (const auto &dp1 : diagram1)
    {
        float min_dist = std::numeric_limits<float>::max();
        for (const auto &dp2 : diagram2)
        {
            float d = std::max(std::abs(dp1.birth - dp2.birth), std::abs(dp1.death - dp2.death));
            min_dist = std::min(min_dist, d);
        }
        max_dist = std::max(max_dist, min_dist);
    }
    return max_dist;
}

} // namespace nerve::approximation

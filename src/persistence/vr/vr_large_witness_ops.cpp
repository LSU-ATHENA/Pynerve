#include "nerve/algebra/complex.hpp"
#include "nerve/persistence/streaming/streaming_vr.hpp"
#include "nerve/persistence/utils/exact_engine.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"
#include "nerve/persistence/vr/vr_landmark_ops.hpp"
#include "nerve/persistence/vr/vr_large_witness_ops.hpp"
#include "nerve/persistence/vr/vr_lazy_witness_ops.hpp"
#include "nerve/persistence/vr/vr_medium_hybrid_ops.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <ranges>
#include <vector>

namespace nerve::persistence
{

namespace
{

constexpr size_t WITNESS_SMALL_THRESHOLD = 10000;
constexpr size_t WITNESS_MEDIUM_THRESHOLD = 50000;
constexpr size_t WITNESS_LARGE_THRESHOLD = 100000;
constexpr size_t WITNESS_MAX_LANDMARKS = 5000;
constexpr size_t WITNESS_MIN_LANDMARKS = 100;
constexpr double WITNESS_LANDMARK_RATIO_SMALL = 0.1;
constexpr double WITNESS_LANDMARK_RATIO_LARGE = 0.05;
constexpr int WITNESS_HIGH_DIM_THRESHOLD = 3;
constexpr double WITNESS_APPROXIMATION_FACTOR = 3.0;

bool isValidLargeWitnessInput(const core::BufferView<const double> &points, Size point_dim,
                              const VRConfig &config)
{
    if (point_dim == 0 || points.empty() || (points.size() % point_dim) != 0)
    {
        return false;
    }
    if (!std::isfinite(config.max_radius) || config.max_radius < 0.0)
    {
        return false;
    }
    if (config.max_dim > static_cast<Size>(std::numeric_limits<Dimension>::max()))
    {
        return false;
    }
    const Size num_points = points.size() / point_dim;
    if (num_points > static_cast<Size>(std::numeric_limits<Index>::max()))
    {
        return false;
    }
    const long double safe_abs =
        std::sqrt(static_cast<long double>(std::numeric_limits<double>::max()) /
                  static_cast<long double>(point_dim)) /
        4.0L;
    for (const double value : points)
    {
        if (!std::isfinite(value) || std::abs(static_cast<long double>(value)) > safe_abs)
        {
            return false;
        }
    }
    return true;
}

} // namespace

std::vector<Pair> computeVrPersistenceLargeWitness(const core::BufferView<const double> &points,
                                                   Size point_dim, const VRConfig &config,
                                                   size_t num_landmarks)
{
    if (!isValidLargeWitnessInput(points, point_dim, config))
    {
        return {};
    }
    const Size num_points = points.size() / point_dim;

    if (num_points <= WITNESS_SMALL_THRESHOLD)
    {
        return computeVrPersistenceMediumHybrid(points, point_dim, config);
    }

    if (num_landmarks == 0)
    {
        num_landmarks =
            std::min(std::max(static_cast<size_t>(static_cast<double>(num_points) * 0.05),
                              WITNESS_MIN_LANDMARKS),
                     WITNESS_MAX_LANDMARKS);
    }

    std::vector<double> point_data(points.begin(), points.end());

    auto landmarks = LandmarkSelector::selectLandmarks(
        point_data, point_dim, num_points, num_landmarks, LandmarkSelector::Strategy::MAXMIN);

    LazyWitnessComplex witness_complex(point_data, point_dim, landmarks, config.max_dim,
                                       config.max_radius);

    algebra::SimplicialComplex complex;
    witness_complex.buildComplex(complex);

    auto exact = computeExactPersistenceZ2(complex, config.max_dim);
    const auto &diagram = exact.pairs;

    std::vector<Pair> pairs;
    pairs.reserve(diagram.size());
    for (const auto &pair : diagram)
    {
        if (pair.dimension <= static_cast<Dimension>(config.max_dim))
        {
            pairs.push_back(pair);
        }
    }

    std::ranges::sort(pairs, {}, &Pair::dimension);

    return pairs;
}

// Configuration helper
WitnessComplexConfig getOptimalWitnessConfig(size_t num_points, size_t point_dim)
{
    WitnessComplexConfig config;

    if (num_points == 0 || point_dim == 0)
    {
        config.approximation_factor = WITNESS_APPROXIMATION_FACTOR;
        return config;
    }

    // Heuristic landmark selection
    if (num_points <= WITNESS_SMALL_THRESHOLD)
    {
        config.num_landmarks = num_points; // Use all points (exact)
    }
    else if (num_points <= WITNESS_MEDIUM_THRESHOLD)
    {
        config.num_landmarks = static_cast<size_t>(static_cast<double>(num_points) *
                                                   WITNESS_LANDMARK_RATIO_SMALL); // 10%
    }
    else if (num_points <= WITNESS_LARGE_THRESHOLD)
    {
        config.num_landmarks = static_cast<size_t>(static_cast<double>(num_points) *
                                                   WITNESS_LANDMARK_RATIO_LARGE); // 5%
    }
    else
    {
        config.num_landmarks = WITNESS_MAX_LANDMARKS; // Cap at 5000 landmarks
    }

    const size_t min_landmarks = std::min(WITNESS_MIN_LANDMARKS, num_points);
    config.num_landmarks = std::min(std::max(config.num_landmarks, min_landmarks), num_points);

    // Selection strategy
    if (point_dim <= WITNESS_HIGH_DIM_THRESHOLD)
    {
        config.strategy = LandmarkSelectionStrategy::MAXMIN; // epsilon-net
    }
    else
    {
        config.strategy = LandmarkSelectionStrategy::DENSITY; // For high-dim
    }

    // Approximation quality
    config.approximation_factor = WITNESS_APPROXIMATION_FACTOR; // 3-approximation of VR
    config.max_witness_distance = 0.0;                          // Auto-compute from data

    return config;
}

ApproximationBounds computeWitnessApproximationBounds(size_t num_points, size_t num_landmarks,
                                                      double max_radius)
{
    const size_t landmarks_used = std::min(num_landmarks, num_points);
    const bool has_valid_radius = std::isfinite(max_radius) && max_radius >= 0.0;
    const bool has_landmarks = num_points > 0 && landmarks_used > 0;

    const double factor = has_landmarks ? WITNESS_APPROXIMATION_FACTOR : 1.0;
    const double estimated_coverage =
        has_landmarks && has_valid_radius
            ? std::min(1.0, static_cast<double>(landmarks_used) / static_cast<double>(num_points))
            : 0.0;

    return {factor, factor, factor, landmarks_used, estimated_coverage};
}

} // namespace nerve::persistence


#include "nerve/persistence/adaptive_acceleration/streaming/approximate_processor.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <ranges>
#include <utility>

namespace nerve::persistence::adaptive_acceleration::approximation
{
namespace
{

constexpr double kMaxFiniteError = 1.0e300;

double boundedFinite(long double value)
{
    if (!std::isfinite(value))
    {
        return kMaxFiniteError;
    }
    if (value <= 0.0L)
    {
        return 0.0;
    }
    return static_cast<double>(std::min(value, static_cast<long double>(kMaxFiniteError)));
}

bool checkedProduct(std::size_t lhs, std::size_t rhs, std::size_t &out)
{
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs)
    {
        out = 0;
        return false;
    }
    out = lhs * rhs;
    return true;
}

bool validPair(const Pair &pair)
{
    const bool finite_death = std::isfinite(pair.death);
    const bool infinite_death = pair.isInfinite();
    return std::isfinite(pair.birth) && (finite_death || infinite_death) && pair.dimension >= 0 &&
           (!finite_death || pair.death >= pair.birth);
}

errors::ErrorResult<void> validatePointBuffer(const core::BufferView<const double> &points,
                                              std::size_t point_dim)
{
    if (point_dim == 0 || points.empty() || points.size() % point_dim != 0)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }
    for (const double value : points)
    {
        if (!std::isfinite(value))
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E54_PH4_INVALID_INPUT);
        }
    }
    return errors::ErrorResult<void>::ok();
}

std::size_t levelStride(ApproximationLevel level)
{
    switch (level)
    {
        case ApproximationLevel::EXACT:
        case ApproximationLevel::HIGH_PRECISION:
            return 1;
        case ApproximationLevel::MEDIUM_PRECISION:
            return 2;
        case ApproximationLevel::LOW_PRECISION:
            return 4;
        case ApproximationLevel::VERY_FAST:
            return 8;
    }
    return 1;
}

double levelErrorBudget(ApproximationLevel level)
{
    switch (level)
    {
        case ApproximationLevel::EXACT:
            return 0.0;
        case ApproximationLevel::HIGH_PRECISION:
            return 1e-8;
        case ApproximationLevel::MEDIUM_PRECISION:
            return 1e-6;
        case ApproximationLevel::LOW_PRECISION:
            return 1e-4;
        case ApproximationLevel::VERY_FAST:
            return 1e-2;
    }
    return 0.0;
}

errors::ErrorResult<std::vector<double>>
downsamplePoints(const core::BufferView<const double> &points, std::size_t point_dim,
                 std::size_t stride)
{
    auto validation = validatePointBuffer(points, point_dim);
    if (validation.isError())
    {
        return errors::ErrorResult<std::vector<double>>::error(validation.errorCode());
    }
    const std::size_t point_count = points.size() / point_dim;
    const std::size_t effective_stride = std::max<std::size_t>(1, stride);
    std::vector<double> reduced;
    const std::size_t sampled_points = (point_count + effective_stride - 1) / effective_stride;
    std::size_t reduced_values = 0;
    if (!checkedProduct(sampled_points, point_dim, reduced_values) ||
        reduced_values > reduced.max_size())
    {
        return errors::ErrorResult<std::vector<double>>::error(
            errors::ErrorCode::E41_RESOURCE_LIMIT);
    }
    reduced.reserve(reduced_values);
    for (std::size_t point = 0; point < point_count; point += effective_stride)
    {
        const std::size_t offset = point * point_dim;
        for (std::size_t dim = 0; dim < point_dim; ++dim)
        {
            reduced.push_back(points[offset + dim]);
        }
    }
    return errors::ErrorResult<std::vector<double>>::success(std::move(reduced));
}

errors::ErrorResult<std::vector<Pair>> scalePairs(const std::vector<Pair> &pairs, double factor)
{
    if (!std::isfinite(factor))
    {
        return errors::ErrorResult<std::vector<Pair>>::error(
            errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }
    std::vector<Pair> scaled;
    if (pairs.size() > scaled.max_size())
    {
        return errors::ErrorResult<std::vector<Pair>>::error(errors::ErrorCode::E41_RESOURCE_LIMIT);
    }
    scaled.reserve(pairs.size());
    for (const auto &pair : pairs)
    {
        Pair out = pair;
        if (!validPair(out))
        {
            return errors::ErrorResult<std::vector<Pair>>::error(
                errors::ErrorCode::E54_PH4_INVALID_INPUT);
        }
        out.birth = boundedFinite(static_cast<long double>(out.birth) * factor);
        if (std::isfinite(out.death))
        {
            out.death = boundedFinite(static_cast<long double>(out.death) * factor);
        }
        scaled.push_back(out);
    }
    return errors::ErrorResult<std::vector<Pair>>::success(std::move(scaled));
}

} // namespace

class ApproximateProcessor::Impl
{
public:
    explicit Impl(const ApproximationConfig &config)
        : config(config)
        , stats()
    {}

    errors::ErrorResult<std::vector<Pair>> compute(const core::BufferView<const double> &points,
                                                   std::size_t point_dim, ApproximationLevel level,
                                                   const VRConfig &config_override)
    {
        auto validation = validatePointBuffer(points, point_dim);
        if (validation.isError())
        {
            return errors::ErrorResult<std::vector<Pair>>::error(validation.errorCode());
        }

        const auto start = std::chrono::steady_clock::now();
        const std::size_t stride = levelStride(level);
        auto reduced_result = downsamplePoints(points, point_dim, stride);
        if (reduced_result.isError())
        {
            return errors::ErrorResult<std::vector<Pair>>::error(reduced_result.errorCode());
        }
        const std::vector<double> reduced_points = reduced_result.moveValue();
        if (reduced_points.empty())
        {
            return errors::ErrorResult<std::vector<Pair>>::error(
                errors::ErrorCode::E54_PH4_INVALID_INPUT);
        }

        VRConfig effective = config_override;
        effective.use_adaptive_acceleration = false;
        effective.auto_detect_adaptive_acceleration = false;
        effective.enable_approximation = false;
        const core::BufferView<const double> reducedView(reduced_points.data(),
                                                         reduced_points.size());
        std::vector<Pair> exact_on_reduced =
            computeVrPersistenceFast(reducedView, point_dim, effective);
        std::vector<Pair> result_pairs;
        if (stride > 1)
        {
            auto scaled = scalePairs(exact_on_reduced, static_cast<double>(stride));
            if (scaled.isError())
            {
                return errors::ErrorResult<std::vector<Pair>>::error(scaled.errorCode());
            }
            result_pairs = scaled.moveValue();
        }
        else
        {
            result_pairs = std::move(exact_on_reduced);
        }
        std::ranges::sort(result_pairs, {},
                          [](const Pair &p) { return std::tuple(p.dimension, p.birth, p.death); });

        const auto end = std::chrono::steady_clock::now();
        stats.computation_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        stats.speedup_factor = std::max(1.0, static_cast<double>(points.size()) /
                                                 static_cast<double>(reduced_points.size()));
        stats.approximation_error = levelErrorBudget(level);
        stats.error_bound = stats.approximation_error;
        stats.iterations_used = 1;
        stats.converged = true;
        stats.approximation_details = "downsampled exact baseline";

        return errors::ErrorResult<std::vector<Pair>>::success(std::move(result_pairs));
    }

    ApproximationConfig config;
    ApproximationStats stats;
};

errors::ErrorResult<std::unique_ptr<ApproximateProcessor>>
ApproximateProcessor::create(const ApproximationConfig &config)
{
    if (config.max_iterations == 0 || !std::isfinite(config.max_error) || config.max_error < 0.0 ||
        !std::isfinite(config.convergence_threshold) || config.convergence_threshold < 0.0 ||
        !std::isfinite(config.target_speedup) || config.target_speedup < 0.0)
    {
        return errors::ErrorResult<std::unique_ptr<ApproximateProcessor>>::error(
            errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }
    std::unique_ptr<ApproximateProcessor> processor(new ApproximateProcessor(config));
    return errors::ErrorResult<std::unique_ptr<ApproximateProcessor>>::success(
        std::move(processor));
}

errors::ErrorResult<std::vector<Pair>>
ApproximateProcessor::computeApproximate(const core::BufferView<const double> &points,
                                         std::size_t point_dim, ApproximationLevel level,
                                         const VRConfig &config)
{
    auto result = impl_->compute(points, point_dim, level, config);
    if (result.isError())
    {
        return errors::ErrorResult<std::vector<Pair>>::error(result.errorCode());
    }
    approximation_stats_ = impl_->stats;
    return result;
}

errors::ErrorResult<std::vector<Pair>> ApproximateProcessor::computeProgressive(
    const core::BufferView<const double> &points, std::size_t point_dim, const VRConfig &config,
    std::function<void(const std::vector<Pair> &, double)> callback)
{
    const std::vector<ApproximationLevel> schedule = {
        ApproximationLevel::VERY_FAST, ApproximationLevel::LOW_PRECISION,
        ApproximationLevel::MEDIUM_PRECISION, ApproximationLevel::HIGH_PRECISION,
        ApproximationLevel::EXACT};

    std::vector<Pair> current;
    std::size_t iteration = 0;
    for (ApproximationLevel level : schedule)
    {
        if (iteration >= config_.max_iterations)
        {
            break;
        }
        auto result = computeApproximate(points, point_dim, level, config);
        if (result.isError())
        {
            return errors::ErrorResult<std::vector<Pair>>::error(result.errorCode());
        }
        current = result.value();
        const double error = getApproximationError(level);
        if (callback)
        {
            callback(current, error);
        }
        ++iteration;
        if (error <= config_.max_error || level == ApproximationLevel::EXACT)
        {
            approximation_stats_.iterations_used = iteration;
            approximation_stats_.converged = true;
            return errors::ErrorResult<std::vector<Pair>>::success(std::move(current));
        }
    }
    approximation_stats_.iterations_used = iteration;
    approximation_stats_.converged = false;
    return errors::ErrorResult<std::vector<Pair>>::success(std::move(current));
}

ApproximateProcessor::ApproximateProcessor(const ApproximationConfig &config)
    : impl_(std::make_unique<Impl>(config))
    , config_(config)
    , approximation_stats_()
{}

ApproximateProcessor::~ApproximateProcessor() = default;

errors::ErrorResult<std::vector<Pair>>
ApproximateProcessor::approximateHighPrecision(const core::BufferView<const double> &points,
                                               std::size_t point_dim, const VRConfig &config)
{
    return computeApproximate(points, point_dim, ApproximationLevel::HIGH_PRECISION, config);
}

errors::ErrorResult<std::vector<Pair>>
ApproximateProcessor::approximateMediumPrecision(const core::BufferView<const double> &points,
                                                 std::size_t point_dim, const VRConfig &config)
{
    return computeApproximate(points, point_dim, ApproximationLevel::MEDIUM_PRECISION, config);
}

errors::ErrorResult<std::vector<Pair>>
ApproximateProcessor::approximateLowPrecision(const core::BufferView<const double> &points,
                                              std::size_t point_dim, const VRConfig &config)
{
    return computeApproximate(points, point_dim, ApproximationLevel::LOW_PRECISION, config);
}

errors::ErrorResult<std::vector<Pair>>
ApproximateProcessor::approximateVeryFast(const core::BufferView<const double> &points,
                                          std::size_t point_dim, const VRConfig &config)
{
    return computeApproximate(points, point_dim, ApproximationLevel::VERY_FAST, config);
}

double ApproximateProcessor::getApproximationError(ApproximationLevel level)
{
    return levelErrorBudget(level);
}

double ApproximateProcessor::getSpeedupFactor(ApproximationLevel level)
{
    const std::size_t stride = levelStride(level);
    return static_cast<double>(stride);
}

std::size_t ApproximateProcessor::getMaxIterations(ApproximationLevel level)
{
    return level == ApproximationLevel::EXACT ? 1 : config_.max_iterations;
}

bool ApproximateProcessor::shouldUseProgressiveRefinement(ApproximationLevel level)
{
    return config_.enable_progressive_refinement && level != ApproximationLevel::EXACT;
}

// ProgressiveRefinementProcessor and ErrorBoundsCalculator split to approximate_processor_ops.cpp

} // namespace nerve::persistence::adaptive_acceleration::approximation

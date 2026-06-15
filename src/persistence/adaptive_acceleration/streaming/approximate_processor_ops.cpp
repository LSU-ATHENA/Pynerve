// Approximate processing operations  --  progressive refinement and error bounds computation.

#include "nerve/persistence/adaptive_acceleration/streaming/approximate_processor.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nerve::persistence::adaptive_acceleration::approximation
{
namespace
{

constexpr double kMaxFiniteError = 1.0e300;
constexpr int SINKHORN_MAX_ITERATIONS = 200;

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

bool validPair(const Pair &pair)
{
    const bool finite_death = std::isfinite(pair.death);
    const bool infinite_death = pair.isInfinite();
    return std::isfinite(pair.birth) && (finite_death || infinite_death) && pair.dimension >= 0 &&
           (!finite_death || pair.death >= pair.birth);
}

errors::ErrorResult<void> validatePairs(const std::vector<Pair> &pairs)
{
    for (const auto &pair : pairs)
    {
        if (!validPair(pair))
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E54_PH4_INVALID_INPUT);
        }
    }
    return errors::ErrorResult<void>::ok();
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

double boundedAbsDifference(double lhs, double rhs)
{
    return boundedFinite(std::abs(static_cast<long double>(lhs) - static_cast<long double>(rhs)));
}

double boundedSum(double lhs, double rhs)
{
    return boundedFinite(static_cast<long double>(lhs) + static_cast<long double>(rhs));
}

std::vector<double> sortedLifetimes(const std::vector<Pair> &pairs)
{
    std::vector<double> lifetimes;
    lifetimes.reserve(pairs.size());
    for (const auto &pair : pairs)
    {
        if (!std::isfinite(pair.birth) || std::isnan(pair.death))
        {
            lifetimes.push_back(kMaxFiniteError);
            continue;
        }
        if (!std::isfinite(pair.death))
        {
            lifetimes.push_back(kMaxFiniteError);
            continue;
        }
        lifetimes.push_back(boundedFinite(static_cast<long double>(pair.death) -
                                          static_cast<long double>(pair.birth)));
    }
    std::ranges::sort(lifetimes);
    return lifetimes;
}

} // namespace

class ProgressiveRefinementProcessor::Impl
{
public:
    explicit Impl(const ApproximationConfig &cfg)
        : config(cfg)
    {}

    ApproximationConfig config;
};

errors::ErrorResult<std::unique_ptr<ProgressiveRefinementProcessor>>
ProgressiveRefinementProcessor::create(const ApproximationConfig &config)
{
    std::unique_ptr<ProgressiveRefinementProcessor> processor(
        new ProgressiveRefinementProcessor(config));
    return errors::ErrorResult<std::unique_ptr<ProgressiveRefinementProcessor>>::success(
        std::move(processor));
}

errors::ErrorResult<std::vector<Pair>> ProgressiveRefinementProcessor::refineProgressive(
    const core::BufferView<const double> &points, std::size_t point_dim, const VRConfig &config,
    std::function<void(const std::vector<Pair> &, std::size_t, double)> callback)
{
    auto base_processor = ApproximateProcessor::create(config_);
    if (base_processor.isError())
    {
        return errors::ErrorResult<std::vector<Pair>>::error(base_processor.errorCode());
    }
    std::size_t iteration = 0;
    auto result = base_processor.value()->computeProgressive(
        points, point_dim, config,
        [&callback, &iteration](const std::vector<Pair> &pairs, double error) {
            if (callback)
            {
                callback(pairs, iteration, error);
            }
            ++iteration;
        });
    if (result.isError())
    {
        return errors::ErrorResult<std::vector<Pair>>::error(result.errorCode());
    }
    return result;
}

ProgressiveRefinementProcessor::ProgressiveRefinementProcessor(const ApproximationConfig &config)
    : impl_(std::make_unique<Impl>(config))
    , config_(config)
{}

ProgressiveRefinementProcessor::~ProgressiveRefinementProcessor() = default;

errors::ErrorResult<ErrorBounds>
ErrorBoundsCalculator::calculateErrorBounds(const std::vector<Pair> &approximate_result,
                                            const std::vector<Pair> &exact_result,
                                            ApproximationLevel level)
{
    auto approximate_validation = validatePairs(approximate_result);
    if (approximate_validation.isError())
    {
        return errors::ErrorResult<ErrorBounds>::error(approximate_validation.errorCode());
    }
    auto exact_validation = validatePairs(exact_result);
    if (exact_validation.isError())
    {
        return errors::ErrorResult<ErrorBounds>::error(exact_validation.errorCode());
    }

    ErrorBounds bounds;
    bounds.bottleneck_distance = computeBottleneckDistance(approximate_result, exact_result);
    bounds.wasserstein_distance = computeWassersteinDistance(approximate_result, exact_result);
    bounds.hausdorff_distance = std::max(bounds.bottleneck_distance, bounds.wasserstein_distance);
    bounds.total_persistence_error = bounds.wasserstein_distance;
    bounds.max_persistence_error = bounds.bottleneck_distance;
    bounds.isValid = true;
    bounds.error_details =
        "computed_from_exact_reference level=" + std::to_string(static_cast<int>(level));
    return errors::ErrorResult<ErrorBounds>::success(std::move(bounds));
}

errors::ErrorResult<ErrorBounds>
ErrorBoundsCalculator::estimateErrorBounds(const std::vector<Pair> &approximate_result,
                                           ApproximationLevel level,
                                           const ProblemCharacteristics &problem_characteristics)
{
    auto validation = validatePairs(approximate_result);
    if (validation.isError())
    {
        return errors::ErrorResult<ErrorBounds>::error(validation.errorCode());
    }

    ErrorBounds bounds;
    bounds.max_persistence_error = levelErrorBudget(level);
    bounds.bottleneck_distance = bounds.max_persistence_error;
    bounds.wasserstein_distance =
        boundedFinite(static_cast<long double>(bounds.max_persistence_error) *
                      static_cast<long double>(approximate_result.size()));
    bounds.hausdorff_distance = bounds.bottleneck_distance;
    bounds.total_persistence_error = bounds.wasserstein_distance;
    bounds.isValid = true;
    bounds.error_details = "estimated_without_exact_reference points=" +
                           std::to_string(problem_characteristics.num_points) +
                           " columns=" + std::to_string(problem_characteristics.estimated_columns);
    return errors::ErrorResult<ErrorBounds>::success(std::move(bounds));
}

double ErrorBoundsCalculator::computeBottleneckDistance(const std::vector<Pair> &pairs1,
                                                        const std::vector<Pair> &pairs2)
{
    const std::vector<double> life1 = sortedLifetimes(pairs1);
    const std::vector<double> life2 = sortedLifetimes(pairs2);
    const std::size_t common = std::min(life1.size(), life2.size());
    double max_gap = 0.0;
    for (std::size_t i = 0; i < common; ++i)
    {
        max_gap = std::max(max_gap, boundedAbsDifference(life1[i], life2[i]));
    }
    if (life1.size() != life2.size())
    {
        max_gap = std::max(max_gap, kMaxFiniteError);
    }
    return std::isfinite(max_gap) ? max_gap : kMaxFiniteError;
}

double ErrorBoundsCalculator::computeWassersteinDistance(const std::vector<Pair> &pairs1,
                                                         const std::vector<Pair> &pairs2)
{
    const std::vector<double> life1 = sortedLifetimes(pairs1);
    const std::vector<double> life2 = sortedLifetimes(pairs2);
    const std::size_t common = std::min(life1.size(), life2.size());
    double sum = 0.0;
    for (std::size_t i = 0; i < common; ++i)
    {
        sum = boundedSum(sum, boundedAbsDifference(life1[i], life2[i]));
    }
    if (life1.size() > common)
    {
        for (auto it = life1.begin() + static_cast<std::ptrdiff_t>(common); it != life1.end(); ++it)
        {
            sum = boundedSum(sum, *it);
        }
    }
    if (life2.size() > common)
    {
        for (auto it = life2.begin() + static_cast<std::ptrdiff_t>(common); it != life2.end(); ++it)
        {
            sum = boundedSum(sum, *it);
        }
    }
    return std::isfinite(sum) ? sum : kMaxFiniteError;
}

} // namespace nerve::persistence::adaptive_acceleration::approximation

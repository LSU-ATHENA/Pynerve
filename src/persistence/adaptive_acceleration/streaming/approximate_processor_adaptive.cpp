
#include "nerve/persistence/adaptive_acceleration/streaming/approximate_processor.hpp"

#include <algorithm>
#include <memory>
#include <utility>

namespace nerve::persistence::adaptive_acceleration::approximation
{

class AdaptivePrecisionManager::Impl
{
public:
    explicit Impl(const ApproximationConfig &cfg)
        : config(cfg)
    {}

    ApproximationConfig config;
};

errors::ErrorResult<std::unique_ptr<AdaptivePrecisionManager>>
AdaptivePrecisionManager::create(const ApproximationConfig &config)
{
    std::unique_ptr<AdaptivePrecisionManager> manager(new AdaptivePrecisionManager(config));
    return errors::ErrorResult<std::unique_ptr<AdaptivePrecisionManager>>::success(
        std::move(manager));
}

ApproximationLevel AdaptivePrecisionManager::selectOptimalPrecision(
    const ProblemCharacteristics &problem_characteristics,
    const PerformanceRequirements &performance_requirements)
{
    if (performance_requirements.min_precision <= 1e-10)
    {
        return ApproximationLevel::EXACT;
    }
    if (performance_requirements.max_time_ms < 50.0 ||
        problem_characteristics.estimated_columns > 10000)
    {
        return ApproximationLevel::VERY_FAST;
    }
    if (problem_characteristics.sparsity_ratio < 0.1)
    {
        return ApproximationLevel::LOW_PRECISION;
    }
    return ApproximationLevel::MEDIUM_PRECISION;
}

ApproximationLevel AdaptivePrecisionManager::adjustPrecision(ApproximationLevel current_level,
                                                             double actual_speedup,
                                                             double target_speedup)
{
    if (actual_speedup >= target_speedup)
    {
        return current_level;
    }
    switch (current_level)
    {
        case ApproximationLevel::EXACT:
            return ApproximationLevel::HIGH_PRECISION;
        case ApproximationLevel::HIGH_PRECISION:
            return ApproximationLevel::MEDIUM_PRECISION;
        case ApproximationLevel::MEDIUM_PRECISION:
            return ApproximationLevel::LOW_PRECISION;
        case ApproximationLevel::LOW_PRECISION:
            return ApproximationLevel::VERY_FAST;
        case ApproximationLevel::VERY_FAST:
            break;
    }
    return ApproximationLevel::VERY_FAST;
}

AdaptivePrecisionManager::AdaptivePrecisionManager(const ApproximationConfig &config)
    : impl_(std::make_unique<Impl>(config))
    , config_(config)
{}

AdaptivePrecisionManager::~AdaptivePrecisionManager() = default;

errors::ErrorResult<std::unique_ptr<ApproximateProcessor>>
ApproximateProcessorFactory::createForLevel(ApproximationLevel level,
                                            const ApproximationConfig &config)
{
    ApproximationConfig configured = config;
    configured.level = level;
    return ApproximateProcessor::create(configured);
}

errors::ErrorResult<std::unique_ptr<ApproximateProcessor>>
ApproximateProcessorFactory::createForExploratory(const ApproximationConfig &config)
{
    ApproximationConfig configured = config;
    configured.level = ApproximationLevel::VERY_FAST;
    configured.target_speedup = std::max(2.0, configured.target_speedup);
    return ApproximateProcessor::create(configured);
}

errors::ErrorResult<std::unique_ptr<ApproximateProcessor>>
ApproximateProcessorFactory::createForHighPrecision(const ApproximationConfig &config)
{
    ApproximationConfig configured = config;
    configured.level = ApproximationLevel::HIGH_PRECISION;
    configured.max_error = std::min(configured.max_error, 1e-8);
    return ApproximateProcessor::create(configured);
}

} // namespace nerve::persistence::adaptive_acceleration::approximation

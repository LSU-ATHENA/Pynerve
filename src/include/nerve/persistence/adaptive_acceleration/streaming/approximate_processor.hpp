
#pragma once

#include "nerve/core.hpp"
#include "nerve/persistence/adaptive_acceleration/adaptive_acceleration_engine.hpp"
#include "nerve/persistence/adaptive_acceleration/adaptive_algorithm_selector.hpp"
#include "nerve/persistence/core/core_types.hpp"

#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

namespace nerve::persistence::adaptive_acceleration::approximation
{

enum class ApproximationLevel
{
    EXACT,
    HIGH_PRECISION,
    MEDIUM_PRECISION,
    LOW_PRECISION,
    VERY_FAST
};

struct ApproximationConfig
{
    ApproximationLevel level = ApproximationLevel::EXACT;
    double max_error = 1e-6;
    bool enable_progressive_refinement = true;
    std::size_t max_iterations = 8;
    double convergence_threshold = 1e-8;
    bool enable_adaptive_precision = true;
    double target_speedup = 1.0;
};

struct ApproximationStats
{
    double computation_time_ms = 0.0;
    double speedup_factor = 1.0;
    double approximation_error = 0.0;
    double error_bound = 0.0;
    std::size_t iterations_used = 0;
    bool converged = false;
    std::string approximation_details;
};

struct ErrorBounds
{
    double bottleneck_distance = 0.0;
    double wasserstein_distance = 0.0;
    double hausdorff_distance = 0.0;
    double total_persistence_error = 0.0;
    double max_persistence_error = 0.0;
    bool isValid = false;
    std::string error_details;
};

class ApproximateProcessor
{
public:
    static errors::ErrorResult<std::unique_ptr<ApproximateProcessor>>
    create(const ApproximationConfig &config);
    ~ApproximateProcessor();

    errors::ErrorResult<std::vector<Pair>> computeApproximate(core::BufferView<const double> points,
                                                              std::size_t point_dim,
                                                              ApproximationLevel level,
                                                              const VRConfig &config);

    errors::ErrorResult<std::vector<Pair>>
    computeProgressive(core::BufferView<const double> points, std::size_t point_dim,
                       const VRConfig &config,
                       std::function<void(const std::vector<Pair> &, double)> callback);

    const ApproximationStats &getApproximationStats() const { return approximation_stats_; }

private:
    explicit ApproximateProcessor(const ApproximationConfig &config);

    class Impl;
    std::unique_ptr<Impl> impl_;
    ApproximationConfig config_;
    ApproximationStats approximation_stats_;

    errors::ErrorResult<std::vector<Pair>>
    approximateHighPrecision(core::BufferView<const double> points, std::size_t point_dim,
                             const VRConfig &config);

    errors::ErrorResult<std::vector<Pair>>
    approximateMediumPrecision(core::BufferView<const double> points, std::size_t point_dim,
                               const VRConfig &config);

    errors::ErrorResult<std::vector<Pair>>
    approximateLowPrecision(core::BufferView<const double> points, std::size_t point_dim,
                            const VRConfig &config);

    errors::ErrorResult<std::vector<Pair>>
    approximateVeryFast(core::BufferView<const double> points, std::size_t point_dim,
                        const VRConfig &config);

    double getApproximationError(ApproximationLevel level);
    double getSpeedupFactor(ApproximationLevel level);
    std::size_t getMaxIterations(ApproximationLevel level);
    bool shouldUseProgressiveRefinement(ApproximationLevel level);
};

class ProgressiveRefinementProcessor
{
public:
    static errors::ErrorResult<std::unique_ptr<ProgressiveRefinementProcessor>>
    create(const ApproximationConfig &config);
    ~ProgressiveRefinementProcessor();

    errors::ErrorResult<std::vector<Pair>>
    refineProgressive(core::BufferView<const double> points, std::size_t point_dim,
                      const VRConfig &config,
                      std::function<void(const std::vector<Pair> &, std::size_t, double)> callback);

private:
    explicit ProgressiveRefinementProcessor(const ApproximationConfig &config);

    class Impl;
    std::unique_ptr<Impl> impl_;
    ApproximationConfig config_;
};

class ErrorBoundsCalculator
{
public:
    static errors::ErrorResult<ErrorBounds>
    calculateErrorBounds(const std::vector<Pair> &approximate_result,
                         const std::vector<Pair> &exact_result, ApproximationLevel level);

    static errors::ErrorResult<ErrorBounds>
    estimateErrorBounds(const std::vector<Pair> &approximate_result, ApproximationLevel level,
                        const ProblemCharacteristics &problem_characteristics);

private:
    static double computeBottleneckDistance(const std::vector<Pair> &pairs1,
                                            const std::vector<Pair> &pairs2);

    static double computeWassersteinDistance(const std::vector<Pair> &pairs1,
                                             const std::vector<Pair> &pairs2);
};

class AdaptivePrecisionManager
{
public:
    static errors::ErrorResult<std::unique_ptr<AdaptivePrecisionManager>>
    create(const ApproximationConfig &config);
    ~AdaptivePrecisionManager();

    ApproximationLevel
    selectOptimalPrecision(const ProblemCharacteristics &problem_characteristics,
                           const PerformanceRequirements &performance_requirements);

    ApproximationLevel adjustPrecision(ApproximationLevel current_level, double actual_speedup,
                                       double target_speedup);

private:
    explicit AdaptivePrecisionManager(const ApproximationConfig &config);

    class Impl;
    std::unique_ptr<Impl> impl_;
    ApproximationConfig config_;
};

class ApproximateProcessorFactory
{
public:
    static errors::ErrorResult<std::unique_ptr<ApproximateProcessor>>
    createForLevel(ApproximationLevel level, const ApproximationConfig &config);

    static errors::ErrorResult<std::unique_ptr<ApproximateProcessor>>
    createForExploratory(const ApproximationConfig &config);

    static errors::ErrorResult<std::unique_ptr<ApproximateProcessor>>
    createForHighPrecision(const ApproximationConfig &config);
};

} // namespace nerve::persistence::adaptive_acceleration::approximation


#include "nerve/persistence/adaptive_acceleration/adaptive_algorithm_selector.hpp"
#include "nerve/persistence/adaptive_acceleration/adaptive_selector_calibration.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"
#include "nerve/runtime/hardware_probe.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <utility>

namespace nerve::persistence::adaptive_acceleration
{
namespace
{
constexpr double kMaxFinitePrediction = 1.0e300;

double finiteNonnegativePrediction(long double value)
{
    if (!std::isfinite(value))
    {
        return kMaxFinitePrediction;
    }
    if (value <= 0.0L)
    {
        return 0.0;
    }
    return static_cast<double>(std::min(value, static_cast<long double>(kMaxFinitePrediction)));
}

double finitePositiveInput(double value, double fallback)
{
    if (!std::isfinite(value) || value <= 0.0)
    {
        return fallback;
    }
    return value;
}

double finiteAtLeast(double value, double minimum, double nonfinite_fallback)
{
    if (!std::isfinite(value))
    {
        return nonfinite_fallback;
    }
    return std::max(minimum, value);
}

double finiteUnitRatio(double value, double fallback)
{
    if (!std::isfinite(value))
    {
        return fallback;
    }
    return std::clamp(value, 0.0, 1.0);
}

double boundedConfidence(double value)
{
    if (!std::isfinite(value))
    {
        return 0.05;
    }
    return std::clamp(value, 0.05, 0.99);
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

errors::ErrorResult<void> validateExecutionInput(core::BufferView<const double> points,
                                                 std::size_t point_dim, const VRConfig &config)
{
    if (point_dim == 0 || points.size() == 0 || points.size() % point_dim != 0 ||
        config.max_dim == 0)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }
    if (!std::isfinite(config.performance_target_ms) || config.performance_target_ms < 0.0 ||
        !std::isfinite(config.memory_limit_mb) || config.memory_limit_mb < 0.0)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }
    for (double value : points)
    {
        if (!std::isfinite(value))
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E54_PH4_INVALID_INPUT);
        }
    }
    return errors::ErrorResult<void>::ok();
}

double timeComplexityFactor(const ProblemCharacteristics &problem)
{
    const long double n =
        static_cast<long double>(std::max<std::size_t>(1, problem.estimated_columns));
    return finiteNonnegativePrediction(n * std::log2(n + 1.0L));
}

double powerTimeEstimate(std::size_t columns, long double exponent, long double scale)
{
    const long double n = static_cast<long double>(std::max<std::size_t>(1, columns));
    return finiteNonnegativePrediction(std::pow(n, exponent) * scale);
}

double predictionSelectionScore(const PerformancePredictor::Prediction &prediction)
{
    const double confidence = boundedConfidence(prediction.confidence_score);
    return finiteNonnegativePrediction(static_cast<long double>(finitePositiveInput(
                                           prediction.estimated_time_ms, kMaxFinitePrediction)) /
                                       static_cast<long double>(confidence));
}

void sanitizePrediction(PerformancePredictor::Prediction &prediction)
{
    prediction.estimated_time_ms = finiteNonnegativePrediction(prediction.estimated_time_ms);
    prediction.estimated_memory_mb = finiteNonnegativePrediction(prediction.estimated_memory_mb);
    prediction.confidence_score = boundedConfidence(prediction.confidence_score);
}

} // namespace
std::vector<PerformancePredictor::Prediction>
PerformancePredictor::predictPerformance(const ProblemCharacteristics &problem,
                                         const SystemCapabilities &system)
{
    std::vector<Prediction> predictions;
    predictions.reserve(6);

    const runtime::HardwareSnapshot snapshot = runtime::collectHardwareSnapshot();
    const std::string fingerprint = runtime::getHardwareFingerprint(snapshot);

    const double baseline = std::max(1.0, timeComplexityFactor(problem) * 1e-4);
    const double memory_scale = std::max(1.0, problem.memory_requirement_mb);

    Prediction matrix;
    matrix.algorithm_type = AdaptiveAlgorithmType::MATRIX_MULTIPLICATION_CPU;
    matrix.estimated_time_ms = estimateMatrixMultiplicationTime(problem);
    matrix.estimated_memory_mb = estimateMemoryUsage(problem, matrix.algorithm_type);
    matrix.confidence_score = boundedConfidence(0.75 + 0.1 * (problem.is_dense ? 1.0 : 0.0));
    matrix.reasoning = "dense-friendly blocked reduction model";
    blendPredictionWithCalibration(matrix, problem, fingerprint);
    sanitizePrediction(matrix);
    predictions.push_back(matrix);

    Prediction sparse;
    sparse.algorithm_type = AdaptiveAlgorithmType::SPARSIFIED_REDUCTION_CPU;
    sparse.estimated_time_ms = estimateSparsificationTime(problem);
    sparse.estimated_memory_mb = estimateMemoryUsage(problem, sparse.algorithm_type);
    sparse.confidence_score = boundedConfidence(0.72 + 0.12 * (problem.is_sparse ? 1.0 : 0.0));
    sparse.reasoning = "sparsity-aware reduction model";
    blendPredictionWithCalibration(sparse, problem, fingerprint);
    sanitizePrediction(sparse);
    predictions.push_back(sparse);

    Prediction lockfree;
    lockfree.algorithm_type = AdaptiveAlgorithmType::LOCKFREE_MULTICORE_CPU;
    lockfree.estimated_time_ms = estimate_lockfree_multicore_time(problem, system);
    lockfree.estimated_memory_mb = estimateMemoryUsage(problem, lockfree.algorithm_type);
    lockfree.confidence_score =
        boundedConfidence(0.7 + 0.2 * (system.num_cpu_cores >= 8 ? 1.0 : 0.0));
    lockfree.reasoning = "multicore scaling from runtime-probed CPU topology";
    blendPredictionWithCalibration(lockfree, problem, fingerprint);
    sanitizePrediction(lockfree);
    predictions.push_back(lockfree);

    Prediction standard;
    standard.algorithm_type = AdaptiveAlgorithmType::STANDARD_CPU;
    standard.estimated_time_ms = estimateStandardCpuTime(problem);
    standard.estimated_memory_mb = estimateMemoryUsage(problem, standard.algorithm_type);
    standard.confidence_score = 0.92;
    standard.reasoning = "exact baseline execution";
    blendPredictionWithCalibration(standard, problem, fingerprint);
    sanitizePrediction(standard);
    predictions.push_back(standard);

    (void)baseline;
    (void)memory_scale;

    return predictions;
}
PerformancePredictor::Prediction
PerformancePredictor::getOptimalAlgorithm(const ProblemCharacteristics &problem,
                                          const SystemCapabilities &system,
                                          const PerformanceRequirements &requirements)
{
    auto predictions = predictPerformance(problem, system);
    predictions.erase(
        std::remove_if(predictions.begin(), predictions.end(),
                       [&requirements](const Prediction &prediction) {
                           return prediction.estimated_time_ms > requirements.max_time_ms ||
                                  prediction.estimated_memory_mb > requirements.max_memory_mb;
                       }),
        predictions.end());
    if (predictions.empty())
    {
        predictions = predictPerformance(problem, system);
    }
    return *std::min_element(
        predictions.begin(), predictions.end(), [](const Prediction &lhs, const Prediction &rhs) {
            return predictionSelectionScore(lhs) < predictionSelectionScore(rhs);
        });
}
double PerformancePredictor::estimateMatrixMultiplicationTime(const ProblemCharacteristics &problem)
{
    return powerTimeEstimate(problem.estimated_columns, 2.65L, 1e-5L);
}
double PerformancePredictor::estimateSparsificationTime(const ProblemCharacteristics &problem)
{
    const long double sparsity = static_cast<long double>(
        std::clamp(finiteUnitRatio(problem.sparsity_ratio, 1.0), 0.01, 1.0));
    return finiteNonnegativePrediction(
        static_cast<long double>(powerTimeEstimate(problem.estimated_columns, 2.25L, 1e-5L)) *
        sparsity);
}
double PerformancePredictor::estimate_lockfree_multicore_time(const ProblemCharacteristics &problem,
                                                              const SystemCapabilities &system)
{
    const long double cores =
        static_cast<long double>(std::max<std::size_t>(1, system.num_cpu_cores));
    const long double parallel_efficiency = 1.0L - (1.0L / (cores + 1.0L));
    const long double denominator = std::max(0.2L, parallel_efficiency * cores);
    return finiteNonnegativePrediction(
        static_cast<long double>(powerTimeEstimate(problem.estimated_columns, 2.3L, 1e-5L)) /
        denominator);
}
double PerformancePredictor::estimateGpuTime(const ProblemCharacteristics &problem,
                                             const SystemCapabilities &system)
{
    const long double capability_scale =
        std::max(1.0L, static_cast<long double>(system.compute_capability) / 20.0L);
    const long double memory_bandwidth =
        static_cast<long double>(finitePositiveInput(system.memory_bandwidth, 1.0));
    const long double bandwidth_scale = std::max(1.0L, std::sqrt(memory_bandwidth));
    return finiteNonnegativePrediction(
        static_cast<long double>(powerTimeEstimate(problem.estimated_columns, 2.15L, 1e-5L)) /
        (capability_scale * bandwidth_scale));
}
double PerformancePredictor::estimateStandardCpuTime(const ProblemCharacteristics &problem)
{
    return powerTimeEstimate(problem.estimated_columns, 2.75L, 1e-5L);
}
double PerformancePredictor::estimateMemoryUsage(const ProblemCharacteristics &problem,
                                                 AdaptiveAlgorithmType algorithm)
{
    const double base = finiteAtLeast(problem.memory_requirement_mb, 1.0, kMaxFinitePrediction);
    switch (algorithm)
    {
        case AdaptiveAlgorithmType::MATRIX_MULTIPLICATION_CPU:
            return finiteNonnegativePrediction(static_cast<long double>(base) * 1.40L);
        case AdaptiveAlgorithmType::SPARSIFIED_REDUCTION_CPU:
            return finiteNonnegativePrediction(
                static_cast<long double>(base) *
                static_cast<long double>(
                    std::clamp(finiteUnitRatio(problem.sparsity_ratio, 0.8) + 0.2, 0.2, 1.0)));
        case AdaptiveAlgorithmType::LOCKFREE_MULTICORE_CPU:
            return finiteNonnegativePrediction(static_cast<long double>(base) * 1.15L);
        case AdaptiveAlgorithmType::GPU_ACCELERATED:
            return finiteNonnegativePrediction(static_cast<long double>(base) * 1.30L);
        case AdaptiveAlgorithmType::HYBRID_GPU_CPU:
            return finiteNonnegativePrediction(static_cast<long double>(base) * 1.20L);
        case AdaptiveAlgorithmType::STANDARD_CPU:
        default:
            return base;
    }
}

class AdaptiveAlgorithmSelector::Impl
{
public:
    explicit Impl(const AdaptiveConfig &config)
        : config_(config)
        , performance_stats_()
    {}

    AdaptiveAlgorithmType selectOptimalAlgorithm(const ProblemCharacteristics &problem,
                                                 const SystemCapabilities &system,
                                                 const PerformanceRequirements &requirements)
    {
        auto predictions = PerformancePredictor::predictPerformance(problem, system);
        performance_stats_.all_predictions = predictions;

        std::vector<PerformancePredictor::Prediction> admissible;
        for (const auto &prediction : predictions)
        {
            if (prediction.estimated_time_ms <= requirements.max_time_ms &&
                prediction.estimated_memory_mb <= requirements.max_memory_mb)
            {
                admissible.push_back(prediction);
            }
        }
        if (admissible.empty())
        {
            admissible = predictions;
        }

        const auto best = std::min_element(
            admissible.begin(), admissible.end(), [](const auto &lhs, const auto &rhs) {
                return predictionSelectionScore(lhs) < predictionSelectionScore(rhs);
            });

        performance_stats_.algorithm_used = best->algorithm_type;
        performance_stats_.selection_reasoning = best->reasoning;
        return best->algorithm_type;
    }

    errors::ErrorResult<std::vector<Pair>> executeAdaptive(AdaptiveAlgorithmSelector &owner,
                                                           core::BufferView<const double> points,
                                                           std::size_t point_dim,
                                                           const VRConfig &config)
    {
        auto validation = validateExecutionInput(points, point_dim, config);
        if (validation.isError())
        {
            return errors::ErrorResult<std::vector<Pair>>::error(validation.errorCode());
        }
        std::size_t input_bytes = 0;
        if (!checkedProduct(points.size(), sizeof(double), input_bytes))
        {
            return errors::ErrorResult<std::vector<Pair>>::error(
                errors::ErrorCode::E41_RESOURCE_LIMIT);
        }

        const auto start = std::chrono::high_resolution_clock::now();

        const ProblemCharacteristics problem = ProblemAnalyzer::analyzeProblem(points, point_dim);
        const SystemCapabilities system = SystemCapabilities::detectCapabilities();

        PerformanceRequirements requirements;
        requirements.max_time_ms = config.performance_target_ms;
        requirements.max_memory_mb = config.memory_limit_mb;
        requirements.min_precision = config.min_precision;
        requirements.prefer_gpu = false;
        requirements.prefer_multicore = config.prefer_multicore;

        const AdaptiveAlgorithmType selected =
            selectOptimalAlgorithm(problem, system, requirements);

        errors::ErrorResult<std::vector<Pair>> result =
            errors::ErrorResult<std::vector<Pair>>::error(errors::ErrorCode::UNKNOWN);
        switch (selected)
        {
            case AdaptiveAlgorithmType::MATRIX_MULTIPLICATION_CPU:
                result = owner.executeMatrixMultiplicationCpu(points, point_dim, config);
                break;
            case AdaptiveAlgorithmType::SPARSIFIED_REDUCTION_CPU:
                result = owner.executeSparsifiedReductionCpu(points, point_dim, config);
                break;
            case AdaptiveAlgorithmType::LOCKFREE_MULTICORE_CPU:
                result = owner.execute_lockfree_multicore_cpu(points, point_dim, config);
                break;
            case AdaptiveAlgorithmType::GPU_ACCELERATED:
                result = owner.executeGpuAccelerated(points, point_dim, config);
                break;
            case AdaptiveAlgorithmType::HYBRID_GPU_CPU:
                result = owner.executeHybridGpuCpu(points, point_dim, config);
                break;
            case AdaptiveAlgorithmType::STANDARD_CPU:
            default:
                result = owner.executeStandardCpu(points, point_dim, config);
                break;
        }

        const auto end = std::chrono::high_resolution_clock::now();
        const double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        performance_stats_.computation_time_ms = elapsed_ms;
        performance_stats_.memory_used_mb = static_cast<double>(input_bytes) / (1024.0 * 1024.0);

        const auto selected_prediction = std::find_if(
            performance_stats_.all_predictions.begin(), performance_stats_.all_predictions.end(),
            [selected](const auto &prediction) { return prediction.algorithm_type == selected; });
        if (selected_prediction != performance_stats_.all_predictions.end() && elapsed_ms > 0.0)
        {
            const double ratio = finiteNonnegativePrediction(
                static_cast<long double>(selected_prediction->estimated_time_ms) /
                static_cast<long double>(elapsed_ms));
            performance_stats_.prediction_accuracy =
                std::isfinite(ratio) ? std::clamp(ratio, 0.0, 10.0) : 0.0;
        }

        recordAdaptiveCalibrationObservation(problem, selected,
                                             selected_prediction !=
                                                     performance_stats_.all_predictions.end()
                                                 ? &(*selected_prediction)
                                                 : nullptr,
                                             elapsed_ms, points.size());

        return result;
    }

    const AdaptiveStats &getPerformanceStats() const { return performance_stats_; }

private:
    AdaptiveConfig config_;
    AdaptiveStats performance_stats_;
};

errors::ErrorResult<std::unique_ptr<AdaptiveAlgorithmSelector>>
AdaptiveAlgorithmSelector::create(const AdaptiveConfig &config)
{
    auto selector =
        std::unique_ptr<AdaptiveAlgorithmSelector>(new AdaptiveAlgorithmSelector(config));
    return errors::ErrorResult<std::unique_ptr<AdaptiveAlgorithmSelector>>::success(
        std::move(selector));
}

AdaptiveAlgorithmSelector::AdaptiveAlgorithmSelector(const AdaptiveConfig &config)
    : impl_(std::make_unique<Impl>(config))
    , config_(config)
    , performance_stats_()
{}

AdaptiveAlgorithmSelector::~AdaptiveAlgorithmSelector() = default;

AdaptiveAlgorithmType
AdaptiveAlgorithmSelector::selectOptimalAlgorithm(const ProblemCharacteristics &problem,
                                                  const SystemCapabilities &system,
                                                  const PerformanceRequirements &requirements)
{
    return impl_->selectOptimalAlgorithm(problem, system, requirements);
}

errors::ErrorResult<std::vector<Pair>>
AdaptiveAlgorithmSelector::executeAdaptive(core::BufferView<const double> points,
                                           std::size_t point_dim, const VRConfig &config)
{
    return impl_->executeAdaptive(*this, points, point_dim, config);
}

const AdaptiveStats &AdaptiveAlgorithmSelector::getPerformanceStats() const
{
    return impl_->getPerformanceStats();
}

} // namespace nerve::persistence::adaptive_acceleration

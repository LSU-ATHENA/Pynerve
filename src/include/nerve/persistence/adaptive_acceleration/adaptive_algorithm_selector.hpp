
#pragma once

#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core_types.hpp"
#include "nerve/errors/errors.hpp"
#include "nerve/persistence/adaptive_acceleration/adaptive_acceleration_problem_analysis.hpp"
#include "nerve/persistence/adaptive_acceleration/adaptive_acceleration_system_capabilities.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"

#include <memory>
#include <string>
#include <vector>

namespace nerve::persistence::adaptive_acceleration
{

enum class AdaptiveAlgorithmType
{
    MATRIX_MULTIPLICATION_CPU,
    SPARSIFIED_REDUCTION_CPU,
    LOCKFREE_MULTICORE_CPU,
    GPU_ACCELERATED,
    HYBRID_GPU_CPU,
    STANDARD_CPU
};

struct AdaptiveConfig
{
    bool enable_performance_prediction = true;
    bool enable_hybrid_optimization = false;
    bool enable_streaming = false;
    bool enable_approximation = false;
    double performance_target_ms = 100.0;
    double memory_limit_mb = 8192.0;
    double min_precision = 1e-10;
};

struct PerformanceRequirements
{
    double max_time_ms = 1000.0;
    double max_memory_mb = 8192.0;
    double min_precision = 1e-10;
    bool require_representatives = false;
    bool require_streaming = false;
    bool require_approximation = false;
    bool prefer_gpu = false;
    bool prefer_multicore = false;
};

class PerformancePredictor
{
public:
    struct Prediction
    {
        AdaptiveAlgorithmType algorithm_type = AdaptiveAlgorithmType::STANDARD_CPU;
        double estimated_time_ms = 0.0;
        double estimated_memory_mb = 0.0;
        double confidence_score = 0.0;
        std::string reasoning;
    };

    static std::vector<Prediction> predictPerformance(const ProblemCharacteristics &problem,
                                                      const SystemCapabilities &system);

    static Prediction getOptimalAlgorithm(const ProblemCharacteristics &problem,
                                          const SystemCapabilities &system,
                                          const PerformanceRequirements &requirements);

private:
    static double estimateMatrixMultiplicationTime(const ProblemCharacteristics &problem);
    static double estimateSparsificationTime(const ProblemCharacteristics &problem);
    static double estimate_lockfree_multicore_time(const ProblemCharacteristics &problem,
                                                   const SystemCapabilities &system);
    static double estimateGpuTime(const ProblemCharacteristics &problem,
                                  const SystemCapabilities &system);
    static double estimateStandardCpuTime(const ProblemCharacteristics &problem);
    static double estimateMemoryUsage(const ProblemCharacteristics &problem,
                                      AdaptiveAlgorithmType algorithm);
};

struct AdaptiveStats
{
    double computation_time_ms = 0.0;
    double memory_used_mb = 0.0;
    AdaptiveAlgorithmType algorithm_used = AdaptiveAlgorithmType::STANDARD_CPU;
    double prediction_accuracy = 0.0;
    double scaling_efficiency = 1.0;
    std::string selection_reasoning;
    std::vector<PerformancePredictor::Prediction> all_predictions;
};

class AdaptiveAlgorithmSelector
{
public:
    static errors::ErrorResult<std::unique_ptr<AdaptiveAlgorithmSelector>>
    create(const AdaptiveConfig &config);

    ~AdaptiveAlgorithmSelector();

    AdaptiveAlgorithmType selectOptimalAlgorithm(const ProblemCharacteristics &problem,
                                                 const SystemCapabilities &system,
                                                 const PerformanceRequirements &requirements);

    errors::ErrorResult<std::vector<Pair>>
    executeAdaptive(const core::BufferView<const double> &points, std::size_t point_dim,
                    const VRConfig &config);

    const AdaptiveStats &getPerformanceStats() const;

private:
    explicit AdaptiveAlgorithmSelector(const AdaptiveConfig &config);

    errors::ErrorResult<std::vector<Pair>>
    executeMatrixMultiplicationCpu(const core::BufferView<const double> &points,
                                   std::size_t point_dim, const VRConfig &config);

    errors::ErrorResult<std::vector<Pair>>
    executeSparsifiedReductionCpu(const core::BufferView<const double> &points,
                                  std::size_t point_dim, const VRConfig &config);

    errors::ErrorResult<std::vector<Pair>>
    execute_lockfree_multicore_cpu(const core::BufferView<const double> &points,
                                   std::size_t point_dim, const VRConfig &config);

    errors::ErrorResult<std::vector<Pair>>
    executeGpuAccelerated(const core::BufferView<const double> &points, std::size_t point_dim,
                          const VRConfig &config);

    errors::ErrorResult<std::vector<Pair>>
    executeHybridGpuCpu(const core::BufferView<const double> &points, std::size_t point_dim,
                        const VRConfig &config);

    errors::ErrorResult<std::vector<Pair>>
    executeStandardCpu(const core::BufferView<const double> &points, std::size_t point_dim,
                       const VRConfig &config);

    class Impl;
    std::unique_ptr<Impl> impl_;
    AdaptiveConfig config_;
    AdaptiveStats performance_stats_;
};

} // namespace nerve::persistence::adaptive_acceleration

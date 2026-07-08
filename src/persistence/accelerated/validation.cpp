
#include "nerve/persistence/accelerated/accelerated_api.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <sstream>
#include <string>
#include <vector>

namespace nerve::persistence::accelerated
{

struct ValidationResult
{
    bool isValid = true;
    std::vector<std::string> errors;
    std::vector<std::string> warnings;

    void addError(std::string message)
    {
        isValid = false;
        errors.push_back(std::move(message));
    }

    void addWarning(std::string message) { warnings.push_back(std::move(message)); }

    std::string getSummary() const
    {
        std::ostringstream oss;
        oss << (isValid ? "valid" : "invalid");
        if (!errors.empty())
        {
            oss << " errors=" << errors.size();
        }
        if (!warnings.empty())
        {
            oss << " warnings=" << warnings.size();
        }
        return oss.str();
    }
};

namespace validation
{

ValidationResult validateMathematicalCorrectness(const std::vector<Pair> &diagram,
                                                 const core::DeterminismContract &contract)
{
    ValidationResult result;
    for (const auto &pair : diagram)
    {
        if (!pair.isInfinite() && pair.death < pair.birth)
        {
            result.addError("death before birth");
            break;
        }
        if (contract.level == core::DeterminismLevel::STRICT &&
            (std::isnan(pair.birth) || std::isnan(pair.death) || std::isinf(pair.birth) ||
             std::isinf(pair.death)))
        {
            result.addError("non-finite value in strict mode");
            break;
        }
    }
    return result;
}

ValidationResult validateBettiNumbers(const std::vector<Pair> &diagram,
                                      const std::vector<int> &expected_betti)
{
    ValidationResult result;
    std::vector<int> observed(std::max<size_t>(1, expected_betti.size()), 0);
    for (const auto &pair : diagram)
    {
        if (pair.isInfinite() && pair.dimension >= 0 &&
            static_cast<size_t>(pair.dimension) < observed.size())
        {
            observed[static_cast<size_t>(pair.dimension)] += 1;
        }
    }
    if (!expected_betti.empty() && observed != expected_betti)
    {
        result.addWarning("betti profile mismatch");
    }
    return result;
}

ValidationResult validateTopologicalInvariants(const std::vector<Pair> &diagram)
{
    ValidationResult result;
    for (const auto &pair : diagram)
    {
        if (pair.dimension < 0)
        {
            result.addError("negative homology dimension");
            break;
        }
    }
    return result;
}

ValidationResult validateComputationDeterminism(const std::vector<Pair> &lhs,
                                                const std::vector<Pair> &rhs,
                                                const core::DeterminismContract &contract,
                                                double tolerance = 1e-12)
{
    ValidationResult result;
    if (contract.level != core::DeterminismLevel::STRICT)
    {
        return result;
    }
    if (lhs.size() != rhs.size())
    {
        result.addError("diagram size mismatch");
        return result;
    }
    for (size_t i = 0; i < lhs.size(); ++i)
    {
        const auto &a = lhs[i];
        const auto &b = rhs[i];
        if (a.dimension != b.dimension || std::abs(a.birth - b.birth) > tolerance ||
            std::abs(a.death - b.death) > tolerance)
        {
            result.addError("diagram mismatch under strict determinism");
            break;
        }
    }
    return result;
}

ValidationResult validateGpuDeterminism()
{
    ValidationResult result;
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess)
    {
        result.addWarning("cuda device probe failed");
        return result;
    }
    if (device_count == 0)
    {
        result.addWarning("no cuda devices available");
    }
    return result;
}

ValidationResult validateInputDeterminism(core::BufferView<const double>points,
                                          const core::DeterminismContract &contract)
{
    ValidationResult result;
    if (contract.level == core::DeterminismLevel::STRICT)
    {
        for (size_t i = 0; i < points.size(); ++i)
        {
            if (!std::isfinite(points[i]))
            {
                result.addError("non-finite input value");
                break;
            }
        }
    }
    return result;
}

ValidationResult validatePerformanceMetrics(const AcceleratedPerformanceStats &stats)
{
    ValidationResult result;
    const bool finite_metrics =
        std::isfinite(stats.total_time_ms) && std::isfinite(stats.gpu_time_ms) &&
        std::isfinite(stats.cpu_time_ms) && std::isfinite(stats.memory_usage_mb) &&
        std::isfinite(stats.gpu_utilization) && std::isfinite(stats.speedup) &&
        std::isfinite(stats.average_speedup) && std::isfinite(stats.peak_memory_usage_mb) &&
        std::isfinite(stats.gpu_stage_ops);
    if (!finite_metrics)
    {
        result.addError("non-finite performance metric");
        return result;
    }
    for (const auto &metric : stats.detailed_metrics)
    {
        const bool finite_detail =
            std::isfinite(metric.total_time_ms) && std::isfinite(metric.gpu_time_ms) &&
            std::isfinite(metric.cpu_time_ms) && std::isfinite(metric.max_radius) &&
            std::isfinite(metric.gpu_work_ratio) && std::isfinite(metric.problem_ops) &&
            std::isfinite(metric.gpu_bytes) && std::isfinite(metric.gpu_stage_ops);
        if (!finite_detail)
        {
            result.addError("non-finite detailed performance metric");
            return result;
        }
        if (metric.total_time_ms < 0.0 || metric.gpu_time_ms < 0.0 || metric.cpu_time_ms < 0.0 ||
            metric.max_radius < 0.0 || metric.gpu_work_ratio < 0.0 || metric.problem_ops < 0.0 ||
            metric.gpu_bytes < 0.0 || metric.gpu_stage_ops < 0.0)
        {
            result.addError("negative detailed runtime metric");
        }
    }
    if (stats.total_time_ms < 0.0 || stats.gpu_time_ms < 0.0 || stats.cpu_time_ms < 0.0 ||
        stats.memory_usage_mb < 0.0 || stats.gpu_utilization < 0.0 || stats.speedup < 0.0 ||
        stats.average_speedup < 0.0 || stats.peak_memory_usage_mb < 0.0 ||
        stats.gpu_stage_ops < 0.0)
    {
        result.addError("negative runtime metric");
    }
    if (stats.gpu_time_ms + stats.cpu_time_ms > stats.total_time_ms * 1.05)
    {
        result.addWarning("component times exceed total runtime");
    }
    return result;
}

ValidationResult validatePerformanceRegression(const AcceleratedPerformanceStats &current,
                                               const AcceleratedPerformanceStats &baseline,
                                               double regression_threshold = 0.05)
{
    ValidationResult result;
    if (baseline.total_time_ms <= 0.0)
    {
        return result;
    }
    const double delta = (current.total_time_ms - baseline.total_time_ms) / baseline.total_time_ms;
    if (delta > regression_threshold)
    {
        result.addWarning("runtime regression above threshold");
    }
    return result;
}

ValidationResult validateApiIntegration(const core::BufferView<const double> &points,
                                        size_t point_dim, const VRConfig &config)
{
    ValidationResult result;
    if (point_dim == 0 || points.size() == 0 || points.size() % point_dim != 0)
    {
        result.addError("invalid point buffer shape");
    }
    for (size_t i = 0; i < points.size(); ++i)
    {
        if (!std::isfinite(points[i]))
        {
            result.addError("point coordinates must be finite");
            break;
        }
    }
    if (config.max_dim > 8)
    {
        result.addWarning("high max_dim may increase runtime significantly");
    }
    return result;
}

ValidationResult validateAcceleratedResult(const std::vector<Pair> &accelerated_result,
                                           const std::vector<Pair> &cpu_result,
                                           double tolerance = 1e-12)
{
    ValidationResult result;
    if (accelerated_result.size() != cpu_result.size())
    {
        result.addError("pair-count mismatch");
        return result;
    }
    for (size_t i = 0; i < accelerated_result.size(); ++i)
    {
        if (accelerated_result[i].dimension != cpu_result[i].dimension ||
            std::abs(accelerated_result[i].birth - cpu_result[i].birth) > tolerance ||
            std::abs(accelerated_result[i].death - cpu_result[i].death) > tolerance)
        {
            result.addWarning("pair mismatch within tolerance gate");
            break;
        }
    }
    return result;
}

ValidationResult validateComprehensive(const std::vector<Pair> &diagram,
                                       core::BufferView<const double>points,
                                       size_t point_dim, const VRConfig &config,
                                       const core::DeterminismContract &contract,
                                       const AcceleratedPerformanceStats &stats)
{
    ValidationResult result = validateMathematicalCorrectness(diagram, contract);
    auto input_result = validateApiIntegration(points, point_dim, config);
    auto perf_result = validatePerformanceMetrics(stats);

    if (!input_result.isValid)
    {
        result.isValid = false;
        result.errors.insert(result.errors.end(), input_result.errors.begin(),
                             input_result.errors.end());
    }
    if (!perf_result.isValid)
    {
        result.isValid = false;
        result.errors.insert(result.errors.end(), perf_result.errors.begin(),
                             perf_result.errors.end());
    }
    result.warnings.insert(result.warnings.end(), input_result.warnings.begin(),
                           input_result.warnings.end());
    result.warnings.insert(result.warnings.end(), perf_result.warnings.begin(),
                           perf_result.warnings.end());
    return result;
}

} // namespace validation
} // namespace nerve::persistence::accelerated

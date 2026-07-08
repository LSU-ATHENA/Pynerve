
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core/rng/determinism_contract.hpp"
#include "nerve/core_types.hpp"
#include "nerve/persistence/accelerated/work_distributor.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace nerve::persistence::accelerated
{

struct DeterminismValidationResult
{
    bool is_deterministic = true;
    std::vector<std::string> warnings;
    std::vector<std::string> errors;
    std::unordered_map<std::string, double> metrics;

    void addWarning(const std::string &message) { warnings.push_back(message); }

    void addError(const std::string &message)
    {
        errors.push_back(message);
        is_deterministic = false;
    }

    void addMetric(const std::string &key, double value) { metrics[key] = value; }

    std::string getSummary() const
    {
        std::ostringstream stream;
        stream << "deterministic=" << (is_deterministic ? "true" : "false");
        stream << " warnings=" << warnings.size();
        stream << " errors=" << errors.size();
        return stream.str();
    }
};

namespace determinism_validation
{

void mergeValidationResults(DeterminismValidationResult &target,
                            const DeterminismValidationResult &source)
{
    target.is_deterministic = target.is_deterministic && source.is_deterministic;
    target.warnings.insert(target.warnings.end(), source.warnings.begin(), source.warnings.end());
    target.errors.insert(target.errors.end(), source.errors.begin(), source.errors.end());
    for (const auto &[name, value] : source.metrics)
    {
        target.metrics[name] = value;
    }
}

DeterminismValidationResult validateInputDeterminism(core::BufferView<const double>points,
                                                     Size point_dim,
                                                     const core::DeterminismContract &contract)
{
    DeterminismValidationResult result;
    result.addMetric("point_count",
                     point_dim == 0 ? 0.0 : static_cast<double>(points.size() / point_dim));
    result.addMetric("point_dim", static_cast<double>(point_dim));

    if (points.size() == 0 || point_dim == 0 || points.size() % point_dim != 0)
    {
        result.addError("invalid input shape");
        return result;
    }

    if (contract.level >= core::DeterminismLevel::STRICT)
    {
        for (Size index = 0; index < points.size(); ++index)
        {
            if (!std::isfinite(points[index]))
            {
                result.addError("non-finite value at index " + std::to_string(index));
            }
        }
    }
    return result;
}

DeterminismValidationResult validateOutputConsistency(const std::vector<Pair> &diagram,
                                                      const std::vector<Pair> &reference_diagram,
                                                      double tolerance = 1e-10)
{
    DeterminismValidationResult result;
    result.addMetric("diagram_size", static_cast<double>(diagram.size()));
    result.addMetric("reference_size", static_cast<double>(reference_diagram.size()));

    if (diagram.size() != reference_diagram.size())
    {
        result.addError("diagram size mismatch");
        return result;
    }

    for (Size index = 0; index < diagram.size(); ++index)
    {
        const auto &lhs = diagram[index];
        const auto &rhs = reference_diagram[index];
        if (lhs.dimension != rhs.dimension || std::abs(lhs.birth - rhs.birth) > tolerance ||
            std::abs(lhs.death - rhs.death) > tolerance)
        {
            result.addError("diagram entry mismatch at index " + std::to_string(index));
            break;
        }
    }
    return result;
}

DeterminismValidationResult validateMathematicalProperties(const std::vector<Pair> &diagram)
{
    DeterminismValidationResult result;
    Size finite_pairs = 0;
    for (const auto &pair : diagram)
    {
        if (pair.death < pair.birth)
        {
            result.addError("death < birth for pair");
        }
        if (!pair.isInfinite())
        {
            ++finite_pairs;
        }
    }
    result.addMetric("finite_pairs", static_cast<double>(finite_pairs));
    result.addMetric("infinite_pairs", static_cast<double>(diagram.size() - finite_pairs));
    return result;
}

DeterminismValidationResult validateGpuDeterminism()
{
    DeterminismValidationResult result;
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess)
    {
        result.addWarning("cuda probe failed");
    }
    result.addMetric("cuda_device_count", static_cast<double>(device_count));
    return result;
}

DeterminismValidationResult validateGpuComputationConsistency(const std::vector<double> &input_data,
                                                              const std::vector<double> &gpu_result,
                                                              const std::vector<double> &cpu_result,
                                                              double tolerance = 1e-10)
{
    DeterminismValidationResult result;
    result.addMetric("input_size", static_cast<double>(input_data.size()));
    if (gpu_result.size() != cpu_result.size())
    {
        result.addError("gpu/cpu result size mismatch");
        return result;
    }
    for (Size index = 0; index < gpu_result.size(); ++index)
    {
        if (std::abs(gpu_result[index] - cpu_result[index]) > tolerance)
        {
            result.addError("gpu/cpu mismatch at index " + std::to_string(index));
            break;
        }
    }
    return result;
}

DeterminismValidationResult
validateWorkDistributionDeterminism(const WorkDistribution &distribution, Size total_columns,
                                    Size n_points, Size point_dim)
{
    DeterminismValidationResult result;
    result.addMetric("total_columns", static_cast<double>(total_columns));
    result.addMetric("n_points", static_cast<double>(n_points));
    result.addMetric("point_dim", static_cast<double>(point_dim));
    result.addMetric("gpuColumns", static_cast<double>(distribution.gpuColumns));
    result.addMetric("cpuColumns", static_cast<double>(distribution.cpuColumns));
    result.addMetric("confidence_score", distribution.confidence_score);

    if (distribution.gpuColumns + distribution.cpuColumns != total_columns)
    {
        result.addError("distribution columns do not sum to total");
    }
    if (distribution.confidence_score < 0.0 || distribution.confidence_score > 1.0)
    {
        result.addError("distribution confidence out of range");
    }
    return result;
}

DeterminismValidationResult
validateWorkDistributionConsistency(const std::vector<WorkDistribution> &distributions)
{
    DeterminismValidationResult result;
    if (distributions.empty())
    {
        result.addWarning("no distributions provided");
        return result;
    }
    double confidence_sum = 0.0;
    for (const auto &distribution : distributions)
    {
        confidence_sum += distribution.confidence_score;
        if (distribution.confidence_score < 0.0 || distribution.confidence_score > 1.0)
        {
            result.addError("confidence out of range in sequence");
        }
    }
    result.addMetric("average_confidence",
                     confidence_sum / static_cast<double>(distributions.size()));
    return result;
}

DeterminismValidationResult comprehensiveValidation(core::BufferView<const double>points,
                                                    Size point_dim,
                                                    const core::DeterminismContract &contract,
                                                    const std::vector<Pair> &diagram,
                                                    const std::vector<Pair> &reference_diagram = {})
{
    DeterminismValidationResult result;
    mergeValidationResults(result, validateInputDeterminism(points, point_dim, contract));
    mergeValidationResults(result, validateMathematicalProperties(diagram));
    if (!reference_diagram.empty())
    {
        mergeValidationResults(result, validateOutputConsistency(diagram, reference_diagram));
    }
    return result;
}

} // namespace determinism_validation

} // namespace nerve::persistence::accelerated

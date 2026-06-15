
#include "nerve/core/detail/error_event_extensions.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

namespace nerve::core
{

namespace
{

void appendUint64(std::vector<uint8_t> &buffer, uint64_t value)
{
    for (unsigned shift = 0; shift < 64; shift += 8)
    {
        buffer.push_back(static_cast<uint8_t>(value >> shift));
    }
}

[[nodiscard]] bool readUint64(const std::vector<uint8_t> &buffer, size_t *pos, uint64_t *value)
{
    if (*pos + sizeof(uint64_t) > buffer.size())
    {
        return false;
    }
    uint64_t decoded = 0;
    for (unsigned shift = 0; shift < 64; shift += 8)
    {
        decoded |= static_cast<uint64_t>(buffer[(*pos)++]) << shift;
    }
    *value = decoded;
    return true;
}

[[nodiscard]] uint64_t roundedNonNegative(double value)
{
    if (!std::isfinite(value) || value <= 0.0)
    {
        return 0;
    }
    if (value >= static_cast<double>(std::numeric_limits<uint64_t>::max()))
    {
        return std::numeric_limits<uint64_t>::max();
    }
    return static_cast<uint64_t>(std::llround(value));
}

} // namespace

HighDimErrorEvent::HighDimErrorEvent()
    : dimension_(0)
    , complex_size_(0)
    , memory_usage_mb_(0)
{}

HighDimErrorEvent::HighDimErrorEvent(ErrorCode code, const std::string &message)
    : ErrorEvent(code, message)
    , dimension_(0)
    , complex_size_(0)
    , memory_usage_mb_(0)
{}

void HighDimErrorEvent::setDimension(size_t dimension)
{
    dimension_ = dimension;
}

size_t HighDimErrorEvent::getDimension() const
{
    return dimension_;
}

void HighDimErrorEvent::setComplexSize(size_t complex_size)
{
    complex_size_ = complex_size;
}

size_t HighDimErrorEvent::getComplexSize() const
{
    return complex_size_;
}

void HighDimErrorEvent::setMemoryUsage(size_t memory_usage_mb)
{
    memory_usage_mb_ = memory_usage_mb;
}

size_t HighDimErrorEvent::getMemoryUsage() const
{
    return memory_usage_mb_;
}

void HighDimErrorEvent::setAlgorithmPhase(const std::string &phase)
{
    algorithm_phase_ = phase;
}

const std::string &HighDimErrorEvent::getAlgorithmPhase() const
{
    return algorithm_phase_;
}

void HighDimErrorEvent::setComputationStage(const std::string &stage)
{
    computation_stage_ = stage;
}

const std::string &HighDimErrorEvent::getComputationStage() const
{
    return computation_stage_;
}

void HighDimErrorEvent::setPerformanceMetrics(const PerformanceMetrics &metrics)
{
    performance_metrics_ = metrics;
}

const PerformanceMetrics &HighDimErrorEvent::getPerformanceMetrics() const
{
    return performance_metrics_;
}

void HighDimErrorEvent::setResourceUtilization(const ResourceUtilization &utilization)
{
    resource_utilization_ = utilization;
}

const ResourceUtilization &HighDimErrorEvent::getResourceUtilization() const
{
    return resource_utilization_;
}

void HighDimErrorEvent::setAlgorithmContext(const AlgorithmContext &context)
{
    algorithm_context_ = context;
}

const AlgorithmContext &HighDimErrorEvent::getAlgorithmContext() const
{
    return algorithm_context_;
}

void HighDimErrorEvent::setFailureAnalysis(const FailureAnalysis &analysis)
{
    failure_analysis_ = analysis;
}

const FailureAnalysis &HighDimErrorEvent::getFailureAnalysis() const
{
    return failure_analysis_;
}

void HighDimErrorEvent::setRecoverySuggestions(const std::vector<std::string> &suggestions)
{
    recovery_suggestions_ = suggestions;
}

const std::vector<std::string> &HighDimErrorEvent::getRecoverySuggestions() const
{
    return recovery_suggestions_;
}

void HighDimErrorEvent::setDebugInfo(const DebugInfo &info)
{
    debug_info_ = info;
}

const DebugInfo &HighDimErrorEvent::getDebugInfo() const
{
    return debug_info_;
}

void HighDimErrorEvent::serializeHighdimData(std::vector<uint8_t> &buffer) const
{
    appendUint64(buffer, static_cast<uint64_t>(dimension_));
    appendUint64(buffer, static_cast<uint64_t>(complex_size_));
    appendUint64(buffer, static_cast<uint64_t>(memory_usage_mb_));
    appendUint64(buffer, roundedNonNegative(performance_metrics_.computation_time_ms));
    appendUint64(buffer, static_cast<uint64_t>(resource_utilization_.memory_allocated_mb));
}

void HighDimErrorEvent::deserializeHighdimData(const std::vector<uint8_t> &buffer)
{
    size_t pos = 0;
    if (buffer.size() >= 5 * sizeof(uint64_t))
    {
        uint64_t decoded = 0;
        if (readUint64(buffer, &pos, &decoded))
        {
            dimension_ = static_cast<size_t>(decoded);
        }
        if (readUint64(buffer, &pos, &decoded))
        {
            complex_size_ = static_cast<size_t>(decoded);
        }
        if (readUint64(buffer, &pos, &decoded))
        {
            memory_usage_mb_ = static_cast<size_t>(decoded);
        }
        if (readUint64(buffer, &pos, &decoded))
        {
            performance_metrics_.computation_time_ms = static_cast<double>(decoded);
        }
        if (readUint64(buffer, &pos, &decoded))
        {
            resource_utilization_.memory_allocated_mb = static_cast<size_t>(decoded);
        }
        return;
    }
    if (pos < buffer.size())
    {
        dimension_ = buffer[pos++];
    }
    if (pos + 1 < buffer.size())
    {
        const size_t low = buffer[pos++];
        const size_t high = buffer[pos++];
        complex_size_ = low + (high * 256);
    }
    if (pos + 1 < buffer.size())
    {
        const size_t low = buffer[pos++];
        const size_t high = buffer[pos++];
        memory_usage_mb_ = low + (high * 256);
    }
    if (pos + 1 < buffer.size())
    {
        const size_t low = buffer[pos++];
        const size_t high = buffer[pos++];
        performance_metrics_.computation_time_ms = static_cast<double>(low + (high * 256));
    }
    if (pos + 1 < buffer.size())
    {
        const size_t low = buffer[pos++];
        const size_t high = buffer[pos++];
        resource_utilization_.memory_allocated_mb = low + (high * 256);
    }
}

bool HighDimErrorEvent::validateHighdimData() const
{
    if (dimension_ > 100)
    {
        return false;
    }
    if (complex_size_ > 10000000)
    {
        return false;
    }
    if (memory_usage_mb_ > 100000)
    {
        return false;
    }
    return true;
}

std::string HighDimErrorEvent::generateFailureSummary() const
{
    std::ostringstream summary;
    summary << "High-Dimensional Failure Summary:\n";
    summary << "  Error Code: " << static_cast<int>(getErrorCode()) << "\n";
    summary << "  Dimension: " << dimension_ << "\n";
    summary << "  Complex Size: " << complex_size_ << "\n";
    summary << "  Memory Usage: " << memory_usage_mb_ << "MB\n";
    summary << "  Algorithm Phase: " << algorithm_phase_ << "\n";
    summary << "  Computation Stage: " << computation_stage_ << "\n";
    if (!failure_analysis_.root_cause.empty())
    {
        summary << "  Root Cause: " << failure_analysis_.root_cause << "\n";
    }
    if (!recovery_suggestions_.empty())
    {
        summary << "  Recovery Suggestions:\n";
        for (const auto &suggestion : recovery_suggestions_)
        {
            summary << "    - " << suggestion << "\n";
        }
    }
    return summary.str();
}

std::string HighDimErrorEvent::generateDiagnosticReport() const
{
    std::ostringstream report;
    report << "=== High-Dimensional Error Diagnostic Report ===\n\n";
    report << "Basic Information:\n";
    report << "  Error Code: " << static_cast<int>(getErrorCode()) << "\n";
    report << "  Error Message: " << getMessage() << "\n";
    report << "  Timestamp: " << getTimestamp().time_since_epoch().count() << "\n";
    report << "  Dimension: " << dimension_ << "\n";
    report << "  Complex Size: " << complex_size_ << "\n";
    report << "  Memory Usage: " << memory_usage_mb_ << "MB\n";
    report << "  Algorithm Phase: " << algorithm_phase_ << "\n";
    report << "  Computation Stage: " << computation_stage_ << "\n\n";
    report << "Performance Metrics:\n";
    report << "  Computation Time: " << performance_metrics_.computation_time_ms << "ms\n";
    report << "  Memory Usage: " << performance_metrics_.memory_usage_mb << "MB\n";
    report << "  CPU Utilization: " << performance_metrics_.cpu_utilization * 100 << "%\n";
    report << "  Progress: " << performance_metrics_.progress_percentage * 100 << "%\n";
    report << "  Operations: " << performance_metrics_.operations_completed << "/"
           << performance_metrics_.operations_total << "\n\n";
    report << "Resource Utilization:\n";
    report << "  Memory Allocated: " << resource_utilization_.memory_allocated_mb << "MB\n";
    report << "  Memory Peak: " << resource_utilization_.memory_peak_mb << "MB\n";
    report << "  CPU Cores Used: " << resource_utilization_.cpu_cores_used << "\n";
    report << "  GPU Utilization: " << resource_utilization_.gpu_utilization * 100 << "%\n\n";
    report << "Algorithm Context:\n";
    report << "  Algorithm: " << algorithm_context_.algorithm_name << "\n";
    report << "  Version: " << algorithm_context_.algorithm_version << "\n";
    report << "  Optimization: " << algorithm_context_.optimization_level << "\n";
    report << "  Parallel: " << (algorithm_context_.parallel_execution ? "Yes" : "No") << "\n\n";
    report << "Failure Analysis:\n";
    report << "  Root Cause: " << failure_analysis_.root_cause << "\n";
    report << "  Category: " << failure_analysis_.failure_category << "\n";
    report << "  Severity: " << failure_analysis_.severity_level << "\n";
    report << "  Recoverable: " << (failure_analysis_.is_recoverable ? "Yes" : "No") << "\n";
    report << "  Confidence: " << failure_analysis_.confidence_score * 100 << "%\n\n";
    if (!recovery_suggestions_.empty())
    {
        report << "Recovery Suggestions:\n";
        for (const auto &suggestion : recovery_suggestions_)
        {
            report << "  - " << suggestion << "\n";
        }
        report << "\n";
    }
    report << "Debug Information:\n";
    report << "  Iteration Count: " << debug_info_.iteration_count << "\n";
    report << "  Convergence Metric: " << debug_info_.convergence_metric << "\n";
    report << "  Last Successful Operation: " << debug_info_.last_successful_operation << "\n";
    if (!debug_info_.stack_trace.empty())
    {
        report << "  Stack Trace:\n";
        for (const auto &frame : debug_info_.stack_trace)
        {
            report << "    " << frame << "\n";
        }
    }
    report << "\n=== End Diagnostic Report ===\n";
    return report.str();
}

PH5ErrorEvent::PH5ErrorEvent()
    : HighDimErrorEvent()
{}

PH5ErrorEvent::PH5ErrorEvent(ErrorCode code, const std::string &message)
    : HighDimErrorEvent(code, message)
{}

void PH5ErrorEvent::setCohomologyContext(const CohomologyContext &context)
{
    cohomology_context_ = context;
}

const CohomologyContext &PH5ErrorEvent::getCohomologyContext() const
{
    return cohomology_context_;
}

void PH5ErrorEvent::setReductionContext(const ReductionContext &context)
{
    reduction_context_ = context;
}

const ReductionContext &PH5ErrorEvent::getReductionContext() const
{
    return reduction_context_;
}

void PH5ErrorEvent::setOrderingContext(const OrderingContext &context)
{
    ordering_context_ = context;
}

const OrderingContext &PH5ErrorEvent::getOrderingContext() const
{
    return ordering_context_;
}

void PH5ErrorEvent::setMatrixContext(const MatrixContext &context)
{
    matrix_context_ = context;
}

const MatrixContext &PH5ErrorEvent::getMatrixContext() const
{
    return matrix_context_;
}

PH6ErrorEvent::PH6ErrorEvent()
    : HighDimErrorEvent()
{}

PH6ErrorEvent::PH6ErrorEvent(ErrorCode code, const std::string &message)
    : HighDimErrorEvent(code, message)
{}

void PH6ErrorEvent::setWitnessContext(const WitnessContext &context)
{
    witness_context_ = context;
}

const WitnessContext &PH6ErrorEvent::getWitnessContext() const
{
    return witness_context_;
}

void PH6ErrorEvent::setSamplingContext(const SamplingContext &context)
{
    sampling_context_ = context;
}

const SamplingContext &PH6ErrorEvent::getSamplingContext() const
{
    return sampling_context_;
}

void PH6ErrorEvent::setLandmarkContext(const LandmarkContext &context)
{
    landmark_context_ = context;
}

const LandmarkContext &PH6ErrorEvent::getLandmarkContext() const
{
    return landmark_context_;
}

void PH6ErrorEvent::setTruncationContext(const TruncationContext &context)
{
    truncation_context_ = context;
}

const TruncationContext &PH6ErrorEvent::getTruncationContext() const
{
    return truncation_context_;
}

} // namespace nerve::core

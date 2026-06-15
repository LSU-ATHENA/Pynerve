
#include "nerve/core/detail/error_event_extensions.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace nerve::core
{

// Confidence score constants for error analysis
constexpr double CONFIDENCE_HIGH = 0.95;
constexpr double CONFIDENCE_MEDIUM = 0.90;
constexpr double CONFIDENCE_MEDIUM_LOW = 0.85;
constexpr double CONFIDENCE_LOW = 0.80;
constexpr double MILLISECONDS_PER_SECOND = 1000.0;
constexpr int PH6_DIMENSION = 6;
constexpr double WITNESS_QUALITY_UNKNOWN = 0.0;

std::unique_ptr<PH5ErrorEvent>
ErrorEventFactory::createPh5MemoryError(size_t dimension, size_t complex_size, size_t memory_usage,
                                        const std::string &phase, const std::string &stage)
{
    auto event =
        std::make_unique<PH5ErrorEvent>(ErrorCode::E11_PH5_OVERFLOW, "PH5 memory budget exceeded");
    setCommonHighdimMetadata(*event, dimension, complex_size, phase, stage);
    event->setMemoryUsage(memory_usage);
    PerformanceMetrics metrics;
    metrics.memory_usage_mb = static_cast<double>(memory_usage);
    metrics.computation_time_ms = std::sqrt(std::max(1.0, static_cast<double>(complex_size))) *
                                  std::max(1.0, static_cast<double>(dimension));
    metrics.progress_percentage = 0.0;
    event->setPerformanceMetrics(metrics);
    FailureAnalysis analysis;
    analysis.root_cause = "Memory budget exceeded during PH5 computation";
    analysis.failure_category = "Resource Limitation";
    analysis.severity_level = "High";
    analysis.is_recoverable = true;
    analysis.recovery_strategy = "Use CompactSummary recovery or reduce problem size";
    analysis.confidence_score = CONFIDENCE_HIGH;
    event->setFailureAnalysis(analysis);
    std::vector<std::string> suggestions;
    suggestions.push_back("Reduce input size or use sparse representations");
    suggestions.push_back("Increase memory budget if available");
    suggestions.push_back("Enable memory optimization features");
    suggestions.push_back("Use lower-dimensional approximation");
    event->setRecoverySuggestions(suggestions);
    return event;
}

std::unique_ptr<PH5ErrorEvent>
ErrorEventFactory::createPh5TimeError(size_t dimension, size_t complex_size, double time_used,
                                      const std::string &phase, const std::string &stage)
{
    auto event =
        std::make_unique<PH5ErrorEvent>(ErrorCode::E11_PH5_OVERFLOW, "PH5 time budget exceeded");
    setCommonHighdimMetadata(*event, dimension, complex_size, phase, stage);
    PerformanceMetrics metrics;
    metrics.computation_time_ms = time_used * MILLISECONDS_PER_SECOND;
    metrics.memory_usage_mb = 0.0;
    metrics.progress_percentage = 0.0;
    event->setPerformanceMetrics(metrics);
    FailureAnalysis analysis;
    analysis.root_cause = "Time budget exceeded during PH5 computation";
    analysis.failure_category = "Performance Limitation";
    analysis.severity_level = "Medium";
    analysis.is_recoverable = true;
    analysis.recovery_strategy = "Use CompactSummary recovery or optimize algorithm";
    analysis.confidence_score = CONFIDENCE_MEDIUM;
    event->setFailureAnalysis(analysis);
    std::vector<std::string> suggestions;
    suggestions.push_back("Increase time budget if available");
    suggestions.push_back("Enable algorithm optimizations");
    suggestions.push_back("Use parallel processing if available");
    suggestions.push_back("Reduce problem complexity");
    event->setRecoverySuggestions(suggestions);
    return event;
}

std::unique_ptr<PH5ErrorEvent> ErrorEventFactory::createPh5ComplexityError(size_t dimension,
                                                                           size_t complex_size,
                                                                           double complexity_score,
                                                                           const std::string &phase,
                                                                           const std::string &stage)
{
    auto event = std::make_unique<PH5ErrorEvent>(ErrorCode::E54_PH4_INVALID_INPUT,
                                                 "PH5 complexity too high");
    setCommonHighdimMetadata(*event, dimension, complex_size, phase, stage);
    FailureAnalysis analysis;
    analysis.root_cause = "Problem complexity score " + std::to_string(complexity_score) +
                          " exceeds algorithm capabilities";
    analysis.failure_category = "Complexity Limitation";
    analysis.severity_level = "High";
    analysis.is_recoverable = true;
    analysis.recovery_strategy = "Use approximation or baseline algorithm";
    analysis.confidence_score = CONFIDENCE_MEDIUM_LOW;
    event->setFailureAnalysis(analysis);
    std::vector<std::string> suggestions;
    suggestions.push_back("Use approximation algorithms");
    suggestions.push_back("Reduce maximum dimension");
    suggestions.push_back("Apply simplification techniques");
    suggestions.push_back("Use hierarchical approach");
    event->setRecoverySuggestions(suggestions);
    return event;
}

std::unique_ptr<PH5ErrorEvent>
ErrorEventFactory::createPh5NumericalError(size_t dimension, const std::string &numerical_issue,
                                           double stability_score)
{
    auto event = std::make_unique<PH5ErrorEvent>(ErrorCode::E54_PH4_INVALID_INPUT,
                                                 "PH5 numerical instability detected");
    event->setDimension(dimension);
    event->setAlgorithmPhase("Cohomology Computation");
    event->setComputationStage("Numerical Processing");
    CohomologyContext cohomology_context;
    cohomology_context.cohomology_dimension = dimension;
    cohomology_context.cohomology_complexity = stability_score;
    cohomology_context.cohomology_method = "Cohomology Reducer";
    event->setCohomologyContext(cohomology_context);
    FailureAnalysis analysis;
    analysis.root_cause =
        numerical_issue.empty()
            ? "Numerical instability in cohomology computation"
            : "Numerical instability in cohomology computation: " + numerical_issue;
    analysis.failure_category = "Numerical Issue";
    analysis.severity_level = "High";
    analysis.is_recoverable = true;
    analysis.recovery_strategy = "Use higher precision or alternative algorithm";
    analysis.confidence_score = CONFIDENCE_LOW;
    event->setFailureAnalysis(analysis);
    std::vector<std::string> suggestions;
    suggestions.push_back("Use higher precision arithmetic");
    suggestions.push_back("Apply numerical stabilization techniques");
    suggestions.push_back("Use alternative cohomology method");
    suggestions.push_back("Reduce problem conditioning");
    event->setRecoverySuggestions(suggestions);
    return event;
}

std::unique_ptr<PH6ErrorEvent>
ErrorEventFactory::createPh6MemoryError(size_t dimension, size_t num_points, size_t memory_usage,
                                        const std::string &phase, const std::string &stage)
{
    auto event =
        std::make_unique<PH6ErrorEvent>(ErrorCode::E12_PH6_OVERFLOW, "PH6 memory budget exceeded");
    setCommonHighdimMetadata(*event, dimension, num_points, phase, stage);
    event->setMemoryUsage(memory_usage);
    WitnessContext witness_context;
    witness_context.witness_complex_size = num_points;
    witness_context.witness_quality = WITNESS_QUALITY_UNKNOWN;
    witness_context.witness_construction_method = "Hierarchical Witness";
    event->setWitnessContext(witness_context);
    std::vector<std::string> suggestions;
    suggestions.push_back("Reduce landmark ratio");
    suggestions.push_back("Increase memory budget if available");
    suggestions.push_back("Use adaptive truncation");
    suggestions.push_back("Apply witness complex optimization");
    event->setRecoverySuggestions(suggestions);
    return event;
}

std::unique_ptr<PH6ErrorEvent>
ErrorEventFactory::createPh6TimeError(size_t dimension, size_t num_points, double time_used,
                                      const std::string &phase, const std::string &stage)
{
    auto event =
        std::make_unique<PH6ErrorEvent>(ErrorCode::E12_PH6_OVERFLOW, "PH6 time budget exceeded");
    setCommonHighdimMetadata(*event, dimension, num_points, phase, stage);
    PerformanceMetrics metrics;
    metrics.computation_time_ms = time_used * MILLISECONDS_PER_SECOND;
    metrics.progress_percentage = 0.0;
    event->setPerformanceMetrics(metrics);
    std::vector<std::string> suggestions;
    suggestions.push_back("Increase time budget if available");
    suggestions.push_back("Reduce landmark ratio");
    suggestions.push_back("Optimize sampling strategy");
    suggestions.push_back("Use parallel processing");
    event->setRecoverySuggestions(suggestions);
    return event;
}

std::unique_ptr<PH6ErrorEvent>
ErrorEventFactory::createPh6SamplingError(size_t num_points, size_t num_landmarks,
                                          double sampling_quality,
                                          const std::string &sampling_strategy)
{
    auto event = std::make_unique<PH6ErrorEvent>(ErrorCode::E54_PH4_INVALID_INPUT,
                                                 "PH6 sampling quality too low");
    event->setDimension(PH6_DIMENSION);
    event->setComplexSize(num_points);
    event->setAlgorithmPhase("Landmark Selection");
    event->setComputationStage("Sampling");
    SamplingContext sampling_context;
    sampling_context.sampling_strategy = sampling_strategy;
    sampling_context.samples_taken = num_landmarks;
    sampling_context.sampling_quality = sampling_quality;
    sampling_context.adaptive_sampling = false;
    event->setSamplingContext(sampling_context);
    std::vector<std::string> suggestions;
    suggestions.push_back("Increase landmark ratio");
    suggestions.push_back("Use adaptive sampling");
    suggestions.push_back("Improve landmark selection strategy");
    suggestions.push_back("Use alternative sampling method");
    event->setRecoverySuggestions(suggestions);
    return event;
}

std::unique_ptr<PH6ErrorEvent>
ErrorEventFactory::createPh6WitnessError(size_t num_landmarks, size_t witness_complex_size,
                                         double witness_quality, const std::string &witness_issue)
{
    auto event = std::make_unique<PH6ErrorEvent>(ErrorCode::E54_PH4_INVALID_INPUT,
                                                 "PH6 witness construction failed");
    event->setDimension(PH6_DIMENSION);
    event->setComplexSize(witness_complex_size);
    event->setAlgorithmPhase("Witness Construction");
    event->setComputationStage("Complex Building");
    WitnessContext witness_context;
    witness_context.witness_complex_size = witness_complex_size;
    witness_context.witness_quality = witness_quality;
    witness_context.witness_construction_method = "Hierarchical Witness";
    witness_context.witness_complexity =
        num_landmarks == 0
            ? 0.0
            : static_cast<double>(witness_complex_size) / static_cast<double>(num_landmarks);
    event->setWitnessContext(witness_context);
    FailureAnalysis analysis;
    analysis.root_cause = witness_issue.empty() ? "Witness construction failed" : witness_issue;
    analysis.failure_category = "Witness Construction";
    analysis.severity_level = "High";
    analysis.is_recoverable = true;
    analysis.recovery_strategy = "Adjust landmark selection or witness-complex thresholds";
    analysis.confidence_score = std::clamp(witness_quality, 0.0, 1.0);
    event->setFailureAnalysis(analysis);
    std::vector<std::string> suggestions;
    suggestions.push_back("Improve landmark selection");
    suggestions.push_back("Use alternative witness construction");
    suggestions.push_back("Apply witness complex optimization");
    suggestions.push_back("Reduce witness complexity requirements");
    event->setRecoverySuggestions(suggestions);
    return event;
}

std::unique_ptr<HighDimErrorEvent>
ErrorEventFactory::createIncrementalError(const std::string &operation_type, size_t num_points,
                                          const std::vector<size_t> &affected_simplices,
                                          const std::string &failure_reason)
{
    auto event = std::make_unique<HighDimErrorEvent>(
        ErrorCode::E54_PH4_INVALID_INPUT, "Incremental operation failed: " + failure_reason);
    event->setComplexSize(num_points);
    event->setAlgorithmPhase("Incremental Update");
    event->setComputationStage(operation_type);
    PerformanceMetrics metrics;
    metrics.operations_total = affected_simplices.size();
    metrics.operations_completed = 0;
    event->setPerformanceMetrics(metrics);
    std::vector<std::string> suggestions;
    suggestions.push_back("Use full recompute recovery");
    suggestions.push_back("Reduce affected simplex count");
    suggestions.push_back("Optimize hint computation");
    suggestions.push_back("Use batch processing instead");
    event->setRecoverySuggestions(suggestions);
    return event;
}

std::unique_ptr<HighDimErrorEvent>
ErrorEventFactory::createStreamingError(size_t window_size, size_t stream_position,
                                        double processing_time, const std::string &streaming_issue)
{
    auto event = std::make_unique<HighDimErrorEvent>(
        ErrorCode::E54_PH4_INVALID_INPUT, "Streaming processing failed: " + streaming_issue);
    event->setComplexSize(window_size);
    event->setAlgorithmPhase("Stream Processing");
    event->setComputationStage("Window Update");
    PerformanceMetrics metrics;
    metrics.computation_time_ms = processing_time * 1000;
    event->setPerformanceMetrics(metrics);
    DebugInfo debug;
    debug.variable_states["stream_position"] = std::to_string(stream_position);
    event->setDebugInfo(debug);
    std::vector<std::string> suggestions;
    suggestions.push_back("Increase window processing time budget");
    suggestions.push_back("Reduce window size");
    suggestions.push_back("Optimize stream processing algorithm");
    suggestions.push_back("Use adaptive window sizing");
    event->setRecoverySuggestions(suggestions);
    return event;
}

std::unique_ptr<HighDimErrorEvent>
ErrorEventFactory::createLaplacianError(size_t matrix_size, size_t matrix_rank,
                                        double condition_number, const std::string &laplacian_issue)
{
    auto event = std::make_unique<HighDimErrorEvent>(
        ErrorCode::E54_PH4_INVALID_INPUT, "Laplacian computation failed: " + laplacian_issue);
    event->setComplexSize(matrix_size);
    event->setAlgorithmPhase("Laplacian Integration");
    event->setComputationStage("Matrix Computation");
    DebugInfo debug;
    debug.variable_states["matrix_rank"] = std::to_string(matrix_rank);
    debug.variable_states["condition_number"] = std::to_string(condition_number);
    event->setDebugInfo(debug);
    std::vector<std::string> suggestions;
    suggestions.push_back("Use regularization techniques");
    suggestions.push_back("Apply matrix preconditioning");
    suggestions.push_back("Use alternative eigenvalue method");
    suggestions.push_back("Reduce matrix size through truncation");
    event->setRecoverySuggestions(suggestions);
    return event;
}

void ErrorEventFactory::setCommonHighdimMetadata(HighDimErrorEvent &event, size_t dimension,
                                                 size_t complex_size, const std::string &phase,
                                                 const std::string &stage)
{
    event.setDimension(dimension);
    event.setComplexSize(complex_size);
    event.setAlgorithmPhase(phase);
    event.setComputationStage(stage);
}

void ErrorEventFactory::setPerformanceMetrics(HighDimErrorEvent &event, double time_used,
                                              size_t memory_usage)
{
    PerformanceMetrics metrics;
    metrics.computation_time_ms = time_used * MILLISECONDS_PER_SECOND;
    metrics.memory_usage_mb = static_cast<double>(memory_usage);
    metrics.progress_percentage = 0.0;
    event.setPerformanceMetrics(metrics);
}

void ErrorEventFactory::generateRecoverySuggestions(HighDimErrorEvent &event, ErrorCode code,
                                                    const std::string &context)
{
    std::vector<std::string> suggestions;
    if (context.find("memory") != std::string::npos)
    {
        suggestions.push_back("Reduce problem size or memory requirements");
        suggestions.push_back("Enable memory optimization features");
        suggestions.push_back("Use sparse representations");
    }
    else if (context.find("time") != std::string::npos)
    {
        suggestions.push_back("Increase time budget if available");
        suggestions.push_back("Enable parallel processing");
        suggestions.push_back("Use algorithm optimizations");
    }
    else if (context.find("complexity") != std::string::npos)
    {
        suggestions.push_back("Use approximation algorithms");
        suggestions.push_back("Reduce problem complexity");
        suggestions.push_back("Apply simplification techniques");
    }
    if (code == ErrorCode::E41_RESOURCE_LIMIT)
    {
        suggestions.push_back("Inspect configured resource budgets");
    }
    else if (code == ErrorCode::E20_NUM_NAN)
    {
        suggestions.push_back("Validate numeric inputs for NaN or infinity");
    }
    else if (code == ErrorCode::E30_DET_MISMATCH)
    {
        suggestions.push_back("Review determinism contract settings");
    }
    suggestions.push_back("Check input parameters and constraints");
    suggestions.push_back("Consider alternative algorithms");
    suggestions.push_back("Use recovery mechanisms if available");
    event.setRecoverySuggestions(suggestions);
}

} // namespace nerve::core

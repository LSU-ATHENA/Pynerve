
#pragma once

#include "nerve/core/error/error_event.hpp"
#include "nerve/core/persistence.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace nerve::core
{

struct PerformanceMetrics
{
    double computation_time_ms = 0.0;
    double memory_usage_mb = 0.0;
    double cpu_utilization = 0.0;
    double io_wait_time = 0.0;
    std::size_t operations_completed = 0;
    std::size_t operations_total = 0;
    double progress_percentage = 0.0;
};

struct ResourceUtilization
{
    std::size_t memory_allocated_mb = 0;
    std::size_t memory_peak_mb = 0;
    std::size_t disk_space_used_mb = 0;
    double cpu_cores_used = 0.0;
    double gpu_utilization = 0.0;
    std::size_t network_bytes_transferred = 0;
};

struct AlgorithmContext
{
    std::string algorithm_name;
    std::string algorithm_version;
    std::unordered_map<std::string, std::string> parameters;
    std::unordered_map<std::string, double> hyperparameters;
    std::string optimization_level;
    bool parallel_execution = false;
};

struct FailureAnalysis
{
    std::string root_cause;
    std::string failure_category;
    double confidence_score = 0.0;
    std::vector<std::string> contributing_factors;
    std::string severity_level;
    bool is_recoverable = false;
    std::string recovery_strategy;
};

struct DebugInfo
{
    std::vector<std::string> stack_trace;
    std::unordered_map<std::string, std::string> variable_states;
    std::vector<std::string> checkpoint_locations;
    std::string last_successful_operation;
    std::size_t iteration_count = 0;
    double convergence_metric = 0.0;
};

struct CohomologyContext
{
    std::size_t cohomology_groups = 0;
    std::size_t cohomology_dimension = 0;
    double cohomology_complexity = 0.0;
    std::string cohomology_method;
    bool cohomology_optimization_enabled = false;
};

struct ReductionContext
{
    std::size_t original_matrix_size = 0;
    std::size_t reduced_matrix_size = 0;
    double reduction_ratio = 0.0;
    std::size_t elimination_steps = 0;
    double reduction_time = 0.0;
    std::string reduction_algorithm;
};

struct OrderingContext
{
    std::string ordering_strategy;
    double ordering_time = 0.0;
    std::size_t ordering_improvement = 0;
    bool clever_ordering_used = false;
    double ordering_efficiency = 0.0;
};

struct MatrixContext
{
    std::size_t matrix_rows = 0;
    std::size_t matrix_columns = 0;
    std::size_t matrix_nonzeros = 0;
    double matrix_density = 0.0;
    double matrix_condition_number = 0.0;
    std::string matrix_format;
};

struct WitnessContext
{
    std::size_t witness_complex_size = 0;
    double witness_quality = 0.0;
    std::string witness_construction_method;
    double witness_complexity = 0.0;
    bool witness_optimization_enabled = false;
};

struct SamplingContext
{
    std::string sampling_strategy;
    std::size_t samples_taken = 0;
    double sampling_quality = 0.0;
    double sampling_time = 0.0;
    bool adaptive_sampling = false;
    double sampling_efficiency = 0.0;
};

struct LandmarkContext
{
    std::vector<std::size_t> landmark_indices;
    std::size_t num_landmarks = 0;
    double landmark_ratio = 0.0;
    double landmark_coverage = 0.0;
    double landmark_selection_time = 0.0;
    std::string landmark_selection_method;
};

struct TruncationContext
{
    bool truncation_applied = false;
    double truncation_threshold = 0.0;
    std::size_t elements_truncated = 0;
    double truncation_ratio = 0.0;
    double truncation_quality = 0.0;
    std::string truncation_method;
};

class HighDimErrorEvent : public ErrorEvent
{
public:
    HighDimErrorEvent();
    HighDimErrorEvent(ErrorCode code, const std::string &message);

    void setDimension(std::size_t dimension);
    std::size_t getDimension() const;
    void setComplexSize(std::size_t complex_size);
    std::size_t getComplexSize() const;
    void setMemoryUsage(std::size_t memory_usage_mb);
    std::size_t getMemoryUsage() const;
    void setAlgorithmPhase(const std::string &phase);
    const std::string &getAlgorithmPhase() const;
    void setComputationStage(const std::string &stage);
    const std::string &getComputationStage() const;
    void setPerformanceMetrics(const PerformanceMetrics &metrics);
    const PerformanceMetrics &getPerformanceMetrics() const;
    void setResourceUtilization(const ResourceUtilization &utilization);
    const ResourceUtilization &getResourceUtilization() const;
    void setAlgorithmContext(const AlgorithmContext &context);
    const AlgorithmContext &getAlgorithmContext() const;
    void setFailureAnalysis(const FailureAnalysis &analysis);
    const FailureAnalysis &getFailureAnalysis() const;
    void setRecoverySuggestions(const std::vector<std::string> &suggestions);
    const std::vector<std::string> &getRecoverySuggestions() const;
    void setDebugInfo(const DebugInfo &info);
    const DebugInfo &getDebugInfo() const;
    void serializeHighdimData(std::vector<std::uint8_t> &buffer) const;
    void deserializeHighdimData(const std::vector<std::uint8_t> &buffer);
    bool validateHighdimData() const;
    std::string generateFailureSummary() const;
    std::string generateDiagnosticReport() const;

private:
    std::size_t dimension_ = 0;
    std::size_t complex_size_ = 0;
    std::size_t memory_usage_mb_ = 0;
    std::string algorithm_phase_;
    std::string computation_stage_;
    PerformanceMetrics performance_metrics_;
    ResourceUtilization resource_utilization_;
    AlgorithmContext algorithm_context_;
    FailureAnalysis failure_analysis_;
    std::vector<std::string> recovery_suggestions_;
    DebugInfo debug_info_;
};

class PH5ErrorEvent : public HighDimErrorEvent
{
public:
    PH5ErrorEvent();
    PH5ErrorEvent(ErrorCode code, const std::string &message);
    void setCohomologyContext(const CohomologyContext &context);
    const CohomologyContext &getCohomologyContext() const;
    void setReductionContext(const ReductionContext &context);
    const ReductionContext &getReductionContext() const;
    void setOrderingContext(const OrderingContext &context);
    const OrderingContext &getOrderingContext() const;
    void setMatrixContext(const MatrixContext &context);
    const MatrixContext &getMatrixContext() const;

private:
    CohomologyContext cohomology_context_;
    ReductionContext reduction_context_;
    OrderingContext ordering_context_;
    MatrixContext matrix_context_;
};

class PH6ErrorEvent : public HighDimErrorEvent
{
public:
    PH6ErrorEvent();
    PH6ErrorEvent(ErrorCode code, const std::string &message);
    void setWitnessContext(const WitnessContext &context);
    const WitnessContext &getWitnessContext() const;
    void setSamplingContext(const SamplingContext &context);
    const SamplingContext &getSamplingContext() const;
    void setLandmarkContext(const LandmarkContext &context);
    const LandmarkContext &getLandmarkContext() const;
    void setTruncationContext(const TruncationContext &context);
    const TruncationContext &getTruncationContext() const;

private:
    WitnessContext witness_context_;
    SamplingContext sampling_context_;
    LandmarkContext landmark_context_;
    TruncationContext truncation_context_;
};

class ErrorEventFactory
{
public:
    static std::unique_ptr<PH5ErrorEvent>
    createPh5MemoryError(std::size_t dimension, std::size_t complex_size, std::size_t memory_usage,
                         const std::string &phase, const std::string &stage);
    static std::unique_ptr<PH5ErrorEvent>
    createPh5TimeError(std::size_t dimension, std::size_t complex_size, double time_used,
                       const std::string &phase, const std::string &stage);
    static std::unique_ptr<PH5ErrorEvent> createPh5ComplexityError(std::size_t dimension,
                                                                   std::size_t complex_size,
                                                                   double complexity_score,
                                                                   const std::string &phase,
                                                                   const std::string &stage);
    static std::unique_ptr<PH5ErrorEvent>
    createPh5NumericalError(std::size_t dimension, const std::string &numerical_issue,
                            double stability_score);
    static std::unique_ptr<PH6ErrorEvent>
    createPh6MemoryError(std::size_t dimension, std::size_t num_points, std::size_t memory_usage,
                         const std::string &phase, const std::string &stage);
    static std::unique_ptr<PH6ErrorEvent>
    createPh6TimeError(std::size_t dimension, std::size_t num_points, double time_used,
                       const std::string &phase, const std::string &stage);
    static std::unique_ptr<PH6ErrorEvent>
    createPh6SamplingError(std::size_t num_points, std::size_t num_landmarks,
                           double sampling_quality, const std::string &sampling_strategy);
    static std::unique_ptr<PH6ErrorEvent> createPh6WitnessError(std::size_t num_landmarks,
                                                                std::size_t witness_complex_size,
                                                                double witness_quality,
                                                                const std::string &witness_issue);
    static std::unique_ptr<HighDimErrorEvent>
    createIncrementalError(const std::string &operation_type, std::size_t num_points,
                           const std::vector<std::size_t> &affected_simplices,
                           const std::string &failure_reason);
    static std::unique_ptr<HighDimErrorEvent>
    createStreamingError(std::size_t window_size, std::size_t stream_position,
                         double processing_time, const std::string &streaming_issue);
    static std::unique_ptr<HighDimErrorEvent>
    createLaplacianError(std::size_t matrix_size, std::size_t matrix_rank, double condition_number,
                         const std::string &laplacian_issue);

private:
    static void setCommonHighdimMetadata(HighDimErrorEvent &event, std::size_t dimension,
                                         std::size_t complex_size, const std::string &phase,
                                         const std::string &stage);
    static void setPerformanceMetrics(HighDimErrorEvent &event, double time_used,
                                      std::size_t memory_usage);
    static void generateRecoverySuggestions(HighDimErrorEvent &event, ErrorCode code,
                                            const std::string &context);
};

} // namespace nerve::core

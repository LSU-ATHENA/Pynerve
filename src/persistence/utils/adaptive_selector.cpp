#include "nerve/persistence/utils/adaptive_selector.hpp"
#include "nerve/runtime/calibration_model.hpp"
#include "nerve/runtime/hardware_probe.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>
namespace nerve::persistence
{
namespace
{
// Memory Unit Constants
constexpr size_t BYTES_PER_MB = 1024ULL * 1024ULL;
// Dataset size thresholds for problem bucketing
constexpr size_t DATASET_SMALL_THRESHOLD = 1000;
constexpr size_t DATASET_MEDIUM_THRESHOLD = 10000;
constexpr size_t DATASET_LARGE_THRESHOLD = 100000;
// Sparsity thresholds for bucketing
constexpr double SPARSITY_VERY_SPARSE_THRESHOLD = 0.75;
constexpr double SPARSITY_SPARSE_THRESHOLD = 0.45;
constexpr double SPARSITY_MIXED_THRESHOLD = 0.2;
// Dimension thresholds for bucketing
constexpr int DIMENSION_HIGH_THRESHOLD = 4;
constexpr int DIMENSION_MEDIUM_THRESHOLD = 2;
// Algorithm optimal dataset sizes
constexpr size_t ALGORITHM_OPTIMAL_STANDARD = 2048;
constexpr size_t ALGORITHM_OPTIMAL_TWISTED = 4096;
constexpr size_t ALGORITHM_OPTIMAL = 8192;
constexpr size_t ALGORITHM_OPTIMAL_GPU = 16384;
constexpr size_t ALGORITHM_OPTIMAL_SIMD = 12288;
// Scoring formula weights
constexpr double SUITABILITY_WEIGHT_SIZE = 0.6;
constexpr double SUITABILITY_WEIGHT_SPARSITY = 0.4;
constexpr double SUITABILITY_PENALTY_BASE = 2.0;
constexpr double SUITABILITY_PENALTY_MIN = 0.25;
constexpr double SUITABILITY_PENALTY_MAX = 3.0;
constexpr double MEMORY_PENALTY_MIN = 0.2;
constexpr double SCORE_NORMALIZATION_FACTOR = 1000.0;
// Size suitability minimum ratio
constexpr double SIZE_RATIO_MIN = 0.1;
// Default sparsity suitability for standard algorithms
constexpr double DEFAULT_SPARSITY_SUITABILITY = 0.75;
constexpr double FAST_SPARSITY_BASE = 0.5;
// Performance model update weights
constexpr double PERFORMANCE_EMA_ALPHA = 0.2;
constexpr double PERFORMANCE_EMA_ONE_MINUS_ALPHA = 0.8;
constexpr double MEMORY_NORMALIZATION_DIVISOR = 128.0;
constexpr double MS_PER_SECOND = 1000.0;
constexpr double MINIMUM_NONZERO_VALUE = 1e-9;
constexpr double MAX_FINITE_SCORE = 1.0e300;

std::size_t saturatingMul(std::size_t lhs, std::size_t rhs)
{
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs)
    {
        return std::numeric_limits<std::size_t>::max();
    }
    return lhs * rhs;
}

std::size_t saturatingAdd(std::size_t lhs, std::size_t rhs)
{
    if (rhs > std::numeric_limits<std::size_t>::max() - lhs)
    {
        return std::numeric_limits<std::size_t>::max();
    }
    return lhs + rhs;
}

double finiteNonnegative(double value, double fallback)
{
    if (!std::isfinite(value))
    {
        return fallback;
    }
    return std::max(0.0, value);
}

double finitePositive(double value, double fallback)
{
    if (!std::isfinite(value) || value <= 0.0)
    {
        return fallback;
    }
    return value;
}

double finiteUnit(double value, double fallback)
{
    if (!std::isfinite(value))
    {
        return fallback;
    }
    return std::clamp(value, 0.0, 1.0);
}

double finiteScore(long double value)
{
    if (!std::isfinite(value))
    {
        return MAX_FINITE_SCORE;
    }
    if (value <= 0.0L)
    {
        return 0.0;
    }
    return static_cast<double>(std::min(value, static_cast<long double>(MAX_FINITE_SCORE)));
}

std::string problemBucketFromData(const DataCharacteristics &data)
{
    const char *size_bucket = "small";
    if (data.num_simplices >= DATASET_LARGE_THRESHOLD)
    {
        size_bucket = "xlarge";
    }
    else if (data.num_simplices >= DATASET_MEDIUM_THRESHOLD)
    {
        size_bucket = "large";
    }
    else if (data.num_simplices >= DATASET_SMALL_THRESHOLD)
    {
        size_bucket = "medium";
    }
    const char *sparsity_bucket = "dense";
    const double sparsity = finiteUnit(data.sparsity, 0.0);
    if (sparsity >= SPARSITY_VERY_SPARSE_THRESHOLD)
    {
        sparsity_bucket = "very_sparse";
    }
    else if (sparsity >= SPARSITY_SPARSE_THRESHOLD)
    {
        sparsity_bucket = "sparse";
    }
    else if (sparsity >= SPARSITY_MIXED_THRESHOLD)
    {
        sparsity_bucket = "mixed";
    }
    const char *dim_bucket = "lowdim";
    if (data.max_dimension >= DIMENSION_HIGH_THRESHOLD)
    {
        dim_bucket = "highdim";
    }
    else if (data.max_dimension >= DIMENSION_MEDIUM_THRESHOLD)
    {
        dim_bucket = "middim";
    }
    return std::string(size_bucket) + "|" + dim_bucket + "|" + sparsity_bucket;
}
} // namespace
AdaptiveSelector::AdaptiveSelector()
{
    const auto snapshot = runtime::collectHardwareSnapshot();
    hardware_fingerprint_ = runtime::getHardwareFingerprint(snapshot);
    initializePerformanceModels();
    const auto load_status = runtime::loadCalibrationModel();
    if (load_status.isError())
    {
        return;
    }
}
void AdaptiveSelector::initializePerformanceModels()
{
    algorithm_registry_.clear();
    algorithm_registry_["standard"] =
        AlgorithmInfo{"standard", 1.0, 1.0, ALGORITHM_OPTIMAL_STANDARD, 1.0};
    algorithm_registry_["twisted"] =
        AlgorithmInfo{"twisted", 1.0, 1.0, ALGORITHM_OPTIMAL_TWISTED, 1.0};
    algorithm_registry_["_accelerated"] =
        AlgorithmInfo{"_accelerated", 1.0, 1.0, ALGORITHM_OPTIMAL, 1.0};
    algorithm_registry_["gpu_accelerated"] =
        AlgorithmInfo{"gpu_accelerated", 1.0, 1.0, ALGORITHM_OPTIMAL_GPU, 1.0};
    algorithm_registry_["simd_vectorized"] =
        AlgorithmInfo{"simd_vectorized", 1.0, 1.0, ALGORITHM_OPTIMAL_SIMD, 1.0};
}
DataCharacteristics AdaptiveSelector::analyzeData(const algebra::BoundaryMatrix &matrix)
{
    DataCharacteristics characteristics{};
    characteristics.num_simplices = matrix.cols();
    characteristics.num_vertices = matrix.rows();
    characteristics.sparsity = computeSparsity(matrix);
    characteristics.max_dimension = estimateMaxDimension(matrix);
    characteristics.avg_simplex_size = computeAvgSimplexSize(matrix);
    characteristics.memory_footprint = estimateMemoryUsage(matrix);
    last_problem_bucket_ = problemBucketFromData(characteristics);
    return characteristics;
}
double AdaptiveSelector::computeSparsity(const algebra::BoundaryMatrix &matrix)
{
    const std::size_t rows = matrix.rows();
    const std::size_t cols = matrix.cols();
    if (rows == 0 || cols == 0)
    {
        return 1.0;
    }
    const std::size_t total_entries = matrix.numNonzeros();
    const double max_entries = static_cast<double>(rows) * static_cast<double>(cols);
    if (max_entries <= 0.0)
    {
        return 1.0;
    }
    return std::clamp(1.0 - (static_cast<double>(total_entries) / max_entries), 0.0, 1.0);
}
size_t AdaptiveSelector::estimateMaxDimension(const algebra::BoundaryMatrix &matrix)
{
    std::size_t max_dim = 0;
    for (std::size_t col = 0; col < matrix.cols(); ++col)
    {
        const int dim = matrix.getColSimplexDimension(col);
        if (dim >= 0)
        {
            max_dim = std::max(max_dim, static_cast<std::size_t>(dim));
        }
    }
    return max_dim;
}
double AdaptiveSelector::computeAvgSimplexSize(const algebra::BoundaryMatrix &matrix)
{
    if (matrix.cols() == 0)
    {
        return 0.0;
    }
    std::size_t total_size = 0;
    for (std::size_t col = 0; col < matrix.cols(); ++col)
    {
        const int dim = matrix.getColSimplexDimension(col);
        if (dim >= 0)
        {
            total_size += static_cast<std::size_t>(dim + 1);
        }
    }
    return static_cast<double>(total_size) / static_cast<double>(matrix.cols());
}
size_t AdaptiveSelector::estimateMemoryUsage(const algebra::BoundaryMatrix &matrix)
{
    const std::size_t sparse_entries = matrix.numNonzeros();
    const std::size_t entry_bytes =
        saturatingMul(sparse_entries, sizeof(std::size_t) * 2 + sizeof(double));
    const std::size_t metadata_bytes = saturatingMul(matrix.cols(), sizeof(std::size_t));
    return saturatingAdd(entry_bytes, metadata_bytes);
}
std::string AdaptiveSelector::selectOptimalAlgorithm(const DataCharacteristics &data)
{
    last_predictions_ms_.clear();
    const std::string baseline_algorithm = "standard";
    std::string selected_algorithm = baseline_algorithm;
    double selected_time_ms = MAX_FINITE_SCORE;
    SelectionDiagnostics selected_diagnostics{};
    selected_diagnostics.gate_passed = false;
    selected_diagnostics.gate_reason = "gate_not_evaluated";
    for (const auto &[name, info] : algorithm_registry_)
    {
        const double predicted_time_ms =
            finitePositive(computeAlgorithmScore(data, info), MAX_FINITE_SCORE);
        last_predictions_ms_[name] = predicted_time_ms;
        runtime::PredictionKey key;
        key.hardware_fingerprint = hardware_fingerprint_;
        key.problem_bucket =
            last_problem_bucket_.empty() ? problemBucketFromData(data) : last_problem_bucket_;
        key.algorithm = name;
        const runtime::PredictionWithBounds calibrated = runtime::queryCalibratedPrediction(key);
        const runtime::SelectionGateDiagnostics gate = runtime::evaluatePredictionGate(calibrated);
        const bool use_calibrated =
            gate.gate_passed && calibrated.available && std::isfinite(calibrated.predicted_time_ms);
        const double candidate_time_ms =
            use_calibrated ? calibrated.predicted_time_ms : predicted_time_ms;
        if (gate.gate_passed && candidate_time_ms < selected_time_ms)
        {
            selected_time_ms = candidate_time_ms;
            selected_algorithm = name;
            selected_diagnostics.gate_passed = gate.gate_passed;
            selected_diagnostics.gate_reason = gate.gate_reason;
            selected_diagnostics.confidence = finiteUnit(gate.confidence, 0.0);
            selected_diagnostics.error_bound = finiteNonnegative(gate.error_bound, 1.0);
            selected_diagnostics.sample_count = gate.sample_count;
            selected_diagnostics.calibration_bucket_id = gate.calibration_bucket_id;
        }
    }
    if (selected_algorithm == baseline_algorithm && !selected_diagnostics.gate_passed)
    {
        selected_diagnostics.gate_reason = "fail_closed_baseline_cpu_exact";
        selected_diagnostics.confidence = 0.0;
        selected_diagnostics.error_bound = 1.0;
        selected_diagnostics.sample_count = 0;
        selected_diagnostics.calibration_bucket_id.clear();
    }
    last_selection_diagnostics_ = selected_diagnostics;
    return selected_algorithm;
}
double AdaptiveSelector::computeAlgorithmScore(const DataCharacteristics &data,
                                               const AlgorithmInfo &algorithm)
{
    const double data_scale = std::max<double>(1.0, static_cast<double>(data.num_simplices));
    const double simplex_scale = std::max(1.0, finiteNonnegative(data.avg_simplex_size, 0.0) + 1.0);
    const double dimension_scale =
        std::max<double>(1.0, static_cast<double>(data.max_dimension) + 1.0);
    const double workload = data_scale * simplex_scale * dimension_scale;
    const double normalized_performance =
        std::max(finitePositive(algorithm.performance_factor, 1.0), MINIMUM_NONZERO_VALUE);
    const double size_suitability = computeSizeSuitability(data, algorithm);
    const double sparsity_suitability = computeSparsitySuitability(data, algorithm);
    const double suitability_penalty =
        std::clamp(SUITABILITY_PENALTY_BASE - (SUITABILITY_WEIGHT_SIZE * size_suitability +
                                               SUITABILITY_WEIGHT_SPARSITY * sparsity_suitability),
                   SUITABILITY_PENALTY_MIN, SUITABILITY_PENALTY_MAX);
    const double memory_penalty =
        std::max(MEMORY_PENALTY_MIN, finitePositive(algorithm.memory_factor, 1.0));
    return finiteScore((static_cast<long double>(workload) * normalized_performance *
                        suitability_penalty * memory_penalty) /
                       SCORE_NORMALIZATION_FACTOR);
}
double AdaptiveSelector::computeSizeSuitability(const DataCharacteristics &data,
                                                const AlgorithmInfo &algorithm)
{
    const double data_size = static_cast<double>(std::max<std::size_t>(1, data.num_simplices));
    const double optimal_size =
        static_cast<double>(std::max<std::size_t>(1, algorithm.optimal_dataset_size));
    if (data_size <= optimal_size)
    {
        return data_size / optimal_size;
    }
    return std::max(SIZE_RATIO_MIN, optimal_size / data_size);
}
double AdaptiveSelector::computeSparsitySuitability(const DataCharacteristics &data,
                                                    const AlgorithmInfo &algorithm)
{
    const double sparsity = finiteUnit(data.sparsity, 1.0);
    if (algorithm.name == "gpu_accelerated")
    {
        return 1.0 - sparsity;
    }
    if (algorithm.name == "_accelerated")
    {
        return FAST_SPARSITY_BASE + FAST_SPARSITY_BASE * sparsity;
    }
    return DEFAULT_SPARSITY_SUITABILITY;
}
void AdaptiveSelector::updatePerformanceModel(const std::string &algorithm_name,
                                              const PerformanceMetrics &metrics)
{
    const auto it = algorithm_registry_.find(algorithm_name);
    if (it == algorithm_registry_.end())
    {
        return;
    }
    AlgorithmInfo &info = it->second;
    const double throughput =
        std::max(finitePositive(metrics.throughput_ops_per_sec, MINIMUM_NONZERO_VALUE),
                 MINIMUM_NONZERO_VALUE);
    const double ms_per_op = MS_PER_SECOND / throughput;
    const double normalized_memory =
        std::max(1.0, static_cast<double>(metrics.memory_usage_mb) / MEMORY_NORMALIZATION_DIVISOR);
    info.performance_factor = PERFORMANCE_EMA_ONE_MINUS_ALPHA * info.performance_factor +
                              PERFORMANCE_EMA_ALPHA * ms_per_op;
    info.memory_factor = PERFORMANCE_EMA_ONE_MINUS_ALPHA * info.memory_factor +
                         PERFORMANCE_EMA_ALPHA * normalized_memory;
    const double observed_time_ms = std::max(
        finitePositive(metrics.computation_time_ms, MINIMUM_NONZERO_VALUE), MINIMUM_NONZERO_VALUE);
    const double predicted_time_ms = [&]() {
        const auto pred_it = last_predictions_ms_.find(algorithm_name);
        if (pred_it == last_predictions_ms_.end())
        {
            return observed_time_ms;
        }
        return std::max(finitePositive(pred_it->second, observed_time_ms), MINIMUM_NONZERO_VALUE);
    }();
    const double relative_error = finiteScore(std::abs(observed_time_ms - predicted_time_ms) /
                                              std::max(predicted_time_ms, MINIMUM_NONZERO_VALUE));
    const double confidence = std::clamp(1.0 - relative_error, 0.0, 1.0);
    runtime::CalibrationSample sample;
    sample.hardware_fingerprint = hardware_fingerprint_;
    sample.problem_bucket = last_problem_bucket_.empty() ? "unknown" : last_problem_bucket_;
    sample.algorithm = algorithm_name;
    sample.predicted_time_ms = predicted_time_ms;
    sample.predicted_memory_mb = static_cast<double>(metrics.memory_usage_mb);
    sample.observed_time_ms = observed_time_ms;
    sample.observed_memory_mb = static_cast<double>(metrics.memory_usage_mb);
    sample.confidence = confidence;
    sample.relative_error = relative_error;
    const auto record_status = runtime::recordCalibrationSample(sample);
    if (record_status.isError())
    {
        return;
    }
}
PerformanceMetrics AdaptiveSelector::benchmarkAlgorithm(const std::string &algorithm_name,
                                                        const algebra::BoundaryMatrix &matrix)
{
    const auto start = std::chrono::high_resolution_clock::now();
    executeAlgorithmWorkload(algorithm_name, matrix);
    const auto end = std::chrono::high_resolution_clock::now();
    const double elapsed_ms =
        std::max(std::chrono::duration<double, std::milli>(end - start).count(), 1e-6);
    const std::size_t operations = matrix.numNonzeros() + matrix.cols();
    PerformanceMetrics metrics{};
    metrics.computation_time_ms = elapsed_ms;
    metrics.memory_usage_mb = estimateMemoryUsage(matrix) / BYTES_PER_MB;
    metrics.throughput_ops_per_sec = static_cast<double>(operations) / (elapsed_ms / 1000.0);
    return metrics;
}
void AdaptiveSelector::executeAlgorithmWorkload(const std::string &algorithm_name,
                                                const algebra::BoundaryMatrix &matrix)
{
    const core::DeterminismContract strict_contract;
    auto &mutable_matrix = const_cast<algebra::BoundaryMatrix &>(matrix);
    const auto pairs = mutable_matrix.computePersistencePairs(strict_contract);
    if (pairs.isError())
    {
        throw std::runtime_error("benchmark workload failed to compute persistence pairs");
    }
    if (algorithm_name == "twisted" || algorithm_name == "_accelerated")
    {
        const auto essential = mutable_matrix.findEssentialCycles(strict_contract);
        if (essential.isError())
        {
            throw std::runtime_error("benchmark workload failed to compute essential cycles");
        }
    }
    if (algorithm_name == "simd_vectorized" || algorithm_name == "gpu_accelerated")
    {
        std::vector<double> ones(std::max<std::size_t>(matrix.cols(), 1), 1.0);
        core::BufferView<const double> chainView(ones.data(), ones.size());
        const auto boundary_values = matrix.multiply(chainView);
        if (boundary_values.size() != matrix.rows() ||
            !std::all_of(boundary_values.begin(), boundary_values.end(),
                         [](double value) { return std::isfinite(value); }))
        {
            throw std::runtime_error("benchmark workload produced invalid boundary values");
        }
    }
}
std::vector<std::string> AdaptiveSelector::getAvailableAlgorithms() const
{
    std::vector<std::string> algorithms;
    algorithms.reserve(algorithm_registry_.size());
    for (const auto &entry : algorithm_registry_)
    {
        algorithms.push_back(entry.first);
    }
    return algorithms;
}
AlgorithmInfo AdaptiveSelector::getAlgorithmInfo(const std::string &algorithm_name) const
{
    const auto it = algorithm_registry_.find(algorithm_name);
    if (it == algorithm_registry_.end())
    {
        return AlgorithmInfo{};
    }
    return it->second;
}
SelectionDiagnostics AdaptiveSelector::getLastSelectionDiagnostics() const
{
    return last_selection_diagnostics_;
}
} // namespace nerve::persistence

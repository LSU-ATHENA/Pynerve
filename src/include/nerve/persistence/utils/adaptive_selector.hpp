
#pragma once
#include "nerve/algebra/boundary.hpp"
#include "nerve/core_types.hpp"

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace nerve::persistence
{

struct DataCharacteristics
{
    size_t num_simplices;
    size_t num_vertices;
    double sparsity;
    size_t max_dimension;
    double avg_simplex_size;
    size_t memory_footprint;
};

struct AlgorithmInfo
{
    std::string name;
    double performance_factor;   // Relative performance (lower is better)
    double memory_factor;        // Relative memory usage (lower is better)
    size_t optimal_dataset_size; // Dataset size where algorithm performs best
    double accuracy;             // Algorithm accuracy (0-1)
};

struct PerformanceMetrics
{
    double computation_time_ms;
    size_t memory_usage_mb;
    double throughput_ops_per_sec;
};

struct SelectionDiagnostics
{
    bool gate_passed = false;
    std::string gate_reason;
    double confidence = 0.0;
    double error_bound = 1.0;
    std::size_t sample_count = 0;
    std::string calibration_bucket_id;
};

class AdaptiveSelector
{
public:
    AdaptiveSelector();

    DataCharacteristics analyzeData(const algebra::BoundaryMatrix &matrix);
    std::string selectOptimalAlgorithm(const DataCharacteristics &data);

    void updatePerformanceModel(const std::string &algorithm_name,
                                const PerformanceMetrics &metrics);

    PerformanceMetrics benchmarkAlgorithm(const std::string &algorithm_name,
                                          const algebra::BoundaryMatrix &matrix);

    std::vector<std::string> getAvailableAlgorithms() const;
    AlgorithmInfo getAlgorithmInfo(const std::string &algorithm_name) const;
    SelectionDiagnostics getLastSelectionDiagnostics() const;

private:
    std::map<std::string, AlgorithmInfo> algorithm_registry_;
    std::unordered_map<std::string, double> last_predictions_ms_;
    std::string hardware_fingerprint_;
    std::string last_problem_bucket_;
    SelectionDiagnostics last_selection_diagnostics_;

    void initializePerformanceModels();

    double computeSparsity(const algebra::BoundaryMatrix &matrix);
    size_t estimateMaxDimension(const algebra::BoundaryMatrix &matrix);
    double computeAvgSimplexSize(const algebra::BoundaryMatrix &matrix);
    size_t estimateMemoryUsage(const algebra::BoundaryMatrix &matrix);

    double computeAlgorithmScore(const DataCharacteristics &data, const AlgorithmInfo &algorithm);
    double computeSizeSuitability(const DataCharacteristics &data, const AlgorithmInfo &algorithm);
    double computeSparsitySuitability(const DataCharacteristics &data,
                                      const AlgorithmInfo &algorithm);

    void executeAlgorithmWorkload(const std::string &algorithm_name,
                                  const algebra::BoundaryMatrix &matrix);
};

} // namespace nerve::persistence


#pragma once

#include "nerve/core/compact_summary/compact_summary.hpp"
#include "nerve/core/persistence.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace nerve::core
{

struct LifetimeStats
{
    double min_lifetime = 0.0;
    double max_lifetime = 0.0;
    double mean_lifetime = 0.0;
    double median_lifetime = 0.0;
    double std_deviation = 0.0;
    std::vector<double> lifetime_distribution;
};

struct CompressionMetrics
{
    double compression_ratio = 0.0;
    double space_savings = 0.0;
    double time_savings = 0.0;
    std::size_t original_size = 0;
    std::size_t compressed_size = 0;
    double compression_quality = 0.0;
};

struct HighDimMetrics
{
    std::size_t max_dimension_computed = 0;
    double highdim_complexity = 0.0;
    std::size_t critical_simplices = 0;
    double simplification_ratio = 0.0;
    double approximation_error = 0.0;
};

struct DimensionInfo
{
    std::size_t dimension = 0;
    std::size_t num_simplices = 0;
    std::size_t betti_number = 0;
    double avg_lifetime = 0.0;
    double complexity_score = 0.0;
    bool is_dominant = false;
};

struct CohomologyMetrics
{
    std::size_t cohomology_groups = 0;
    double reduction_time = 0.0;
    std::size_t reduction_operations = 0;
    double ordering_efficiency = 0.0;
    bool clever_ordering_used = false;
};

struct ReductionStats
{
    std::size_t original_matrix_size = 0;
    std::size_t reduced_matrix_size = 0;
    double reduction_ratio = 0.0;
    std::size_t elimination_steps = 0;
    double reduction_time = 0.0;
};

struct OrderingInfo
{
    std::string ordering_strategy;
    double ordering_time = 0.0;
    std::size_t ordering_improvement = 0;
    bool was_accelerated = false;
};

struct BudgetInfo
{
    std::size_t memory_limit_mb = 0;
    std::size_t time_limit_ms = 0;
    std::size_t memory_used_mb = 0;
    std::size_t time_used_ms = 0;
    bool budget_exceeded = false;
    std::string exceeded_resource;
};

struct WitnessMetrics
{
    std::size_t num_landmarks = 0;
    double landmark_ratio = 0.0;
    std::size_t witness_complex_size = 0;
    double witness_quality = 0.0;
    double sampling_efficiency = 0.0;
};

struct SamplingInfo
{
    std::string sampling_strategy;
    double sampling_time = 0.0;
    std::size_t samples_taken = 0;
    double sampling_quality = 0.0;
    bool adaptive_sampling = false;
};

struct LandmarkInfo
{
    std::vector<std::size_t> landmark_indices;
    double landmark_selection_time = 0.0;
    double landmark_coverage = 0.0;
    double landmark_quality = 0.0;
};

struct TruncationInfo
{
    bool truncation_applied = false;
    double truncation_threshold = 0.0;
    std::size_t elements_truncated = 0;
    double truncation_ratio = 0.0;
    double truncation_quality = 0.0;
};

class HighDimCompactSummary : public CompactSummary
{
public:
    HighDimCompactSummary();
    explicit HighDimCompactSummary(const CompactSummary &base_summary);

    void setHighdimBettiNumber(std::size_t dimension, std::size_t value);
    std::size_t getHighdimBettiNumber(std::size_t dimension) const;
    const std::unordered_map<std::size_t, std::size_t> &getAllHighdimBettiNumbers() const;
    void setHighdimBettiTop8(const std::vector<std::size_t> &top8);
    const std::vector<std::size_t> &getHighdimBettiTop8() const;

    void setLifetimeStatistics(const LifetimeStats &stats);
    const LifetimeStats &getLifetimeStatistics() const;
    void setCompressionMetrics(const CompressionMetrics &metrics);
    const CompressionMetrics &getCompressionMetrics() const;
    void setHighdimMetrics(const HighDimMetrics &metrics);
    const HighDimMetrics &getHighdimMetrics() const;

    void setMemoryEfficiency(double efficiency);
    double getMemoryEfficiency() const;
    void setComputationalEfficiency(double efficiency);
    double getComputationalEfficiency() const;
    void setApproximationQuality(double quality);
    double getApproximationQuality() const;

    void setDimensionInfo(std::size_t dimension, const DimensionInfo &info);
    const DimensionInfo &getDimensionInfo(std::size_t dimension) const;
    void serializeHighdimData(std::vector<std::uint8_t> &buffer) const;
    void deserializeHighdimData(const std::vector<std::uint8_t> &buffer);
    bool validateHighdimData() const;
    std::size_t getTotalHighdimBettiNumber() const;
    double getAverageLifetime() const;
    std::size_t getDominantDimension() const;

private:
    void updateTop8BettiNumbers();

    std::unordered_map<std::size_t, std::size_t> highdim_betti_numbers_;
    std::vector<std::size_t> highdim_betti_top8_;
    LifetimeStats lifetime_stats_;
    CompressionMetrics compression_metrics_;
    HighDimMetrics highdim_metrics_;
    double memory_efficiency_ = 0.0;
    double computational_efficiency_ = 0.0;
    double approximation_quality_ = 0.0;
    std::unordered_map<std::size_t, DimensionInfo> dimension_info_;
};

class PH5CompactSummary : public HighDimCompactSummary
{
public:
    PH5CompactSummary();
    void setCohomologyMetrics(const CohomologyMetrics &metrics);
    const CohomologyMetrics &getCohomologyMetrics() const;
    void setReductionStats(const ReductionStats &stats);
    const ReductionStats &getReductionStats() const;
    void setOrderingInfo(const OrderingInfo &info);
    const OrderingInfo &getOrderingInfo() const;
    void setBudgetInfo(const BudgetInfo &info);
    const BudgetInfo &getBudgetInfo() const;

private:
    CohomologyMetrics cohomology_metrics_;
    ReductionStats reduction_stats_;
    OrderingInfo ordering_info_;
    BudgetInfo budget_info_;
};

class PH6CompactSummary : public HighDimCompactSummary
{
public:
    PH6CompactSummary();
    void setWitnessMetrics(const WitnessMetrics &metrics);
    const WitnessMetrics &getWitnessMetrics() const;
    void setSamplingInfo(const SamplingInfo &info);
    const SamplingInfo &getSamplingInfo() const;
    void setLandmarkInfo(const LandmarkInfo &info);
    const LandmarkInfo &getLandmarkInfo() const;
    void setTruncationInfo(const TruncationInfo &info);
    const TruncationInfo &getTruncationInfo() const;

private:
    WitnessMetrics witness_metrics_;
    SamplingInfo sampling_info_;
    LandmarkInfo landmark_info_;
    TruncationInfo truncation_info_;
};

} // namespace nerve::core


#pragma once
#include "nerve/core/rng/determinism_contract.hpp"
#include "nerve/core_types.hpp"
#include "nerve/errors/errors.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace nerve::summary
{
struct CompactSummary
{
    static constexpr size_t MAX_LIFETIMES = 10;
    static constexpr size_t MAX_BETTI_DIM = 5;
    static constexpr size_t MAX_EIGENVALUES = 10;
    static constexpr size_t TARGET_SIZE_BYTES = 1024;
    struct Lifetime
    {
        float birth;
        float death;
        uint8_t dimension;
        float persistence;
    };
    std::array<Lifetime, MAX_LIFETIMES> top_lifetimes;
    uint8_t lifetime_count = 0;
    std::array<uint16_t, MAX_BETTI_DIM> betti_counts;
    uint8_t betti_dimension_count = 0;
    struct Eigenvalue
    {
        float value;
        uint16_t multiplicity;
    };
    std::array<Eigenvalue, MAX_EIGENVALUES> top_eigenvalues;
    uint8_t eigenvalue_count = 0;
    float persistence_entropy = 0.0f;
    float betti_entropy = 0.0f;
    float spectral_entropy = 0.0f;
    int64_t timestamp_ns;
    int64_t symbol_id;
    uint32_t computation_time_us;
    uint16_t data_points_count;
    float noise_level;
    struct HighDimExtension
    {
        std::array<uint16_t, 8> highdim_betti_top8;
        uint8_t highdim_betti_count = 0;
        struct LifetimeStats
        {
            float mean_lifetime = 0.0f;
            float std_deviation = 0.0f;
            float max_lifetime = 0.0f;
            uint32_t feature_count = 0;
        };
        std::array<LifetimeStats, 8> highdim_lifetime_stats;
        std::array<float, 8> dimension_complexity;
        std::array<uint32_t, 8> simplex_counts;
        bool truncated_by_budget = false;
        uint8_t max_dimension_attempted = 0;
        uint32_t num_boundary_ops = 0;
        float budget_utilization = 0.0f;
        float compression_ratio = 0.0f;
        float memory_efficiency = 0.0f;
        float computational_efficiency = 0.0f;
        std::vector<uint8_t> serializeExtension() const;
        bool deserializeExtension(const std::vector<uint8_t> &data);
        size_t extensionSizeBytes() const;
    };
    mutable bool has_highdim_extension = false;
    mutable HighDimExtension highdim_extension;
    std::vector<uint8_t> serialize() const;
    bool deserialize(const std::vector<uint8_t> &data);
    size_t sizeBytes() const;
    bool isValid() const;
    bool isUnderSizeLimit() const;
    bool hasHighdimData() const { return has_highdim_extension; }
    const HighDimExtension &getHighdimExtension() const;
    void setHighdimExtension(const HighDimExtension &extension);
    void clearHighdimExtension();
    uint16_t getHighdimBetti(uint8_t dimension) const;
    const HighDimExtension::LifetimeStats &getLifetimeStats(uint8_t dimension) const;
    float getDimensionComplexity(uint8_t dimension) const;
};
class CompactSummaryPipeline
{
public:
    struct PipelineConfig
    {
        float max_computation_time_ms = 5.0f;
        size_t max_data_points = 10000;
        float sampling_rate = 0.1f;
        bool enable_approximation = true;
        uint32_t random_seed = 42;
        Size max_persistence_dim = 2;
        double max_filtration_radius = 1.0;
    };
    explicit CompactSummaryPipeline(const PipelineConfig &config);
    errors::ErrorResult<CompactSummary>
    computeSummary(const std::vector<std::vector<float>> &points, int64_t timestamp_ns,
                   int64_t data_id, const core::DeterminismContract &contract = {}) const;
    CompactSummary computeApproximateSummary(const std::vector<std::vector<float>> &points,
                                             int64_t timestamp_ns, int64_t data_id,
                                             const core::DeterminismContract &contract = {}) const;
    void updateSummary(const std::vector<std::vector<float>> &new_points,
                       CompactSummary &existing_summary) const;
    struct PerformanceMetrics
    {
        float last_computation_time_ms;
        size_t points_processed;
        bool used_approximation;
        float accuracy_estimate;
    };
    PerformanceMetrics getLastMetrics() const;

private:
    PipelineConfig config_;
    mutable PerformanceMetrics last_metrics_;
    errors::ErrorResult<CompactSummary>
    computeFromPointCloud(const std::vector<std::vector<float>> &points, int64_t timestamp_ns,
                          int64_t data_id, const core::DeterminismContract &contract) const;
    static float
    bettiDistributionEntropy(const std::array<uint16_t, CompactSummary::MAX_BETTI_DIM> &counts,
                             uint8_t betti_dimension_count);
    static float
    spectralEntropyFromEigenvalues(const std::vector<CompactSummary::Eigenvalue> &eigenvalues);
    std::vector<CompactSummary::Eigenvalue>
    computeTopEigenvalues(const std::vector<std::vector<float>> &points) const;
    float computePersistenceEntropy(const std::vector<CompactSummary::Lifetime> &lifetimes) const;
    std::vector<std::vector<float>>
    downsamplePoints(const std::vector<std::vector<float>> &points) const;
    static float accuracyVsReferenceBetti(
        const std::array<uint16_t, CompactSummary::MAX_BETTI_DIM> &ref_betti, uint8_t ref_dim_count,
        const std::array<uint16_t, CompactSummary::MAX_BETTI_DIM> &approx_betti,
        uint8_t approx_dim_count);
};
class SummaryFactory
{
public:
    enum class Strategy
    {
        EXACT,
        APPROXIMATE,
        INCREMENTAL
    };
    static std::unique_ptr<CompactSummaryPipeline>
    createPipeline(Strategy strategy, const CompactSummaryPipeline::PipelineConfig &config);
    static errors::ErrorResult<CompactSummary>
    computeSummary(const std::vector<std::vector<float>> &points, int64_t timestamp_ns,
                   int64_t data_id, float max_time_ms,
                   const core::DeterminismContract &contract = {});
};
class SummaryValidator
{
public:
    static bool validateSummary(const CompactSummary &summary);
    static bool validateSizeConstraints(const CompactSummary &summary);
    static bool validateComputationTime(const CompactSummary &summary, float max_time_ms);
    static float estimateAccuracy(const CompactSummary &exact_summary,
                                  const CompactSummary &approximate_summary);
    static std::string generateQualityReport(const CompactSummary &summary);
};
} // namespace nerve::summary

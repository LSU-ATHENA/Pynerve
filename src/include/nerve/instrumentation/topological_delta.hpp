
#pragma once
#include "nerve/core/rng/determinism_contract.hpp"
#include "nerve/core_types.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>
namespace nerve
{
namespace instrumentation
{
struct alignas(32) TopologicalDelta
{
    uint32_t simplex_id;
    uint8_t dimension;
    float lifetime_delta;
    float birth_delta;
    float death_delta;
    uint8_t change_type;
    uint8_t confidence;
    uint16_t reserved;
    uint64_t timestamp_ns;
    enum class ChangeType : uint8_t
    {
        APPEARED = 0,
        DISAPPEARED = 1,
        LIFETIME_INCREASED = 2,
        LIFETIME_DECREASED = 3,
        BIRTH_SHIFTED = 4,
        DEATH_SHIFTED = 5,
        MERGED = 6,
        SPLIT = 7
    };
    bool isValid() const
    {
        return simplex_id > 0 && dimension <= 3 && std::isfinite(lifetime_delta) &&
               std::isfinite(birth_delta) && std::isfinite(death_delta) && timestamp_ns > 0 &&
               change_type <= 7;
    }
    ChangeType getChangeType() const { return static_cast<ChangeType>(change_type); }
    void setChangeType(ChangeType type) { change_type = static_cast<uint8_t>(type); }
    bool isSignificant() const
    {
        if (!isValid())
        {
            return false;
        }
        return std::abs(lifetime_delta) > 1e-6f || std::abs(birth_delta) > 1e-6f ||
               std::abs(death_delta) > 1e-6f;
    }
    bool isHighConfidence() const { return confidence >= 200; }
    float getImpactScore() const
    {
        if (!isValid())
        {
            return 0.0f;
        }
        double impact = static_cast<double>(std::abs(lifetime_delta));
        if (dimension > 0)
            impact *= (dimension + 1);
        if (isHighConfidence())
            impact *= 1.5f;
        const double max_float = static_cast<double>(std::numeric_limits<float>::max());
        if (impact >= max_float)
        {
            return std::numeric_limits<float>::max();
        }
        return static_cast<float>(impact);
    }
};
static_assert(sizeof(TopologicalDelta) == 32, "TopologicalDelta must be 32 bytes");
class TopologicalChangeDetector
{
public:
    struct DetectionConfig
    {
        float min_lifetime_change = 1e-6f;
        float min_birth_change = 1e-6f;
        float min_death_change = 1e-6f;
        uint8_t min_confidence = 100;
        bool track_appearances = true;
        bool track_disappearances = true;
        bool track_lifetime_changes = true;
        bool track_birth_death_shifts = true;
    };
    TopologicalChangeDetector();
    explicit TopologicalChangeDetector(const DetectionConfig &config);
    std::vector<TopologicalDelta> detectChanges(const std::vector<std::pair<float, float>> &old_pd,
                                                const std::vector<std::pair<float, float>> &new_pd,
                                                const core::DeterminismContract &contract);
    std::vector<TopologicalDelta> detectEigenpairChanges(const std::vector<double> &old_eigenvalues,
                                                         const std::vector<double> &new_eigenvalues,
                                                         const core::DeterminismContract &contract);
    std::vector<TopologicalDelta>
    detectStreamingChanges(const std::vector<std::pair<float, float>> &previous_pd,
                           const std::vector<std::pair<float, float>> &current_pd,
                           const core::DeterminismContract &contract);
    const DetectionConfig &getConfig() const { return config_; }
    void setConfig(const DetectionConfig &config) { config_ = config; }

private:
    DetectionConfig config_;
    std::vector<TopologicalDelta>
    detectLifetimeChanges(const std::vector<std::pair<float, float>> &old_pd,
                          const std::vector<std::pair<float, float>> &new_pd);
    std::vector<TopologicalDelta>
    detectAppearancesDisappearances(const std::vector<std::pair<float, float>> &old_pd,
                                    const std::vector<std::pair<float, float>> &new_pd);
    float computeLifetimeChange(const std::pair<float, float> &old_point,
                                const std::pair<float, float> &new_point);
    uint8_t computeConfidence(float change_magnitude, float threshold);
    uint64_t generateSimplexId(const std::pair<float, float> &point);
};
class TopologicalChangeAnalyzer
{
public:
    struct AnalysisConfig
    {
        float significance_threshold = 1e-5f;
        size_t max_changes_to_track = 1000;
        bool enable_trend_analysis = true;
        bool enable_impact_scoring = true;
    };
    TopologicalChangeAnalyzer();
    explicit TopologicalChangeAnalyzer(const AnalysisConfig &config);
    void addChanges(const std::vector<TopologicalDelta> &changes);
    struct AnalysisResults
    {
        size_t total_changes;
        size_t significant_changes;
        float average_impact_score;
        float max_impact_score;
        std::array<size_t, 8> changes_by_dimension;
        std::array<size_t, 8> changes_by_type;
        float trend_direction;
        float stability_score;
    };
    AnalysisResults analyze() const;
    std::vector<TopologicalDelta> getTopChanges(size_t count = 10) const;
    std::vector<TopologicalDelta> getChangesByDimension(uint8_t dimension) const;
    std::vector<TopologicalDelta> getChangesByType(TopologicalDelta::ChangeType type) const;
    void clear();
    const AnalysisConfig &getConfig() const { return config_; }
    void setConfig(const AnalysisConfig &config) { config_ = config; }

private:
    AnalysisConfig config_;
    std::vector<TopologicalDelta> changes_;
    void computeTrendAnalysis(AnalysisResults &results) const;
    void computeImpactScores(AnalysisResults &results) const;
    void computeAttribution(AnalysisResults &results) const;
};
#define DETECT_TOPOLOGICAL_CHANGES(old_pd, new_pd, contract)                                       \
    nerve::instrumentation::TopologicalChangeDetector().detectChanges(old_pd, new_pd, contract)
#define CREATE_DETERMINISM_CONTRACT(seed, params_hash)                                             \
    nerve::core::DeterminismContract::create(seed, params_hash)
#define ANALYZE_CHANGES(changes) nerve::instrumentation::TopologicalChangeAnalyzer().analyze()
} // namespace instrumentation
} // namespace nerve

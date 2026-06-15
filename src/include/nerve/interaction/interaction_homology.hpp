
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
namespace nerve
{
namespace interaction
{
struct HigherOrderInteraction
{
    std::vector<uint32_t> participants;
    uint8_t order;
    float strength;
    int64_t start_time_ns;
    int64_t end_time_ns;
    float lifetime;
    std::vector<float> attributes;
    std::string interaction_type;
    bool isValid() const
    {
        return !participants.empty() && order == participants.size() && std::isfinite(strength) &&
               strength >= 0 && std::isfinite(lifetime) && lifetime >= 0 &&
               std::all_of(attributes.begin(), attributes.end(),
                           [](float value) { return std::isfinite(value); });
    }
};
struct InteractionConfig
{
    size_t max_order = 4;
    float min_strength_threshold = 0.01f;
    float min_lifetime_threshold = 1e-6f;
    bool enable_temporal_filtering = true;
    int64_t max_time_window_ns = 1'000'000'000;
    std::vector<std::string> required_attributes;
    bool enable_online_computation = true;
    size_t online_summary_size = 1000;
};
struct InteractionSummary
{
    std::unordered_map<uint8_t, size_t> interaction_counts;
    std::unordered_map<uint8_t, float> total_strengths;
    std::unordered_map<uint8_t, float> average_lifetimes;
    std::unordered_map<uint8_t, float> max_strengths;
    std::unordered_map<uint8_t, float> max_lifetimes;
    int64_t earliest_time_ns;
    int64_t latest_time_ns;
    std::unordered_map<uint8_t, std::vector<float>> lifetime_distributions;
    std::unordered_map<std::string, std::vector<float>> attribute_statistics;
    std::unordered_map<uint8_t, std::vector<float>> betti_numbers;
    std::unordered_map<uint8_t, std::vector<float>> persistence_summary;
    size_t total_interactions;
    double computation_time_ms;
    bool is_online;
};
class InteractionHomology
{
public:
    explicit InteractionHomology(const InteractionConfig &config);
    void addInteraction(const HigherOrderInteraction &interaction);
    void addInteractions(const std::vector<HigherOrderInteraction> &interactions);
    void removeInteraction(const std::vector<uint32_t> &participants);
    void updateInteraction(const HigherOrderInteraction &interaction);
    InteractionSummary computeHomologySummary();
    InteractionSummary computeExactHomology();
    InteractionSummary computeOnlineSummary();
    InteractionSummary analyzePairInteractions();
    InteractionSummary analyzeTriadicInteractions();
    InteractionSummary analyzeQuadrupleInteractions();
    InteractionSummary analyzeTemporalWindow(int64_t start_time_ns, int64_t end_time_ns);
    InteractionSummary analyzeByAttribute(const std::string &attribute_name,
                                          float min_value = -std::numeric_limits<float>::max(),
                                          float max_value = std::numeric_limits<float>::max());
    std::vector<float> extractBettiNumbers(const std::vector<HigherOrderInteraction> &interactions,
                                           uint8_t max_order = 4);
    std::vector<float>
    extractPersistenceSummary(const std::vector<HigherOrderInteraction> &interactions,
                              uint8_t max_order = 4);
    struct InteractionCluster
    {
        std::vector<std::vector<uint32_t>> clusters;
        std::vector<float> cluster_strengths;
        std::vector<float> cluster_lifetimes;
    };
    InteractionCluster clusterInteractions(const std::vector<HigherOrderInteraction> &interactions,
                                           size_t num_clusters = 10);

private:
    InteractionConfig config_;
    std::vector<HigherOrderInteraction> interactions_;
    InteractionSummary online_summary_;
    std::vector<HigherOrderInteraction>
    filterInteractions(const std::vector<HigherOrderInteraction> &interactions);
    std::vector<std::vector<uint32_t>>
    buildSimplicialComplex(const std::vector<HigherOrderInteraction> &interactions);
    std::vector<float>
    computeBettiNumbersFromComplex(const std::vector<std::vector<uint32_t>> &complex);
    void updateOnlineSummary(const HigherOrderInteraction &interaction);
    std::vector<float>
    computeInteractionFiltration(const std::vector<uint32_t> &simplex,
                                 const std::vector<HigherOrderInteraction> &interactions);
};
class OnlineInteractionProcessor
{
public:
    struct OnlineConfig
    {
        size_t buffer_size = 10000;
        size_t batch_size = 100;
        double processing_interval_ms = 100.0;
        bool enable_incremental_updates = true;
        bool enableStreamingOutput = true;
    };
    explicit OnlineInteractionProcessor(const OnlineConfig &config,
                                        const InteractionConfig &interaction_config);
    void processInteractionStream(const HigherOrderInteraction &interaction);
    void processInteractionBatch(const std::vector<HigherOrderInteraction> &interactions);
    void updateSummaryIncremental(const HigherOrderInteraction &interaction);
    InteractionSummary getCurrentSummary() const;
    void enableStreamingOutput(const std::string &output_file);
    void disableStreamingOutput();
    void flushBuffer();
    void clearBuffer();
    size_t getBufferSize() const;

private:
    OnlineConfig config_;
    InteractionConfig interaction_config_;
    std::vector<HigherOrderInteraction> interaction_buffer_;
    InteractionSummary current_summary_;
    std::unique_ptr<std::ofstream> output_stream_;
    void processBuffer();
    void writeSummaryToStream(const InteractionSummary &summary);
};
class ExactInteractionHomology
{
public:
    struct ExactConfig
    {
        bool enable_full_persistence_diagrams = true;
        size_t max_complex_size = 10000;
        bool enable_multiparameter_homology = true;
        bool enable_persistent_cohomology = true;
        std::string output_directory = "./exact_results";
    };
    explicit ExactInteractionHomology(const ExactConfig &config);
    struct ExactResult
    {
        std::vector<std::vector<std::pair<double, double>>> persistence_diagrams;
        std::vector<std::vector<float>> betti_numbers;
        std::vector<std::vector<float>> cohomology_groups;
        std::vector<std::vector<std::vector<float>>> multiparameter_pds;
        double computation_time_ms;
        size_t memory_usage_mb;
    };
    ExactResult computeExactHomology(const std::vector<HigherOrderInteraction> &interactions);
    ExactResult
    computeMultiparameterHomology(const std::vector<HigherOrderInteraction> &interactions,
                                  const std::vector<std::string> &parameter_names);
    ExactResult
    computePersistentCohomology(const std::vector<HigherOrderInteraction> &interactions);
    void saveResults(const ExactResult &result, const std::string &filename);
    void visualizePersistenceDiagrams(const ExactResult &result, const std::string &output_dir);

private:
    ExactConfig config_;
    std::vector<std::vector<uint32_t>>
    buildFilteredComplex(const std::vector<HigherOrderInteraction> &interactions);
    std::vector<std::vector<std::pair<double, double>>>
    computePersistenceDiagrams(const std::vector<std::vector<uint32_t>> &complex);
    std::vector<float> computeCohomologyGroups(const std::vector<std::vector<uint32_t>> &complex);
};
class InteractionHomologyManager
{
public:
    static InteractionHomologyManager &instance();
    void setInteractionConfig(const InteractionConfig &config);
    void setOnlineConfig(const OnlineInteractionProcessor::OnlineConfig &config);
    void setExactConfig(const ExactInteractionHomology::ExactConfig &config);
    std::shared_ptr<InteractionHomology> getInteractionHomology();
    std::shared_ptr<OnlineInteractionProcessor> getOnlineProcessor();
    std::shared_ptr<ExactInteractionHomology> getExactHomology();
    void addInteraction(const HigherOrderInteraction &interaction);
    void addInteractions(const std::vector<HigherOrderInteraction> &interactions);
    InteractionSummary getCurrentSummary();
    InteractionSummary computeExactSummary();
    InteractionSummary analyzeTemporalWindow(int64_t start_time_ns, int64_t end_time_ns);
    std::vector<float> extractTopologicalFeatures();
    InteractionHomology::InteractionCluster detectInteractionClusters(size_t num_clusters = 10);
    ExactInteractionHomology::ExactResult
    runExactAnalysis(const std::vector<HigherOrderInteraction> &interactions);
    struct PerformanceStats
    {
        double average_processing_time_ms;
        size_t total_interactions_processed;
        size_t total_exact_computations;
        double average_exact_computation_time_ms;
        size_t memory_usage_mb;
    };
    PerformanceStats getPerformanceStats() const;
    void resetPerformanceStats();

private:
    InteractionHomologyManager() = default;
    InteractionConfig interaction_config_;
    OnlineInteractionProcessor::OnlineConfig online_config_;
    ExactInteractionHomology::ExactConfig exact_config_;
    std::shared_ptr<InteractionHomology> interaction_homology_;
    std::shared_ptr<OnlineInteractionProcessor> online_processor_;
    std::shared_ptr<ExactInteractionHomology> exact_homology_;
    mutable std::shared_mutex mutex_;
    mutable PerformanceStats performance_stats_;
    void updatePerformanceStats(const std::string &operation, double time_ms);
};
} // namespace interaction
} // namespace nerve

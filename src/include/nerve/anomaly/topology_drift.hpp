
#pragma once
#include <cmath>
#include <cstdint>
#include <memory>
#include <shared_mutex>
#include <string>
#include <vector>
namespace nerve
{
namespace anomaly
{

struct PersistenceProfile
{
    size_t num_bins = 0;
    size_t total_pairs = 0;
    double birth_mean = 0.0;
    double death_mean = 0.0;
    double persistence_mean = 0.0;
    double birth_variance = 0.0;
    double death_variance = 0.0;
    double persistence_variance = 0.0;
    std::vector<size_t> birth_histogram;
    std::vector<size_t> death_histogram;
    std::vector<size_t> persistence_histogram;
};
class BettiChangeDetector
{
public:
    struct ChangeConfig
    {
        double significance_level = 0.05;
        size_t min_window_size = 100;
        size_t max_window_size = 1000;
        bool enable_trend_detection = true;
        double min_change_threshold = 0.1;
    };
    struct ChangePoint
    {
        int64_t timestamp_ns;
        size_t window_index;
        std::vector<double> betti_before;
        std::vector<double> betti_after;
        double change_magnitude;
        double p_value;
        std::string change_type;
    };
    explicit BettiChangeDetector(const ChangeConfig &config);
    std::vector<ChangePoint> detectChanges(const std::vector<std::vector<double>> &betti_sequences,
                                           const std::vector<int64_t> &timestamps);
    ChangePoint detectSingleChange(const std::vector<double> &betti_before,
                                   const std::vector<double> &betti_after, int64_t timestamp_ns);
    bool updateAndDetect(const std::vector<double> &new_betti, int64_t timestamp_ns,
                         ChangePoint &detected_change);
    double computeStatisticalDistance(const std::vector<double> &seq1,
                                      const std::vector<double> &seq2);
    double computePValue(const std::vector<double> &seq1, const std::vector<double> &seq2);
    struct TrendInfo
    {
        bool is_trending;
        double trend_slope;
        double trend_strength;
        std::string trend_direction;
    };
    TrendInfo analyzeTrend(const std::vector<double> &betti_sequence) const;

private:
    ChangeConfig config_;
    std::vector<std::vector<double>> historical_betti_;
    std::vector<int64_t> historical_timestamps_;
    size_t current_window_size_;
    double computeKolmogorovSmirnovStatistic(const std::vector<double> &seq1,
                                             const std::vector<double> &seq2) const;
    double computeChiSquaredStatistic(const std::vector<double> &seq1,
                                      const std::vector<double> &seq2) const;
    void updateWindowSize();
};
class LifetimeDriftDetector
{
public:
    struct DriftConfig
    {
        double drift_threshold = 0.1;
        size_t reference_window_size = 500;
        size_t detection_window_size = 100;
        bool enable_adaptive_threshold = true;
        std::string distance_metric = "wasserstein";
        double wasserstein_order = 2.0;
    };
    struct DriftPoint
    {
        int64_t timestamp_ns;
        double drift_score;
        double p_value;
        std::vector<float> reference_lifetimes;
        std::vector<float> current_lifetimes;
        double distance_value;
        bool is_drift_detected;
    };
    explicit LifetimeDriftDetector(const DriftConfig &config);
    std::vector<DriftPoint> detectDrift(const std::vector<std::vector<float>> &lifetime_sequences,
                                        const std::vector<int64_t> &timestamps);
    DriftPoint detectSingleDrift(const std::vector<float> &reference_lifetimes,
                                 const std::vector<float> &current_lifetimes, int64_t timestamp_ns);
    bool updateAndDetect(const std::vector<float> &new_lifetimes, int64_t timestamp_ns,
                         DriftPoint &detected_drift);
    double computeSlicedWassersteinDistance(const std::vector<float> &dist1,
                                            const std::vector<float> &dist2,
                                            size_t num_projections = 100);
    double computeEmdDistance(const std::vector<float> &dist1, const std::vector<float> &dist2);
    double computeKlDivergence(const std::vector<float> &dist1, const std::vector<float> &dist2);
    void updateReferenceDistribution(const std::vector<float> &new_reference);
    void resetReference();

private:
    DriftConfig config_;
    std::vector<float> reference_lifetimes_;
    std::vector<std::vector<float>> historical_lifetimes_;
    std::vector<int64_t> historical_timestamps_;
    double adaptive_threshold_;
    std::vector<float> computeHistogram(const std::vector<float> &lifetimes,
                                        size_t num_bins = 50) const;
    void updateAdaptiveThreshold(double current_drift_score);
};
class MarketAnomalyDetector
{
public:
    struct MarketConfig
    {
        double price_change_threshold = 0.02;
        double volume_spike_threshold = 3.0;
        double topology_anomaly_threshold = 0.15;
        size_t lookback_window = 100;
        bool enable_volume_analysis = true;
        bool enable_price_analysis = true;
        bool enable_topology_analysis = true;
    };
    struct AnomalyEvent
    {
        int64_t timestamp_ns;
        std::string anomaly_type;
        double anomaly_score;
        double p_value;
        std::string description;
        std::vector<double> contributing_factors;
        bool is_critical;
    };
    explicit MarketAnomalyDetector(const MarketConfig &config);
    std::vector<AnomalyEvent>
    detectAnomalies(const std::vector<int64_t> &timestamps, const std::vector<double> &prices,
                    const std::vector<float> &volumes,
                    const std::vector<std::vector<float>> &topological_features);
    AnomalyEvent detectSingleAnomaly(int64_t timestamp_ns, double price, float volume,
                                     const std::vector<float> &topological_features);
    bool updateAndDetect(int64_t timestamp_ns, double price, float volume,
                         const std::vector<float> &topological_features,
                         AnomalyEvent &detected_anomaly);
    AnomalyEvent detectPriceAnomaly(double current_price, const std::vector<double> &price_history);
    AnomalyEvent detectVolumeAnomaly(float current_volume,
                                     const std::vector<float> &volume_history);
    AnomalyEvent detectTopologyAnomaly(const std::vector<float> &current_features,
                                       const std::vector<std::vector<float>> &feature_history);
    AnomalyEvent detectCombinedAnomaly(const std::vector<double> &factor_scores,
                                       const std::vector<std::string> &factor_names);
    void updateNormalBehavior(const std::vector<double> &prices, const std::vector<float> &volumes,
                              const std::vector<std::vector<float>> &topological_features);
    void resetNormalBehavior();

private:
    MarketConfig config_;
    std::vector<double> price_history_;
    std::vector<float> volume_history_;
    std::vector<std::vector<float>> topology_history_;
    std::vector<int64_t> timestamp_history_;
    double computePriceZscore(double current_price) const;
    double computeVolumeZscore(float current_volume) const;
    double computeTopologyAnomalyScore(const std::vector<float> &current_features) const;
    std::vector<double> normalizeFeatures(const std::vector<double> &features) const;
};
class OnlinePValueCalculator
{
public:
    struct PValueConfig
    {
        double significance_level = 0.05;
        size_t min_samples = 30;
        bool enable_fdr_control = true;
        double fdr_rate = 0.1;
    };
    explicit OnlinePValueCalculator(const PValueConfig &config);
    double computePValue(double test_statistic, const std::vector<double> &null_distribution);
    double computeEmpiricalPValue(double test_statistic,
                                  const std::vector<double> &sample_distribution);
    std::vector<double> bonferroniCorrection(const std::vector<double> &p_values);
    std::vector<double> benjaminiHochbergFdr(const std::vector<double> &p_values);
    void updateNullDistribution(double new_sample);
    void updateSampleDistribution(double new_sample);
    bool isSignificant(double p_value) const;
    std::vector<bool> multipleTestingSignificance(const std::vector<double> &p_values);
    double computeEffectSize(double sample_mean, double null_mean, double sample_std,
                             double null_std);
    double computeConfidenceInterval(const std::vector<double> &samples,
                                     double confidence_level = 0.95);

private:
    PValueConfig config_;
    std::vector<double> null_distribution_;
    std::vector<double> sample_distribution_;
    mutable std::shared_mutex mutex_;
    double computeTwoSidedPValue(double test_statistic,
                                 const std::vector<double> &distribution) const;
};
class RegimeChangeDetector
{
public:
    struct RegimeConfig
    {
        double regime_change_threshold = 0.2;
        size_t min_regime_duration = 50;
        bool enable_hmm_detection = true;
        size_t num_regimes = 3;
        double transition_probability = 0.01;
    };
    struct Regime
    {
        int regime_id;
        std::vector<double> characteristic_features;
        double stability_score;
        int64_t start_timestamp_ns;
        int64_t end_timestamp_ns;
        size_t duration_points;
        std::string description;
    };
    struct RegimeChange
    {
        int64_t timestamp_ns;
        int from_regime_id;
        int to_regime_id;
        double change_confidence;
        std::vector<double> transition_features;
        std::string change_description;
    };
    explicit RegimeChangeDetector(const RegimeConfig &config);
    std::vector<Regime> detectRegimes(const std::vector<std::vector<float>> &topological_features,
                                      const std::vector<int64_t> &timestamps);
    std::vector<RegimeChange>
    detectRegimeChanges(const std::vector<std::vector<float>> &topological_features,
                        const std::vector<int64_t> &timestamps);
    bool updateAndDetect(const std::vector<float> &new_features, int64_t timestamp_ns,
                         RegimeChange &detected_change);
    std::vector<double>
    extractRegimeFeatures(const std::vector<std::vector<float>> &feature_window);
    std::string characterizeRegime(const std::vector<double> &features);
    struct HMMModel
    {
        std::vector<std::vector<double>> emission_means;
        std::vector<std::vector<double>> emission_covariances;
        std::vector<std::vector<double>> transition_matrix;
        std::vector<double> initial_probabilities;
    };
    HMMModel trainHmm(const std::vector<std::vector<float>> &features,
                      const std::vector<int> &regime_labels);
    std::vector<int> predictRegimesHmm(const HMMModel &model,
                                       const std::vector<std::vector<float>> &features);

private:
    RegimeConfig config_;
    std::vector<Regime> current_regimes_;
    std::vector<std::vector<float>> feature_history_;
    std::vector<int64_t> timestamp_history_;
    double computeRegimeSimilarity(const std::vector<double> &features1,
                                   const std::vector<double> &features2);
    std::vector<std::vector<float>> clusterFeatures(const std::vector<std::vector<float>> &features,
                                                    size_t num_clusters);
};
class AnomalyDetectionManager
{
public:
    static AnomalyDetectionManager &instance();
    void setBettiDetectorConfig(const BettiChangeDetector::ChangeConfig &config);
    void setDriftDetectorConfig(const LifetimeDriftDetector::DriftConfig &config);
    void setMarketDetectorConfig(const MarketAnomalyDetector::MarketConfig &config);
    void setPvalueConfig(const OnlinePValueCalculator::PValueConfig &config);
    void setRegimeConfig(const RegimeChangeDetector::RegimeConfig &config);
    std::shared_ptr<BettiChangeDetector> getBettiDetector();
    std::shared_ptr<LifetimeDriftDetector> getDriftDetector();
    std::shared_ptr<MarketAnomalyDetector> getMarketDetector();
    std::shared_ptr<OnlinePValueCalculator> getPvalueCalculator();
    std::shared_ptr<RegimeChangeDetector> getRegimeDetector();
    struct AnomalyReport
    {
        std::vector<BettiChangeDetector::ChangePoint> betti_changes;
        std::vector<LifetimeDriftDetector::DriftPoint> drift_points;
        std::vector<MarketAnomalyDetector::AnomalyEvent> market_anomalies;
        std::vector<RegimeChangeDetector::RegimeChange> regime_changes;
        double overall_anomaly_score;
        std::string summary_report;
    };
    AnomalyReport detectAllAnomalies(const std::vector<int64_t> &timestamps,
                                     const std::vector<double> &prices,
                                     const std::vector<float> &volumes,
                                     const std::vector<std::vector<double>> &betti_sequences,
                                     const std::vector<std::vector<float>> &lifetime_sequences,
                                     const std::vector<std::vector<float>> &topological_features);
    std::vector<std::string> generateAlerts(const AnomalyReport &report);
    bool sendAlerts(const std::vector<std::string> &alerts);

private:
    AnomalyDetectionManager() = default;
    std::shared_ptr<BettiChangeDetector> betti_detector_;
    std::shared_ptr<LifetimeDriftDetector> drift_detector_;
    std::shared_ptr<MarketAnomalyDetector> market_detector_;
    std::shared_ptr<OnlinePValueCalculator> pvalue_calculator_;
    std::shared_ptr<RegimeChangeDetector> regime_detector_;
    mutable std::shared_mutex mutex_;
    std::string generateSummaryReport(const AnomalyReport &report);
};
} // namespace anomaly
} // namespace nerve

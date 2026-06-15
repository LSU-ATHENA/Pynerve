
#pragma once
#include <cstdint>
#include <memory>
#include <random>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>
namespace nerve
{
namespace approximate
{
struct Pair
{
    float birth;
    float death;
    float persistence;
    uint8_t dimension;
    float error_bound;
    float confidence;
    uint64_t sample_id;
};
struct StreamingConfig
{
    double error_budget = 0.01;
    double confidence_threshold = 0.95;
    size_t sketch_size = 1000;
    size_t sample_rate = 100;
    bool enable_error_tracking = true;
    uint32_t random_seed = 42;
};
class WitnessComplex
{
public:
    struct WitnessPoint
    {
        std::vector<float> coordinates;
        uint64_t timestamp_ns;
        uint64_t point_id;
        std::vector<uint32_t> landmark_indices;
        std::vector<float> distances;
    };
    struct Landmark
    {
        std::vector<float> coordinates;
        uint64_t point_id;
        std::vector<uint64_t> witness_indices;
        float radius;
    };
    explicit WitnessComplex(size_t max_landmarks = 1000);
    void addPoint(const std::vector<float> &point, uint64_t timestamp_ns);
    void addPointsBatch(const std::vector<std::vector<float>> &points,
                        const std::vector<uint64_t> &timestamps);
    void updateLandmarks();
    std::vector<Landmark> getLandmarks() const;
    std::vector<WitnessPoint> getWitnesses() const;
    std::vector<Pair> compute() const;
    float estimateMaxError() const;
    std::vector<Pair> getPairsWithErrorBounds() const;

private:
    std::vector<Landmark> landmarks_;
    std::vector<WitnessPoint> witnesses_;
    std::vector<std::vector<float>> point_buffer_;
    size_t max_landmarks_;
    mutable std::shared_mutex mutex_;
    void selectLandmarks();
    std::vector<uint32_t> findNearestLandmarks(const std::vector<float> &point, size_t k) const;
};
class SketchBasedPH
{
public:
    struct SketchConfig
    {
        size_t sketch_dim = 100;
        double sparsity = 0.1;
        bool enable_compression = true;
        bool preserve_diagonal = true;
    };
    explicit SketchBasedPH(const SketchConfig &config);
    void addPoint(const std::vector<float> &point, uint64_t timestamp_ns);
    void addPointsBatch(const std::vector<std::vector<float>> &points,
                        const std::vector<uint64_t> &timestamps);
    void updateSketch();
    std::vector<Pair> computePersistenceFromSketch() const;
    double computeErrorBound() const;
    std::vector<Pair> getPairsWithConfidence() const;

private:
    SketchConfig config_;
    std::vector<std::vector<float>> sketch_matrix_;
    std::vector<std::vector<float>> point_buffer_;
    mutable std::shared_mutex mutex_;
    void compressSketch();
    std::vector<Pair> extractPersistenceFromSketch() const;
};
class AdaptiveSampler
{
public:
    struct SamplingConfig
    {
        size_t initial_sample_rate = 100;
        size_t min_sample_rate = 10;
        size_t max_sample_rate = 1000;
        double error_target = 0.01;
        double adaptation_rate = 0.1;
        bool enable_importance_sampling = true;
    };
    explicit AdaptiveSampler(const SamplingConfig &config);
    bool shouldSamplePoint(const std::vector<float> &point, uint64_t timestamp_ns);
    std::vector<size_t> sampleBatch(const std::vector<std::vector<float>> &points,
                                    const std::vector<uint64_t> &timestamps);
    void updateSamplingRate(double current_error);
    void updateImportanceRegions(const std::vector<Pair> &pairs);
    double getCurrentSampleRate() const;
    double getEstimatedError() const;
    size_t getPointsSampled() const;
    size_t getPointsProcessed() const;

private:
    SamplingConfig config_;
    size_t current_sample_rate_;
    double estimated_error_;
    size_t points_sampled_;
    size_t points_processed_;
    std::vector<std::vector<float>> importance_regions_;
    mutable std::shared_mutex mutex_;
    bool isInImportantRegion(const std::vector<float> &point) const;
    void adjustSampleRate(double error_diff);
};
class ErrorTracker
{
public:
    struct ErrorMetrics
    {
        double max_error;
        double average_error;
        double confidence_interval;
        size_t samples_compared;
        bool bounds_are_tight;
    };
    explicit ErrorTracker(double error_budget = 0.01);
    void recordExactPair(const Pair &exact_pair);
    void recordApproximatePair(const Pair &approx_pair);
    void recordComparison(const Pair &exact, const Pair &approximate);
    ErrorMetrics getErrorMetrics() const;
    double computeErrorBound(double confidence = 0.95) const;
    bool isWithinBudget(const Pair &pair) const;
    std::vector<Pair> filterByErrorBudget(const std::vector<Pair> &pairs) const;
    void reset();

private:
    double error_budget_;
    std::vector<Pair> exact_pairs_;
    std::vector<Pair> approximate_pairs_;
    std::vector<double> error_samples_;
    mutable std::shared_mutex mutex_;
    double computePercentile(const std::vector<double> &values, double percentile) const;
};
class StreamingPH
{
public:
    explicit StreamingPH(const StreamingConfig &config);
    void addPoint(const std::vector<float> &point, uint64_t timestamp_ns);
    void addPointsBatch(const std::vector<std::vector<float>> &points,
                        const std::vector<uint64_t> &timestamps);
    std::vector<Pair> compute();
    std::vector<Pair> computePersistenceWithBounds();
    void updateErrorBudget(double new_budget);
    void updateConfidenceThreshold(double new_threshold);
    void updateThroughputTarget(size_t points_per_second);
    struct PerformanceMetrics
    {
        double points_processed_per_second;
        double average_computation_time_ms;
        double current_error_estimate;
        size_t points_in_buffer;
        bool using_approximation;
    };
    PerformanceMetrics getPerformanceMetrics() const;
    bool validateErrorBounds() const;
    std::vector<Pair> getHighConfidencePairs() const;

private:
    StreamingConfig config_;
    std::unique_ptr<WitnessComplex> witness_complex_;
    std::unique_ptr<SketchBasedPH> sketch_ph_;
    std::unique_ptr<AdaptiveSampler> sampler_;
    std::unique_ptr<ErrorTracker> error_tracker_;
    std::vector<std::vector<float>> point_buffer_;
    std::vector<uint64_t> timestamp_buffer_;
    mutable std::shared_mutex mutex_;
    void processBuffer();
    std::vector<Pair> combineResults(const std::vector<Pair> &witness_pairs,
                                     const std::vector<Pair> &sketch_pairs) const;
    void adaptToPerformance(const PerformanceMetrics &metrics);
};
class ThroughputAccuracyOptimizer
{
public:
    struct OptimizationTarget
    {
        double min_throughput;
        double max_error;
        double min_confidence;
    };
    explicit ThroughputAccuracyOptimizer(const OptimizationTarget &target);
    StreamingConfig optimizeConfig(const StreamingPH::PerformanceMetrics &current_metrics);
    void updateTarget(const OptimizationTarget &new_target);
    std::vector<StreamingConfig> exploreConfigSpace() const;
    double predictThroughput(const StreamingConfig &config) const;
    double predictError(const StreamingConfig &config) const;

private:
    OptimizationTarget target_;
    std::vector<std::pair<StreamingConfig, StreamingPH::PerformanceMetrics>> performance_history_;
    StreamingConfig findBestConfig() const;
    void updatePerformanceHistory(const StreamingConfig &config,
                                  const StreamingPH::PerformanceMetrics &metrics);
};
class StreamingPHManager
{
public:
    static StreamingPHManager &instance();
    void createStream(const std::string &name, const StreamingConfig &config);
    std::shared_ptr<StreamingPH> getStream(const std::string &name);
    void removeStream(const std::string &name);
    std::vector<std::string> getStreamNames() const;
    void addPointsToAllStreams(const std::vector<std::vector<float>> &points,
                               const std::vector<uint64_t> &timestamps);
    void computeAllPersistence();
    void optimizeAllForThroughput(double target_throughput);
    void optimizeAllForAccuracy(double target_error);

private:
    StreamingPHManager() = default;
    std::unordered_map<std::string, std::shared_ptr<StreamingPH>> streams_;
    std::unique_ptr<ThroughputAccuracyOptimizer> optimizer_;
    mutable std::shared_mutex mutex_;
};
} // namespace approximate
} // namespace nerve

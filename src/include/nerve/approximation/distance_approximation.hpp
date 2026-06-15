
#pragma once
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <random>
#include <shared_mutex>
#include <unordered_map>
#include <vector>
namespace nerve
{
namespace approximation
{
struct DiagramPoint
{
    float birth;
    float death;
    float persistence;
    uint8_t dimension;
    bool isValid() const
    {
        return std::isfinite(birth) && std::isfinite(death) && std::isfinite(persistence) &&
               death >= birth && persistence >= 0;
    }
    float getLifetime() const { return persistence; }
    std::pair<float, float> toPair() const { return {birth, death}; }
};
struct ApproximationConfig
{
    size_t num_projections = 100;
    uint32_t random_seed = 42;
    size_t num_hash_tables = 10;
    size_t num_hash_functions = 5;
    float hash_bandwidth = 1.0f;
    bool enable_multi_resolution_hashing = true;
    float approximation_tolerance = 0.01f;
    size_t max_points_per_diagram = 1000;
    bool enable_outlier_removal = true;
    float outlier_threshold = 3.0f;
    bool enable_parallel_computation = true;
    size_t num_threads = 4;
};
class SlicedWasserstein
{
public:
    explicit SlicedWasserstein(const ApproximationConfig &config);
    float computeDistance(const std::vector<DiagramPoint> &diagram1,
                          const std::vector<DiagramPoint> &diagram2);
    std::vector<float> computeDistances(const std::vector<std::vector<DiagramPoint>> &diagrams);
    std::vector<std::vector<float>>
    computeDistanceMatrix(const std::vector<std::vector<DiagramPoint>> &diagrams);
    float computeAdaptiveDistance(const std::vector<DiagramPoint> &diagram1,
                                  const std::vector<DiagramPoint> &diagram2,
                                  float target_error = 0.01f);
    std::vector<std::vector<float>> generateProjections(size_t num_projections);
    void setProjections(const std::vector<std::vector<float>> &projections);
    struct DistanceStats
    {
        float mean_distance;
        float std_distance;
        float min_distance;
        float max_distance;
        size_t num_computations;
        float average_computation_time_ms;
    };
    DistanceStats getStats() const;
    void resetStats();

private:
    ApproximationConfig config_;
    std::vector<std::vector<float>> projections_;
    std::mt19937 rng_;
    mutable std::mutex stats_mutex_;
    DistanceStats stats_;
    float compute1dWasserstein(const std::vector<float> &values1,
                               const std::vector<float> &values2);
    std::vector<float> projectDiagram(const std::vector<DiagramPoint> &diagram,
                                      const std::vector<float> &projection);
    std::vector<DiagramPoint> preprocessDiagram(const std::vector<DiagramPoint> &diagram);
    void updateStats(float distance, float computation_time_ms);
};
class DiagramLSH
{
public:
    explicit DiagramLSH(const ApproximationConfig &config);
    void buildHashTables(const std::vector<std::vector<DiagramPoint>> &diagrams);
    void addDiagram(size_t diagram_id, const std::vector<DiagramPoint> &diagram);
    void removeDiagram(size_t diagram_id);
    std::vector<size_t> findSimilarDiagrams(const std::vector<DiagramPoint> &query_diagram,
                                            size_t max_results = 10);
    std::vector<std::pair<size_t, float>>
    findSimilarDiagramsWithDistance(const std::vector<DiagramPoint> &query_diagram,
                                    size_t max_results = 10);
    std::vector<std::vector<size_t>>
    clusterDiagrams(const std::vector<std::vector<DiagramPoint>> &diagrams,
                    float similarity_threshold = 0.8f);
    struct HashFunction
    {
        std::vector<float> projection;
        float offset;
        float bandwidth;
    };
    void generateHashFunctions();
    std::vector<uint32_t> computeHash(const std::vector<DiagramPoint> &diagram, size_t table_id);
    struct LSHStats
    {
        size_t total_diagrams;
        size_t total_hash_tables;
        float average_query_time_ms;
        float average_hash_computation_time_ms;
        float hash_table_load_factor;
    };
    LSHStats getStats() const;
    void resetStats();

private:
    ApproximationConfig config_;
    struct HashTable
    {
        std::unordered_map<uint32_t, std::vector<size_t>> buckets;
        std::vector<HashFunction> hash_functions;
    };
    std::vector<HashTable> hash_tables_;
    std::unordered_map<size_t, std::vector<DiagramPoint>> diagram_storage_;
    mutable std::mutex stats_mutex_;
    LSHStats stats_;
    std::vector<float> extractFeatures(const std::vector<DiagramPoint> &diagram);
    uint32_t computeSingleHash(const std::vector<float> &features, const HashFunction &hash_func);
    float estimateDiagramSimilarity(const std::vector<DiagramPoint> &diagram1,
                                    const std::vector<DiagramPoint> &diagram2);
    void updateStats(float query_time_ms, float hash_time_ms);
};
class CoarseGrainedMatcher
{
public:
    struct CoarseConfig
    {
        size_t grid_resolution = 50;
        float birth_range_min = 0.0f;
        float birth_range_max = 1.0f;
        float death_range_min = 0.0f;
        float death_range_max = 1.0f;
        float grid_adaptation_factor = 2.0f;
    };
    explicit CoarseGrainedMatcher(const CoarseConfig &config);
    std::vector<std::vector<float>> discretizeDiagram(const std::vector<DiagramPoint> &diagram);
    float computeGridDistance(const std::vector<std::vector<float>> &grid1,
                              const std::vector<std::vector<float>> &grid2);
    void adaptGridResolution(const std::vector<std::vector<DiagramPoint>> &diagrams);
    std::vector<size_t> findMatches(const std::vector<DiagramPoint> &query_diagram,
                                    const std::vector<std::vector<DiagramPoint>> &database,
                                    size_t max_results = 10);
    void buildIndex(const std::vector<std::vector<DiagramPoint>> &diagrams);
    void addToIndex(size_t diagram_id, const std::vector<DiagramPoint> &diagram);
    void removeFromIndex(size_t diagram_id);

private:
    CoarseConfig config_;
    std::vector<std::vector<float>> birth_grid_;
    std::vector<std::vector<float>> death_grid_;
    std::unordered_map<size_t, std::vector<std::vector<float>>> index_storage_;
    void initializeGrid();
    std::vector<float> computeGridCoordinates(float birth, float death);
};
class ApproximateBottleneck
{
public:
    struct BottleneckConfig
    {
        float approximation_factor = 2.0f;
        size_t num_landmark_points = 100;
        bool enable_random_sampling = true;
        float sampling_ratio = 0.1f;
    };
    explicit ApproximateBottleneck(const BottleneckConfig &config);
    float computeDistance(const std::vector<DiagramPoint> &diagram1,
                          const std::vector<DiagramPoint> &diagram2);
    float computeLandmarkDistance(const std::vector<DiagramPoint> &diagram1,
                                  const std::vector<DiagramPoint> &diagram2);
    float computeSamplingDistance(const std::vector<DiagramPoint> &diagram1,
                                  const std::vector<DiagramPoint> &diagram2);
    float computeMultiscaleDistance(const std::vector<DiagramPoint> &diagram1,
                                    const std::vector<DiagramPoint> &diagram2);

private:
    BottleneckConfig config_;
    std::mt19937 rng_;
    std::vector<DiagramPoint> selectLandmarkPoints(const std::vector<DiagramPoint> &diagram);
    std::vector<DiagramPoint> sampleDiagram(const std::vector<DiagramPoint> &diagram, float ratio);
    float computeExactBottleneck(const std::vector<DiagramPoint> &diagram1,
                                 const std::vector<DiagramPoint> &diagram2);
};
class DistanceApproximationManager
{
public:
    static DistanceApproximationManager &instance();
    void setApproximationConfig(const ApproximationConfig &config);
    void setBottleneckConfig(const ApproximateBottleneck::BottleneckConfig &config);
    void setCoarseConfig(const CoarseGrainedMatcher::CoarseConfig &config);
    std::shared_ptr<SlicedWasserstein> getSlicedWasserstein();
    std::shared_ptr<DiagramLSH> getDiagramLsh();
    std::shared_ptr<CoarseGrainedMatcher> getCoarseMatcher();
    std::shared_ptr<ApproximateBottleneck> getApproximateBottleneck();
    float computeFastDistance(const std::vector<DiagramPoint> &diagram1,
                              const std::vector<DiagramPoint> &diagram2,
                              const std::string &method = "sliced_wasserstein");
    std::vector<float> computeDistanceMatrix(const std::vector<std::vector<DiagramPoint>> &diagrams,
                                             const std::string &method = "sliced_wasserstein");
    std::vector<size_t> findSimilarDiagrams(const std::vector<DiagramPoint> &query_diagram,
                                            const std::vector<std::vector<DiagramPoint>> &database,
                                            size_t max_results = 10);
    std::vector<std::vector<size_t>>
    clusterDiagrams(const std::vector<std::vector<DiagramPoint>> &diagrams,
                    const std::string &method = "lsh", float similarity_threshold = 0.8f);
    struct PerformanceStats
    {
        double average_computation_time_ms;
        double total_computations;
        std::unordered_map<std::string, double> method_times;
        double memory_usage_mb;
    };
    PerformanceStats getPerformanceStats() const;
    void resetPerformanceStats();

private:
    DistanceApproximationManager() = default;
    ApproximationConfig approximation_config_;
    ApproximateBottleneck::BottleneckConfig bottleneck_config_;
    CoarseGrainedMatcher::CoarseConfig coarse_config_;
    std::shared_ptr<SlicedWasserstein> sliced_wasserstein_;
    std::shared_ptr<DiagramLSH> diagram_lsh_;
    std::shared_ptr<CoarseGrainedMatcher> coarse_matcher_;
    std::shared_ptr<ApproximateBottleneck> approximate_bottleneck_;
    mutable std::shared_mutex mutex_;
    mutable PerformanceStats performance_stats_;
    void updatePerformanceStats(const std::string &method, double time_ms);
};
} // namespace approximation
} // namespace nerve

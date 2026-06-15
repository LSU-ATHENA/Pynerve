
#pragma once
#include "nerve/precision/precision_policy.hpp"
#include "nerve/streaming/incremental.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include <vector>

namespace nerve::streaming
{
class NUMAMemoryPool
{
public:
    struct PoolConfig
    {
        size_t numa_node = 0;
        size_t pool_size_mb = 1024;
        size_t block_size_bytes = 4096;
        bool enable_huge_pages = false;
        bool preallocate = true;
    };
    NUMAMemoryPool();
    explicit NUMAMemoryPool(const PoolConfig &config);
    ~NUMAMemoryPool();
    void *allocate(size_t size);
    void deallocate(void *ptr);
    void reset();
    size_t getAllocatedBytes() const;
    size_t get_free_bytes() const;
    size_t getPoolSize() const;
    void bindToNumaNode(int node);
    int getCurrentNumaNode() const;

private:
    PoolConfig config_;
    std::vector<void *> memory_blocks_;
    std::vector<bool> block_used_;
    std::mutex pool_mutex_;
    size_t allocated_bytes_;
    int current_numa_node_;
    void initializePool();
    void *allocateFromPool(size_t size);
    void deallocateToPool(void *ptr);
};
class PartialRecomputeHeuristic
{
public:
    enum class RecomputeStrategy
    {
        FULL_RECOMPUTE = 0,
        INCREMENTAL_UPDATE = 1,
        HYBRID = 2,
        ADAPTIVE = 3
    };
    struct HeuristicConfig
    {
        RecomputeStrategy strategy = RecomputeStrategy::ADAPTIVE;
        double change_threshold = 0.1;
        size_t min_recompute_size = 100;
        double performance_weight = 0.7;
        bool enable_change_detection = true;
        size_t max_partial_updates = 1000;
    };
    PartialRecomputeHeuristic();
    explicit PartialRecomputeHeuristic(const HeuristicConfig &config);
    RecomputeStrategy determineStrategy(const std::vector<algebra::Simplex> &added_simplices,
                                        const std::vector<algebra::Simplex> &removed_simplices,
                                        const Diagram &current_diagram) const;
    double computeChangeMagnitude(const std::vector<algebra::Simplex> &added_simplices,
                                  const std::vector<algebra::Simplex> &removed_simplices) const;
    double estimateRecomputeCost(RecomputeStrategy strategy, size_t window_size) const;
    double estimateUpdateAccuracy(RecomputeStrategy strategy) const;
    void setConfig(const HeuristicConfig &config);
    HeuristicConfig getConfig() const;

private:
    HeuristicConfig config_;
    bool shouldUseIncremental(const std::vector<algebra::Simplex> &changes) const;
    bool shouldUseHybrid(size_t window_size, double change_magnitude) const;
    double computeComplexityScore(const Diagram &diagram) const;
};
class AcceleratedWindowedPH
{
public:
    struct OptimizationConfig
    {
        bool enable_numa_optimization = true;
        NUMAMemoryPool::PoolConfig numa_pool_config;
        PartialRecomputeHeuristic::HeuristicConfig heuristic_config;
        bool enable_witness_mode = false;
        bool enable_approximate_mode = false;
        double approximation_error_budget = 0.01;
        bool enable_parallel_processing = true;
        size_t num_threads = std::thread::hardware_concurrency();
        bool enable_batch_processing = true;
        size_t batch_size = 100;
        bool obey_no_alloc_invariant = true;
        bool obey_hot_path_invariant = true;
        size_t max_hot_path_latency_ns = 10000;
        precision::PrecisionLevel precision_level = precision::PrecisionLevel::BALANCED;
    };
    struct PerformanceMetrics
    {
        double last_window_update_time_ms;
        double average_update_time_ms;
        size_t total_updates_processed;
        size_t partial_recomputes_used;
        size_t full_recomputes_used;
        double memory_usage_mb;
        double numa_efficiency_score;
        bool hot_path_invariant_violated;
    };
    AcceleratedWindowedPH();
    explicit AcceleratedWindowedPH(const OptimizationConfig &config);
    ~AcceleratedWindowedPH();
    void addSimplex(const algebra::Simplex &simplex);
    void addSimplicesBatch(const std::vector<algebra::Simplex> &simplices);
    void slideWindow(size_t new_window_size);
    void advanceWindow(size_t step_size = 1);
    Diagram compute();
    Diagram computePersistenceApproximate();
    /// Truncates the current diagram to the pairs with largest finite lifetimes (not a witness
    /// complex).
    Diagram computePersistenceTopLifetimeTruncation();
    void setWindowSize(size_t window_size);
    void setOptimizationConfig(const OptimizationConfig &config);
    OptimizationConfig getOptimizationConfig() const;
    PerformanceMetrics getPerformanceMetrics() const;
    void resetPerformanceMetrics();
    void bindToNumaNode(int node);
    size_t getMemoryUsage() const;
    void compactMemory();
    bool checkHotPathInvariants() const;
    void enableHotPathMode(bool enable);

private:
    OptimizationConfig config_;
    std::unique_ptr<NUMAMemoryPool> memory_pool_;
    std::unique_ptr<PartialRecomputeHeuristic> heuristic_;
    std::unique_ptr<WindowedPH> windowed_ph_;
    std::queue<algebra::Simplex> simplex_window_;
    size_t current_window_size_;
    Diagram current_diagram_;
    PerformanceMetrics metrics_;
    mutable std::mutex window_mutex_;
    mutable std::mutex metrics_mutex_;
    std::vector<std::thread> worker_threads_;
    std::queue<std::vector<algebra::Simplex>> work_queue_;
    std::mutex work_queue_mutex_;
    std::condition_variable work_queue_cv_;
    std::atomic<bool> shutdown_flag_;
    std::atomic<bool> hot_path_mode_enabled_;
    std::vector<algebra::Simplex> previous_window_snapshot_;
    bool window_engine_synced_ = false;
    void initializeOptimizations();
    void processWindowUpdate();
    void handleWindowSlide(size_t old_size, size_t new_size);
    Diagram recomputeIncremental(const std::vector<algebra::Simplex> &added,
                                 const std::vector<algebra::Simplex> &removed);
    Diagram recomputePartial(const std::vector<algebra::Simplex> &added,
                             const std::vector<algebra::Simplex> &removed);
    Diagram recomputeFull();
    Diagram computeTopLifetimeTruncation();
    Diagram computeSparseApproximation();
    void startWorkerThreads();
    void stopWorkerThreads();
    void workerThreadFunction();
    void processBatch(const std::vector<algebra::Simplex> &batch);
    void updatePerformanceMetrics(double update_time_ms,
                                  PartialRecomputeHeuristic::RecomputeStrategy strategy);
    void checkHotPathViolations(double update_time_ms);
    void optimizeMemoryLayout();
    void preallocate_memory_pools();
};
class StreamingPHPerformanceHarness
{
public:
    struct BenchmarkConfig
    {
        std::array<size_t, 4> window_sizes{100, 1000, 10000, 100000};
        size_t num_iterations = 100;
        bool enable_hot_path_validation = true;
        double performance_target_ms = 10.0;
    };
    struct BenchmarkResults
    {
        std::vector<double> update_times_ms;
        std::vector<double> memory_usage_mb;
        std::vector<bool> hot_path_violations;
        double average_update_time_ms;
        double max_update_time_ms;
        double memory_efficiency_score;
        bool meets_performance_targets;
    };
    StreamingPHPerformanceHarness();
    explicit StreamingPHPerformanceHarness(const BenchmarkConfig &config);
    BenchmarkResults benchmarkWindowedPh(const AcceleratedWindowedPH::OptimizationConfig &config);
    BenchmarkResults benchmarkWithBaseline(const AcceleratedWindowedPH::OptimizationConfig &config);
    bool validateHotPathInvariants(const BenchmarkResults &results);
    bool validateMemoryEfficiency(const BenchmarkResults &results);
    bool validatePerformanceTargets(const BenchmarkResults &results);
    BenchmarkResults
    compareBeforeAfter(const AcceleratedWindowedPH::OptimizationConfig &before_config,
                       const AcceleratedWindowedPH::OptimizationConfig &after_config);

private:
    BenchmarkConfig config_;
    std::vector<algebra::Simplex> generateTestComplex(size_t size);
    std::vector<std::vector<algebra::Simplex>> generateWindowSequence(size_t window_size,
                                                                      size_t num_windows);
    double measureWindowUpdateTime(AcceleratedWindowedPH &ph,
                                   const std::vector<algebra::Simplex> &simplices);
    double measureMemoryUsage(AcceleratedWindowedPH &ph);
};
} // namespace nerve::streaming

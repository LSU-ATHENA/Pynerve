
#pragma once
#include "nerve/cpu/x86_intrinsics.hpp"

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <vector>
namespace nerve
{
namespace persistence_homology
{
struct PersistenceHomologyOptimizationConfig
{
    bool enable_ph0 = true;
    bool enable_ph1 = true;
    bool enable_ph2 = false;
    bool enable_ph3 = false;
    double ph0_time_budget_ms = 0.1;
    double ph1_time_budget_ms = 1.0;
    double ph2_time_budget_ms = 10.0;
    double ph3_time_budget_ms = 100.0;
    bool use_soa_numeric_arrays = true;
    bool use_aos_metadata = true;
    size_t memory_pool_size = 1024 * 1024;
    bool use_aligned_memory = true;
    bool use_pinned_memory = true;
    bool use_sse_avx = true;
    bool enable_fma = true;
    size_t vectorization_threshold = 64;
    size_t gpu_batch_size = 32;
    bool use_mixed_precision = false;
    bool enable_lock_free = true;
    size_t ring_buffer_size = 1024;
    bool use_hazard_pointers = true;
    size_t max_producers = 4;
    size_t max_consumers = 8;
    bool enable_streaming = true;
    bool use_witness_complexes = true;
    bool enable_incremental_updates = true;
    double streaming_error_budget = 0.01;
    size_t top_k_eigenpairs = 10;
    bool enable_warm_start = true;
    double convergence_tolerance = 1e-8;
    size_t max_matrix_size = 10000;
    size_t max_consecutive_failures = 5;
    bool enable_circuit_breaker = true;
    bool enable_monitoring = true;
    double latency_sla_ms = 5.0;
    double precision_event_rate_threshold = 0.05;
};
class AcceleratedMemoryPool
{
public:
    explicit AcceleratedMemoryPool(size_t pool_size, bool aligned = true);
    ~AcceleratedMemoryPool();
    void *allocate(size_t size, size_t alignment = 64);
    void deallocate(void *ptr);
    template <typename T>
    T *allocateTyped(size_t count = 1);
    template <typename T>
    void deallocateTyped(T *ptr);
    void reset();
    size_t getAllocatedSize() const;
    size_t get_free_size() const;

private:
    struct MemoryBlock
    {
        void *ptr;
        size_t size;
        bool in_use;
    };
    std::vector<MemoryBlock> blocks_;
    std::mutex mutex_;
    size_t pool_size_;
    bool aligned_;
    void *allocateAligned(size_t size, size_t alignment);
    void *allocateUnaligned(size_t size);
};
class LockFreeRingBuffer
{
public:
    struct RingBufferEntry
    {
        std::array<float, 128> features;
        uint64_t sequence_number;
        int64_t timestamp_ns;
        uint32_t symbol_id;
        uint32_t flags;
        std::array<uint8_t, 64> metadata;
    };
    explicit LockFreeRingBuffer(size_t size, bool aligned = true);
    ~LockFreeRingBuffer();
    bool tryProduce(const RingBufferEntry &entry);
    bool tryProduceBatch(const std::vector<RingBufferEntry> &entries);
    bool tryConsume(RingBufferEntry &entry);
    bool tryConsumeBatch(std::vector<RingBufferEntry> &entries, size_t max_count);
    size_t size() const;
    bool empty() const;
    bool full() const;
    static constexpr size_t entry_size = sizeof(RingBufferEntry);
    static constexpr size_t entry_alignment = 64;

private:
    struct alignas(64) AlignedEntry
    {
        RingBufferEntry entry;
        std::atomic<uint64_t> sequence_number;
        std::atomic<uint32_t> generation;
    };
    AlignedEntry *buffer_;
    size_t buffer_size_;
    std::atomic<size_t> head_;
    std::atomic<size_t> tail_;
    struct HazardPointer
    {
        std::atomic<AlignedEntry *> pointer;
        std::atomic<uint64_t> counter;
    };
    std::array<HazardPointer, 8> hazard_pointers_;
    std::atomic<uint32_t> global_epoch_{0};

    HazardPointer *getThreadHazardPointer()
    {
        static thread_local size_t idx =
            std::hash<std::thread::id>{}(std::this_thread::get_id()) % hazard_pointers_.size();
        return &hazard_pointers_[idx];
    }

    bool isEntryValid(const AlignedEntry *entry, uint64_t expected_gen)
    {
        if (!entry)
            return false;
        uint32_t gen = entry->generation.load(std::memory_order_acquire);
        return gen == static_cast<uint32_t>(expected_gen);
    }

    void publishEntry(AlignedEntry *entry)
    {
        entry->generation.store(global_epoch_.load(std::memory_order_relaxed),
                                std::memory_order_release);
        std::atomic_thread_fence(std::memory_order_release);
    }

    AlignedEntry *acquireEntry()
    {
        HazardPointer *hp = getThreadHazardPointer();
        size_t tail = tail_.load(std::memory_order_acquire);
        if (tail >= head_.load(std::memory_order_acquire))
            return nullptr;
        AlignedEntry *entry = &buffer_[tail % buffer_size_];
        hp->pointer.store(entry, std::memory_order_release);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        if (tail_.load(std::memory_order_acquire) != tail)
        {
            hp->pointer.store(nullptr, std::memory_order_release);
            return nullptr;
        }
        return entry;
    }

    void releaseEntry(AlignedEntry *entry)
    {
        if (!entry)
            return;
        HazardPointer *hp = getThreadHazardPointer();
        if (hp->pointer.load(std::memory_order_acquire) == entry)
        {
            hp->pointer.store(nullptr, std::memory_order_release);
        }
    }
};
class PH0ConnectedComponents
{
public:
    struct PH0Result
    {
        uint32_t num_components;
        std::vector<uint32_t> component_sizes;
        float clustering_coefficient;
        uint32_t largest_component_size;
        double computation_time_ms;
        bool is_approximate;
    };
    explicit PH0ConnectedComponents(const PersistenceHomologyOptimizationConfig &config);
    PH0Result computeConnectedComponents(const std::vector<std::vector<float>> &points);
    PH0Result computeConnectedComponentsStreaming(const std::vector<std::vector<float>> &points,
                                                  const std::vector<uint64_t> &timestamps);
    PH0Result computeConnectedComponentsVectorized(const std::vector<std::vector<float>> &points);

private:
    PersistenceHomologyOptimizationConfig config_;
    std::unique_ptr<AcceleratedMemoryPool> memory_pool_;
    std::vector<uint32_t> unionFindVectorized(const std::vector<std::vector<float>> &points);
    void pathCompressVectorized(std::vector<uint32_t> &parent, std::vector<uint32_t> &rank);
};
class PH1Loops
{
public:
    struct LoopResult
    {
        std::vector<std::vector<uint32_t>> loops;
        std::vector<float> loop_lengths;
        uint32_t num_loops;
        float average_loop_length;
        double computation_time_ms;
        bool is_approximate;
    };
    explicit PH1Loops(const PersistenceHomologyOptimizationConfig &config);
    LoopResult detectLoops(const std::vector<std::vector<float>> &points,
                           const std::vector<std::vector<uint32_t>> &edges);
    LoopResult detectLoopsStreaming(const std::vector<std::vector<float>> &points,
                                    const std::vector<std::vector<uint32_t>> &edges,
                                    const std::vector<uint64_t> &timestamps);

private:
    PersistenceHomologyOptimizationConfig config_;
    std::unique_ptr<AcceleratedMemoryPool> memory_pool_;
    std::vector<std::pair<float, float>>
    computePersistenceDiagram(const std::vector<std::vector<float>> &points);
};
class PH2Voids
{
public:
    struct VoidResult
    {
        std::vector<std::vector<std::pair<float, float>>> voids;
        std::vector<uint32_t> betti_numbers;
        std::vector<float> void_volumes;
        uint32_t total_voids;
        double computation_time_ms;
        bool is_approximate;
    };
    explicit PH2Voids(const PersistenceHomologyOptimizationConfig &config);
    VoidResult detectVoids(const std::vector<std::vector<float>> &surface_points, size_t width,
                           size_t height);
    VoidResult detectVoidsApproximate(const std::vector<std::vector<float>> &surface_points,
                                      size_t width, size_t height, double error_budget);

private:
    PersistenceHomologyOptimizationConfig config_;
    std::unique_ptr<AcceleratedMemoryPool> memory_pool_;
    std::vector<std::vector<float>>
    downsampleSurface(const std::vector<std::vector<float>> &surface_points, size_t target_width,
                      size_t target_height);
};
class PH3HigherDimensions
{
public:
    struct HigherDimResult
    {
        std::vector<std::vector<std::pair<float, float>>> persistence_diagrams;
        std::vector<uint32_t> betti_numbers;
        uint8_t max_dimension;
        double computation_time_ms;
        bool is_approximate;
    };
    explicit PH3HigherDimensions(const PersistenceHomologyOptimizationConfig &config);
    HigherDimResult computeHigherHomology(const std::vector<std::vector<float>> &points,
                                          uint8_t max_dimension = 3);
    HigherDimResult computeHigherHomologyApproximate(const std::vector<std::vector<float>> &points,
                                                     uint8_t max_dimension = 3,
                                                     double error_budget = 0.01);

private:
    PersistenceHomologyOptimizationConfig config_;
    std::unique_ptr<AcceleratedMemoryPool> memory_pool_;
    std::vector<std::vector<float>>
    reduceDimensionality(const std::vector<std::vector<float>> &points, uint8_t target_dimension);
};
class CompactSummaryFeatures
{
public:
    struct CompactSummary
    {
        static constexpr size_t SIZE_BYTES = 128;
        std::array<float, 8> betti_numbers;
        std::array<float, 8> top_lifetimes;
        float persistence_entropy;
        std::array<float, 4> laplacian_top4;
        uint32_t num_points;
        uint64_t timestamp_ns;
        uint32_t params_hash_low;
        uint32_t params_hash_high;
        uint16_t computation_time_us;
        uint8_t flags;
        uint8_t reserved[15];
    };
    static_assert(sizeof(CompactSummary) == CompactSummary::SIZE_BYTES,
                  "CompactSummary size mismatch");
    explicit CompactSummaryFeatures(const PersistenceHomologyOptimizationConfig &config);
    CompactSummary computeCompactSummary(const std::vector<std::vector<float>> &points);
    std::vector<uint8_t> serialize(const CompactSummary &summary);
    CompactSummary deserialize(const std::vector<uint8_t> &data);
    bool isValid(const CompactSummary &summary) const;

private:
    PersistenceHomologyOptimizationConfig config_;
    std::array<float, 8> computeBettiNumbersFast(const std::vector<std::vector<float>> &points);
    std::array<float, 8> extractTopLifetimes(const std::vector<std::vector<float>> &points);
    float computePersistenceEntropy(const std::vector<std::vector<float>> &points);
};
enum class PHErrorCode
{
    SUCCESS = 0,
    TRANSIENT_GPU_OOM = 1,
    TRANSIENT_TIMEOUT = 2,
    PERMANENT_CORRUPT_DATA = 3,
    PERMANENT_NAN_DATA = 4,
    INVARIANT_SCHEMA_MISMATCH = 5,
    INVARIANT_PARAMS_HASH_CHANGE = 6,
    COMPUTATION_TIMEOUT = 7,
    MEMORY_ALLOCATION_FAILED = 8
};
class ErrorHandler
{
public:
    explicit ErrorHandler(const PersistenceHomologyOptimizationConfig &config);
    bool isTransientError(PHErrorCode error) const;
    bool isPermanentError(PHErrorCode error) const;
    bool isInvariantViolation(PHErrorCode error) const;
    struct ErrorAction
    {
        bool retry_with_backoff;
        bool route_to_cpu;
        bool skip_window;
        bool mark_bad_window;
        bool stop_pipeline;
        bool alert_operator;
        uint32_t backoff_ms;
    };
    ErrorAction handleError(PHErrorCode error, const std::string &context);
    bool shouldHaltAfterFailures() const;
    void incrementFailureCount();
    void resetFailureCount();
    bool enforceQuotas(size_t memory_usage_mb, double cpu_usage_percent);

private:
    PersistenceHomologyOptimizationConfig config_;
    std::atomic<uint32_t> consecutive_failures_;
    std::atomic<bool> circuit_breaker_tripped_;
    std::chrono::steady_clock::time_point last_reset_;
    uint32_t calculateBackoff(uint32_t attempt);
};
struct MonitoringMetrics
{
    std::atomic<double> artifact_latency_ms{0.0};
    std::atomic<size_t> pd_size{0};
    std::atomic<double> image_raster_ms{0.0};
    std::atomic<double> eigen_ms{0.0};
    std::atomic<double> cache_hit_rate{0.0};
    std::atomic<double> precision_event_rate{0.0};
    std::atomic<size_t> error_count{0};
    std::atomic<double> drift_score{0.0};
    std::atomic<bool> drift_detected{false};
    std::atomic<uint64_t> total_computations{0};
    std::atomic<uint64_t> successful_computations{0};
    std::atomic<uint64_t> approximate_computations{0};
    void reset();
    std::string exportJson() const;
};
class MonitoringSystem
{
public:
    explicit MonitoringSystem(const PersistenceHomologyOptimizationConfig &config);
    void recordArtifactLatency(double latency_ms);
    void recordPdSize(size_t size);
    void recordCacheHitRate(double rate);
    void recordPrecisionEventRate(double rate);
    void recordError(PHErrorCode error);
    bool checkAlerts();
    std::vector<std::string> getAlertMessages();
    void updateDriftScore(double score);
    bool isDriftDetected() const;
    MonitoringMetrics getMetrics() const;
    void resetMetrics();

private:
    PersistenceHomologyOptimizationConfig config_;
    MonitoringMetrics metrics_;
    double latency_sla_ms_;
    double precision_event_rate_threshold_;
    std::vector<std::string> alert_messages_;
    mutable std::mutex alert_mutex_;
    void checkLatencySla();
    void checkPrecisionEventRate();
    void checkErrorRate();
};
class PersistenceHomologyOptimizationManager
{
public:
    static PersistenceHomologyOptimizationManager &instance();
    void setConfig(const PersistenceHomologyOptimizationConfig &config);
    PersistenceHomologyOptimizationConfig getConfig() const;
    enum class RuntimeMode
    {
        DETERMINISTIC_CPU,
        LOW_LATENCY,
        THROUGHPUT,
        CUSTOM
    };
    void setRuntimeMode(RuntimeMode mode);
    RuntimeMode getRuntimeMode() const;
    void setTimeBudgetMs(double budget_ms);
    double getTimeBudgetMs() const;
    PH0ConnectedComponents::PH0Result computePh0(const std::vector<std::vector<float>> &points);
    PH1Loops::LoopResult computePh1(const std::vector<std::vector<float>> &points,
                                    const std::vector<std::vector<uint32_t>> &edges);
    PH2Voids::VoidResult computePh2(const std::vector<std::vector<float>> &surface_points,
                                    size_t width, size_t height);
    PH3HigherDimensions::HigherDimResult computePh3(const std::vector<std::vector<float>> &points,
                                                    uint8_t max_dimension = 3);
    CompactSummaryFeatures::CompactSummary
    computeCompactSummary(const std::vector<std::vector<float>> &points);
    std::shared_ptr<LockFreeRingBuffer> getRingBuffer();
    std::shared_ptr<AcceleratedMemoryPool> getMemoryPool();
    std::shared_ptr<ErrorHandler> getErrorHandler();
    std::shared_ptr<MonitoringSystem> getMonitoring();
    struct PerformanceStats
    {
        double average_ph0_time_ms;
        double average_ph1_time_ms;
        double average_ph2_time_ms;
        double average_ph3_time_ms;
        size_t total_computations;
        size_t approximate_computations;
        double cache_hit_rate;
        double error_rate;
    };
    PerformanceStats getPerformanceStats() const;
    void resetPerformanceStats();

private:
    PersistenceHomologyOptimizationManager() = default;
    PersistenceHomologyOptimizationConfig config_;
    RuntimeMode runtime_mode_;
    double time_budget_ms_;
    std::shared_ptr<PH0ConnectedComponents> ph0_;
    std::shared_ptr<PH1Loops> ph1_;
    std::shared_ptr<PH2Voids> ph2_;
    std::shared_ptr<PH3HigherDimensions> ph3_;
    std::shared_ptr<CompactSummaryFeatures> compact_summary_features_;
    std::shared_ptr<LockFreeRingBuffer> ring_buffer_;
    std::shared_ptr<AcceleratedMemoryPool> memory_pool_;
    std::shared_ptr<ErrorHandler> error_handler_;
    std::shared_ptr<MonitoringSystem> monitoring_;
    mutable std::shared_mutex mutex_;
    mutable PerformanceStats performance_stats_;
    void updatePerformanceStats(const std::string &operation, double time_ms);
};
} // namespace persistence_homology
} // namespace nerve

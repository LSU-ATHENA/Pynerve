
#pragma once
#include "nerve/gpu/compute_manager.hpp"

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(__CUDACC__)
#include <cuda_fp16.h>
#endif

#if !defined(__CUDACC__) && !defined(__global__)
#define __global__
#endif

#if !defined(__CUDACC__) && !defined(__half)
struct __half
{
    uint16_t x;
};
#endif

namespace nerve::optimization
{
enum class ErrorCode : uint32_t
{
    IO_TIMEOUT = 0x00000100,
    IO_READ_ERROR = 0x00000101,
    IO_WRITE_ERROR = 0x00000102,
    GPU_OOM = 0x00000200,
    GPU_TIMEOUT = 0x00000201,
    GPU_KERNEL_ERROR = 0x00000202,
    NUM_NAN = 0x00000300,
    NUM_INF = 0x00000301,
    NUM_CONVERGENCE_FAILURE = 0x00000302,
    DET_MISMATCH = 0x00000400,
    DET_SEED_MISMATCH = 0x00000401,
    DET_THREAD_ORDER_MISMATCH = 0x00000402,
    CPU_OVERLOAD = 0x00000500,
    CPU_QUEUE_SATURATION = 0x00000501,
    CPU_AFFINITY_ERROR = 0x00000502,
    PH_ABORT = 0x00000600,
    PH_TIME_BUDGET_EXCEEDED = 0x00000601,
    PH_WINDOW_TOO_LARGE = 0x00000602,
    SUCCESS = 0x00000000
};
struct CallContract
{
    double time_budget_ms;
    bool strict_time_enforcement;
    std::string operation_name;
    uint64_t params_hash;
    int64_t window_start_ns;
    int64_t window_end_ns;
};
struct CircuitBreakerConfig
{
    size_t max_consecutive_failures = 5;
    uint64_t cooldown_ms = 5000;
    bool enable_automatic_recovery = true;
    double recovery_sample_ratio = 0.1;
};
class AcceleratedFeatureCache
{
public:
    struct CacheConfig
    {
        size_t max_symbols = 100;
        size_t entries_per_symbol = 1024;
        size_t entry_size = 512;
        bool use_hugepages = true;
        bool pin_memory = true;
        size_t cache_line_size = 64;
        bool enable_numa_binding = true;
        int numa_node = -1;
    };
    explicit AcceleratedFeatureCache(const CacheConfig &config);
    ~AcceleratedFeatureCache();
    struct alignas(64) CacheEntry
    {
        std::array<float, 128> features;
        std::array<double, 32> weights;
        uint64_t sequence_number;
        int64_t timestamp_ns;
        uint32_t symbol_id;
        uint32_t flags;
        uint16_t generation;
        uint8_t feature_count;
        uint8_t reserved[13];
    };
    static constexpr size_t ENTRY_SIZE = sizeof(CacheEntry);
    static constexpr size_t CACHE_LINE_SIZE = 64;
    struct alignas(64) RingBuffer
    {
        std::atomic<uint64_t> write_sequence{0};
        std::atomic<uint64_t> read_sequence{0};
        CacheEntry *entries;
        size_t capacity;
        size_t mask;
        bool tryProduce(uint32_t symbol_id, const CacheEntry &entry);
        bool tryConsume(uint32_t symbol_id, CacheEntry &entry);
    };
    bool getFeature(uint32_t symbol_id, CacheEntry &entry);
    bool putFeature(uint32_t symbol_id, const CacheEntry &entry);
    struct CacheStats
    {
        double l1_miss_rate;
        double l2_miss_rate;
        double average_latency_ns;
        double p99_latency_ns;
        uint64_t total_operations;
        uint64_t cache_hits;
    };
    CacheStats getCacheStats() const;
    bool validateSla() const;

private:
    CacheConfig config_;
    std::vector<std::unique_ptr<RingBuffer>> ring_buffers_;
    void *allocateSharedMemory(size_t size);
    void deallocateSharedMemory(void *ptr);
    void bindToNumaNode(int node);
    void setCpuAffinity(const std::vector<int> &cpus);
    mutable std::atomic<uint64_t> total_operations_{0};
    mutable std::atomic<uint64_t> cache_hits_{0};
    mutable std::atomic<uint64_t> total_latency_ns_{0};
    void measureCacheMissRates() const;
};
class AcceleratedCompactSummaries
{
public:
    struct SummaryConfig
    {
        bool enable_avx512 = true;
        bool use_per_thread_allocators = true;
        size_t thread_allocator_size = 1024ULL * 1024ULL;
        bool precomputeHeavyReductions = true;
        bool enable_vectorization = true;
        size_t summary_size = 128;
        bool enable_serialization_optimization = true;
    };
    explicit AcceleratedCompactSummaries(const SummaryConfig &config);
    struct CompactSummary
    {
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
    static_assert(sizeof(CompactSummary) == 128, "CompactSummary must be 128 bytes");
    CompactSummary computeSummary(const std::vector<std::vector<float>> &points,
                                  const CallContract &contract);
    void *allocateThreadMemory(size_t size);
    void deallocateThreadMemory(void *ptr);
    bool serializeSummary(const CompactSummary &summary, std::vector<uint8_t> &buffer);
    bool validatePerformance() const;
    size_t estimateMemoryRequirement(size_t num_points) const;
    size_t getCurrentMemoryUsage() const;
    size_t getPeakMemoryUsage() const;
    void resetPeakMemoryUsage();

private:
    SummaryConfig config_;
    thread_local static std::unique_ptr<char[]> thread_allocator_;
    thread_local static size_t thread_allocator_offset_;
    mutable std::atomic<size_t> peak_memory_usage_bytes_{0};
    void computeBettiNumbers(const std::vector<std::vector<float>> &points,
                             std::array<float, 8> &betti_numbers);
    void computeTopLifetimes(const std::vector<std::vector<float>> &points,
                             std::array<float, 8> &lifetimes);
    struct PrecomputedReductions
    {
        std::vector<float> precomputed_betti_bases;
        std::vector<float> precomputed_lifetimes_bases;
        bool isValid;
    };
    PrecomputedReductions precomputed_reductions_;
    void precomputeHeavyReductions();
};
class AcceleratedGpuPrimitives
{
public:
    struct GPUConfig
    {
        size_t min_batch_size = 16;
        size_t optimal_batch_size = 32;
        size_t max_batch_size = 64;
        bool enable_async_operations = true;
        size_t num_cuda_streams = 4;
        int num_streams = 4;
        bool use_pinned_memory = true;
        bool enable_mixed_precision = false;
        double mixed_precision_error_threshold = 1e-6;
        bool enable_one_kernel_pipeline = true;
    };
    explicit AcceleratedGpuPrimitives(const GPUConfig &config);
    ~AcceleratedGpuPrimitives();
    struct BatchedOperation
    {
        std::vector<float *> input_buffers;
        std::vector<float *> output_buffers;
        std::vector<size_t> batch_sizes;
        std::vector<cudaStream_t> streams;
        std::vector<cudaEvent_t> events;
    };
    ErrorCode executeBatchedOperation(const BatchedOperation &operation,
                                      const CallContract &contract);
    ErrorCode executeOneKernelPipeline(const float *points_a, const float *points_b,
                                       float *distances, float *image, size_t batch_size,
                                       size_t num_points_a, size_t num_points_b,
                                       const CallContract &contract);
    ErrorCode executeWarpReduction(const float *input, float *output, size_t size,
                                   const CallContract &contract);
    bool validatePerformance() const;

    void computeDistanceMatrixBatch(float *d_points, size_t n_points, size_t point_dim);
    void reduceColumnGpu(uint32_t *column_data, size_t size);
    void sparseMatrixVectorMultiply(float *d_matrix, float *d_vector, float *d_result, size_t rows,
                                    size_t cols);
    size_t getPeakMemoryUsage() const;

private:
    GPUConfig config_;
    std::vector<cudaStream_t> streams_;
    std::vector<cudaEvent_t> events_;
#if defined(__CUDACC__)
    cudaStream_t *stream_pool_;
#endif
    void setupAsyncOperations();
    void cleanupAsyncOperations();
    bool validateMixedPrecision(const float *double_result, const __half *fp16_result, size_t size);
    void measureKernelLatency();
    void measureTransferLatency();
};
class AcceleratedStreamingPh
{
public:
    struct StreamingConfig
    {
        bool use_witness_complexes = true;
        bool enable_sparse_filtrations = true;
        double error_budget = 0.01;
        bool enable_incrementality = true;
        size_t max_active_simplex_growth = 10000;
        bool enable_coarsening = true;
        size_t coarsening_threshold = 5000;
        bool enable_summary_cap = true;
    };
    explicit AcceleratedStreamingPh(const StreamingConfig &config);
    ~AcceleratedStreamingPh();
    struct StreamingResult
    {
        std::vector<std::pair<float, float>> persistence_diagram;
        double actual_error;
        bool used_approximation;
        double computation_time_ms;
        ErrorCode error_code;
    };
    StreamingResult computeStreamingPh(const std::vector<std::vector<float>> &points);
    void updateStreamingPh(const std::vector<std::vector<float>> &new_points,
                           const std::vector<uint32_t> &changed_indices);
    void capSimplexGrowth();
    void applyCoarsening();
    StreamingResult computeSummaryOnly(const std::vector<std::vector<float>> &points,
                                       const CallContract &contract);
    std::vector<std::pair<float, float>> compute(const std::vector<std::vector<float>> &window);
    std::vector<std::pair<float, float>>
    computeExact(const std::vector<std::vector<float>> &window);
    std::vector<std::pair<float, float>>
    computeIncremental(const std::vector<std::vector<float>> &window);
    bool isCacheValid() const;

private:
    StreamingConfig config_;
    std::vector<std::vector<float>> landmark_points_;
    std::vector<std::vector<uint32_t>> point_to_simplices_;
    std::vector<uint32_t> changed_neighborhoods_;
    std::vector<bool> point_changed_;
    void measureUpdateTime();
    void measureErrorBudget();
    size_t window_size_;
    std::vector<float> persistence_cache_;
    bool cache_valid_;
    size_t cached_num_points_;
    size_t cached_point_dim_;
    uint64_t cached_window_hash_;
    double last_update_time_ms_ = 0.0;
    double last_error_budget_observed_ = 0.0;
    bool refreshDistanceCache(const std::vector<std::vector<float>> &window);
    uint64_t computeWindowHash(const std::vector<std::vector<float>> &window) const;
    std::vector<std::pair<float, float>> computePairsFromDistanceMatrix(const float *distances,
                                                                        size_t n) const;
};
class AcceleratedIncrementalLaplacian
{
public:
    struct LaplacianConfig
    {
        bool enable_warm_start = true;
        size_t top_k_eigenpairs = 10;
        bool use_csr_csc = true;
        bool enable_incremental_reordering = true;
        bool enable_lightweight_preconditioner = true;
        double convergence_tolerance = 1e-6;
        size_t max_iterations = 1000;
        bool enable_residual_confidence = true;
    };
    explicit AcceleratedIncrementalLaplacian(const LaplacianConfig &config);
    struct EigenpairResult
    {
        std::vector<double> eigenvalues;
        std::vector<std::vector<double>> eigenvectors;
        std::vector<double> residual_estimates;
        std::vector<double> confidence_scores;
        double computation_time_ms;
        ErrorCode error_code;
    };
    EigenpairResult
    computeIncrementalLaplacian(const std::vector<std::vector<uint32_t>> &adjacency_list,
                                const std::vector<double> &edge_weights,
                                const CallContract &contract,
                                const EigenpairResult &warm_start = {});
    void updateMatrixStructure(const std::vector<std::vector<uint32_t>> &new_edges);
    void updateMatrixValues(const std::vector<double> &new_weights);
    void applyLightweightPreconditioner();
    bool validatePerformance() const;

private:
    LaplacianConfig config_;
    std::vector<double> csr_data_;
    std::vector<int> csr_row_ptr_;
    std::vector<int> csr_col_ind_;
    std::vector<std::vector<double>> krylov_subspace_;
    std::vector<double> krylov_residuals_;
    std::vector<int> reordering_;
    std::vector<int> inverse_reordering_;
    std::vector<double> preconditioner_diagonal_;
    void measureConvergenceTime();
    void measureResidualNorm();
};
class AcceleratedDeterministicReplay
{
public:
    struct ReplayConfig
    {
        bool enable_parallel_deterministic = true;
        bool enable_fixed_partitioning = true;
        bool enable_checksum_validation = true;
        size_t max_session_size = 100;
        size_t max_artifacts_per_session = 10;
    };
    explicit AcceleratedDeterministicReplay(const ReplayConfig &config);
    struct ReplayResult
    {
        bool success;
        std::vector<std::string> reproduced_artifact_ids;
        std::vector<std::string> failed_artifact_ids;
        std::vector<ErrorCode> error_codes;
        double total_time_ms;
        bool checksum_mismatch;
    };
    ReplayResult replaySession(const std::string &session_id, const CallContract &contract);
    bool validateChecksums(const std::string &session_id);
    ErrorCode validateArtifactChecksum(const std::string &artifact_id,
                                       const std::array<uint8_t, 32> &expected_checksum);

private:
    ReplayConfig config_;
    void setupDeterministicScheduler();
    void setupParallelDeterministic();
    void setupFixedPartitioning();
    std::unordered_map<std::string, std::array<uint8_t, 32>> artifact_checksums_;
    void measureReplayTime();
    void measureChecksumValidationTime();
};
} // namespace nerve::optimization

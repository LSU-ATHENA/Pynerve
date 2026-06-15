#pragma once
#include <vector>
#include <memory>
#include <cstdint>

#include "nerve/config.hpp"

#if __has_include(<cuda_runtime.h>)
#include <cuda_runtime.h>
#else
using cudaStream_t = void*;
using cudaError_t = int;
#endif

#if (defined(NERVE_HAS_CUDA_RUNTIME) || defined(NERVE_HAS_CUDA) || defined(BUILD_CUDA)) && \
    __has_include(<cublas_v2.h>) && __has_include(<cusparse.h>)
#include <cublas_v2.h>
#include <cusparse.h>
#else
using cublasHandle_t = void*;
using cublasStatus_t = int;
using cusparseHandle_t = void*;
using cusparseStatus_t = int;
#endif
#include <functional>
#include <unordered_map>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <atomic>
#include <chrono>
#include "nerve/errors/errors.hpp"
namespace nerve {
namespace gpu {
class GPUStreamManager {
public:
    enum class StreamOwnership {
        USER_PROVIDED,
        LIBRARY_OWNED,
        SHARED_POOL
    };
    struct StreamConfig {
        StreamOwnership ownership = StreamOwnership::LIBRARY_OWNED;
        cudaStream_t user_stream = nullptr;
        bool enable_deterministic = false;
        uint32_t deterministic_seed = 42;
        int gpu_device = 0;
        bool enable_timing = false;
        size_t stream_id = 0;
    };
    explicit GPUStreamManager(const StreamConfig& config);
    ~GPUStreamManager();
    cudaStream_t get_stream() const;
    bool is_user_owned() const;
    bool is_deterministic() const;
    errors::ErrorCode ensure_deterministic_operation();
    void set_deterministic_seed(uint32_t seed);
    uint32_t get_deterministic_seed() const;
    void synchronize() const;
    bool is_completed() const;
    struct StreamMetrics {
        std::chrono::nanoseconds total_execution_time;
        std::chrono::nanoseconds gpu_execution_time;
        std::chrono::nanoseconds memory_transfer_time;
        size_t operations_count;
        bool determinism_violations;
        errors::ErrorCode last_error;
    };
    StreamMetrics get_metrics() const;
    void reset_metrics();
private:
    StreamConfig config_;
    cudaStream_t stream_;
    mutable std::mutex stream_mutex_;
    std::atomic<bool> deterministic_mode_enabled_{false};
    std::atomic<uint32_t> deterministic_seed_{42};
    std::atomic<bool> determinism_violated_{false};
    mutable StreamMetrics metrics_;
    std::chrono::high_resolution_clock::time_point start_time_;
    void create_library_stream();
    void destroy_stream();
    bool validate_deterministic_preconditions() const;
    void record_determinism_violation();
};
class GPUMemoryManager {
public:
    struct AllocationConfig {
        bool enable_deterministic_allocation = false;
        bool use_memory_pool = true;
        size_t pool_size_mb = 1024;
        bool enable_tracking = true;
        bool pin_memory = true;
        int gpu_device = 0;
    };
    static GPUMemoryManager& instance();
    void* allocate(size_t size, const AllocationConfig& config);
    void deallocate(void* ptr);
    void* allocate_deterministic(size_t size, uint32_t seed = 42);
    bool validate_deterministic_allocation(void* ptr, size_t expected_size) const;
    std::vector<void*> allocate_batch(
        const std::vector<size_t>& sizes,
        const AllocationConfig& config
    );
    void* get_from_pool(size_t size);
    void return_to_pool(void* ptr, size_t size);
    void clear_pool();
    struct MemoryStats {
        size_t total_allocated;
        size_t pool_size;
        size_t peak_memory;
        size_t allocation_count;
        bool deterministic_violations;
        errors::ErrorCode last_error;
    };
    MemoryStats get_stats() const;
    bool validate_no_leaks() const;
private:
    GPUMemoryManager() = default;
    mutable std::shared_mutex mutex_;
    std::unordered_map<size_t, std::vector<void*>> memory_pool_;
    std::unordered_map<void*, size_t> allocation_sizes_;
    std::atomic<bool> deterministic_mode_enabled_{false};
    std::atomic<uint32_t> allocation_counter_{0};
    std::atomic<bool> allocation_violations_{false};
    std::atomic<size_t> total_allocated_{0};
    std::atomic<size_t> peak_allocated_{0};
    std::atomic<size_t> allocation_count_{0};
    void* allocate_from_system(size_t size, bool pinned);
    void deallocate_to_system(void* ptr, bool pinned);
    bool validate_deterministic_preconditions() const;
};
class BatchedDistanceMatrix {
public:
    struct DistanceConfig {
        enum class Metric { EUCLIDEAN, MANHATTAN, COSINE, CHEBYSHEV };
        Metric metric = Metric::EUCLIDEAN;
        bool use_pinned_memory = true;
        bool async_computation = true;
        size_t batch_size = 1000;
        GPUStreamManager::StreamOwnership stream_ownership = GPUStreamManager::StreamOwnership::LIBRARY_OWNED;
        cudaStream_t user_stream = nullptr;
        bool enable_deterministic = false;
        uint32_t deterministic_seed = 42;
        enum class Precision { SINGLE, DOUBLE, MIXED };
        Precision precision = Precision::SINGLE;
        bool enable_mixed_precision_retry = true;
        bool enable_recovery = true;
        bool validate_gpu_results = true;
        size_t max_gpu_retries = 3;
        bool enable_timing = false;
        bool track_memory_usage = false;
    };
    explicit BatchedDistanceMatrix(const DistanceConfig& config);
    ~BatchedDistanceMatrix();
    std::vector<float> compute_distances(
        const std::vector<std::vector<float>>& points_a,
        const std::vector<std::vector<float>>& points_b);
    errors::ErrorCode compute_distances_with_status(
        const std::vector<std::vector<float>>& points_a,
        const std::vector<std::vector<float>>& points_b,
        std::vector<float>& result);
    void compute_distances_async(
        const std::vector<std::vector<float>>& points_a,
        const std::vector<std::vector<float>>& points_b,
        std::vector<float>& result,
        cudaStream_t stream = nullptr);
    errors::ErrorCode compute_distances_deterministic(
        const std::vector<std::vector<float>>& points_a,
        const std::vector<std::vector<float>>& points_b,
        std::vector<float>& result,
        uint32_t deterministic_seed = 42);
    errors::ErrorCode compute_distances_mixed_precision(
        const std::vector<std::vector<float>>& points_a,
        const std::vector<std::vector<float>>& points_b,
        std::vector<float>& result);
    std::vector<std::vector<float>> compute_batch_distances(
        const std::vector<std::vector<std::vector<float>>>& batch_a,
        const std::vector<std::vector<std::vector<float>>>& batch_b);
    errors::ErrorCode compute_batch_distances_checked(
        const std::vector<std::vector<std::vector<float>>>& batch_a,
        const std::vector<std::vector<std::vector<float>>>& batch_b,
        std::vector<std::vector<float>>& results);
    std::vector<float> compute_symmetric_distances(
        const std::vector<std::vector<float>>& points);
    void set_user_stream(cudaStream_t stream);
    cudaStream_t get_stream() const;
    bool is_user_stream_active() const;
    void enable_deterministic_mode(bool enable, uint32_t seed = 42);
    bool is_deterministic_mode() const;
    errors::ErrorCode validate_deterministic_result(
        const std::vector<float>& result,
        const std::vector<std::vector<float>>& points_a,
        const std::vector<std::vector<float>>& points_b) const;
    struct PerformanceMetrics {
        float computation_time_ms;
        float memory_transfer_time_ms;
        float gpu_utilization;
        size_t memory_used_bytes;
        bool used_gpu;
    };
    PerformanceMetrics get_last_metrics() const;
    static std::vector<float> compute_distances_cpu(
        const std::vector<std::vector<float>>& points_a,
        const std::vector<std::vector<float>>& points_b,
        DistanceConfig::Metric metric = DistanceConfig::Metric::EUCLIDEAN);
private:
    DistanceConfig config_;
    cudaStream_t stream_;
    cublasHandle_t cublas_handle_;
    PerformanceMetrics last_metrics_;
    void launch_euclidean_kernel(const float* d_a, const float* d_b,
                                 float* d_result, size_t m, size_t n, size_t k);
    void launch_manhattan_kernel(const float* d_a, const float* d_b,
                                float* d_result, size_t m, size_t n, size_t k);
    void launch_cosine_kernel(const float* d_a, const float* d_b,
                             float* d_result, size_t m, size_t n, size_t k);
};
class BatchedPersistenceImage {
public:
    struct ImageConfig {
        size_t width = 256;
        size_t height = 256;
        float sigma = 1.0f;
        bool use_pinned_memory = true;
        bool normalize_output = true;
        size_t batch_size = 100;
        uint32_t stream_id = 0;
    };
    explicit BatchedPersistenceImage(const ImageConfig& config);
    ~BatchedPersistenceImage();
    std::vector<float> rasterize_persistence_diagram(
        const std::vector<std::pair<float, float>>& persistence_pairs,
        float min_birth, float max_birth,
        float min_death, float max_death);
    void rasterize_persistence_diagram_async(
        const std::vector<std::pair<float, float>>& persistence_pairs,
        float min_birth, float max_birth,
        float min_death, float max_death,
        std::vector<float>& image,
        cudaStream_t stream = nullptr);
    std::vector<std::vector<float>> rasterize_batch(
        const std::vector<std::vector<std::pair<float, float>>>& batch_pairs,
        const std::vector<std::tuple<float, float, float, float>>& bounds);
    std::vector<float> rasterize_multidimensional(
        const std::vector<std::tuple<float, float, uint8_t>>& persistence_pairs,
        float min_birth, float max_birth,
        float min_death, float max_death,
        uint8_t target_dimension);
    struct PerformanceMetrics {
        float rasterization_time_ms;
        float memory_transfer_time_ms;
        float gpu_utilization;
        size_t memory_used_bytes;
        size_t pairs_processed;
        bool used_gpu;
    };
    PerformanceMetrics get_last_metrics() const;
    static std::vector<float> rasterize_cpu(
        const std::vector<std::pair<float, float>>& persistence_pairs,
        size_t width, size_t height, float sigma,
        float min_birth, float max_birth,
        float min_death, float max_death);
private:
    ImageConfig config_;
    cudaStream_t stream_;
    PerformanceMetrics last_metrics_;
    void launch_gaussian_kernel(const float* d_pairs, float* d_image,
                               size_t num_pairs, size_t width, size_t height,
                               float sigma, float min_birth, float max_birth,
                               float min_death, float max_death);
    void launch_normalization_kernel(float* d_image, size_t width, size_t height);
};
class BatchedArnoldi {
public:
    using DistanceConfig = BatchedDistanceMatrix::DistanceConfig;
    struct ArnoldiConfig {
        size_t max_eigenpairs = 10;
        double tolerance = 1e-6;
        size_t max_iterations = 1000;
        bool use_pinned_memory = true;
        size_t batch_size = 50;
        uint32_t stream_id = 0;
    };
    struct Eigenpair {
        double eigenvalue;
        std::vector<float> eigenvector;
        double error_estimate;
        uint32_t iterations;
    };
    explicit BatchedArnoldi(const ArnoldiConfig& config);
    ~BatchedArnoldi();
    std::vector<Eigenpair> compute_eigenpairs(
        const std::vector<std::vector<float>>& matrix);
    void compute_eigenpairs_async(
        const std::vector<std::vector<float>>& matrix,
        std::vector<Eigenpair>& result,
        cudaStream_t stream = nullptr);
    std::vector<std::vector<Eigenpair>> compute_batch_eigenpairs(
        const std::vector<std::vector<std::vector<float>>>& batch_matrices);
    std::vector<Eigenpair> compute_sparse_eigenpairs(
        const std::vector<std::tuple<uint32_t, uint32_t, float>>& sparse_matrix,
        size_t matrix_size);
    struct PerformanceMetrics {
        float computation_time_ms;
        float memory_transfer_time_ms;
        float gpu_utilization;
        size_t memory_used_bytes;
        bool used_gpu;
        bool determinism_violations;
        errors::ErrorCode last_error;
        size_t gpu_retries_used;
        bool routed_to_cpu;
        float mixed_precision_speedup;
        bool validation_passed;
        bool user_stream_used;
        bool library_stream_used;
        bool shared_pool_stream_used;
    };
    PerformanceMetrics get_last_metrics() const;
    void reset_metrics();
    static std::vector<float> compute_distances_cpu(
        const std::vector<std::vector<float>>& points_a,
        const std::vector<std::vector<float>>& points_b,
        DistanceConfig::Metric metric = DistanceConfig::Metric::EUCLIDEAN);
    bool validate_gpu_results_against_cpu(
        const std::vector<float>& gpu_result,
        const std::vector<std::vector<float>>& points_a,
        const std::vector<std::vector<float>>& points_b,
        DistanceConfig::Metric metric = DistanceConfig::Metric::EUCLIDEAN,
        float tolerance = 1e-4f);

private:
    ArnoldiConfig config_;
    cudaStream_t stream_;
    PerformanceMetrics last_metrics_;
};

}  // namespace gpu
}  // namespace nerve

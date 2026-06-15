#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#if __has_include(<cublasLt.h>)
#include <cublasLt.h>
#else
using cublasLtHandle_t = void *;
#endif

#if __has_include(<cublas_v2.h>)
#include <cublas_v2.h>
#else
using cublasHandle_t = void *;
#endif

#if __has_include(<cusparse.h>)
#include <cusparse.h>
#else
using cusparseHandle_t = void *;
#endif

#include "nerve/gpu/batch_manager.hpp"

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <vector>

namespace nerve
{
namespace gpu
{
class BatchedDistanceMatrix
{
public:
    explicit BatchedDistanceMatrix(const GPUBatchConfig &config);
    ~BatchedDistanceMatrix();
    void computeDistancesBatched(const float *points_a_batch, const float *points_b_batch,
                                 float *distances_batch, size_t batch_size, size_t num_points_a,
                                 size_t num_points_b, cudaStream_t stream = 0);
    void computeEuclideanDistancesBatched(const float *points_a_batch, const float *points_b_batch,
                                          float *distances_batch, size_t batch_size,
                                          size_t num_points_a, size_t num_points_b,
                                          cudaStream_t stream = 0);
    void computeManhattanDistancesBatched(const float *points_a_batch, const float *points_b_batch,
                                          float *distances_batch, size_t batch_size,
                                          size_t num_points_a, size_t num_points_b,
                                          cudaStream_t stream = 0);
    void computeCosineDistancesBatched(const float *points_a_batch, const float *points_b_batch,
                                       float *distances_batch, size_t batch_size,
                                       size_t num_points_a, size_t num_points_b,
                                       cudaStream_t stream = 0);
    void computeDistancesBatchedAsync(const float *points_a_batch, const float *points_b_batch,
                                      float *distances_batch, size_t batch_size,
                                      size_t num_points_a, size_t num_points_b,
                                      cudaStream_t stream = 0,
                                      cudaEvent_t *completion_event = nullptr);
    struct DistanceStats
    {
        double average_time_ms;
        size_t total_computations;
        double throughput_gflops;
        size_t memory_bandwidth_gb_s;
        double gpu_utilization;
    };
    DistanceStats getPerformanceStats() const;
    void resetPerformanceStats();

private:
    GPUBatchConfig config_;
    std::unique_ptr<GPUBatchMemoryManager> memory_manager_;
    cublasHandle_t cublas_handle_;
    cusparseHandle_t cusparse_handle_;
    void launchEuclideanKernel(const float *points_a, const float *points_b, float *distances,
                               size_t batch_size, size_t num_points_a, size_t num_points_b,
                               cudaStream_t stream);
    void launchManhattanKernel(const float *points_a, const float *points_b, float *distances,
                               size_t batch_size, size_t num_points_a, size_t num_points_b,
                               cudaStream_t stream);
    void launchCosineKernel(const float *points_a, const float *points_b, float *distances,
                            size_t batch_size, size_t num_points_a, size_t num_points_b,
                            cudaStream_t stream);
    void updatePerformanceStats(double time_ms, size_t batch_size, size_t num_points_a,
                                size_t num_points_b);
};
class BatchedPersistenceImage
{
public:
    struct ImageConfig
    {
        size_t width = 256;
        size_t height = 256;
        float sigma = 1.0f;
        float birth_min = 0.0f;
        float birth_max = 1.0f;
        float death_min = 0.0f;
        float death_max = 1.0f;
        bool normalize = true;
        bool use_antialiasing = true;
        bool use_atomic_operations = true;
    };
    explicit BatchedPersistenceImage(const ImageConfig &image_config, const GPUBatchConfig &config);
    void rasterizeBatched(const float *birth_batch, const float *death_batch,
                          const float *weights_batch, float *image_batch, size_t batch_size,
                          cudaStream_t stream = 0);
    void rasterizeBatchedWithGrid(const float *birth_batch, const float *death_batch,
                                  const float *weights_batch, float *image_batch,
                                  const float *x_grid, const float *y_grid, size_t batch_size,
                                  size_t width, size_t height, cudaStream_t stream = 0);
    void rasterizeBatchedFp16(const float *birth_batch, const float *death_batch,
                              const float *weights_batch, __half *image_batch, size_t batch_size,
                              cudaStream_t stream = 0);
    void rasterizeBatchedAsync(const float *birth_batch, const float *death_batch,
                               const float *weights_batch, float *image_batch, size_t batch_size,
                               cudaStream_t stream = 0, cudaEvent_t *completion_event = nullptr);
    struct RasterizationStats
    {
        double average_time_ms;
        size_t total_rasterizations;
        double throughput_images_per_sec;
        size_t memory_bandwidth_gb_s;
        double gpu_utilization;
    };
    RasterizationStats getPerformanceStats() const;
    void resetPerformanceStats();

private:
    ImageConfig image_config_;
    GPUBatchConfig gpu_config_;
    std::unique_ptr<GPUBatchMemoryManager> memory_manager_;
    void launchGaussianRasterizationKernel(const float *birth, const float *death,
                                           const float *weights, float *image, const float *x_grid,
                                           const float *y_grid, size_t batch_size, size_t width,
                                           size_t height, float sigma, float birth_min,
                                           float birth_max, float death_min, float death_max,
                                           cudaStream_t stream);
    void launchAtomicRasterizationKernel(const float *birth, const float *death,
                                         const float *weights, float *image, const float *x_grid,
                                         const float *y_grid, size_t batch_size, size_t width,
                                         size_t height, float sigma, float birth_min,
                                         float birth_max, float death_min, float death_max,
                                         cudaStream_t stream);
    void updatePerformanceStats(double time_ms, size_t batch_size);
};
class BatchedEigenSolver
{
public:
    struct EigenConfig
    {
        size_t max_matrix_size = 1000;
        size_t max_eigenpairs = 10;
        double tolerance = 1e-6;
        size_t max_iterations = 1000;
        bool use_arnoldi = false;
        bool enable_cusparse = true;
        bool enable_cublaslt = true;
    };
    explicit BatchedEigenSolver(const EigenConfig &eigen_config, const GPUBatchConfig &config);
    void computeEigenpairsBatched(const double *matrix_data_batch, const int *row_indices_batch,
                                  const int *col_indices_batch, const size_t *matrix_sizes_batch,
                                  double *eigenvalues_batch, double *eigenvectors_batch,
                                  size_t batch_size, size_t max_eigenpairs,
                                  cudaStream_t stream = 0);
    void computeSymmetricEigenpairsBatched(
        const double *matrix_data_batch, const int *row_indices_batch, const int *col_indices_batch,
        const size_t *matrix_sizes_batch, double *eigenvalues_batch, double *eigenvectors_batch,
        size_t batch_size, size_t max_eigenpairs, cudaStream_t stream = 0);
    void computeGeneralEigenpairsBatched(const double *matrix_data_batch,
                                         const int *row_indices_batch, const int *col_indices_batch,
                                         const size_t *matrix_sizes_batch,
                                         double *eigenvalues_real_batch,
                                         double *eigenvalues_imag_batch,
                                         double *eigenvectors_real_batch,
                                         double *eigenvectors_imag_batch, size_t batch_size,
                                         size_t max_eigenpairs, cudaStream_t stream = 0);
    void computeEigenpairsBatchedAsync(const double *matrix_data_batch,
                                       const int *row_indices_batch, const int *col_indices_batch,
                                       const size_t *matrix_sizes_batch, double *eigenvalues_batch,
                                       double *eigenvectors_batch, size_t batch_size,
                                       size_t max_eigenpairs, cudaStream_t stream = 0,
                                       cudaEvent_t *completion_event = nullptr);
    struct EigenStats
    {
        double average_time_ms;
        size_t total_computations;
        double throughput_eigenpairs_per_sec;
        size_t average_matrix_size;
        double gpu_utilization;
    };
    EigenStats getPerformanceStats() const;
    void resetPerformanceStats();

private:
    EigenConfig eigen_config_;
    GPUBatchConfig gpu_config_;
    std::unique_ptr<GPUBatchMemoryManager> memory_manager_;
    cusparseHandle_t cusparse_handle_;
    cublasHandle_t cublas_handle_;
    cublasLtHandle_t cublaslt_handle_;
    void launchLanczosKernel(const double *matrix_data, const int *row_indices,
                             const int *col_indices, const size_t *matrix_sizes,
                             double *eigenvalues, double *eigenvectors, size_t matrix_size,
                             size_t num_eigenpairs, double tolerance, size_t max_iterations,
                             cudaStream_t stream);
    void launchArnoldiKernel(const double *matrix_data, const int *row_indices,
                             const int *col_indices, const size_t *matrix_sizes,
                             double *eigenvalues_real, double *eigenvalues_imag,
                             double *eigenvectors_real, double *eigenvectors_imag,
                             size_t matrix_size, size_t num_eigenpairs, double tolerance,
                             size_t max_iterations, cudaStream_t stream);
    void updatePerformanceStats(double time_ms, size_t batch_size, size_t avg_matrix_size);
};
class GPUBatchManager
{
public:
    static GPUBatchManager &instance();
    void setConfig(const GPUBatchConfig &config);
    GPUBatchConfig getConfig() const;
    std::shared_ptr<BatchedDistanceMatrix> getDistanceMatrix();
    std::shared_ptr<BatchedPersistenceImage> getPersistenceImage();
    std::shared_ptr<BatchedEigenSolver> getEigenSolver();
    std::shared_ptr<GPUBatchMemoryManager> getMemoryManager();
    void computeDistancesAndImages(const float *points_a_batch, const float *points_b_batch,
                                   const float *birth_batch, const float *death_batch,
                                   float *distances_batch, float *image_batch, size_t batch_size,
                                   size_t num_points_a, size_t num_points_b, size_t image_width,
                                   size_t image_height, cudaStream_t stream = 0);
    void computeEigenpairsAndFeatures(const double *matrix_data_batch, const int *row_indices_batch,
                                      const int *col_indices_batch,
                                      const size_t *matrix_sizes_batch, double *eigenvalues_batch,
                                      double *eigenvectors_batch, float *features_batch,
                                      size_t batch_size, size_t max_eigenpairs, size_t feature_dim,
                                      cudaStream_t stream = 0);
    class MicroBatchAccumulator
    {
    public:
        explicit MicroBatchAccumulator(size_t target_batch_size, double target_latency_ms);
        bool addBatch(const std::vector<float> &data);
        void flush();
        bool isReady() const;

    private:
        size_t target_batch_size_;
        double target_latency_ms_;
        std::vector<std::vector<float>> batch_buffer_;
        std::chrono::steady_clock::time_point last_flush_;
        std::atomic<bool> ready_;
    };
    std::unique_ptr<MicroBatchAccumulator> createMicroBatchAccumulator(size_t target_batch_size,
                                                                       double target_latency_ms);
    struct GPUBatchStats
    {
        double average_batch_time_ms;
        size_t total_batches_processed;
        double throughput_batches_per_sec;
        double gpu_utilization;
        size_t memory_usage_mb;
        double fp16_speedup;
        size_t active_streams;
    };
    GPUBatchStats getGpuStats() const;
    void resetGpuStats();
    bool initializeGPU();
    void cleanupGPU();
    bool isAvailable() const;
    std::string getGpuInfo() const;
    std::vector<cudaStream_t> getAvailableStreams();
    void returnStream(cudaStream_t stream);

private:
    GPUBatchManager() = default;
    GPUBatchConfig config_;
    std::shared_ptr<BatchedDistanceMatrix> distance_matrix_;
    std::shared_ptr<BatchedPersistenceImage> persistence_image_;
    std::shared_ptr<BatchedEigenSolver> eigen_solver_;
    std::shared_ptr<GPUBatchMemoryManager> memory_manager_;
    std::vector<cudaStream_t> available_streams_;
    std::vector<bool> stream_in_use_;
    mutable std::shared_mutex mutex_;
    mutable GPUBatchStats stats_;
    void initializeStreams();
    void cleanupStreams();
    void updateGpuStats(const std::string &operation, double time_ms);
};
} // namespace gpu
} // namespace nerve

#pragma once

#include "nerve/types.hpp"

#include <cuda_runtime.h>

#include <map>
#include <string>
#include <vector>

namespace nerve::persistence::accelerated
{

/// Automatically detects GPU compute capability and selects
/// optimal kernel configurations.
struct GPUArchitecture
{
    int major;
    int minor;
    int computeCapability;
    int multiProcessorCount;
    size_t sharedMemPerBlock;
    size_t totalGlobalMem;
    int maxThreadsPerBlock;
    int maxThreadsPerMultiProcessor;

    enum class Family
    {
        Fermi,     // 2.x
        Kepler,    // 3.x
        Maxwell,   // 5.x
        Pascal,    // 6.x
        Volta,     // 7.0
        Turing,    // 7.5
        Ampere,    // 8.x
        Hopper,    // 9.0
        Ada,       // 8.9
        Blackwell, // 10.x
        Unknown
    } family;

    /// Detect current GPU architecture
    static GPUArchitecture detect();

    /// Check feature support
    bool supportsTensorCores() const;
    bool supportsAsyncCopy() const;
    bool supportsCooperativeGroups() const;
    bool supportsMultiInstanceGPU() const;

    /// Get optimal tile size for shared memory
    int getOptimalTileSize() const;

    /// Get recommended block size for occupancy
    int getOptimalBlockSize() const;

    /// Get optimal number of streams
    int getOptimalStreamCount() const;
};

/// High-performance distance matrix computation with automatic
/// architecture selection and mixed precision support.
struct DistanceMatrixOptimizer
{
    /// Compute double-precision distance matrix with the optimized CUDA launcher.
    static int compute(const double *points, double *distances, Size n_points, Size point_dim,
                       double max_radius, cudaStream_t stream = 0);

    /// Run the same distance kernel for each batch entry.
    static int computeBatch(const double *const *pointsBatch, double **distancesBatch,
                            const Size *n_points, Size point_dim, Size batch_size,
                            cudaStream_t stream = 0);

    /// Single-precision compatibility path. This does not imply tensor-core use.
    static int computeFP16(const float *points, float *distances, Size n_points, Size point_dim,
                           float max_radius, cudaStream_t stream = 0);
};

/// Warp-level optimized boundary matrix reduction with shared
/// memory caching and persistent kernels.
struct MatrixReductionOptimizer
{
    /// Reduce boundary matrix with optimizations
    static cudaError_t reduce(const uint64_t *boundaryMatrix, uint64_t *columns, int n_cols,
                              int n_words_per_col, int *pivotColumn, uint64_t *reduced,
                              cudaStream_t stream = 0);

    /// Batch reduction with CUDA Graphs
    static cudaError_t reduceBatch(const uint64_t *const *boundaryMatrices, uint64_t **columnsArray,
                                   const int *n_cols_array, int n_words_per_col, int **pivotTables,
                                   int batchSize, cudaStream_t stream = 0);

    /// Clearing optimization (Chen & Kerber)
    static cudaError_t applyClearing(const int2 *pairs, int n_pairs, uint64_t *columns, int n_cols,
                                     int n_words_per_col, bool *cleared, cudaStream_t stream = 0);
};

/// Benchmarks different kernel configurations at runtime and
/// selects optimal parameters for the current GPU.
class CudaAutoTuner
{
public:
    struct KernelConfig
    {
        dim3 block;
        dim3 grid;
        int sharedMemBytes;
        int tileSize;
        float measuredTimeMs;
    };

    /// Run auto-tuning for distance matrix kernel
    static KernelConfig tuneDistanceMatrix(Size n_points, Size point_dim, int numTrials = 10);

    /// Run auto-tuning for matrix reduction
    static KernelConfig tuneMatrixReduction(int n_cols, int n_words_per_col, int numTrials = 10);

    /// Save/load tuned configurations
    static void saveConfig(const std::string &filename);
    static void loadConfig(const std::string &filename);

private:
    static std::map<std::string, KernelConfig> configs_;
};

/// Efficiently captures and replays repeated kernel execution
/// patterns to reduce CPU launch overhead.
class CudaGraphManager
{
public:
    CudaGraphManager();
    ~CudaGraphManager();

    /// Capture distance matrix workflow
    cudaError_t captureDistanceMatrix(const double *points, double *distances, Size n_points,
                                      Size point_dim, double max_radius, cudaStream_t stream);

    /// Capture matrix reduction workflow
    cudaError_t captureMatrixReduction(uint64_t *columns, int n_cols, int n_words_per_col,
                                       int *pivotColumn, cudaStream_t stream);

    /// Replay captured graph
    cudaError_t launch(cudaStream_t stream);

    /// Check if graph is captured
    bool isCaptured() const { return captured_; }

private:
    cudaGraph_t graph_;
    cudaGraphExec_t instance_;
    bool captured_;
};

/// Manages multiple CUDA streams for overlapping data transfers
/// and kernel execution.
class StreamPool
{
public:
    explicit StreamPool(int numStreams = 0); // 0 = auto-detect
    ~StreamPool();

    /// Get stream for specific operation type
    cudaStream_t getComputeStream(int index = 0);
    cudaStream_t getTransferStream(int index = 0);
    cudaStream_t getH2DStream(int index = 0);
    cudaStream_t getD2HStream(int index = 0);

    /// Synchronize all streams
    void synchronizeAll();

    /// Get number of streams
    int getNumStreams() const { return numStreams_; }

    /// Wait for computation stream and transfer stream to complete
    void waitForComputeTransferPair(int index);

private:
    std::vector<cudaStream_t> computeStreams_;
    std::vector<cudaStream_t> transferStreams_;
    int numStreams_;
};

/// Nsight Compute compatible profiling annotations and metrics.
class CudaProfiler
{
public:
    /// Begin NVTX range for profiling
    static void beginRange(const char *name);

    /// End NVTX range
    static void endRange();

    /// Mark kernel execution
    static void markKernel(const char *name, cudaStream_t stream);

    /// Get kernel occupancy
    static float getKernelOccupancy(const void *func, int blockSize, size_t dynamicSmemSize);

    /// Print performance metrics
    static void printMetrics(cudaStream_t stream = 0);
};

} // namespace nerve::persistence::accelerated

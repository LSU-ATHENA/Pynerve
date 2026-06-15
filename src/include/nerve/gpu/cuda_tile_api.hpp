#pragma once

/// @file cuda_tile_api.hpp
/// The API exposes tile/cluster/tuning configuration and provides two execution
/// backends for distance matrix operations:
/// - CUDA distance-kernel backend for full-precision distance matrix launches.
/// - Deterministic host tiled proxy for quantized precision modes and
///   degraded-device paths.
/// Device capability queries are best-effort and degrade to conservative
/// defaults when CUDA runtime queries fail.

#include "nerve/gpu/distance_kernels.hpp"
#include "nerve/types.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace nerve::gpu::tile
{

template <typename T>
class TileBuffer;
template <typename T>
class TileView;
class TileLaunchConfig;
class TileAutoTuner;

// Tile Data Types

enum class TileDataType
{
    kFP32,     // float
    kFP64,     // double
    kFP16,     // half
    kBF16,     // bfloat16
    kFP8_E4M3, // Hopper/Blackwell FP8
    kFP8_E5M2, // Hopper/Blackwell FP8
    kFP4,      // Blackwell FP4
    kINT8,
    kINT32
};

enum class TileLayout
{
    kRowMajor,    // Standard C/C++ layout
    kColumnMajor, // Fortran/Matlab layout
    kSwizzled     // Bank-conflict-free layout for shared memory
};

enum class TilePrecision
{
    // Full-precision accumulation without quantization.
    kFull,
    // Quantize differences to float-like precision before accumulation.
    kMixed,
    // Coarser quantization step than kMixed.
    kLow,
    // Most aggressive quantization used by this API.
    kUltraLow
};

// Tile Configuration

struct TileConfig
{
    int tileSizeM = 64;   // Rows per tile
    int tileSizeN = 64;   // Cols per tile
    int tileSizeK = 16;   // Inner dimension (for matmul)
    int clusterSizeX = 1; // Thread blocks in cluster X
    int clusterSizeY = 1; // Thread blocks in cluster Y
    int clusterSizeZ = 1; // Thread blocks in cluster Z
    TileDataType dataType = TileDataType::kFP32;
    TileLayout layout = TileLayout::kRowMajor;
    TilePrecision precision = TilePrecision::kFull;
    // Tuning hints retained in config/tuner state. The current implementation
    // uses standard CUDA kernels or host implementation paths, so these default off.
    bool useTMA = false;
    bool useTensorCores = false;

    // Validation
    bool isValid() const;
    size_t sharedMemSize() const;
    int totalClusterSize() const
    {
        if (clusterSizeX <= 0 || clusterSizeY <= 0 || clusterSizeZ <= 0)
        {
            return 0;
        }
        const int max_value = std::numeric_limits<int>::max();
        if (clusterSizeX > max_value / clusterSizeY)
        {
            return max_value;
        }
        const int xy = clusterSizeX * clusterSizeY;
        return xy > max_value / clusterSizeZ ? max_value : xy * clusterSizeZ;
    }
};

struct DistanceTileConfig
{
    int pointTileSize = 256;    // Points per tile
    int pointDim = 3;           // Dimensionality
    int clusterSize = 1;        // Requested cluster size for future device paths
    float maxRadius = 1.0f;     // Distance threshold
    bool useClustering = false; // Keep default launch semantics single-block

    // Convert to generic TileConfig
    TileConfig toTileConfig(TileDataType dtype = TileDataType::kFP32) const;
};

// Tile Buffer Management

template <typename T>
class TileBuffer
{
public:
    /// @brief Create tile buffer with specified dimensions
    TileBuffer(int rows, int cols, TileLayout layout = TileLayout::kRowMajor);

    /// @brief Create from existing device pointer (non-owning)
    TileBuffer(T *data, int rows, int cols, bool owning = false);

    ~TileBuffer();

    // Move semantics only
    TileBuffer(TileBuffer &&other) noexcept;
    TileBuffer &operator=(TileBuffer &&other) noexcept;

    TileBuffer(const TileBuffer &) = delete;
    TileBuffer &operator=(const TileBuffer &) = delete;

    // Access
    T *data() { return data_; }
    const T *data() const { return data_; }
    int rows() const { return rows_; }
    int cols() const { return cols_; }
    size_t size() const
    {
        if (rows_ <= 0 || cols_ <= 0)
        {
            return 0;
        }
        const size_t row_count = static_cast<size_t>(rows_);
        const size_t col_count = static_cast<size_t>(cols_);
        if (row_count > std::numeric_limits<size_t>::max() / col_count)
        {
            return 0;
        }
        return row_count * col_count;
    }
    size_t bytes() const
    {
        const size_t elements = size();
        if (elements > std::numeric_limits<size_t>::max() / sizeof(T))
        {
            return 0;
        }
        return elements * sizeof(T);
    }
    TileLayout layout() const { return layout_; }

    // Pitch (for pitched allocations)
    size_t pitch() const { return pitch_; }

    // Copy operations
    void copyFromHost(const T *hostData);
    void copyToHost(T *hostData) const;
    void copyFromDevice(const T *deviceData);

    // Views
    TileView<T> view(int rowStart, int colStart, int rows, int cols);
    TileView<const T> view(int rowStart, int colStart, int rows, int cols) const;

private:
    T *data_ = nullptr;
    int rows_ = 0;
    int cols_ = 0;
    size_t pitch_ = 0;
    TileLayout layout_ = TileLayout::kRowMajor;
    bool owning_ = true;
};

// Tile Views

template <typename T>
class TileView
{
public:
    TileView(T *data, int rows, int cols, size_t pitch = 0);

    // Element access
    T &at(int row, int col);
    const T &at(int row, int col) const;

    // Row access
    T *row(int r);
    const T *row(int r) const;

    // Properties
    int rows() const { return rows_; }
    int cols() const { return cols_; }
    size_t pitch() const { return pitch_; }
    T *data() { return data_; }
    const T *data() const { return data_; }

    // Sub-view
    TileView<T> subview(int rowStart, int colStart, int rows, int cols);
    TileView<const T> subview(int rowStart, int colStart, int rows, int cols) const;

private:
    T *data_ = nullptr;
    int rows_ = 0;
    int cols_ = 0;
    size_t pitch_ = 0;
};

// Tile Kernel Launcher

class TileLauncher
{
public:
    explicit TileLauncher(const TileConfig &config);

    /// @brief Launch distance matrix computation
    template <typename T>
    cudaError_t launchDistanceMatrix(const TileBuffer<T> &points, TileBuffer<T> &distances,
                                     float maxRadius, cudaStream_t stream = 0);

    /// @brief Launch reduction (host implementation implementation).
    template <typename T>
    cudaError_t launchReduction(const TileBuffer<T> &input, TileBuffer<T> &output,
                                cudaStream_t stream = 0);

    /// @brief Launch transpose (host implementation implementation).
    template <typename T>
    cudaError_t launchTranspose(const TileBuffer<T> &input, TileBuffer<T> &output,
                                cudaStream_t stream = 0);

    /// @brief Launch elementwise tile op (host implementation implementation).
    template <typename T, typename Func>
    cudaError_t launchTileOp(const TileBuffer<T> &input, TileBuffer<T> &output, Func operation,
                             cudaStream_t stream = 0);

    /// @brief Get required shared memory size
    size_t getSharedMemSize() const;

    /// @brief Get optimal grid dimensions for workload
    dim3 getGridDim(int rows, int cols) const;

    /// @brief Get optimal block dimensions
    dim3 getBlockDim() const;

private:
    TileConfig config_;
    void *kernelArgs_[16]; // Maximum 16 kernel arguments
};

// Tile Auto-Tuner

class TileAutoTuner
{
public:
    struct TuningResult
    {
        TileConfig config;
        float timeMs;
        float throughput;
        float occupancy;
    };

    /// @brief Tune for specific workload
    static TileConfig tuneDistanceMatrix(uint32_t nPoints, uint32_t pointDim,
                                         TileDataType dtype = TileDataType::kFP32,
                                         int deviceId = 0);

    /// @brief Tune for matrix multiplication
    static TileConfig tuneMatMul(uint32_t m, uint32_t n, uint32_t k,
                                 TileDataType dtype = TileDataType::kFP32, int deviceId = 0);

    /// @brief Tune for reduction
    static TileConfig tuneReduction(uint32_t nElements, TileDataType dtype = TileDataType::kFP32,
                                    int deviceId = 0);

    /// @brief Get all benchmarked configurations
    static std::vector<TuningResult>
    benchmarkAll(const void *workloadData, std::function<float(const TileConfig &)> benchmarkFunc);

    /// @brief Save tuned configuration to cache
    static void saveTunedConfig(const std::string &key, const TileConfig &config);

    /// @brief Load tuned configuration from cache
    static std::optional<TileConfig> loadTunedConfig(const std::string &key);
};

// Cluster Utilities

/// host semantics.
class ClusterUtils
{
public:
    /// @brief Get current cluster rank in host mode (always 0).
    static int getClusterRank();

    /// @brief Get total cluster size in host mode (always 1).
    static int getClusterSize();

    /// @brief Synchronize the active CUDA device in host mode.
    static void clusterSync();

    /// @brief Get distributed shared-memory pointer (unsupported in host mode).
    template <typename T>
    static T *getDistributedSmem(int targetBlockRank);

    /// @brief Query whether device reports thread-block cluster support.
    static bool clustersSupported(int deviceId = 0);

    /// @brief Get maximum cluster size for device
    static int getMaxClusterSize(int deviceId = 0);

    /// @brief Check if 16-block clusters supported (B200 only)
    static bool supports16BlockClusters(int deviceId = 0);

    /// @brief Calculate optimal cluster size for workload
    static int calculateOptimalClusterSize(int nBlocks, int smCount, int computeCapability);
};

// Convenience API

template <typename T>
TileBuffer<T> tileDistanceMatrix(const T *points, uint32_t nPoints, uint32_t pointDim,
                                 float maxRadius, TilePrecision precision = TilePrecision::kFull,
                                 int deviceId = 0);

template <typename T>
T tileReduce(const T *data, uint32_t nElements, int deviceId = 0);

inline bool tileApiAvailable()
{
    int device_count = 0;
    return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

std::string getTileApiVersion();

} // namespace nerve::gpu::tile

#ifndef NERVE_CUDA_TILE_API_DECLARATIONS_ONLY
#include "nerve/gpu/detail/cuda_tile_buffer_impl.hpp"
#include "nerve/gpu/detail/cuda_tile_cluster_impl.hpp"
#include "nerve/gpu/detail/cuda_tile_config_impl.hpp"
#include "nerve/gpu/detail/cuda_tile_detail_impl.hpp"
#include "nerve/gpu/detail/cuda_tile_launcher_impl.hpp"
#include "nerve/gpu/detail/cuda_tile_tuner_impl.hpp"
#endif

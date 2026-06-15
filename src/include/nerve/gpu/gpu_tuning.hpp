#pragma once

/// @file gpu_tuning.hpp

#include "gpu_capability_core.hpp"

#include <string>
#include <vector>

namespace nerve::gpu::advanced
{

struct PipelineConfig
{
    int numStages = 3;  // 2-4 optimal
    int tileSizeM = 64; // M dimension tile
    int tileSizeN = 64; // N dimension tile
    int tileSizeK = 16; // K dimension tile
    bool useTMA = true;
    bool useAsyncBarriers = true;
};

class PipelineLauncher
{
public:
    explicit PipelineLauncher(const PipelineConfig &config);

    [[nodiscard]] cudaError_t launchDistanceMatrix(const void *points, void *distances, int nPoints,
                                                   int pointDim, cudaStream_t stream);

    [[nodiscard]] cudaError_t launchMatrixMultiply(const void *A, const void *B, void *C, int m,
                                                   int n, int k, cudaStream_t stream);

private:
    PipelineConfig config_;
};

struct PersistentConfig
{
    int threadsPerBlock = 256;
    int blocksPerSM = 8;       // Maximum occupancy
    int warpsPerBlock = 8;     // 256/32
    int workQueueSize = 10000; // Total work items
};

struct ReductionWorkItem
{
    uint64_t *columnData;
    int columnIndex;
    int wordsPerColumn;
};

class PersistentWorkQueue
{
public:
    void initialize(int numItems);
    void setItem(int index, const ReductionWorkItem &item);

    [[nodiscard]] ReductionWorkItem *getItems() const { return items_; }
    [[nodiscard]] int getNumItems() const { return numItems_; }

private:
    ReductionWorkItem *items_ = nullptr;
    int numItems_ = 0;
};

struct WgmmaConfig
{
    int m = 64;              // Accumulator M dimension
    int n = 8;               // Accumulator N dimension
    int k = 16;              // Operand K dimension
    bool useTF32 = true;     // TF32 operands
    bool accumInFP32 = true; // FP32 accumulators
};

class WgmmaEngine
{
public:
    explicit WgmmaEngine(const WgmmaConfig &config);

    /// Initialize accumulator registers
    __device__ void initAccumulator(float *accum, int size);

    /// Issue WGMMA instruction
    __device__ void mma(float *accum, uint64_t descA, uint64_t descB, int accumOffset);

    /// Commit WGMMA group
    __device__ void commitGroup();

    /// Wait for WGMMA completion
    __device__ void waitGroup(int n);

private:
    WgmmaConfig config_;
};

struct ClusterConfig
{
    int clusterDimX = 4; // 4 blocks in X
    int clusterDimY = 1;
    int clusterDimZ = 1;
    int sharedMemPerBlock = 64 * 1024; // 64KB
};

class DistributedSharedMemory
{
public:
    explicit DistributedSharedMemory(const ClusterConfig &config);

    [[nodiscard]] int getClusterRank() const;
    [[nodiscard]] int getClusterSize() const;

    /// Access another block's shared memory
    template <typename T>
    __device__ T *getBlockSharedMem(int blockRank) const;

    /// Synchronize entire cluster
    __device__ void clusterSync() const;

private:
    ClusterConfig config_;
};

class AdvancedProfiler
{
public:
    static void beginRange(const char *name);
    static void endRange();
    static void markKernel(const char *name);
    static void markMemory(const char *name, size_t bytes);

    [[nodiscard]] static float getTheoreticalOccupancy(const void *kernel, int blockSize,
                                                       size_t dynamicSmem);

    [[nodiscard]] static float getAchievedOccupancy();
};

struct TmaTileConfig
{
    int pointTileSize = 64;
    int dimTileSize = 16;
    int clusterSizeX = 4;
    int clusterSizeY = 1;
    int smemSwizzle = 32;
    int l2Promotion = 128;
    int numStages = 3;
};

class TmaTileTuner
{
public:
    struct WorkloadProfile
    {
        uint32_t nPoints;
        uint32_t pointDim;
        float sparsityEstimate;
        int gpuComputeCapability;
        size_t availableSmem;
    };

    [[nodiscard]] static TmaTileConfig tune(const WorkloadProfile &profile, int numTrials = 10);
    [[nodiscard]] static TmaTileConfig getOrTune(const std::string &cacheKey,
                                                 const WorkloadProfile &profile);
};

} // namespace nerve::gpu::advanced

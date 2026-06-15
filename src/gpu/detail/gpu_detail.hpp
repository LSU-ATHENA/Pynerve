#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace nerve
{

using Dimension = int;
using Index = std::int64_t;
using Size = std::size_t;

namespace gpu
{

class ComputeManager
{
public:
    static ComputeManager &getInstance();
    void initialize();
    void shutdown();
    bool isAvailable() const;
    void setMemoryLimitMB(size_t limit);
    void setUseOptimizedKernels(bool use);
    void setUseCudaGraphs(bool use);
    std::string getGPUArchitectureName() const;

    // computeDistanceMatrix returns some error type with isError()
    struct Error
    {
        bool isError() const { return false; }
    };
    Error computeDistanceMatrix(const std::vector<std::vector<double>> &points,
                                std::vector<std::vector<double>> &out_distances);

private:
    ComputeManager() = default;
    ~ComputeManager();
    ComputeManager(const ComputeManager &) = delete;
    ComputeManager &operator=(const ComputeManager &) = delete;
};

struct GPUBatchConfig
{
    size_t memory_pool_size = 0;
    bool enable_pinned_memory = false;
};

class GPUBatchMemoryManager
{
public:
    explicit GPUBatchMemoryManager(const GPUBatchConfig &config);
    ~GPUBatchMemoryManager();

    GPUBatchMemoryManager(const GPUBatchMemoryManager &) = delete;
    GPUBatchMemoryManager &operator=(const GPUBatchMemoryManager &) = delete;

    void *allocate(size_t size, size_t alignment = 256);
    void deallocate(void *ptr);
    void reset();

    static bool isPowerOfTwo(size_t value) noexcept;
    static size_t alignUp(size_t value, size_t alignment) noexcept;
};

namespace tuning
{

struct GpuSignature
{
    int computeCapability = 0;
    int smCount = 0;
    int clockRate = 0;
    size_t totalMemory = 0;
    std::string uuid;
    std::string name;
};

struct WorkloadFingerprint
{
    unsigned nPoints = 0;
    unsigned pointDim = 0;
    float sparsityEstimate = 0.5f;
    unsigned problemType = 0;
    unsigned flags = 0;
};

struct TunedConfig
{
    int blockSize = 256;
    int tileSize = 64;
    int clusterSize = 4;
    int numStages = 3;
    bool useWGMMA = true;
};

struct TuningStats
{
    size_t totalEntries = 0;
};

class GpuTuningDatabase
{
public:
    static GpuTuningDatabase &instance();
    void initialize(const std::string &cacheDir);
    std::optional<TunedConfig> lookup(const GpuSignature &gpu, const WorkloadFingerprint &workload);
    void store(const GpuSignature &gpu, const WorkloadFingerprint &workload,
               const TunedConfig &config);
    void clear();
    TuningStats getStats() const;
};

} // namespace tuning

namespace async
{

class CUDAStream
{
public:
    explicit CUDAStream(int device);
    void synchronize();
};

class CUDAEvent
{
public:
    CUDAEvent();
    void record(CUDAStream &stream);
    void synchronize();
};

class AsyncExecutor
{
public:
    explicit AsyncExecutor(size_t max_workers);
    std::future<void> submit(std::function<void()> task);
    void synchronizeAll();
};

} // namespace async

} // namespace gpu
} // namespace nerve

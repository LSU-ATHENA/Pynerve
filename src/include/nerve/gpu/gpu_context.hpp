#pragma once

#include "nerve/types.hpp"

#include <cuda_runtime.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(CUDA_VERSION) && CUDA_VERSION >= 13010
#define NERVE_HAS_GREEN_CONTEXTS 1
#else
#define NERVE_HAS_GREEN_CONTEXTS 0
#endif

namespace nerve::gpu
{

constexpr size_t BYTES_PER_KB = 1024ULL;
constexpr size_t BYTES_PER_MB = 1024ULL * 1024ULL;
constexpr size_t BYTES_PER_GB = 1024ULL * 1024ULL * 1024ULL;

constexpr size_t DEFAULT_WORKLOAD_MEMORY = BYTES_PER_GB;             // 1GB default
constexpr size_t DEFAULT_CONTEXT_MEMORY_LIMIT = 8ULL * BYTES_PER_GB; // 8GB

class GreenContextManager;
class MultiTenantManager;

using GreenContextId = uint64_t;
using WorkloadId = uint64_t;

struct GreenContext
{
    GreenContextId id = 0;
    int gpuDevice = 0;
    float computeFraction = 1.0f; // 0.0-1.0 of GPU SMs
    int allocatedSMs = 0;
    size_t memoryLimit = 0;
    std::atomic<size_t> currentMemoryUsage{0};
    std::unordered_map<void *, size_t> allocatedMemoryMap;
    int priority = 0;            // Higher = more priority
    float throttleFactor = 1.0f; // 0.0-1.0 dynamic throttling
    bool active = false;
    std::vector<cudaStream_t> streams;
    std::atomic<int> nextStream{0};
    cudaEvent_t startEvent = nullptr;
    cudaEvent_t endEvent = nullptr;

    GreenContext() = default;
    ~GreenContext() = default;

    GreenContext(const GreenContext &) = delete;
    GreenContext &operator=(const GreenContext &) = delete;

    GreenContext(GreenContext &&other) noexcept
        : id(other.id)
        , gpuDevice(other.gpuDevice)
        , computeFraction(other.computeFraction)
        , allocatedSMs(other.allocatedSMs)
        , memoryLimit(other.memoryLimit)
        , currentMemoryUsage(other.currentMemoryUsage.load())
        , allocatedMemoryMap(std::move(other.allocatedMemoryMap))
        , priority(other.priority)
        , throttleFactor(other.throttleFactor)
        , active(other.active)
        , streams(std::move(other.streams))
        , nextStream(other.nextStream.load())
        , startEvent(other.startEvent)
        , endEvent(other.endEvent)
    {
        other.id = 0;
        other.currentMemoryUsage.store(0);
        other.nextStream.store(0);
        other.startEvent = nullptr;
        other.endEvent = nullptr;
    }

    GreenContext &operator=(GreenContext &&other) noexcept
    {
        if (this != &other)
        {
            id = other.id;
            gpuDevice = other.gpuDevice;
            computeFraction = other.computeFraction;
            allocatedSMs = other.allocatedSMs;
            memoryLimit = other.memoryLimit;
            currentMemoryUsage.store(other.currentMemoryUsage.load());
            allocatedMemoryMap = std::move(other.allocatedMemoryMap);
            priority = other.priority;
            throttleFactor = other.throttleFactor;
            active = other.active;
            streams = std::move(other.streams);
            nextStream.store(other.nextStream.load());
            startEvent = other.startEvent;
            endEvent = other.endEvent;
            other.id = 0;
            other.currentMemoryUsage.store(0);
            other.nextStream.store(0);
            other.startEvent = nullptr;
            other.endEvent = nullptr;
        }
        return *this;
    }
};

struct GreenContextMetrics
{
    GreenContextId contextId = 0;
    int gpuDevice = 0;

    int allocatedSMs = 0;
    int totalSMsOnGpu = 0;
    float computeUtilization = 0.0f; // 0.0-1.0
    size_t memoryUsed = 0;
    size_t memoryLimit = 0;
    float memoryUtilization = 0.0f; // 0.0-1.0
    float powerUsageWatts = 0.0f;
    float temperatureCelsius = 0.0f;
    uint64_t bytesTransferred = 0;

    bool active = false;
};

struct ClusterMetrics
{
    int numGpus = 0;
    int numActiveContexts = 0;
    float aggregateComputeUtilization = 0.0f;
    size_t aggregateMemoryUsed = 0;
    size_t aggregateMemoryAvailable = 0;
    float nvLinkBandwidthUtilization = 0.0f;
};

class GreenContextManager
{
public:
    GreenContextManager();
    ~GreenContextManager();

    GreenContextManager(const GreenContextManager &) = delete;
    GreenContextManager &operator=(const GreenContextManager &) = delete;

    GreenContext createContext(float computeFraction, size_t memoryLimit, int priority = 0);

    GreenContext createContextOnDevice(int deviceId, float computeFraction, size_t memoryLimit,
                                       int priority = 0);

    void destroyContext(GreenContextId id);

    cudaError_t launchInContext(GreenContextId ctxId, void *kernel, dim3 grid, dim3 block,
                                void **args, size_t sharedMem = 0);

    cudaError_t launchInContextAsync(GreenContextId ctxId, void *kernel, dim3 grid, dim3 block,
                                     void **args, cudaStream_t stream, size_t sharedMem = 0);

    cudaError_t synchronizeContext(GreenContextId ctxId);
    void *allocateInContext(GreenContextId ctxId, size_t size);
    void freeInContext(GreenContextId ctxId, void *ptr);
    void setContextPriority(GreenContextId ctxId, int priority);
    void throttleContext(GreenContextId ctxId, float throttleFactor);
    GreenContextMetrics getContextMetrics(GreenContextId ctxId);
    std::vector<GreenContextId> getActiveContexts() const;
    void enableContext(GreenContextId ctxId);
    void disableContext(GreenContextId ctxId);
    size_t getTotalGpuMemory() const;
    ClusterMetrics getClusterMetrics() const;
    void printContextStats(GreenContextId ctxId) const;
    static bool isAvailable();

private:
    void initialize();
    void destroyContextInternal(GreenContext &ctx);
    int calculateAllocatedSMs(float fraction, int deviceId);

    std::unordered_map<GreenContextId, GreenContext> contexts_;
    std::vector<int> gpuDevices_;
    GreenContextId nextContextId_;
};

enum class WorkloadStatus
{
    kPending,
    kRunning,
    kCompleted,
    kCancelled,
    kError
};

struct WorkloadDescription
{
    std::string name;
    float computeFraction = 0.1f;
    size_t memoryRequirement = DEFAULT_WORKLOAD_MEMORY; // 1GB default
    int priority = 0;
    int minGpuCount = 1;
    int maxGpuCount = 1;
    bool requiresNVLink = false;
    bool cancelled = false;
};

struct WorkloadHandle
{
    WorkloadId id = 0;
    GreenContextId assignedContext = 0;
    WorkloadStatus status = WorkloadStatus::kPending;
    float progress = 0.0f;
    std::string errorMessage;
};

class MultiTenantManager
{
public:
    explicit MultiTenantManager(GreenContextManager &manager);

    WorkloadHandle submitWorkload(const WorkloadDescription &desc,
                                  GreenContextId preferredContext = 0);

    void cancelWorkload(WorkloadHandle handle);
    void prioritizeWorkload(WorkloadHandle handle, int priority);
    WorkloadStatus getWorkloadStatus(WorkloadId id) const;
    void rebalanceWorkloads();
    std::vector<WorkloadHandle> getActiveWorkloads() const;
    bool waitForWorkload(WorkloadHandle handle, int timeoutMs = -1);
    void setRebalancingPolicy(bool autoRebalance, int rebalanceIntervalMs = 100);

private:
    GreenContextId findBestContext(const WorkloadDescription &desc);

    GreenContextManager &contextManager_;
    std::unordered_map<WorkloadId, WorkloadDescription> workloads_;
    std::unordered_map<WorkloadId, WorkloadHandle> handles_;
    std::atomic<WorkloadId> nextWorkloadId_{1};
    bool autoRebalance_ = false;
    int rebalanceIntervalMs_ = 100;
};

struct QoSConfig
{
    float minComputeFraction = 0.1f; // Minimum guaranteed compute
    float maxComputeFraction = 1.0f; // Maximum allowed compute
    size_t minMemoryMB = 1024;       // Minimum guaranteed memory
    size_t maxMemoryMB = 0;          // 0 = unlimited
    int maxLatencyMs = 100;          // Maximum kernel launch latency
    float targetUtilization = 0.8f;  // Target GPU utilization
};

class QoSManager
{
public:
    explicit QoSManager(GreenContextManager &manager);

    void setQoS(GreenContextId ctxId, const QoSConfig &config);
    bool checkQoS(GreenContextId ctxId) const;
    std::string getQoSReport(GreenContextId ctxId) const;
    void enforceQoS(GreenContextId ctxId);
    void enableAutoQoS(int checkIntervalMs = 50);

private:
    GreenContextManager &manager_;
    std::unordered_map<GreenContextId, QoSConfig> qosConfigs_;
};

GreenContext createGreenContext(float computeFraction = 0.25f,
                                size_t memoryLimit = DEFAULT_CONTEXT_MEMORY_LIMIT, // 8GB
                                int priority = 0);

void destroyGreenContext(GreenContextId id);

template <typename KernelFunc, typename... Args>
cudaError_t greenLaunch(GreenContextId ctxId, KernelFunc kernel, dim3 grid, dim3 block,
                        std::tuple<Args...> args, size_t sharedMem = 0)
{
    static GreenContextManager manager;
    return std::apply(
        [&](auto &...unpacked) {
            std::array<void *, sizeof...(Args)> argPtrs{static_cast<void *>(&unpacked)...};
            return manager.launchInContext(ctxId, (void *)kernel, grid, block,
                                           argPtrs.empty() ? nullptr : argPtrs.data(), sharedMem);
        },
        args);
}

inline bool greenContextsAvailable()
{
#if NERVE_HAS_GREEN_CONTEXTS
    return GreenContextManager::isAvailable();
#else
    return false;
#endif
}

std::string getGreenContextVersion();

inline bool GreenContextManager::isAvailable()
{
#if NERVE_HAS_GREEN_CONTEXTS
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    return (err == cudaSuccess && deviceCount > 0);
#else
    return false;
#endif
}

} // namespace nerve::gpu

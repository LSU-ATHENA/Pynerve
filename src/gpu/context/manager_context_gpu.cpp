#include "nerve/gpu/gpu_context.hpp"

#include <cuda_runtime.h>
#include <nvml.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <thread>

#if defined(CUDA_VERSION) && CUDA_VERSION >= 13010
namespace nerve::gpu
{

GreenContextManager::GreenContextManager()
{
    initialize();
}
GreenContextManager::~GreenContextManager()
{
    for (auto &[id, ctx] : contexts_)
    {
        destroyContextInternal(ctx);
    }
    contexts_.clear();
    nvmlShutdown();
}

void GreenContextManager::initialize()
{
    (void)nvmlInit();

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err == cudaSuccess)
    {
        for (int i = 0; i < deviceCount; ++i)
        {
            gpuDevices_.push_back(i);
        }
    }

    nextContextId_ = 1;
}

GreenContext GreenContextManager::createContext(float computeFraction, size_t memoryLimit,
                                                int priority)
{
    const int deviceId = gpuDevices_.empty() ? 0 : gpuDevices_.front();
    return createContextOnDevice(deviceId, computeFraction, memoryLimit, priority);
}

GreenContext GreenContextManager::createContextOnDevice(int deviceId, float computeFraction,
                                                        size_t memoryLimit, int priority)
{
    if (computeFraction <= 0.0f || computeFraction > 1.0f)
    {
        throw std::invalid_argument("computeFraction must be in (0, 1]");
    }
    if (memoryLimit == 0)
    {
        throw std::invalid_argument("memoryLimit must be > 0");
    }
    if (!gpuDevices_.empty() &&
        std::find(gpuDevices_.begin(), gpuDevices_.end(), deviceId) == gpuDevices_.end())
    {
        throw std::invalid_argument("Invalid device ID: " + std::to_string(deviceId));
    }

    GreenContext ctx;
    ctx.id = nextContextId_++;
    ctx.computeFraction = computeFraction;
    ctx.memoryLimit = memoryLimit;
    ctx.priority = priority;
    ctx.active = true;
    ctx.gpuDevice = deviceId;
    ctx.allocatedSMs = calculateAllocatedSMs(computeFraction, ctx.gpuDevice);

    cudaError_t err = cudaSetDevice(ctx.gpuDevice);
    if (err != cudaSuccess)
    {
        throw std::runtime_error("Failed to set CUDA device: " +
                                 std::string(cudaGetErrorString(err)));
    }

    ctx.streams.resize(4);
    for (auto &stream : ctx.streams)
    {
        cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, priority);
    }

    (void)cudaDeviceSetLimit(cudaLimitMallocHeapSize, memoryLimit);

    contexts_[ctx.id] = ctx;
    return contexts_.at(ctx.id);
}

void GreenContextManager::destroyContext(GreenContextId id)
{
    auto it = contexts_.find(id);
    if (it == contexts_.end())
    {
        return;
    }

    destroyContextInternal(it->second);
    contexts_.erase(it);
}

void GreenContextManager::destroyContextInternal(GreenContext &ctx)
{
    for (auto &stream : ctx.streams)
    {
        if (stream)
        {
            cudaStreamDestroy(stream);
            stream = nullptr;
        }
    }

    for (auto &[ptr, size] : ctx.allocatedMemoryMap)
    {
        cudaFree(ptr);
    }
    ctx.allocatedMemoryMap.clear();

    ctx.active = false;
}

cudaError_t GreenContextManager::launchInContext(GreenContextId ctxId, void *kernel, dim3 grid,
                                                 dim3 block, void **args, size_t sharedMem)
{
    auto it = contexts_.find(ctxId);
    if (it == contexts_.end())
    {
        return cudaErrorInvalidDevice;
    }

    GreenContext &ctx = it->second;
    cudaError_t err = cudaSetDevice(ctx.gpuDevice);
    if (err != cudaSuccess)
    {
        return err;
    }

    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);
    if (freeMem < ctx.memoryLimit / 10)
    {
        return cudaErrorMemoryAllocation;
    }

    cudaStream_t stream = nullptr;
    if (!ctx.streams.empty())
    {
        const int streamIdx = ctx.nextStream.fetch_add(1) % static_cast<int>(ctx.streams.size());
        stream = ctx.streams[streamIdx];
    }

    return cudaLaunchKernel(kernel, grid, block, args, sharedMem, stream);
}

cudaError_t GreenContextManager::launchInContextAsync(GreenContextId ctxId, void *kernel, dim3 grid,
                                                      dim3 block, void **args, cudaStream_t stream,
                                                      size_t sharedMem)
{
    auto it = contexts_.find(ctxId);
    if (it == contexts_.end())
    {
        return cudaErrorInvalidDevice;
    }

    GreenContext &ctx = it->second;
    cudaError_t err = cudaSetDevice(ctx.gpuDevice);
    if (err != cudaSuccess)
    {
        return err;
    }

    return cudaLaunchKernel(kernel, grid, block, args, sharedMem, stream);
}

cudaError_t GreenContextManager::synchronizeContext(GreenContextId ctxId)
{
    auto it = contexts_.find(ctxId);
    if (it == contexts_.end())
    {
        return cudaErrorInvalidDevice;
    }

    GreenContext &ctx = it->second;
    cudaError_t err = cudaSetDevice(ctx.gpuDevice);
    if (err != cudaSuccess)
    {
        return err;
    }
    for (auto &stream : ctx.streams)
    {
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess)
        {
            return err;
        }
    }

    return cudaSuccess;
}

void *GreenContextManager::allocateInContext(GreenContextId ctxId, size_t size)
{
    auto it = contexts_.find(ctxId);
    if (it == contexts_.end())
    {
        return nullptr;
    }

    GreenContext &ctx = it->second;
    size_t currentUsage = ctx.currentMemoryUsage.load();
    if (size > ctx.memoryLimit || currentUsage > ctx.memoryLimit - size)
    {
        return nullptr;
    }

    cudaError_t err = cudaSetDevice(ctx.gpuDevice);
    if (err != cudaSuccess)
    {
        return nullptr;
    }

    void *ptr = nullptr;
    err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess)
    {
        return nullptr;
    }

    try
    {
        ctx.allocatedMemoryMap[ptr] = size;
    }
    catch (...)
    {
        cudaFree(ptr);
        return nullptr;
    }
    ctx.currentMemoryUsage.fetch_add(size);
    return ptr;
}

void GreenContextManager::freeInContext(GreenContextId ctxId, void *ptr)
{
    auto it = contexts_.find(ctxId);
    if (it == contexts_.end())
    {
        return;
    }

    GreenContext &ctx = it->second;
    auto memIt = ctx.allocatedMemoryMap.find(ptr);
    if (memIt == ctx.allocatedMemoryMap.end())
    {
        return;
    }

    size_t size = memIt->second;
    cudaSetDevice(ctx.gpuDevice);
    cudaFree(ptr);

    ctx.allocatedMemoryMap.erase(memIt);
    ctx.currentMemoryUsage.fetch_sub(size);
}

void GreenContextManager::setContextPriority(GreenContextId ctxId, int priority)
{
    auto it = contexts_.find(ctxId);
    if (it == contexts_.end())
    {
        return;
    }

    GreenContext &ctx = it->second;
    ctx.priority = priority;
    for (auto &stream : ctx.streams)
    {
        cudaStreamDestroy(stream);
        cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, priority);
    }
}

void GreenContextManager::throttleContext(GreenContextId ctxId, float throttleFactor)
{
    auto it = contexts_.find(ctxId);
    if (it == contexts_.end())
    {
        return;
    }

    GreenContext &ctx = it->second;
    ctx.throttleFactor = std::max(0.0f, std::min(1.0f, throttleFactor));
    float newFraction = ctx.computeFraction * ctx.throttleFactor;
    ctx.allocatedSMs = calculateAllocatedSMs(newFraction, ctx.gpuDevice);
}

int GreenContextManager::calculateAllocatedSMs(float fraction, int deviceId)
{
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, deviceId);
    if (err != cudaSuccess)
    {
        return 0;
    }

    const int allocated = static_cast<int>(prop.multiProcessorCount * fraction);
    return prop.multiProcessorCount > 0 ? std::max(1, allocated) : 0;
}

GreenContextMetrics GreenContextManager::getContextMetrics(GreenContextId ctxId)
{
    GreenContextMetrics metrics = {};

    auto it = contexts_.find(ctxId);
    if (it == contexts_.end())
    {
        return metrics;
    }

    GreenContext &ctx = it->second;

    metrics.contextId = ctxId;
    metrics.gpuDevice = ctx.gpuDevice;
    metrics.allocatedSMs = ctx.allocatedSMs;
    metrics.memoryUsed = ctx.currentMemoryUsage.load();
    metrics.memoryLimit = ctx.memoryLimit;

    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, ctx.gpuDevice) == cudaSuccess)
    {
        metrics.totalSMsOnGpu = prop.multiProcessorCount;
    }

    nvmlDevice_t nvmlDev;
    nvmlReturn_t result = nvmlDeviceGetHandleByIndex(ctx.gpuDevice, &nvmlDev);
    if (result == NVML_SUCCESS)
    {
        nvmlUtilization_t utilization;
        result = nvmlDeviceGetUtilizationRates(nvmlDev, &utilization);
        if (result == NVML_SUCCESS)
        {
            metrics.computeUtilization = utilization.gpu / 100.0f;
            metrics.memoryUtilization = utilization.memory / 100.0f;
        }
    }

    metrics.active = ctx.active;

    return metrics;
}

std::vector<GreenContextId> GreenContextManager::getActiveContexts() const
{
    std::vector<GreenContextId> active;
    for (const auto &[id, ctx] : contexts_)
    {
        if (ctx.active)
        {
            active.push_back(id);
        }
    }
    return active;
}

void GreenContextManager::enableContext(GreenContextId ctxId)
{
    auto it = contexts_.find(ctxId);
    if (it != contexts_.end())
    {
        it->second.active = true;
    }
}

void GreenContextManager::disableContext(GreenContextId ctxId)
{
    auto it = contexts_.find(ctxId);
    if (it != contexts_.end())
    {
        it->second.active = false;
    }
}

size_t GreenContextManager::getTotalGpuMemory() const
{
    size_t totalMemory = 0;
    for (int deviceId : gpuDevices_)
    {
        cudaSetDevice(deviceId);
        size_t freeMem = 0;
        size_t totalMem = 0;
        if (cudaMemGetInfo(&freeMem, &totalMem) == cudaSuccess)
        {
            totalMemory = totalMem > std::numeric_limits<size_t>::max() - totalMemory
                              ? std::numeric_limits<size_t>::max()
                              : totalMemory + totalMem;
        }
    }
    return totalMemory;
}

ClusterMetrics GreenContextManager::getClusterMetrics() const
{
    ClusterMetrics metrics;
    metrics.numGpus = static_cast<int>(gpuDevices_.size());
    metrics.aggregateMemoryAvailable = getTotalGpuMemory();

    for (const auto &[id, ctx] : contexts_)
    {
        if (!ctx.active)
        {
            continue;
        }
        ++metrics.numActiveContexts;
        const size_t contextMemory = ctx.currentMemoryUsage.load();
        metrics.aggregateMemoryUsed =
            contextMemory > std::numeric_limits<size_t>::max() - metrics.aggregateMemoryUsed
                ? std::numeric_limits<size_t>::max()
                : metrics.aggregateMemoryUsed + contextMemory;

        nvmlDevice_t nvmlDev;
        if (nvmlDeviceGetHandleByIndex(ctx.gpuDevice, &nvmlDev) == NVML_SUCCESS)
        {
            nvmlUtilization_t utilization;
            if (nvmlDeviceGetUtilizationRates(nvmlDev, &utilization) == NVML_SUCCESS)
            {
                metrics.aggregateComputeUtilization += utilization.gpu / 100.0f;
            }
        }
    }

    if (metrics.numActiveContexts > 0)
    {
        metrics.aggregateComputeUtilization /= static_cast<float>(metrics.numActiveContexts);
    }
    return metrics;
}

void GreenContextManager::printContextStats(GreenContextId ctxId) const
{
    auto it = contexts_.find(ctxId);
    if (it == contexts_.end())
    {
        std::cout << "Context " << ctxId << " not found\n";
        return;
    }

    const GreenContext &ctx = it->second;
    std::cout << "Context " << ctx.id << " on GPU " << ctx.gpuDevice << ": "
              << ctx.currentMemoryUsage.load() << "/" << ctx.memoryLimit << " bytes, "
              << ctx.allocatedSMs << " SMs, priority " << ctx.priority << "\n";
}

#include "detail/manager_context_tenant_ops.inl"

std::string getGreenContextVersion()
{
    return "green-context-api/1.0";
}

} // namespace nerve::gpu

#endif

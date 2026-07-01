
#ifdef NERVE_HAS_CUDA

#include "nerve/gpu/detail/gpu_detail.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <vector>

namespace
{

using nerve::gpu::ComputeManager;
using nerve::gpu::GPUBatchMemoryManager;
using nerve::gpu::tuning::GpuSignature;
using nerve::gpu::tuning::GpuTuningDatabase;
using nerve::gpu::tuning::TunedConfig;
using nerve::gpu::tuning::WorkloadFingerprint;

constexpr double TOL = 1e-10;

bool check_gpu_compute_manager_init()
{
    ComputeManager &mgr = ComputeManager::getInstance();
    mgr.initialize();
    mgr.setMemoryLimitMB(1024);
    mgr.setUseOptimizedKernels(true);
    mgr.setUseCudaGraphs(false);
    bool has_arch = !mgr.getGPUArchitectureName().empty();
    mgr.shutdown();
    return has_arch || !mgr.isAvailable();
}

bool check_async_executor_config()
{
    nerve::gpu::async::AsyncExecutor exec(2);
    nerve::gpu::async::CUDAStream stream(0);
    nerve::gpu::async::CUDAEvent event;
    auto fut = exec.submit([&stream]() { stream.synchronize(); });
    fut.wait();
    exec.synchronizeAll();
    return true;
}

bool check_matrix_distance_ops()
{
    ComputeManager &mgr = ComputeManager::getInstance();
    mgr.initialize();

    std::vector<std::vector<double>> points = {{0.0, 0.0}, {3.0, 0.0}, {0.0, 4.0}};
    std::vector<std::vector<double>> dist;
    auto err = mgr.computeDistanceMatrix(points, dist);
    if (err.isError())
    {
        return true;
    }
    if (dist.size() != 3)
    {
        return false;
    }
    for (const auto &row : dist)
    {
        if (row.size() != 3)
        {
            return false;
        }
    }
    if (std::abs(dist[0][1] - 3.0) > TOL || std::abs(dist[0][2] - 4.0) > TOL)
    {
        return false;
    }
    if (std::abs(dist[1][2] - 5.0) > TOL)
    {
        return false;
    }
    if (std::abs(dist[1][0] - dist[0][1]) > TOL)
    {
        return false;
    }
    mgr.shutdown();
    return true;
}

bool check_gpu_context_creation()
{
#if defined(CUDA_VERSION) && CUDA_VERSION >= 13010
    nerve::gpu::GreenContextManager ctx_mgr;
    nerve::gpu::GreenContextId cid = ctx_mgr.createContext(0.5f, 1024ULL * 1024ULL * 1024ULL, 0);
    auto metrics = ctx_mgr.getContextMetrics(cid);
    if (metrics.contextId != cid)
    {
        return false;
    }
    ctx_mgr.destroyContext(cid);
#endif
    return true;
}

bool check_batch_memory_manager()
{
    nerve::gpu::GPUBatchConfig config;
    config.memory_pool_size = 0;
    config.enable_pinned_memory = false;
    GPUBatchMemoryManager mgr(config);
    if (!mgr.isPowerOfTwo(256))
    {
        return false;
    }
    if (mgr.isPowerOfTwo(0))
    {
        return false;
    }
    if (mgr.isPowerOfTwo(3))
    {
        return false;
    }
    if (mgr.alignUp(17, 16) != 32)
    {
        return false;
    }
    if (mgr.alignUp(16, 16) != 16)
    {
        return false;
    }
    if (mgr.alignUp(0, 16) != 0)
    {
        return false;
    }
    if (mgr.alignUp(1, 256) != 256)
    {
        return false;
    }
    void *ptr = mgr.allocate(64, 64);
    if (ptr != nullptr)
    {
        mgr.deallocate(ptr);
    }
    mgr.reset();
    return true;
}

bool check_tuning_cache_ops()
{
    GpuTuningDatabase &db = GpuTuningDatabase::instance();
    db.initialize("/tmp");
    auto pre_stats = db.getStats();
    (void)pre_stats;
    GpuSignature sig;
    sig.computeCapability = 89;
    sig.smCount = 128;
    sig.clockRate = 2500;
    sig.totalMemory = 25769803776ULL;
    sig.name = "Test GPU";
    WorkloadFingerprint wl;
    wl.nPoints = 10000;
    wl.pointDim = 128;
    wl.sparsityEstimate = 0.5f;
    wl.problemType = 1;
    wl.flags = 0;
    auto miss = db.lookup(sig, wl);
    if (miss.has_value())
    {
        return false;
    }
    TunedConfig cfg;
    cfg.blockSize = 256;
    cfg.tileSize = 64;
    cfg.clusterSize = 4;
    cfg.numStages = 3;
    cfg.useWGMMA = true;
    db.store(sig, wl, cfg);
    auto hit = db.lookup(sig, wl);
    if (!hit.has_value())
    {
        return false;
    }
    if (hit->blockSize != 256)
    {
        return false;
    }
    auto stats = db.getStats();
    if (stats.totalEntries != 1)
    {
        return false;
    }
    db.clear();
    stats = db.getStats();
    if (stats.totalEntries != 0)
    {
        return false;
    }
    return true;
}

} // namespace

int main()
{
    int failures = 0;

    auto run = [&](const char *name, bool ok) {
        if (!ok)
        {
            std::cerr << "FAIL: " << name << "\n";
            ++failures;
        }
        else
        {
            std::cout << "PASS: " << name << "\n";
        }
    };

    run("gpu_compute_manager_init", check_gpu_compute_manager_init());
    run("async_executor_config", check_async_executor_config());
    run("matrix_distance_ops", check_matrix_distance_ops());
    run("gpu_context_creation", check_gpu_context_creation());
    run("batch_memory_manager", check_batch_memory_manager());
    run("tuning_cache_ops", check_tuning_cache_ops());

    return failures > 0 ? 1 : 0;
}

#else // NERVE_HAS_CUDA

#include <iostream>

int main()
{
    std::cout << "SKIP: internal_gpu_test (NERVE_HAS_CUDA not defined)\n";
    return 0;
}

#endif

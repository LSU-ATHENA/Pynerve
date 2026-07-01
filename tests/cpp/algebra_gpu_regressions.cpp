
#include "nerve/config.hpp"
#include "nerve/core_types.hpp"
#include "nerve/gpu/compute_manager.hpp"

#include <cstddef>
#include <iostream>
#include <random>
#include <vector>

namespace
{

using nerve::Size;
using nerve::gpu::ComputeConfig;
using nerve::gpu::ComputeManager;
using nerve::gpu::OperationType;

constexpr double TOL = 1e-10;

bool check_gpu_availability_check()
{
    auto &mgr = ComputeManager::instance();
    bool available = mgr.isGpuAvailable();
    if (available)
    {
        Size count = mgr.getGpuCount();
        if (count == 0)
        {
            std::cerr << "available but count = 0\n";
            return false;
        }
    }
    return true;
}

bool check_gpu_initialize_shutdown()
{
    auto &mgr = ComputeManager::instance();
    bool init = mgr.initialize(0);
    if (init)
    {
        if (!mgr.isGpuAvailable())
        {
            std::cerr << "init succeeded but not available\n";
            return false;
        }
        mgr.shutdown();
        if (mgr.isGpuAvailable())
        {
            std::cerr << "post-shutdown still available\n";
            return false;
        }
    }
    return true;
}

bool check_gpu_config_validation()
{
    auto &mgr = ComputeManager::instance();
    ComputeConfig cfg;
    cfg.gpu_id = 0;
    cfg.batch_size = 512;
    cfg.use_unified_memory = false;
    cfg.enable_profiling = false;
    mgr.setConfig(cfg);
    auto retrieved = mgr.getConfig();
    if (retrieved.batch_size != cfg.batch_size)
    {
        std::cerr << "config batch_size mismatch\n";
        return false;
    }
    return true;
}

bool check_gpu_submit_operation()
{
    auto &mgr = ComputeManager::instance();
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0};
    bool submitted = mgr.submit(OperationType::MATRIX_REDUCTION, data);
    if (mgr.isGpuAvailable() && !submitted)
    {
        std::cerr << "submit failed when GPU available\n";
        return false;
    }
    if (!mgr.isGpuAvailable() && submitted)
    {
        std::cerr << "submit succeeded when GPU unavailable\n";
        return false;
    }
    return true;
}

bool check_gpu_synchronize()
{
    auto &mgr = ComputeManager::instance();
    bool sync = mgr.synchronize();
    if (mgr.isGpuAvailable() && !sync)
    {
        std::cerr << "synchronize failed when GPU available\n";
        return false;
    }
    return true;
}

bool check_fallback_cpu_no_cuda()
{
#ifndef NERVE_HAS_CUDA
    auto &mgr = ComputeManager::instance();
    if (mgr.isGpuAvailable())
    {
        std::cerr << "without CUDA, GPU should not be available\n";
        return false;
    }
#endif
    return true;
}

} // namespace

int main()
{
    if (!check_gpu_availability_check())
    {
        std::cerr << "FAIL: gpu availability\n";
        return 1;
    }
    if (!check_gpu_initialize_shutdown())
    {
        std::cerr << "FAIL: gpu init/shutdown\n";
        return 1;
    }
    if (!check_gpu_config_validation())
    {
        std::cerr << "FAIL: gpu config\n";
        return 1;
    }
    if (!check_gpu_submit_operation())
    {
        std::cerr << "FAIL: gpu submit\n";
        return 1;
    }
    if (!check_gpu_synchronize())
    {
        std::cerr << "FAIL: gpu synchronize\n";
        return 1;
    }
    if (!check_fallback_cpu_no_cuda())
    {
        std::cerr << "FAIL: fallback cpu\n";
        return 1;
    }
    return 0;
}

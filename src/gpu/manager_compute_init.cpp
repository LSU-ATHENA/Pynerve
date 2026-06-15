#include "nerve/gpu/kernel_launcher.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <mutex>
#include <shared_mutex>
#include <string>

namespace nerve::gpu
{

void ComputeManager::initialize(const Config &config)
{
    std::unique_lock<std::shared_mutex> lock(mutex_);

    config_ = config;
    initialized_ = true;
    available_ = false;
    gpuComputeCapability_ = 0;
    gpuMultiProcessorCount_ = 0;
    gpuSharedMemPerBlock_ = 0;
    last_error_.clear();

    if (!config_.enable_gpu)
    {
        return;
    }

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count <= 0)
    {
        config_.enable_gpu = false;
        last_error_ = "CUDA device missing";
        return;
    }

    available_ = true;
    detectGPUArchitecture();
}

void ComputeManager::detectGPUArchitecture()
{
    int device = 0;
    cudaDeviceProp props{};
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaGetDeviceProperties(&props, device) != cudaSuccess)
    {
        gpuComputeCapability_ = 0;
        gpuMultiProcessorCount_ = 0;
        gpuSharedMemPerBlock_ = 0;
        return;
    }

    gpuComputeCapability_ = props.major * 10 + props.minor;
    gpuMultiProcessorCount_ = props.multiProcessorCount;
    gpuSharedMemPerBlock_ = props.sharedMemPerBlock;
}

} // namespace nerve::gpu

#pragma once

/// @file gpu_capability_core.hpp

#include "nerve/types.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <string>

namespace nerve::gpu::advanced
{

struct AdvancedCapabilities
{
public:
    static AdvancedCapabilities detect();

    [[nodiscard]] bool isTuring() const { return computeCapability() == 75; }
    [[nodiscard]] bool isAmpere() const
    {
        const int cc = computeCapability();
        return cc >= 80 && cc < 89;
    }
    [[nodiscard]] bool isAda() const
    {
        const int cc = computeCapability();
        return cc >= 89 && cc < 90;
    }
    [[nodiscard]] bool isHopper() const
    {
        const int cc = computeCapability();
        return cc >= 90 && cc < 100;
    }
    [[nodiscard]] bool isBlackwell() const
    {
        const int cc = computeCapability();
        return cc >= 100 && cc < 110;
    }
    [[nodiscard]] bool isRubinCpx() const { return computeCapability() >= 110; }

    [[nodiscard]] bool isConsumerGPU() const
    {
        const int cc = computeCapability();
        return cc >= 75 && cc < 90;
    }
    [[nodiscard]] bool isDataCenterGPU() const { return computeCapability() >= 90; }

    [[nodiscard]] bool supportsPTXOptimizations() const { return computeCapability() >= 89; }
    [[nodiscard]] bool supportsWGMMA() const { return supports_wgmma || computeCapability() >= 90; }
    [[nodiscard]] bool supportsClusters() const
    {
        return supports_cluster16 || computeCapability() >= 90;
    }
    [[nodiscard]] bool supportsFP4() const { return computeCapability() >= 100; }

    [[nodiscard]] int computeCapability() const
    {
        return compute_capability_major * 10 + compute_capability_minor;
    }
    [[nodiscard]] std::string architectureName() const;

    bool cuda_available = false;
    int device_count = 0;
    int compute_capability_major = 0;
    int compute_capability_minor = 0;
    int sm_count = 0;
    std::size_t shared_mem_per_block = 0;
    bool supports_tile_api = false;
    bool supports_wgmma = false;
    bool supports_tcgen05 = false;
    bool supports_cluster16 = false;
};

// Inline implementation
inline AdvancedCapabilities AdvancedCapabilities::detect()
{
    AdvancedCapabilities caps;

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0)
    {
        return caps;
    }

    caps.cuda_available = true;
    caps.device_count = device_count;

    int major, minor;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0);

    caps.compute_capability_major = major;
    caps.compute_capability_minor = minor;
    cudaDeviceGetAttribute(&caps.sm_count, cudaDevAttrMultiProcessorCount, 0);

    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess)
    {
        caps.shared_mem_per_block = prop.sharedMemPerBlock;
    }
#if defined(CUDA_VERSION) && CUDA_VERSION >= 13020
    caps.supports_tile_api = true;
#endif
    caps.supports_wgmma = caps.computeCapability() >= 90;
    caps.supports_tcgen05 = caps.computeCapability() >= 100;
    caps.supports_cluster16 = caps.computeCapability() >= 100;

    return caps;
}

inline std::string AdvancedCapabilities::architectureName() const
{
    if (isRubinCpx())
        return "Rubin CPX";
    if (isBlackwell())
        return "Blackwell";
    if (isHopper())
        return "Hopper";
    return "Pre-Hopper";
}

} // namespace nerve::gpu::advanced

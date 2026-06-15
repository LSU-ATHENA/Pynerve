#pragma once

#include "nerve/config.hpp"
#include "nerve/gpu/distance_kernels.hpp"
#include "nerve/gpu/gpu_capability_core.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#if defined(NERVE_HAS_MPI)
#include <mpi.h>
#endif

namespace nerve::gpu::advanced
{

struct GpuTuningHandle
{
    int blockSize = 128;
    int numStages = 2;
    int tileSize = 32;
    bool useWGMMA = false;
    bool useTMA = false;
    bool usePTXOpts = true;
    float measuredTime = 0.0f;

    AdvancedCapabilities capabilities{};
    float tuningTimeMs = 0.0f;
    bool usedCache = false;
};

struct TileKernelConfig
{
    int tileSizeM = 64;
    int tileSizeN = 64;
    int tileSizeK = 16;
    int clusterSize = 1;
    bool useTMA = false;
    bool useTensorCores = false;
    std::string dataType = "fp32"; // fp32 or fp64 for host-computed path
};

struct Tcgen05Config
{
    int shapeM = 128;
    int shapeN = 16;
    int shapeK = 32;
    std::string aFormat = "e4m3";
    std::string bFormat = "e4m3";
    std::string accumType = "fp32";
    int pipelineDepth = 3;
};

struct Cluster16Config
{
    bool useNonPortable = true;
    bool useMulticast = true;
    bool useDistributedL2 = true;
    std::size_t sharedMemPerBlock = 128 * 1024;
};

namespace detail
{

[[nodiscard]] inline bool isSupportedTcgen05Config(const Tcgen05Config &config)
{
    const bool valid_shape =
        (config.shapeM == 64 || config.shapeM == 128 || config.shapeM == 256) &&
        (config.shapeN == 8 || config.shapeN == 16 || config.shapeN == 64) &&
        (config.shapeK == 16 || config.shapeK == 32);
    const bool valid_format = (config.aFormat == "e4m3" || config.aFormat == "e5m2" ||
                               config.aFormat == "fp16" || config.aFormat == "bf16") &&
                              (config.bFormat == "e4m3" || config.bFormat == "e5m2" ||
                               config.bFormat == "fp16" || config.bFormat == "bf16");
    const bool valid_accum = config.accumType == "fp32" || config.accumType == "fp64";
    return valid_shape && valid_format && valid_accum && config.pipelineDepth > 0;
}

[[nodiscard]] inline AdvancedCapabilities probeCapabilities(const int requested_device_id)
{
    AdvancedCapabilities caps{};
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0)
    {
        return caps;
    }

    const int device_id = std::clamp(requested_device_id, 0, device_count - 1);
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device_id) != cudaSuccess)
    {
        return caps;
    }

    caps.cuda_available = true;
    caps.device_count = device_count;
    caps.compute_capability_major = prop.major;
    caps.compute_capability_minor = prop.minor;
    caps.shared_mem_per_block = prop.sharedMemPerBlock;
#if defined(CUDA_VERSION) && CUDA_VERSION >= 13020
    caps.supports_tile_api = true;
#else
    caps.supports_tile_api = false;
#endif
    caps.supports_wgmma = prop.major >= 9;
    caps.supports_tcgen05 = prop.major >= 10;
    caps.supports_cluster16 = prop.major >= 10;
    return caps;
}

} // namespace detail

[[nodiscard]] inline GpuTuningHandle tuneForGpu(const std::uint32_t nPoints,
                                                const std::uint32_t pointDim, const int deviceId)
{
    GpuTuningHandle handle{};
    handle.capabilities = detail::probeCapabilities(deviceId);
    if (!handle.capabilities.cuda_available)
    {
        return handle;
    }

    const double workload = static_cast<double>(nPoints) * static_cast<double>(pointDim);
    if (workload >= 2.0e6)
    {
        handle.blockSize = 512;
        handle.tileSize = 64;
        handle.numStages = 4;
    }
    else if (workload >= 5.0e5)
    {
        handle.blockSize = 256;
        handle.tileSize = 48;
        handle.numStages = 3;
    }
    else
    {
        handle.blockSize = 128;
        handle.tileSize = 32;
        handle.numStages = 2;
    }

    handle.useTMA =
        handle.capabilities.supports_tile_api && handle.capabilities.compute_capability_major >= 9;
    handle.useWGMMA = handle.capabilities.supports_wgmma;
    handle.usePTXOpts = true;
    return handle;
}

[[nodiscard]] inline std::vector<GpuTuningHandle> tuneForMultiGpu(const std::uint32_t nPoints,
                                                                  const std::uint32_t pointDim,
                                                                  const std::vector<int> &deviceIds)
{
    std::vector<GpuTuningHandle> out;
    out.reserve(deviceIds.size());
    for (const int device_id : deviceIds)
    {
        out.push_back(tuneForGpu(nPoints, pointDim, device_id));
    }
    return out;
}

#if defined(NERVE_HAS_MPI)
[[nodiscard]] inline std::vector<GpuTuningHandle>
tuneForDistributed(const std::uint32_t nPoints, const std::uint32_t pointDim, MPI_Comm comm)
{
    int rank = 0;
    int size = 1;
    (void)MPI_Comm_rank(comm, &rank);
    (void)MPI_Comm_size(comm, &size);
    std::vector<int> local_devices;
    local_devices.reserve(size > 0 ? 1 : 0);
    local_devices.push_back(rank % std::max(1, size));
    return tuneForMultiGpu(nPoints, pointDim, local_devices);
}
#endif

[[nodiscard]] inline bool tileApiAvailable()
{
#if defined(CUDA_VERSION) && CUDA_VERSION >= 13020
    int device_count = 0;
    return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
#else
    return false;
#endif
}

[[nodiscard]] inline cudaError_t launchTileDistanceMatrix(const void *points, void *distances,
                                                          const std::uint32_t nPoints,
                                                          const std::uint32_t pointDim,
                                                          const TileKernelConfig &config,
                                                          const cudaStream_t stream)
{
    if (config.tileSizeM <= 0 || config.tileSizeN <= 0 || config.tileSizeK <= 0 ||
        config.clusterSize <= 0 || config.clusterSize > 16)
    {
        return cudaErrorInvalidValue;
    }
    if (nPoints == 0)
    {
        return cudaSuccess;
    }
    if (pointDim == 0)
    {
        return cudaErrorInvalidValue;
    }
    if (points == nullptr || distances == nullptr)
    {
        return cudaErrorInvalidValue;
    }
#if HAS_CUDA
    if (config.dataType == "fp64")
    {
        return launch_pairwise_distance_radius_f64(static_cast<const double *>(points), pointDim,
                                                   static_cast<double *>(distances), nPoints,
                                                   nPoints, pointDim, 0.0, stream);
    }
    if (config.dataType == "fp32")
    {
        return launch_pairwise_distance_radius_f32(static_cast<const float *>(points), pointDim,
                                                   static_cast<float *>(distances), nPoints,
                                                   nPoints, pointDim, 0.0f, stream);
    }
#else
    (void)stream;
#endif
    return cudaErrorInvalidValue;
}

[[nodiscard]] inline bool tcgen05Available()
{
    const auto caps = detail::probeCapabilities(0);
    return caps.supports_tcgen05;
}

[[nodiscard]] inline cudaError_t launchTcgen05DistanceMatrix(const void *points, void *distances,
                                                             const std::uint32_t nPoints,
                                                             const std::uint32_t pointDim,
                                                             const Tcgen05Config &config,
                                                             const cudaStream_t stream)
{
    if (!detail::isSupportedTcgen05Config(config))
    {
        return cudaErrorInvalidValue;
    }
    TileKernelConfig proxy{};
    proxy.tileSizeM = config.shapeM;
    proxy.tileSizeN = config.shapeN;
    proxy.tileSizeK = config.shapeK;
    proxy.dataType = (config.accumType == "fp64") ? "fp64" : "fp32";
    return launchTileDistanceMatrix(points, distances, nPoints, pointDim, proxy, stream);
}

[[nodiscard]] inline bool cluster16Available()
{
    const auto caps = detail::probeCapabilities(0);
    return caps.supports_cluster16;
}

[[nodiscard]] inline cudaError_t launchCluster16DistanceMatrix(const void *points, void *distances,
                                                               const std::uint32_t nPoints,
                                                               const std::uint32_t pointDim,
                                                               const Cluster16Config &config,
                                                               const cudaStream_t stream)
{
    const auto caps = detail::probeCapabilities(0);
    TileKernelConfig proxy{};
    proxy.clusterSize = caps.supports_cluster16 ? (config.useNonPortable ? 16 : 4) : 1;
    proxy.useTMA =
        config.useMulticast && caps.supports_tile_api && caps.compute_capability_major >= 9;
    proxy.dataType = "fp32";
    return launchTileDistanceMatrix(points, distances, nPoints, pointDim, proxy, stream);
}

} // namespace nerve::gpu::advanced

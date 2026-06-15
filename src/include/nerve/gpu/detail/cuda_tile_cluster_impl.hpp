#pragma once

#include "nerve/gpu/detail/cuda_tile_detail_impl.hpp"

namespace nerve::gpu::tile
{

// Inline Implementation: ClusterUtils

inline int ClusterUtils::getClusterRank()
{
    // Host-side implementation: without launched cooperative clusters we expose
    // single-cluster semantics for deterministic control flow.
    return 0;
}

inline int ClusterUtils::getClusterSize()
{
    // Host-side implementation: exactly one block-equivalent participant.
    return 1;
}

inline void ClusterUtils::clusterSync()
{
    (void)cudaDeviceSynchronize();
}

template <typename T>
inline T *ClusterUtils::getDistributedSmem(int targetBlockRank)
{
    if (targetBlockRank != 0)
    {
        return nullptr;
    }
    return nullptr;
}

inline bool ClusterUtils::clustersSupported(int deviceId)
{
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, deviceId) != cudaSuccess)
    {
        return false;
    }
    return prop.major >= 9;
}

inline int ClusterUtils::getMaxClusterSize(int deviceId)
{
    if (!clustersSupported(deviceId))
    {
        return 1;
    }
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, deviceId) != cudaSuccess)
    {
        return 1;
    }
    return prop.major >= 10 ? 16 : 8;
}

inline bool ClusterUtils::supports16BlockClusters(int deviceId)
{
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, deviceId) != cudaSuccess)
    {
        return false;
    }
    return prop.major >= 10;
}

inline int ClusterUtils::calculateOptimalClusterSize(int nBlocks, int smCount,
                                                     int computeCapability)
{
    if (nBlocks <= 0 || smCount <= 0)
    {
        return 1;
    }
    if (computeCapability < 90)
    {
        return 1;
    }
    int preferred = computeCapability >= 100 ? 16 : (computeCapability >= 90 ? 8 : 4);
    preferred = std::min(preferred, nBlocks);
    preferred = std::min(preferred, smCount);
    return std::max(1, preferred);
}

// Inline Implementation: Convenience API

template <typename T>
inline TileBuffer<T> tileDistanceMatrix(const T *points, uint32_t nPoints, uint32_t pointDim,
                                        float maxRadius, TilePrecision precision, int deviceId)
{
    if (points == nullptr || nPoints == 0 || pointDim == 0)
    {
        return TileBuffer<T>(0, 0);
    }
    if (!std::isfinite(maxRadius))
    {
        return TileBuffer<T>(0, 0);
    }
    if (nPoints > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
        pointDim > static_cast<uint32_t>(std::numeric_limits<int>::max()))
    {
        return TileBuffer<T>(0, 0);
    }
    if (nPoints > 0 && static_cast<size_t>(nPoints) >
                           (std::numeric_limits<size_t>::max() / static_cast<size_t>(nPoints)))
    {
        return TileBuffer<T>(0, 0);
    }

    (void)detail::selectDevice(deviceId);
    TileBuffer<T> output(static_cast<int>(nPoints), static_cast<int>(nPoints));
    if (output.data() == nullptr)
    {
        return output;
    }

    if (precision == TilePrecision::kFull)
    {
        TileBuffer<T> device_points(static_cast<int>(nPoints), static_cast<int>(pointDim));
        if (device_points.data() != nullptr)
        {
            device_points.copyFromHost(points);
            const cudaError_t device_status = detail::launchDistanceMatrixDeviceBackend<T>(
                device_points.data(), static_cast<int>(nPoints), static_cast<int>(pointDim),
                output.data(), maxRadius, nullptr);
            if (device_status == cudaSuccess && cudaStreamSynchronize(nullptr) == cudaSuccess)
            {
                return output;
            }
        }
    }

    std::vector<T> host_distances(static_cast<size_t>(nPoints) * static_cast<size_t>(nPoints),
                                  T{0});
    detail::computeSymmetricDistancesHostTiled(points, nPoints, pointDim, maxRadius, precision, 64,
                                               64, host_distances);
    output.copyFromHost(host_distances.data());
    return output;
}

template <typename T>
inline T tileReduce(const T *data, uint32_t nElements, int deviceId)
{
    (void)detail::selectDevice(deviceId);
    if (data == nullptr || nElements == 0)
    {
        return T{0};
    }
    T sum = T{0};
    for (uint32_t i = 0; i < nElements; ++i)
    {
        sum += data[i];
    }
    return sum;
}

inline std::string getTileApiVersion()
{
    return tileApiAvailable() ? "nerve-tile-api:cuda-runtime" : "nerve-tile-api:host-runtime";
}

} // namespace nerve::gpu::tile

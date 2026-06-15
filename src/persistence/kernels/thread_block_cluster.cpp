#include "nerve/persistence/cuda/thread_block_cluster.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <limits>
#include <string>
#include <unordered_map>

namespace nerve::persistence::accelerated
{
namespace
{

static std::unordered_map<std::string, ClusterBenchmarkResult> s_configCache;

bool deviceComputeCapability(int &computeCapability)
{
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess)
    {
        return false;
    }

    int major = 0;
    int minor = 0;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess)
    {
        return false;
    }

    computeCapability = major * 10 + minor;
    return true;
}

bool checkedBytes(uint32_t lhs, uint32_t rhs, size_t elementSize, size_t &bytes)
{
    const auto max = std::numeric_limits<size_t>::max();
    const size_t a = static_cast<size_t>(lhs);
    const size_t b = static_cast<size_t>(rhs);
    if (a != 0 && b > max / a)
    {
        return false;
    }
    const size_t count = a * b;
    if (elementSize != 0 && count > max / elementSize)
    {
        return false;
    }
    bytes = count * elementSize;
    return true;
}

uint32_t ceilDiv(uint32_t value, uint32_t divisor)
{
    return value == 0 ? 0 : 1 + (value - 1) / divisor;
}

cudaError_t launchCluster16(float *points, float *distances, uint32_t nPoints, uint32_t pointDim,
                            float maxRadiusSq, dim3 grid, dim3 block, size_t smem)
{
    void *args[] = {&points, &distances, &nPoints, &pointDim, &maxRadiusSq};
    return cudaLaunchKernel(reinterpret_cast<const void *>(cluster16DistanceMatrixKernel), grid,
                            block, args, smem, nullptr);
}

cudaError_t launchClusterDistributedL2(float *points, float *distances, uint32_t nPoints,
                                       uint32_t pointDim, float maxRadiusSq,
                                       AdvancedClusterConfig config, dim3 grid, dim3 block,
                                       size_t smem)
{
    void *args[] = {&points, &distances, &nPoints, &pointDim, &maxRadiusSq, &config};
    return cudaLaunchKernel(reinterpret_cast<const void *>(clusterDistributedL2Kernel), grid, block,
                            args, smem, nullptr);
}

cudaError_t launchClusterTma(float *points, float *distances, uint32_t nPoints, uint32_t pointDim,
                             float maxRadiusSq, dim3 grid, dim3 block, size_t smem)
{
    void *args[] = {&points, &distances, &nPoints, &pointDim, &maxRadiusSq};
    return cudaLaunchKernel(reinterpret_cast<const void *>(clusterTMAMulticastKernel), grid, block,
                            args, smem, nullptr);
}

void freeDeviceBuffers(float *points, float *distances)
{
    if (points != nullptr)
    {
        cudaFree(points);
    }
    if (distances != nullptr)
    {
        cudaFree(distances);
    }
}

} // namespace

bool AdvancedClusterConfig::isValid(int computeCapability) const
{
    if (clusterSizeX <= 0 || clusterSizeY <= 0 || clusterSizeZ <= 0 || sharedMemPerBlock == 0 ||
        numPipelineStages <= 0)
    {
        return false;
    }
    const int total = totalClusterSize();
    if (total > CLUSTER_MAX_SIZE)
        return false;
    if (total > CLUSTER_MAX_SIZE_PRE_B200 && computeCapability < CLUSTER_B200_COMPUTE_CAPABILITY)
        return false;
    return true;
}

size_t AdvancedClusterConfig::totalDistributedSmem() const
{
    if (sharedMemPerBlock >
        std::numeric_limits<size_t>::max() / static_cast<size_t>(std::max(1, totalClusterSize())))
    {
        return std::numeric_limits<size_t>::max();
    }
    return sharedMemPerBlock * totalClusterSize();
}

bool clusterFeaturesAvailable()
{
    int computeCapability = 0;
    return deviceComputeCapability(computeCapability) && computeCapability >= 90;
}

bool nonPortableClustersAvailable()
{
    int computeCapability = 0;
    return deviceComputeCapability(computeCapability) &&
           computeCapability >= CLUSTER_B200_COMPUTE_CAPABILITY;
}

ClusterBenchmarkResult benchmarkClusterConfig(uint32_t nPoints, uint32_t pointDim,
                                              const ClusterBenchmarkConfig &config)
{
    ClusterBenchmarkResult result;
    int computeCapability = 0;
    if (nPoints == 0 || pointDim == 0 || pointDim > TMA_MAX_POINT_DIMENSIONS ||
        config.numTrials <= 0 || !deviceComputeCapability(computeCapability) ||
        computeCapability < 90)
    {
        return result;
    }

    size_t pointsBytes = 0;
    size_t distancesBytes = 0;
    if (!checkedBytes(nPoints, pointDim, sizeof(float), pointsBytes) ||
        !checkedBytes(nPoints, nPoints, sizeof(float), distancesBytes))
    {
        return result;
    }

    const bool canUseNonPortable = computeCapability >= CLUSTER_B200_COMPUTE_CAPABILITY;
    const dim3 threadsPerBlock(256);
    const dim3 grid(std::max<uint32_t>(1, ceilDiv(nPoints, threadsPerBlock.x)));

    float *d_points = nullptr;
    float *d_distances = nullptr;
    if (cudaMalloc(&d_points, pointsBytes) != cudaSuccess ||
        cudaMalloc(&d_distances, distancesBytes) != cudaSuccess)
    {
        freeDeviceBuffers(d_points, d_distances);
        return result;
    }
    if (cudaMemset(d_points, 0, pointsBytes) != cudaSuccess ||
        cudaMemset(d_distances, 0, distancesBytes) != cudaSuccess)
    {
        freeDeviceBuffers(d_points, d_distances);
        return result;
    }

    const int minClusterX = std::max(1, config.minClusterSizeX);
    const int maxClusterX = std::max(minClusterX, config.maxClusterSizeX);
    const int minClusterY = std::max(1, config.minClusterSizeY);
    const int maxClusterY = std::max(minClusterY, config.maxClusterSizeY);
    for (int cx = minClusterX; cx <= maxClusterX;)
    {
        for (int cy = minClusterY; cy <= maxClusterY; ++cy)
        {
            AdvancedClusterConfig clusterConfig;
            clusterConfig.clusterSizeX = cx;
            clusterConfig.clusterSizeY = cy;
            clusterConfig.clusterSizeZ = 1;
            clusterConfig.useNonPortable =
                clusterConfig.totalClusterSize() > CLUSTER_MAX_SIZE_PRE_B200;
            if (!clusterConfig.isValid(computeCapability))
            {
                continue;
            }
            if (clusterConfig.useNonPortable && (!canUseNonPortable || !config.testNonPortable))
            {
                continue;
            }

            cudaEvent_t start = nullptr;
            cudaEvent_t stop = nullptr;
            if (cudaEventCreate(&start) != cudaSuccess)
            {
                continue;
            }
            if (cudaEventCreate(&stop) != cudaSuccess)
            {
                cudaEventDestroy(start);
                continue;
            }

            float totalTime = 0.0f;
            bool validTrial = true;
            for (int trial = 0; trial < config.numTrials; ++trial)
            {
                if (cudaEventRecord(start, nullptr) != cudaSuccess)
                {
                    validTrial = false;
                    break;
                }

                cudaError_t launchStatus = cudaSuccess;
                if (clusterConfig.useNonPortable)
                {
                    launchStatus =
                        launchCluster16(d_points, d_distances, nPoints, pointDim, 1e20f, grid,
                                        threadsPerBlock, clusterConfig.sharedMemPerBlock);
                }
                else if (clusterConfig.useDistributedL2)
                {
                    launchStatus = launchClusterDistributedL2(
                        d_points, d_distances, nPoints, pointDim, 1e20f, clusterConfig, grid,
                        threadsPerBlock, clusterConfig.sharedMemPerBlock);
                }
                else
                {
                    launchStatus =
                        launchClusterTma(d_points, d_distances, nPoints, pointDim, 1e20f, grid,
                                         threadsPerBlock, clusterConfig.sharedMemPerBlock);
                }
                if (launchStatus != cudaSuccess || cudaGetLastError() != cudaSuccess ||
                    cudaEventRecord(stop, nullptr) != cudaSuccess ||
                    cudaEventSynchronize(stop) != cudaSuccess)
                {
                    validTrial = false;
                    break;
                }

                float elapsed = 0.0f;
                if (cudaEventElapsedTime(&elapsed, start, stop) != cudaSuccess)
                {
                    validTrial = false;
                    break;
                }
                totalTime += elapsed;
            }

            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            if (!validTrial)
            {
                continue;
            }
            const float avgTime = totalTime / static_cast<float>(config.numTrials);
            if (avgTime < result.optimalTimeMs)
            {
                result.optimalTimeMs = avgTime;
                result.optimalClusterSizeX = cx;
                result.optimalClusterSizeY = cy;
                result.optimalClusterSizeZ = clusterConfig.clusterSizeZ;
                result.useNonPortable = clusterConfig.useNonPortable;
                result.blackwellRequired = clusterConfig.useNonPortable;
            }
        }
        if (cx > maxClusterX / 2)
        {
            break;
        }
        cx *= 2;
    }

    freeDeviceBuffers(d_points, d_distances);
    return result;
}

ClusterBenchmarkResult getOrBenchmarkClusterConfig(uint32_t nPoints, uint32_t pointDim,
                                                   const std::string &cacheKey)
{
    auto it = s_configCache.find(cacheKey);
    if (it != s_configCache.end())
    {
        return it->second;
    }

    ClusterBenchmarkResult result = benchmarkClusterConfig(nPoints, pointDim);
    s_configCache[cacheKey] = result;
    return result;
}

} // namespace nerve::persistence::accelerated

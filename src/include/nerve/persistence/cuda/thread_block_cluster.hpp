#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
#include <string>

#if defined(__CUDACC__) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define NERVE_CUDA_CLUSTER_DIMS(x, y, z) __cluster_dims__(x, y, z)
#else
#define NERVE_CUDA_CLUSTER_DIMS(x, y, z)
#endif

namespace nerve::persistence::accelerated
{

constexpr int CLUSTER_MAX_SIZE = 16;
constexpr int CLUSTER_DEFAULT_SIZE_X = 8;
constexpr int CLUSTER_DEFAULT_SIZE_Y = 1;
constexpr int CLUSTER_DEFAULT_SIZE_Z = 1;
constexpr int CLUSTER_MAX_SIZE_PRE_B200 = 8;
constexpr int CLUSTER_B200_COMPUTE_CAPABILITY = 100;
constexpr size_t CLUSTER_DEFAULT_SMEM_PER_BLOCK = 128 * 1024;
constexpr int CLUSTER_L2_CACHE_LINE_SIZE = 128;
constexpr int CLUSTER_DEFAULT_PIPELINE_STAGES = 3;
constexpr int TMA_POINTS_PER_CLUSTER = 512;
constexpr int TMA_MAX_POINT_DIMENSIONS = 8;

struct AdvancedClusterConfig
{
    int clusterSizeX = CLUSTER_DEFAULT_SIZE_X;
    int clusterSizeY = CLUSTER_DEFAULT_SIZE_Y;
    int clusterSizeZ = CLUSTER_DEFAULT_SIZE_Z;
    bool useNonPortable = false;
    bool useMulticast = true;
    bool useDistributedL2 = true;

    size_t sharedMemPerBlock = CLUSTER_DEFAULT_SMEM_PER_BLOCK;
    int l2PromotionSize = CLUSTER_L2_CACHE_LINE_SIZE;
    int numPipelineStages = CLUSTER_DEFAULT_PIPELINE_STAGES;

    [[nodiscard]] __host__ __device__ int totalClusterSize() const
    {
        return clusterSizeX * clusterSizeY * clusterSizeZ;
    }

    [[nodiscard]] bool isValid(int computeCapability) const;
    [[nodiscard]] size_t totalDistributedSmem() const;
};

struct ClusterBenchmarkConfig
{
    int minClusterSizeX = 2;
    int maxClusterSizeX = 16;
    int minClusterSizeY = 1;
    int maxClusterSizeY = 2;
    int numTrials = 10;
    bool testNonPortable = true;
};

struct ClusterBenchmarkResult
{
    int optimalClusterSizeX = 1;
    int optimalClusterSizeY = 1;
    int optimalClusterSizeZ = 1;
    bool useNonPortable = false;
    float optimalTimeMs = std::numeric_limits<float>::infinity();
    bool blackwellRequired = false;
};

[[nodiscard]] ClusterBenchmarkResult
benchmarkClusterConfig(uint32_t nPoints, uint32_t pointDim,
                       const ClusterBenchmarkConfig &config = ClusterBenchmarkConfig{});

[[nodiscard]] ClusterBenchmarkResult
getOrBenchmarkClusterConfig(uint32_t nPoints, uint32_t pointDim, const std::string &cacheKey);

__global__ void NERVE_CUDA_CLUSTER_DIMS(16, 1, 1)
    cluster16DistanceMatrixKernel(const float *__restrict__ points, float *__restrict__ distances,
                                  uint32_t nPoints, uint32_t pointDim, float maxRadiusSq);

__global__ void NERVE_CUDA_CLUSTER_DIMS(4, 2, 1)
    clusterDistributedL2Kernel(const float *__restrict__ points, float *__restrict__ distances,
                               uint32_t nPoints, uint32_t pointDim, float maxRadiusSq,
                               AdvancedClusterConfig config);

__global__ void NERVE_CUDA_CLUSTER_DIMS(8, 1, 1)
    clusterTMAMulticastKernel(const float *__restrict__ points, float *__restrict__ distances,
                              uint32_t nPoints, uint32_t pointDim, float maxRadiusSq);

[[nodiscard]] bool clusterFeaturesAvailable();

[[nodiscard]] bool nonPortableClustersAvailable();

} // namespace nerve::persistence::accelerated

#undef NERVE_CUDA_CLUSTER_DIMS

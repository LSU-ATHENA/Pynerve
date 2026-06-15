#include "nerve/persistence/cuda/thread_block_cluster.hpp"

#if defined(__CUDACC__) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define NERVE_CLUSTER_DIMS(x, y, z) __cluster_dims__(x, y, z)
#else
#define NERVE_CLUSTER_DIMS(x, y, z)
#endif

#if defined(__CUDACC__)
#include <cmath>

namespace nerve::persistence::accelerated
{

__global__ void NERVE_CLUSTER_DIMS(4, 2, 1) __launch_bounds__(256)
    clusterDistributedL2Kernel(const float *__restrict__ points, float *__restrict__ distances,
                               uint32_t nPoints, uint32_t pointDim, float maxRadiusSq,
                               AdvancedClusterConfig config)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if (nPoints == 0 || pointDim == 0 || pointDim > TMA_MAX_POINT_DIMENSIONS ||
        config.totalClusterSize() <= 0)
    {
        return;
    }
    if (!isfinite(maxRadiusSq) || maxRadiusSq < 0.0f)
    {
        return;
    }

    int clusterRank;
    asm volatile("mov.u32 %0, %%clusterid;" : "=r"(clusterRank));

    int tid = threadIdx.x;
    int warpId = tid / 32;

    extern __shared__ char smemBase[];
    float *distributedSmem = reinterpret_cast<float *>(smemBase);

    int pointsInSmem = max(1, static_cast<int>(nPoints / 2));
    int pointsInL2 = static_cast<int>(nPoints) - pointsInSmem;
    int smemPointsPerBlock =
        max(1, (pointsInSmem + config.totalClusterSize() - 1) / config.totalClusterSize());

    int mySmemStart = clusterRank * smemPointsPerBlock;
    int mySmemCount = min(smemPointsPerBlock, pointsInSmem - mySmemStart);
    if (mySmemCount < 0)
    {
        mySmemCount = 0;
    }

    for (int i = tid; i < mySmemCount * pointDim; i += blockDim.x)
    {
        int pointIdx = i / pointDim;
        int dimIdx = i % pointDim;
        int globalIdx = mySmemStart + pointIdx;

        if (globalIdx < nPoints)
        {
            distributedSmem[pointIdx * pointDim + dimIdx] = points[globalIdx * pointDim + dimIdx];
        }
    }

    __syncthreads();
    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;" ::: "memory");

    if (warpId < 4)
    {
        int tileSize = 64;
        int tileRow = (blockIdx.y * 4 + warpId) * tileSize;

        for (int localRow = tid % 32; localRow < tileSize; localRow += 32)
        {
            int globalRow = tileRow + localRow;
            if (globalRow >= nPoints || globalRow >= pointsInSmem)
                continue;

            int sourceBlock = globalRow / smemPointsPerBlock;
            int localIdx = globalRow % smemPointsPerBlock;
            if (sourceBlock >= config.totalClusterSize())
                continue;

            float pointA[8];
            if (sourceBlock == clusterRank)
            {
                for (int d = 0; d < pointDim && d < 8; ++d)
                {
                    pointA[d] = distributedSmem[localIdx * pointDim + d];
                }
            }
            else
            {
                int remoteBlockPoints = smemPointsPerBlock;
                int remoteLocalIdx = globalRow % remoteBlockPoints;

                size_t smemPerBlock = config.sharedMemPerBlock;
                uint32_t remoteBaseAddr = static_cast<uint32_t>(sourceBlock * smemPerBlock);
                uint32_t targetAddr = remoteBaseAddr + remoteLocalIdx * pointDim * sizeof(float);

                for (int d = 0; d < pointDim && d < 8; ++d)
                {
                    uint32_t value;
                    asm volatile("mapa.sync.aligned.b32 %0, [%1], %2;"
                                 : "=r"(value)
                                 : "r"(targetAddr + d * sizeof(float)), "r"(sourceBlock));
                    pointA[d] = __uint_as_float(value);
                }
            }

            for (int j = 0; j < 64 && (tileRow + j) < nPoints; ++j)
            {
                int globalCol = tileRow + j;
                if (globalRow < globalCol)
                    continue;

                float distSq = 0.0f;
                float pointB[8];

                for (int d = 0; d < pointDim && d < 8; ++d)
                {
                    pointB[d] = points[globalCol * pointDim + d];
                }

                for (int d = 0; d < pointDim && d < 8; ++d)
                {
                    float diff = pointA[d] - pointB[d];
                    float contribution = diff * diff;
                    float nextDistSq = distSq + contribution;
                    if (!isfinite(diff) || !isfinite(contribution) || !isfinite(nextDistSq))
                    {
                        distSq = CUDART_INF_F;
                        break;
                    }
                    distSq = nextDistSq;
                }

                if (distSq <= maxRadiusSq)
                {
                    distances[globalRow * nPoints + globalCol] = sqrtf(distSq);
                }
            }
        }
    }
    else
    {
        int l2Start = pointsInSmem;
        int l2Count = pointsInL2;

        for (int i = l2Start + tid; i < l2Start + l2Count; i += blockDim.x)
        {
            asm volatile("prefetch.global.L2 [%0];" ::"l"(&points[i * pointDim]));
        }

        for (int i = l2Start + (warpId - 4) * 32 + (tid % 32); i < l2Start + l2Count;
             i += (8 - 4) * 32)
        {
            int globalRow = i;
            if (globalRow >= nPoints)
                continue;

            float pointA[8];
            for (int d = 0; d < pointDim && d < 8; ++d)
            {
                pointA[d] = points[globalRow * pointDim + d];
            }

            for (int j = 0; j < 64; ++j)
            {
                int globalCol = j;
                if (globalRow < globalCol)
                    continue;
                if (globalCol >= pointsInSmem)
                    continue;

                int sourceBlock = globalCol / smemPointsPerBlock;
                int localIdx = globalCol % smemPointsPerBlock;
                if (sourceBlock >= config.totalClusterSize())
                    continue;

                float pointB[8];
                if (sourceBlock == clusterRank)
                {
                    for (int d = 0; d < pointDim && d < 8; ++d)
                    {
                        pointB[d] = distributedSmem[localIdx * pointDim + d];
                    }
                }
                else
                {
                    continue;
                }

                float distSq = 0.0f;
                for (int d = 0; d < pointDim && d < 8; ++d)
                {
                    float diff = pointA[d] - pointB[d];
                    float contribution = diff * diff;
                    float nextDistSq = distSq + contribution;
                    if (!isfinite(diff) || !isfinite(contribution) || !isfinite(nextDistSq))
                    {
                        distSq = CUDART_INF_F;
                        break;
                    }
                    distSq = nextDistSq;
                }

                if (distSq <= maxRadiusSq)
                {
                    distances[globalRow * nPoints + globalCol] = sqrtf(distSq);
                }
            }
        }
    }
#else
    (void)points;
    (void)distances;
    (void)nPoints;
    (void)pointDim;
    (void)maxRadiusSq;
    (void)config;
#endif
}

} // namespace nerve::persistence::accelerated

#endif

#undef NERVE_CLUSTER_DIMS

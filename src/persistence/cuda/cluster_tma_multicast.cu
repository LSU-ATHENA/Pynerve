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

__global__ void NERVE_CLUSTER_DIMS(8, 1, 1) __launch_bounds__(256)
    clusterTMAMulticastKernel(const float *__restrict__ points, float *__restrict__ distances,
                              uint32_t nPoints, uint32_t pointDim, float maxRadiusSq)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && CUDART_VERSION < 13000
    if (nPoints == 0 || pointDim == 0 || pointDim > TMA_MAX_POINT_DIMENSIONS)
    {
        return;
    }
    if (!isfinite(maxRadiusSq) || maxRadiusSq < 0.0f)
    {
        return;
    }

    int clusterRank;
    asm volatile("mov.u32 %0, %%clusterid;" : "=r"(clusterRank));

    int clusterSize;
    asm volatile("mov.u32 %0, %%clustersize;" : "=r"(clusterSize));
    if (clusterSize <= 0)
    {
        return;
    }

    int tid = threadIdx.x;

    extern __shared__ char smemBase[];
    float *cachedPoints = reinterpret_cast<float *>(smemBase);

    int pointsPerBlock = (nPoints + clusterSize - 1) / clusterSize;
    int myStart = clusterRank * pointsPerBlock;
    int myCount = min(pointsPerBlock, static_cast<int>(nPoints) - myStart);

    __syncthreads();
    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;" ::: "memory");

    for (int i = tid; i < myCount * pointDim; i += blockDim.x)
    {
        int pointIdx = i / pointDim;
        int dimIdx = i % pointDim;
        int globalIdx = myStart + pointIdx;

        if (globalIdx < nPoints)
        {
            cachedPoints[pointIdx * pointDim + dimIdx] = points[globalIdx * pointDim + dimIdx];
        }
    }

    __syncthreads();

    int rowsPerBlock = (nPoints + gridDim.x - 1) / gridDim.x;
    int myRowStart = blockIdx.x * rowsPerBlock;
    int myRowCount = min(rowsPerBlock, static_cast<int>(nPoints) - myRowStart);

    for (int localRow = tid; localRow < myRowCount; localRow += blockDim.x)
    {
        int globalRow = myRowStart + localRow;
        if (globalRow >= nPoints)
            continue;

        float pointA[8];
        int sourceBlockA = globalRow / pointsPerBlock;
        int localIdxA = globalRow % pointsPerBlock;

        if (sourceBlockA == clusterRank)
        {
            for (int d = 0; d < pointDim && d < 8; ++d)
            {
                pointA[d] = cachedPoints[localIdxA * pointDim + d];
            }
        }
        else
        {
            size_t smemPerBlock = CLUSTER_DEFAULT_SMEM_PER_BLOCK;
            uint32_t remoteBaseAddr = static_cast<uint32_t>(sourceBlockA * smemPerBlock);
            uint32_t targetAddr = remoteBaseAddr + localIdxA * pointDim * sizeof(float);

            for (int d = 0; d < pointDim && d < 8; ++d)
            {
                uint32_t value;
                asm volatile("mapa.sync.aligned.b32 %0, [%1], %2;"
                             : "=r"(value)
                             : "r"(static_cast<unsigned int>(targetAddr + d * sizeof(float))), "r"(sourceBlockA));
                pointA[d] = __uint_as_float(value);
            }
        }

        for (int globalCol = 0; globalCol <= globalRow; ++globalCol)
        {
            float pointB[8];
            int sourceBlockB = globalCol / pointsPerBlock;
            int localIdxB = globalCol % pointsPerBlock;

            if (sourceBlockB == clusterRank)
            {
                for (int d = 0; d < pointDim && d < 8; ++d)
                {
                    pointB[d] = cachedPoints[localIdxB * pointDim + d];
                }
            }
            else
            {
                size_t smemPerBlock = CLUSTER_DEFAULT_SMEM_PER_BLOCK;
                uint32_t remoteBaseAddr = static_cast<uint32_t>(sourceBlockB * smemPerBlock);
                uint32_t targetAddr = remoteBaseAddr + localIdxB * pointDim * sizeof(float);

                for (int d = 0; d < pointDim && d < 8; ++d)
                {
                    uint32_t value;
                    asm volatile("mapa.sync.aligned.b32 %0, [%1], %2;"
                                 : "=r"(value)
                                 : "r"(static_cast<unsigned int>(targetAddr + d * sizeof(float))), "r"(sourceBlockB));
                    pointB[d] = __uint_as_float(value);
                }
            }

            float distSq = 0.0f;
            for (int d = 0; d < pointDim && d < 8; ++d)
            {
                float diff = pointA[d] - pointB[d];
                float contribution = diff * diff;
                float nextDistSq = distSq + contribution;
                if (!isfinite(diff) || !isfinite(contribution) || !isfinite(nextDistSq))
                {
                    distSq = __int_as_float(0x7f800000);
                    break;
                }
                distSq = nextDistSq;
            }

            if (distSq <= maxRadiusSq)
            {
                distances[globalRow * nPoints + globalCol] = sqrtf(distSq);
                if (globalRow != globalCol)
                {
                    distances[globalCol * nPoints + globalRow] = sqrtf(distSq);
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
#endif
}

} // namespace nerve::persistence::accelerated

#endif

#undef NERVE_CLUSTER_DIMS

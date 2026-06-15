// Tile-oriented CUDA kernels for distance and reduction workloads.
// The kernels use ordinary CUDA shared-memory staging and grid-level tiling.

#include "nerve/types.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <type_traits>
#include <vector>

namespace nerve::persistence::accelerated
{

[[maybe_unused]] constexpr int TILE_KERNEL_BLOCK_SIZE = 256;

template <typename T, int TileSizeM, int TileSizeN, int PointDim>
struct DistanceTileConfig
{
    static constexpr int kTileSizeM = TileSizeM;
    static constexpr int kTileSizeN = TileSizeN;
    static constexpr int kPointDim = PointDim;
};

template <typename T, typename Config, int ClusterSize = 1>
__global__ void __launch_bounds__(256)
    cudaTileDistanceMatrixKernel(const T *__restrict__ points, T *__restrict__ distances,
                                 uint32_t nPoints, float maxRadiusSq)
{
    (void)ClusterSize;
    if (points == nullptr || distances == nullptr || nPoints == 0 || !isfinite(maxRadiusSq) ||
        maxRadiusSq < 0.0f)
    {
        return;
    }
    const int tileRow = blockIdx.y;
    const int tileCol = blockIdx.x;
    const int globalRow = tileRow * Config::kTileSizeM;
    const int globalCol = tileCol * Config::kTileSizeN;
    if (globalRow >= static_cast<int>(nPoints) || globalCol >= static_cast<int>(nPoints))
    {
        return;
    }
    if (globalRow + Config::kTileSizeM <= globalCol)
    {
        return;
    }

    extern __shared__ unsigned char smemBuffer[];
    T *tilePointsA = reinterpret_cast<T *>(smemBuffer);
    T *tilePointsB = tilePointsA + Config::kTileSizeM * Config::kPointDim;

    const int tid = threadIdx.x;
    const int threadsPerBlock = blockDim.x;

    const int pointsToLoadA = Config::kTileSizeM * Config::kPointDim;
    for (int i = tid; i < pointsToLoadA; i += threadsPerBlock)
    {
        const int pointIdx = i / Config::kPointDim;
        const int dimIdx = i % Config::kPointDim;
        const int globalIdx = globalRow + pointIdx;
        tilePointsA[i] = (globalIdx < static_cast<int>(nPoints))
                             ? points[globalIdx * Config::kPointDim + dimIdx]
                             : static_cast<T>(0);
    }

    const int pointsToLoadB = Config::kTileSizeN * Config::kPointDim;
    for (int i = tid; i < pointsToLoadB; i += threadsPerBlock)
    {
        const int pointIdx = i / Config::kPointDim;
        const int dimIdx = i % Config::kPointDim;
        const int globalIdx = globalCol + pointIdx;
        tilePointsB[i] = (globalIdx < static_cast<int>(nPoints))
                             ? points[globalIdx * Config::kPointDim + dimIdx]
                             : static_cast<T>(0);
    }
    __syncthreads();

    const int pairCount = Config::kTileSizeM * Config::kTileSizeN;
    const int pairsPerThread = (pairCount + threadsPerBlock - 1) / threadsPerBlock;
    const int startPair = tid * pairsPerThread;
    const int endPair = min(startPair + pairsPerThread, pairCount);

    for (int p = startPair; p < endPair; ++p)
    {
        const int localRow = p / Config::kTileSizeN;
        const int localCol = p % Config::kTileSizeN;
        const int rowIdx = globalRow + localRow;
        const int colIdx = globalCol + localCol;
        if (rowIdx >= static_cast<int>(nPoints) || colIdx >= static_cast<int>(nPoints) ||
            rowIdx < colIdx)
        {
            continue;
        }

        const T *pointA = &tilePointsA[localRow * Config::kPointDim];
        const T *pointB = &tilePointsB[localCol * Config::kPointDim];
        float distSq = 0.0f;
        for (int d = 0; d < Config::kPointDim; ++d)
        {
            const float diff = static_cast<float>(pointA[d]) - static_cast<float>(pointB[d]);
            const float contribution = diff * diff;
            const float nextDistSq = distSq + contribution;
            if (!isfinite(diff) || !isfinite(contribution) || !isfinite(nextDistSq))
            {
                distSq = INFINITY;
                break;
            }
            distSq = nextDistSq;
            if (distSq > maxRadiusSq)
            {
                break;
            }
        }

        const T out = (distSq <= maxRadiusSq) ? static_cast<T>(sqrtf(distSq)) : static_cast<T>(-1);
        distances[rowIdx * nPoints + colIdx] = out;
        if (rowIdx != colIdx)
        {
            distances[colIdx * nPoints + rowIdx] = out;
        }
    }
}

template <typename T, int TileSize, int ClusterSize = 1>
__global__ void __launch_bounds__(256)
    cudaTileReductionKernel(const T *__restrict__ input, T *__restrict__ output, uint32_t nElements,
                            T identity)
{
    (void)ClusterSize;
    extern __shared__ T tileData[];
    const int tid = threadIdx.x;
    const int globalStart = blockIdx.x * TileSize;

    for (int i = tid; i < TileSize; i += blockDim.x)
    {
        const int idx = globalStart + i;
        tileData[i] = (idx < static_cast<int>(nElements)) ? input[idx] : identity;
    }
    __syncthreads();

    for (int stride = TileSize / 2; stride > 0; stride /= 2)
    {
        if (tid < stride)
        {
            tileData[tid] += tileData[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        output[blockIdx.x] = tileData[0];
    }
}

template <typename T, int TileSize>
__global__ void __launch_bounds__(1024)
    cudaTileTransposeKernel(const T *__restrict__ input, T *__restrict__ output, uint32_t rows,
                            uint32_t cols)
{
    __shared__ T tile[TileSize][TileSize + 1];
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int rowIn = by * TileSize + ty;
    const int colIn = bx * TileSize + tx;
    if (rowIn < static_cast<int>(rows) && colIn < static_cast<int>(cols))
    {
        tile[ty][tx] = input[rowIn * cols + colIn];
    }
    __syncthreads();

    const int rowOut = bx * TileSize + ty;
    const int colOut = by * TileSize + tx;
    if (rowOut < static_cast<int>(cols) && colOut < static_cast<int>(rows))
    {
        output[rowOut * rows + colOut] = tile[tx][ty];
    }
}

using FP32TileConfig = DistanceTileConfig<float, 64, 64, 3>;
using FP64TileConfig = DistanceTileConfig<double, 32, 32, 3>;

class CudaTileAutoTuner
{
public:
    struct TileBenchmarkResult
    {
        int tileSizeM;
        int tileSizeN;
        int pointDim;
        int clusterSize;
        float timeMs;
        float throughput;
    };

    template <typename T>
    static TileBenchmarkResult tune(uint32_t nPoints, uint32_t pointDim, int computeCapability,
                                    int numTrials = 5)
    {
        (void)computeCapability;
        using Config = std::conditional_t<std::is_same_v<T, float>, FP32TileConfig, FP64TileConfig>;

        TileBenchmarkResult out{
            Config::kTileSizeM, Config::kTileSizeN, Config::kPointDim, 1, 0.0f, 0.0f,
        };

        if (pointDim != static_cast<uint32_t>(Config::kPointDim) || nPoints == 0)
        {
            return out;
        }

        out.timeMs = benchmarkNativeConfig<T, Config>(nPoints, numTrials);
        out.throughput = (out.timeMs > 0.0f)
                             ? (static_cast<float>(nPoints) * static_cast<float>(nPoints)) /
                                   (out.timeMs / 1000.0f)
                             : 0.0f;
        return out;
    }

private:
    template <typename T, typename Config>
    static float benchmarkNativeConfig(uint32_t nPoints, int numTrials)
    {
        const int pointDim = Config::kPointDim;
        const size_t pointsSize =
            static_cast<size_t>(nPoints) * static_cast<size_t>(pointDim) * sizeof(T);
        const size_t distSize = static_cast<size_t>(nPoints) * nPoints * sizeof(T);

        T *d_points = nullptr;
        T *d_distances = nullptr;
        if (cudaMalloc(&d_points, pointsSize) != cudaSuccess)
        {
            return 1000.0f;
        }
        if (cudaMalloc(&d_distances, distSize) != cudaSuccess)
        {
            cudaFree(d_points);
            return 1000.0f;
        }

        std::vector<float> hostPointsFloat(static_cast<size_t>(nPoints) *
                                           static_cast<size_t>(pointDim));
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (float &v : hostPointsFloat)
            v = dist(rng);

        std::vector<T> hostPoints(static_cast<size_t>(nPoints) * static_cast<size_t>(pointDim));
        for (size_t i = 0; i < hostPointsFloat.size(); ++i)
        {
            hostPoints[i] = static_cast<T>(hostPointsFloat[i]);
        }
        if (cudaMemcpy(d_points, hostPoints.data(), pointsSize, cudaMemcpyHostToDevice) !=
                cudaSuccess ||
            cudaMemset(d_distances, 0, distSize) != cudaSuccess)
        {
            cudaFree(d_points);
            cudaFree(d_distances);
            return 1000.0f;
        }

        const dim3 blockDim(TILE_KERNEL_BLOCK_SIZE);
        const dim3 gridDim((nPoints + Config::kTileSizeM - 1) / Config::kTileSizeM,
                           (nPoints + Config::kTileSizeN - 1) / Config::kTileSizeN);
        const size_t smemSize =
            static_cast<size_t>(2) * Config::kTileSizeM * Config::kPointDim * sizeof(T);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        float total = 0.0f;
        for (int i = 0; i < numTrials; ++i)
        {
            cudaEventRecord(start);
            cudaTileDistanceMatrixKernel<T, Config, 1>
                <<<gridDim, blockDim, smemSize>>>(d_points, d_distances, nPoints, 1000.0f);
            if (cudaGetLastError() != cudaSuccess)
            {
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                cudaFree(d_points);
                cudaFree(d_distances);
                return 1000.0f;
            }
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, start, stop);
            total += ms;
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_points);
        cudaFree(d_distances);
        return total / std::max(1, numTrials);
    }
};

template __global__ void
cudaTileDistanceMatrixKernel<float, FP32TileConfig, 1>(const float *, float *, uint32_t, float);
template __global__ void
cudaTileDistanceMatrixKernel<double, FP64TileConfig, 1>(const double *, double *, uint32_t, float);
template __global__ void cudaTileReductionKernel<float, 256, 1>(const float *, float *, uint32_t,
                                                                float);
template __global__ void cudaTileTransposeKernel<float, 32>(const float *, float *, uint32_t,
                                                            uint32_t);

} // namespace nerve::persistence::accelerated

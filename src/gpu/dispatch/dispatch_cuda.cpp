#include "nerve/gpu/cuda_dispatch.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <vector>

namespace nerve::persistence::accelerated
{
namespace
{

bool checkedSizeProduct(Size lhs, Size rhs, Size &out) noexcept
{
    if (lhs != 0 && rhs > std::numeric_limits<Size>::max() / lhs)
    {
        return false;
    }
    out = lhs * rhs;
    return true;
}

bool checkedByteProduct(Size count, std::size_t element_size, std::size_t &out) noexcept
{
    if (element_size != 0 &&
        count > static_cast<Size>(std::numeric_limits<std::size_t>::max() / element_size))
    {
        return false;
    }
    out = static_cast<std::size_t>(count) * element_size;
    return true;
}

bool checkedIntProduct(int lhs, int rhs, int &out) noexcept
{
    if (lhs < 0 || rhs < 0 || (lhs != 0 && rhs > std::numeric_limits<int>::max() / lhs))
    {
        return false;
    }
    out = lhs * rhs;
    return true;
}

} // namespace

GPUArchitecture GPUArchitecture::detect()
{
    GPUArchitecture arch{};
    arch.major = 0;
    arch.minor = 0;
    arch.computeCapability = 0;
    arch.multiProcessorCount = 0;
    arch.sharedMemPerBlock = 0;
    arch.totalGlobalMem = 0;
    arch.maxThreadsPerBlock = 0;
    arch.maxThreadsPerMultiProcessor = 0;
    arch.family = Family::Unknown;

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0)
    {
        return arch;
    }

    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess)
    {
        return arch;
    }

    arch.major = prop.major;
    arch.minor = prop.minor;
    arch.computeCapability = prop.major * 10 + prop.minor;
    arch.multiProcessorCount = prop.multiProcessorCount;
    arch.sharedMemPerBlock = prop.sharedMemPerBlock;
    arch.totalGlobalMem = prop.totalGlobalMem;
    arch.maxThreadsPerBlock = prop.maxThreadsPerBlock;
    arch.maxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;

    switch (arch.computeCapability / 10)
    {
        case 2:
            arch.family = Family::Fermi;
            break;
        case 3:
            arch.family = Family::Kepler;
            break;
        case 5:
            arch.family = Family::Maxwell;
            break;
        case 6:
            arch.family = Family::Pascal;
            break;
        case 7:
            arch.family = (arch.minor == 5) ? Family::Turing : Family::Volta;
            break;
        case 8:
            arch.family = (arch.minor == 9) ? Family::Ada : Family::Ampere;
            break;
        case 9:
            arch.family = Family::Hopper;
            break;
        case 10:
            arch.family = Family::Blackwell;
            break;
        default:
            arch.family = Family::Unknown;
            break;
    }

    return arch;
}

bool GPUArchitecture::supportsTensorCores() const
{
    return computeCapability >= 70;
}

bool GPUArchitecture::supportsAsyncCopy() const
{
    return computeCapability >= 80;
}

bool GPUArchitecture::supportsCooperativeGroups() const
{
    return computeCapability >= 60;
}

bool GPUArchitecture::supportsMultiInstanceGPU() const
{
    return computeCapability >= 80;
}

int GPUArchitecture::getOptimalTileSize() const
{
    size_t memPerTile = 2 * 3 * sizeof(double);
    size_t maxTiles = sharedMemPerBlock / memPerTile;
    int tileSize = static_cast<int>(std::sqrt(static_cast<double>(maxTiles)));

    if (tileSize >= 64)
        return 64;
    if (tileSize >= 32)
        return 32;
    if (tileSize >= 16)
        return 16;
    return 8;
}

int GPUArchitecture::getOptimalBlockSize() const
{
    return computeCapability >= 80 ? 256 : 128;
}

int GPUArchitecture::getOptimalStreamCount() const
{
    int baseStreams = std::max(1, multiProcessorCount);
    if (computeCapability >= 80)
    {
        return baseStreams > std::numeric_limits<int>::max() / 2 ? std::numeric_limits<int>::max()
                                                                 : baseStreams * 2;
    }
    return baseStreams;
}

StreamPool::StreamPool(int numStreams)
{
    if (numStreams <= 0)
    {
        auto arch = GPUArchitecture::detect();
        numStreams = arch.getOptimalStreamCount();
    }

    numStreams_ = std::max(1, numStreams);

    for (int i = 0; i < numStreams_; ++i)
    {
        cudaStream_t stream = nullptr;
        if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess)
        {
            stream = nullptr;
        }
        computeStreams_.push_back(stream);

        stream = nullptr;
        if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess)
        {
            stream = nullptr;
        }
        transferStreams_.push_back(stream);
    }

    if (computeStreams_.empty())
    {
        computeStreams_.push_back(nullptr);
    }
    if (transferStreams_.empty())
    {
        transferStreams_.push_back(nullptr);
    }
    numStreams_ = static_cast<int>(computeStreams_.size());
}

StreamPool::~StreamPool()
{
    for (auto stream : computeStreams_)
    {
        cudaStreamDestroy(stream);
    }
    for (auto stream : transferStreams_)
    {
        cudaStreamDestroy(stream);
    }
}

cudaStream_t StreamPool::getComputeStream(int index)
{
    if (computeStreams_.empty())
    {
        return nullptr;
    }
    const size_t normalized = index >= 0 ? static_cast<size_t>(index)
                                         : static_cast<size_t>(-(static_cast<long long>(index)));
    return computeStreams_[normalized % computeStreams_.size()];
}

cudaStream_t StreamPool::getTransferStream(int index)
{
    if (transferStreams_.empty())
    {
        return nullptr;
    }
    const size_t normalized = index >= 0 ? static_cast<size_t>(index)
                                         : static_cast<size_t>(-(static_cast<long long>(index)));
    return transferStreams_[normalized % transferStreams_.size()];
}

cudaStream_t StreamPool::getH2DStream(int index)
{
    return getComputeStream(index);
}

cudaStream_t StreamPool::getD2HStream(int index)
{
    return getTransferStream(index);
}

void StreamPool::synchronizeAll()
{
    for (auto stream : computeStreams_)
    {
        cudaStreamSynchronize(stream);
    }
    for (auto stream : transferStreams_)
    {
        cudaStreamSynchronize(stream);
    }
}

void StreamPool::waitForComputeTransferPair(int index)
{
    cudaEvent_t event = nullptr;
    if (cudaEventCreate(&event) != cudaSuccess)
    {
        return;
    }

    if (cudaEventRecord(event, getComputeStream(index)) != cudaSuccess)
    {
        cudaEventDestroy(event);
        return;
    }
    (void)cudaStreamWaitEvent(getTransferStream(index), event, 0);

    cudaEventDestroy(event);
}

std::map<std::string, CudaAutoTuner::KernelConfig> CudaAutoTuner::configs_;

CudaAutoTuner::KernelConfig CudaAutoTuner::tuneDistanceMatrix(Size n_points, Size point_dim,
                                                              int numTrials)
{
    KernelConfig bestConfig;
    bestConfig.measuredTimeMs = 1e9f;
    if (n_points == 0 || point_dim == 0 || numTrials <= 0)
    {
        return bestConfig;
    }

    auto arch = GPUArchitecture::detect();

    std::vector<int> tileSizes = {16, 32, 64};
    std::vector<dim3> blockConfigs = {
        dim3(16, 16, 1),
        dim3(32, 16, 1),
        dim3(32, 32, 1),
    };

    Size point_values = 0;
    Size matrix_values = 0;
    size_t pointsSize = 0;
    size_t distSize = 0;
    if (!checkedSizeProduct(n_points, point_dim, point_values) ||
        !checkedSizeProduct(n_points, n_points, matrix_values) ||
        !checkedByteProduct(point_values, sizeof(double), pointsSize) ||
        !checkedByteProduct(matrix_values, sizeof(double), distSize))
    {
        return bestConfig;
    }

    double *d_points = nullptr;
    double *d_distances = nullptr;
    if (cudaMalloc(&d_points, pointsSize) != cudaSuccess ||
        cudaMalloc(&d_distances, distSize) != cudaSuccess)
    {
        cudaFree(d_points);
        cudaFree(d_distances);
        return bestConfig;
    }

    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess)
    {
        cudaFree(d_points);
        cudaFree(d_distances);
        return bestConfig;
    }

    for (int tileSize : tileSizes)
    {
        for (const auto &block : blockConfigs)
        {
            int double_tile = 0;
            int tile_dim_values = 0;
            int sharedMemSize = 0;
            if (point_dim > static_cast<Size>(std::numeric_limits<int>::max()) ||
                !checkedIntProduct(2, tileSize, double_tile) ||
                !checkedIntProduct(double_tile, static_cast<int>(point_dim), tile_dim_values) ||
                !checkedIntProduct(tile_dim_values, static_cast<int>(sizeof(double)),
                                   sharedMemSize))
            {
                continue;
            }

            if (static_cast<size_t>(sharedMemSize) > arch.sharedMemPerBlock)
            {
                continue;
            }

            DistanceMatrixOptimizer::compute(d_points, d_distances, n_points, point_dim, 1e10,
                                             stream);
            cudaStreamSynchronize(stream);

            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            float totalTime = 0.0f;
            for (int trial = 0; trial < numTrials; ++trial)
            {
                cudaEventRecord(start, stream);

                DistanceMatrixOptimizer::compute(d_points, d_distances, n_points, point_dim, 1e10,
                                                 stream);

                cudaEventRecord(stop, stream);
                cudaStreamSynchronize(stream);

                float ms = 0.0f;
                cudaEventElapsedTime(&ms, start, stop);
                totalTime += ms;
            }

            float avgTime = totalTime / numTrials;

            if (avgTime < bestConfig.measuredTimeMs)
            {
                bestConfig.block = block;
                bestConfig.tileSize = tileSize;
                bestConfig.sharedMemBytes = sharedMemSize;
                bestConfig.measuredTimeMs = avgTime;
            }

            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
    }

    cudaFree(d_points);
    cudaFree(d_distances);
    cudaStreamDestroy(stream);

    std::string key =
        "distance_matrix_" + std::to_string(n_points) + "_" + std::to_string(point_dim);
    configs_[key] = bestConfig;

    return bestConfig;
}

void CudaAutoTuner::saveConfig(const std::string &filename)
{
    std::ofstream file(filename);
    for (const auto &[key, config] : configs_)
    {
        file << key << " " << config.block.x << " " << config.block.y << " " << config.tileSize
             << " " << config.sharedMemBytes << " " << config.measuredTimeMs << "\n";
    }
}

void CudaAutoTuner::loadConfig(const std::string &filename)
{
    std::ifstream file(filename);
    std::string key;
    while (file >> key)
    {
        KernelConfig config;
        file >> config.block.x >> config.block.y >> config.tileSize >> config.sharedMemBytes >>
            config.measuredTimeMs;
        config.block.z = 1;
        configs_[key] = config;
    }
}

void CudaProfiler::beginRange(const char *name)
{
#ifdef CUDA_ENABLE_NVTX
    nvtxRangePushA(name);
#else
    (void)name;
#endif
}

void CudaProfiler::endRange()
{
#ifdef CUDA_ENABLE_NVTX
    nvtxRangePop();
#endif
}

void CudaProfiler::markKernel(const char *name, cudaStream_t stream)
{
#ifdef CUDA_ENABLE_NVTX
    nvtxRangePushEx(nullptr, name, 0xFF000000, NVTX_MESSAGE_TYPE_ASCII);
    nvtxStreamMarkEx(nullptr, stream, name, 0xFF000000);
#else
    (void)name;
    (void)stream;
#endif
}

float CudaProfiler::getKernelOccupancy(const void *func, int blockSize, size_t dynamicSmemSize)
{
    int minGridSize = 0;
    int blockSizeOut = 0;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSizeOut, func, dynamicSmemSize, 0);

    int maxActiveBlocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, func, blockSize,
                                                  dynamicSmemSize);

    auto arch = GPUArchitecture::detect();
    int maxWarpsPerSM = std::max(1, arch.maxThreadsPerMultiProcessor / 32);
    int activeWarps = maxActiveBlocks * (blockSize / 32);

    return static_cast<float>(activeWarps) / maxWarpsPerSM;
}

void CudaProfiler::printMetrics(cudaStream_t stream)
{
    cudaStreamSynchronize(stream);

    auto arch = GPUArchitecture::detect();
    std::cout << "GPU Architecture: " << arch.major << "." << arch.minor << "\n";
    std::cout << "SM Count: " << arch.multiProcessorCount << "\n";
    std::cout << "Shared Memory per Block: " << arch.sharedMemPerBlock / 1024 << " KB\n";
}

} // namespace nerve::persistence::accelerated

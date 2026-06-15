#pragma once

#include "nerve/gpu/detail/cuda_tile_detail_impl.hpp"

namespace nerve::gpu::tile
{
namespace detail
{

inline bool productAtLeast(uint32_t lhs, uint32_t rhs, uint32_t depth, uint64_t threshold)
{
    if (lhs == 0 || rhs == 0 || depth == 0)
    {
        return false;
    }
    uint64_t product = lhs;
    if (product >= threshold || rhs > threshold / product)
    {
        return true;
    }
    product *= rhs;
    return product >= threshold || depth > threshold / product;
}

} // namespace detail

// Inline Implementation: TileAutoTuner

inline TileConfig TileAutoTuner::tuneDistanceMatrix(uint32_t nPoints, uint32_t pointDim,
                                                    TileDataType dtype, int deviceId)
{
    const cudaDeviceProp prop = detail::queryDeviceProperties(deviceId);
    TileConfig cfg;
    cfg.dataType = dtype;
    const bool large_memory = static_cast<uint64_t>(prop.totalGlobalMem) >= (8ULL << 30);
    if (nPoints >= 65536 && large_memory)
    {
        cfg.tileSizeM = 128;
        cfg.tileSizeN = 128;
    }
    else if (nPoints >= 8192)
    {
        cfg.tileSizeM = 64;
        cfg.tileSizeN = 64;
    }
    else
    {
        cfg.tileSizeM = 32;
        cfg.tileSizeN = 32;
    }
    const uint32_t preferred_k = std::max<uint32_t>(8, std::min<uint32_t>(pointDim, 128));
    cfg.tileSizeK =
        static_cast<int>(std::max<uint32_t>(static_cast<uint32_t>(prop.warpSize / 2), preferred_k));
    const int compute_capability = std::max(0, prop.major * 10 + prop.minor);
    if (compute_capability >= 90)
    {
        const size_t tile_m = static_cast<size_t>(std::max(1, cfg.tileSizeM));
        const size_t blocks = (static_cast<size_t>(nPoints) + tile_m - 1U) / tile_m;
        const int block_count = blocks > static_cast<size_t>(std::numeric_limits<int>::max())
                                    ? std::numeric_limits<int>::max()
                                    : static_cast<int>(blocks);
        cfg.clusterSizeX = ClusterUtils::calculateOptimalClusterSize(
            block_count, std::max(1, prop.multiProcessorCount), compute_capability);
    }
    else
    {
        cfg.clusterSizeX = 1;
    }
    cfg.useTMA = false;
    cfg.useTensorCores = false;
    return cfg;
}

inline TileConfig TileAutoTuner::tuneMatMul(uint32_t m, uint32_t n, uint32_t k, TileDataType dtype,
                                            int deviceId)
{
    const cudaDeviceProp prop = detail::queryDeviceProperties(deviceId);
    TileConfig cfg;
    cfg.dataType = dtype;
    const bool large_ops = detail::productAtLeast(m, n, k, 1ULL << 30);
    const int max_threads = std::max(64, prop.maxThreadsPerBlock);
    const int edge_tile = large_ops && max_threads >= 512 ? 128 : 64;
    cfg.tileSizeM = edge_tile;
    cfg.tileSizeN = edge_tile;
    cfg.tileSizeK = k >= 256 ? 64 : (k >= 128 ? 32 : 16);
    cfg.clusterSizeX =
        prop.major >= 9 ? std::max(1, std::min(4, prop.multiProcessorCount / 16)) : 1;
    cfg.useTMA = false;
    cfg.useTensorCores = false;
    return cfg;
}

inline TileConfig TileAutoTuner::tuneReduction(uint32_t nElements, TileDataType dtype, int deviceId)
{
    const cudaDeviceProp prop = detail::queryDeviceProperties(deviceId);
    TileConfig cfg;
    cfg.dataType = dtype;
    const int warp = std::max(32, prop.warpSize);
    cfg.tileSizeM = nElements >= (1U << 20) ? std::min(1024, std::max(256, warp * 8))
                                            : std::min(512, std::max(128, warp * 4));
    cfg.tileSizeN = 1;
    cfg.tileSizeK = 1;
    cfg.clusterSizeX = 1;
    return cfg;
}

inline std::vector<TileAutoTuner::TuningResult>
TileAutoTuner::benchmarkAll(const void *workloadData,
                            std::function<float(const TileConfig &)> benchmarkFunc)
{
    std::vector<TuningResult> results;
    const cudaDeviceProp prop = detail::queryDeviceProperties(0);
    float work_items = 1.0f;
    std::vector<TileConfig> candidates = {
        TileConfig{32, 32, 16, 1, 1, 1, TileDataType::kFP32, TileLayout::kRowMajor,
                   TilePrecision::kFull, false, false},
        TileConfig{64, 64, 16, 1, 1, 1, TileDataType::kFP32, TileLayout::kRowMajor,
                   TilePrecision::kFull, false, false},
        TileConfig{128, 64, 16, 1, 1, 1, TileDataType::kFP32, TileLayout::kRowMajor,
                   TilePrecision::kFull, false, false},
    };
    if (workloadData != nullptr)
    {
        const auto *shape = static_cast<const uint32_t *>(workloadData);
        work_items = static_cast<float>(
            std::max<uint64_t>(1ULL, static_cast<uint64_t>(shape[0]) * shape[0]));
        candidates.push_back(tuneDistanceMatrix(shape[0], shape[1], TileDataType::kFP32, 0));
    }
    results.reserve(candidates.size());
    for (const TileConfig &cfg : candidates)
    {
        const float time_ms = benchmarkFunc ? benchmarkFunc(cfg) : 0.0f;
        TuningResult result;
        result.config = cfg;
        result.timeMs = time_ms;
        result.throughput = time_ms > 0.0f ? (work_items / time_ms) : 0.0f;
        result.occupancy = detail::estimateOccupancy(cfg, prop);
        results.push_back(result);
    }
    return results;
}

inline void TileAutoTuner::saveTunedConfig(const std::string &key, const TileConfig &config)
{
    detail::tileConfigCache()[key] = config;
}

inline std::optional<TileConfig> TileAutoTuner::loadTunedConfig(const std::string &key)
{
    auto &cache = detail::tileConfigCache();
    auto it = cache.find(key);
    if (it == cache.end())
    {
        return std::nullopt;
    }
    return it->second;
}

} // namespace nerve::gpu::tile

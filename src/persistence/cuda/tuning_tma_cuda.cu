// Runtime TMA tile tuning utilities.
// This file provides concrete implementations for the public
// nerve::gpu::advanced::TmaTileTuner API.

#include "nerve/gpu/gpu_tuning.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace nerve::gpu::advanced
{

namespace
{

constexpr int kDefaultTrials = 10;
constexpr int kMinTile = 16;
constexpr int kMaxTile = 256;
constexpr int kMaxCluster = 16;
constexpr int kSmemSafetyPadBytes = 128;

[[nodiscard]] int clampInt(int value, int lo, int hi)
{
    return value < lo ? lo : (value > hi ? hi : value);
}

[[nodiscard]] std::size_t estimateSharedMemBytes(const TmaTileConfig &cfg)
{
    const std::size_t tile_points = static_cast<std::size_t>(cfg.pointTileSize);
    const std::size_t tile_dim = static_cast<std::size_t>(cfg.dimTileSize);
    return tile_points * tile_dim * sizeof(float) + kSmemSafetyPadBytes;
}

[[nodiscard]] std::vector<int> tileSearchSpace(int cc)
{
    if (cc >= 100)
        return {32, 64, 128, 256};
    if (cc >= 90)
        return {32, 64, 128};
    if (cc >= 80)
        return {32, 64};
    return {32};
}

[[nodiscard]] std::vector<int> clusterSearchSpace(int cc)
{
    if (cc >= 100)
        return {1, 2, 4, 8, 16};
    if (cc >= 90)
        return {1, 2, 4, 8};
    return {1};
}

[[nodiscard]] bool isConfigValid(const TmaTileConfig &cfg,
                                 const TmaTileTuner::WorkloadProfile &profile)
{
    if (cfg.pointTileSize < kMinTile || cfg.pointTileSize > kMaxTile)
        return false;
    if (cfg.dimTileSize <= 0 || cfg.numStages <= 0)
        return false;
    if (cfg.clusterSizeX <= 0 || cfg.clusterSizeY <= 0)
        return false;
    if (cfg.clusterSizeX * cfg.clusterSizeY > kMaxCluster)
        return false;
    if (profile.gpuComputeCapability < 90 && (cfg.clusterSizeX > 1 || cfg.clusterSizeY > 1))
    {
        return false;
    }
    if (cfg.pointTileSize > static_cast<int>(profile.nPoints) && profile.nPoints > 0)
        return false;
    if (estimateSharedMemBytes(cfg) > profile.availableSmem)
        return false;
    return true;
}

[[nodiscard]] float estimateOccupancy(const TmaTileConfig &cfg,
                                      const TmaTileTuner::WorkloadProfile &profile)
{
    const float tile_pressure = static_cast<float>(cfg.pointTileSize) / 128.0f;
    const float stage_penalty = 1.0f + 0.07f * static_cast<float>(std::max(0, cfg.numStages - 2));
    const float cluster_penalty =
        1.0f + 0.03f * static_cast<float>(std::max(0, cfg.clusterSizeX - 1));
    const float dim_scale = std::max(1.0f, static_cast<float>(profile.pointDim) / 16.0f);
    const float raw = 1.0f / (tile_pressure * stage_penalty * cluster_penalty * dim_scale);
    return std::clamp(raw * 1.6f, 0.05f, 1.0f);
}

[[nodiscard]] float estimateBandwidthUtilization(const TmaTileConfig &cfg,
                                                 const TmaTileTuner::WorkloadProfile &profile)
{
    const float cc_bonus = profile.gpuComputeCapability >= 100  ? 0.10f
                           : profile.gpuComputeCapability >= 90 ? 0.06f
                                                                : 0.0f;
    const float swizzle_bonus = cfg.smemSwizzle >= 64 ? 0.05f : 0.02f;
    const float stage_bonus = 0.03f * static_cast<float>(std::min(cfg.numStages, 4));
    const float sparsity_penalty = std::clamp(profile.sparsityEstimate, 0.0f, 1.0f) * 0.08f;
    const float util = 0.58f + cc_bonus + swizzle_bonus + stage_bonus - sparsity_penalty;
    return std::clamp(util, 0.10f, 0.98f);
}

[[nodiscard]] float estimateRuntimeMs(const TmaTileConfig &cfg,
                                      const TmaTileTuner::WorkloadProfile &profile, int numTrials)
{
    const double n = static_cast<double>(profile.nPoints);
    const double dim = static_cast<double>(profile.pointDim);
    const double work = std::max(1.0, n * n * dim);
    const float occ = estimateOccupancy(cfg, profile);
    const float bw = estimateBandwidthUtilization(cfg, profile);
    const float tile_eff = std::clamp(static_cast<float>(cfg.pointTileSize) / 64.0f, 0.5f, 4.0f);
    const float stage_eff = 1.0f + 0.04f * static_cast<float>(cfg.numStages - 2);
    const float cluster_eff = 1.0f + 0.02f * static_cast<float>(cfg.clusterSizeX - 1);
    const float trial_smoothing = 1.0f + 0.03f / static_cast<float>(std::max(1, numTrials));
    const double perf = std::max(0.02f, occ * bw * tile_eff * stage_eff * cluster_eff);
    return static_cast<float>((work / 1.0e8) / perf) * trial_smoothing;
}

std::mutex g_cache_mutex;
std::unordered_map<std::string, TmaTileConfig> g_cache;

[[nodiscard]] TmaTileConfig defaultConfig(const TmaTileTuner::WorkloadProfile &profile)
{
    TmaTileConfig cfg{};
    cfg.pointTileSize = profile.gpuComputeCapability >= 90 ? 64 : 32;
    cfg.dimTileSize = clampInt(static_cast<int>(profile.pointDim), 8, 64);
    cfg.clusterSizeX = profile.gpuComputeCapability >= 90 ? 4 : 1;
    cfg.clusterSizeY = 1;
    cfg.smemSwizzle = profile.gpuComputeCapability >= 90 ? 64 : 32;
    cfg.l2Promotion = profile.gpuComputeCapability >= 90 ? 128 : 64;
    cfg.numStages = profile.gpuComputeCapability >= 90 ? 3 : 2;
    while (estimateSharedMemBytes(cfg) > profile.availableSmem && cfg.pointTileSize > kMinTile)
    {
        cfg.pointTileSize /= 2;
    }
    return cfg;
}

} // namespace

TmaTileConfig TmaTileTuner::tune(const WorkloadProfile &profile, int numTrials)
{
    const int trials = std::max(1, numTrials > 0 ? numTrials : kDefaultTrials);
    const auto point_tiles = tileSearchSpace(profile.gpuComputeCapability);
    const auto clusters = clusterSearchSpace(profile.gpuComputeCapability);
    const std::vector<int> stages = {2, 3, 4};
    const std::vector<int> swizzles = {32, 64};

    bool found = false;
    float best_time = std::numeric_limits<float>::max();
    float best_bw = 0.0f;
    TmaTileConfig best_cfg = defaultConfig(profile);

    for (int tile : point_tiles)
    {
        for (int cluster : clusters)
        {
            for (int stage : stages)
            {
                for (int swizzle : swizzles)
                {
                    TmaTileConfig cfg{};
                    cfg.pointTileSize = tile;
                    cfg.dimTileSize = clampInt(static_cast<int>(profile.pointDim), 8, 64);
                    cfg.clusterSizeX = cluster;
                    cfg.clusterSizeY = 1;
                    cfg.smemSwizzle = swizzle;
                    cfg.l2Promotion = (profile.gpuComputeCapability >= 90) ? 128 : 64;
                    cfg.numStages = stage;
                    if (!isConfigValid(cfg, profile))
                    {
                        continue;
                    }

                    const float time_ms = estimateRuntimeMs(cfg, profile, trials);
                    const float bw = estimateBandwidthUtilization(cfg, profile);
                    if (!found || time_ms < best_time ||
                        (std::abs(time_ms - best_time) < 0.05f && bw > best_bw))
                    {
                        found = true;
                        best_time = time_ms;
                        best_bw = bw;
                        best_cfg = cfg;
                    }
                }
            }
        }
    }

    return found ? best_cfg : defaultConfig(profile);
}

TmaTileConfig TmaTileTuner::getOrTune(const std::string &cacheKey, const WorkloadProfile &profile)
{
    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        const auto it = g_cache.find(cacheKey);
        if (it != g_cache.end())
        {
            return it->second;
        }
    }

    TmaTileConfig tuned = tune(profile, kDefaultTrials);
    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        g_cache[cacheKey] = tuned;
    }
    return tuned;
}

} // namespace nerve::gpu::advanced

#include "nerve/gpu/hybrid_tuning.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
namespace nerve::gpu::tuning
{
namespace
{

[[nodiscard]] constexpr float clamp01(float x) noexcept
{
    return x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x);
}

[[nodiscard]] constexpr int clamp_int(int value, int lo, int hi) noexcept
{
    return value < lo ? lo : (value > hi ? hi : value);
}

} // namespace

HybridTuner::HybridTuner() = default;

void HybridTuner::setBenchmarkFunction(
    std::function<float(const TunedConfig &, const WorkloadCharacteristics &)> func)
{
    benchmarkFunc_ = std::move(func);
    useExternalBenchmark_ = true;
}

HybridTuner::Prediction HybridTuner::predict(const GpuHardwareSpecs &specs,
                                             const WorkloadCharacteristics &workload) const
{
    Prediction pred{};

    if (specs.computeCapability >= 100)
    {
        pred.config = tuneForBlackwell(specs, workload);
    }
    else if (specs.computeCapability >= 90)
    {
        pred.config = tuneForHopper(specs, workload);
    }
    else
    {
        pred.config.blockSize = selectOptimalBlockSize(specs, workload);
        pred.config.tileSize = selectOptimalTileSize(specs, workload);
        pred.config.clusterSize = selectOptimalClusterSize(specs, workload);
        pred.config.numStages = selectOptimalNumStages(specs, workload.isMemoryBound);
        pred.config.useWGMMA = specs.hasTensorCores && specs.computeCapability >= 90;
        pred.config.useTMA = specs.hasTMA;
        pred.config.usePTXOpts = specs.computeCapability >= 89;
        pred.config.useFP8 = specs.computeCapability >= 89 && workload.pointDim >= 32;
        pred.config.useFP4 = specs.hasFP4 && workload.sparsityEstimate > 0.85f;
        pred.config.useCluster = specs.computeCapability >= 90 && workload.nPoints >= 4096;
        pred.config.useTMAMulticast = pred.config.useCluster && workload.nPoints >= 16384;
    }

    const float memory_ms = predictMemoryTime(pred.config, specs, workload);
    const float compute_ms = predictComputeTime(pred.config, specs, workload);
    const float overhead_ms = predictOverheadTime(pred.config, specs);

    pred.performance.predictedTimeMs = std::max(memory_ms, compute_ms) + overhead_ms;
    pred.performance.estimatedOccupancy = calculateOccupancy(pred.config, specs);
    pred.performance.estimatedBandwidthUtilization =
        calculateBandwidthUtilization(pred.config, specs, workload);
    pred.performance.estimatedComputeUtilization =
        clamp01(0.35f + 0.45f * pred.performance.estimatedOccupancy +
                (pred.config.useWGMMA ? 0.10f : 0.0f) + (pred.config.useTMA ? 0.05f : 0.0f));

    const float spread =
        std::abs(memory_ms - compute_ms) / std::max(1e-4f, std::max(memory_ms, compute_ms));
    pred.performance.confidence =
        clamp01(0.55f + 0.25f * pred.performance.estimatedOccupancy +
                0.10f * pred.performance.estimatedBandwidthUtilization - 0.12f * spread);

    std::ostringstream reasoning;
    reasoning << "model=max(memory,compute)+overhead; " << "occ=" << std::fixed
              << std::setprecision(2) << pred.performance.estimatedOccupancy
              << ", bw=" << pred.performance.estimatedBandwidthUtilization;
    pred.performance.reasoning = reasoning.str();

    pred.config.modelConfidence = pred.performance.confidence;
    pred.config.measuredTime = pred.performance.predictedTimeMs;
    return pred;
}

TunedConfig HybridTuner::verify(const Prediction &prediction, const GpuHardwareSpecs &specs,
                                const WorkloadCharacteristics &workload, int numTrials)
{
    std::vector<TunedConfig> candidates;
    candidates.push_back(prediction.config);

    TunedConfig tile_down = prediction.config;
    tile_down.tileSize = clamp_int(prediction.config.tileSize / 2, 16, 256);
    candidates.push_back(tile_down);

    TunedConfig tile_up = prediction.config;
    tile_up.tileSize = clamp_int(prediction.config.tileSize * 2, 16, 256);
    candidates.push_back(tile_up);

    TunedConfig block_down = prediction.config;
    block_down.blockSize = clamp_int(prediction.config.blockSize / 2, 64, 1024);
    candidates.push_back(block_down);

    TunedConfig block_up = prediction.config;
    block_up.blockSize = clamp_int(prediction.config.blockSize * 2, 64, 1024);
    candidates.push_back(block_up);

    TunedConfig architecture_variant = prediction.config;
    architecture_variant.useTMA = specs.hasTMA;
    architecture_variant.useWGMMA = specs.hasTensorCores && specs.computeCapability >= 90;
    architecture_variant.useCluster = specs.computeCapability >= 90 && workload.nPoints >= 4096;
    architecture_variant.clusterSize = selectOptimalClusterSize(specs, workload);
    candidates.push_back(architecture_variant);

    TunedConfig best = prediction.config;
    float best_time = std::numeric_limits<float>::max();
    for (auto &candidate : candidates)
    {
        const float measured = runMicroBenchmark(candidate, workload, std::max(1, numTrials));
        candidate.measuredTime = measured;
        if (measured < best_time)
        {
            best_time = measured;
            best = candidate;
        }
    }

    best.modelConfidence = prediction.performance.confidence;
    return best;
}

TunedConfig HybridTuner::tune(const GpuHardwareSpecs &specs,
                              const WorkloadCharacteristics &workload, int numTrials,
                              float confidenceThreshold)
{
    const Prediction pred = predict(specs, workload);
    if (!useExternalBenchmark_ && pred.performance.confidence >= confidenceThreshold)
    {
        return pred.config;
    }
    return verify(pred, specs, workload, numTrials);
}

float HybridTuner::predictMemoryTime(const TunedConfig &config, const GpuHardwareSpecs &specs,
                                     const WorkloadCharacteristics &workload) const
{
    const double n = static_cast<double>(workload.nPoints);
    const double bytes =
        std::max<double>(static_cast<double>(workload.workingSetSize),
                         n * static_cast<double>(workload.pointDim) * sizeof(float) * 2.0);
    const double bw = std::max(1.0, static_cast<double>(specs.memoryBandwidthGBps) * 1e9);
    const double util =
        std::max(0.10, static_cast<double>(calculateBandwidthUtilization(config, specs, workload)));
    return static_cast<float>((bytes / (bw * util)) * 1e3);
}

float HybridTuner::predictComputeTime(const TunedConfig &config, const GpuHardwareSpecs &specs,
                                      const WorkloadCharacteristics &workload) const
{
    const double n = static_cast<double>(workload.nPoints);
    const double flops = std::max(1.0, n * n * static_cast<double>(workload.pointDim) * 2.0);
    const double peak = std::max(1e6, static_cast<double>(specs.peakTFlops) * 1e12);
    const double occ = std::max(0.10, static_cast<double>(calculateOccupancy(config, specs)));
    const double feature_gain = (config.useWGMMA ? 1.20 : 1.0) * (config.useTMA ? 1.05 : 1.0);
    return static_cast<float>((flops / (peak * occ * feature_gain)) * 1e3);
}

float HybridTuner::predictOverheadTime(const TunedConfig &config,
                                       const GpuHardwareSpecs &specs) const
{
    float overhead = (specs.computeCapability >= 90 ? 0.04f : 0.06f) +
                     0.004f * static_cast<float>(config.numStages);
    if (config.useCluster)
    {
        overhead += specs.computeCapability >= 90 ? 0.03f : 0.06f;
    }
    if (config.useTMAMulticast)
    {
        overhead += 0.01f;
    }
    return overhead;
}

float HybridTuner::calculateOccupancy(const TunedConfig &config,
                                      const GpuHardwareSpecs &specs) const
{
    const int threads = clamp_int(config.blockSize, 32, std::max(32, specs.maxThreadsPerSm));
    const int warps_per_block = std::max(1, threads / 32);
    const int shared_per_block = std::max(1, config.tileSize * config.tileSize * 4);
    const int blocks_by_threads = std::max(1, specs.maxThreadsPerSm / threads);
    const int blocks_by_smem = std::max(1, specs.sharedMemPerSm / shared_per_block);
    const int active_blocks = std::max(1, std::min(blocks_by_threads, blocks_by_smem));
    const int max_warps = std::max(1, specs.maxThreadsPerSm / 32);
    return clamp01(static_cast<float>(active_blocks * warps_per_block) /
                   static_cast<float>(max_warps));
}

float HybridTuner::calculateBandwidthUtilization(const TunedConfig &config,
                                                 const GpuHardwareSpecs &specs,
                                                 const WorkloadCharacteristics &workload) const
{
    float util = workload.isMemoryBound ? 0.70f : 0.50f;
    if (specs.memoryBandwidthGBps >= 1500)
    {
        util += 0.05f;
    }
    if (config.useTMA)
    {
        util += 0.12f;
    }
    if (config.useTMAMulticast)
    {
        util += 0.05f;
    }
    if (workload.sparsityEstimate > 0.85f)
    {
        util -= 0.07f;
    }
    return clamp01(util);
}

int HybridTuner::selectOptimalTileSize(const GpuHardwareSpecs &specs,
                                       const WorkloadCharacteristics &workload) const
{
    int tile = specs.hasTMA ? 64 : 32;
    if (workload.pointDim <= 8 && specs.computeCapability >= 100)
    {
        tile = 128;
    }
    else if (workload.pointDim <= 16)
    {
        tile = 64;
    }
    else if (workload.pointDim > 128)
    {
        tile = 16;
    }
    return clamp_int(tile, 16, 256);
}

int HybridTuner::selectOptimalBlockSize(const GpuHardwareSpecs &specs,
                                        const WorkloadCharacteristics &workload) const
{
    int block = 256;
    if (workload.isMemoryBound)
    {
        block = 128;
    }
    if (specs.computeCapability >= 100 && !workload.isMemoryBound)
    {
        block = 512;
    }
    return clamp_int(block, 64, 1024);
}

int HybridTuner::selectOptimalClusterSize(const GpuHardwareSpecs &specs,
                                          const WorkloadCharacteristics &workload) const
{
    if (!specs.hasTMA)
    {
        return 1;
    }
    if (workload.nPoints >= 50000)
    {
        return specs.computeCapability >= 100 ? 8 : 4;
    }
    if (workload.nPoints >= 10000)
    {
        return 4;
    }
    return 1;
}

int HybridTuner::selectOptimalNumStages(const GpuHardwareSpecs &specs, bool isMemoryBound) const
{
    int stages = isMemoryBound ? 3 : 2;
    if (specs.computeCapability >= 100)
    {
        stages += 1;
    }
    if (!specs.hasAsyncCopy)
    {
        stages = 1;
    }
    return clamp_int(stages, 1, 6);
}

TunedConfig HybridTuner::tuneForHopper(const GpuHardwareSpecs &specs,
                                       const WorkloadCharacteristics &workload) const
{
    TunedConfig config{};
    config.blockSize = selectOptimalBlockSize(specs, workload);
    config.tileSize = selectOptimalTileSize(specs, workload);
    config.clusterSize = selectOptimalClusterSize(specs, workload);
    config.numStages = selectOptimalNumStages(specs, workload.isMemoryBound);
    config.useWGMMA = specs.hasWGMMA;
    config.useTMA = specs.hasTMA;
    config.usePTXOpts = true;
    config.useFP8 = workload.pointDim >= 16;
    config.useCluster = config.clusterSize > 1;
    config.useTMAMulticast = config.useCluster && workload.nPoints >= 12000;
    config.l2PromotionSize = workload.isMemoryBound ? 256 : 128;
    config.smemSwizzle = 32;
    return config;
}

TunedConfig HybridTuner::tuneForBlackwell(const GpuHardwareSpecs &specs,
                                          const WorkloadCharacteristics &workload) const
{
    TunedConfig config = tuneForHopper(specs, workload);
    config.tileSize = std::max(config.tileSize, 64);
    config.numStages = clamp_int(config.numStages + 1, 2, 6);
    config.useFP8 = true;
    config.useFP4 = specs.hasFP4 && workload.sparsityEstimate > 0.80f;
    config.useNonPortableCluster = config.clusterSize >= 8;
    config.l2PromotionSize = 256;
    config.smemSwizzle = 64;
    return config;
}

float HybridTuner::runMicroBenchmark(const TunedConfig &config,
                                     const WorkloadCharacteristics &workload, int numTrials)
{
    if (useExternalBenchmark_ && benchmarkFunc_)
    {
        float total = 0.0f;
        for (int i = 0; i < std::max(1, numTrials); ++i)
        {
            total += benchmarkFunc_(config, workload);
        }
        return total / static_cast<float>(std::max(1, numTrials));
    }

    const double n = static_cast<double>(workload.nPoints);
    const double work = std::max(1.0, n * static_cast<double>(workload.pointDim));
    const float block_factor = 256.0f / static_cast<float>(std::max(64, config.blockSize));
    const float tile_factor = 64.0f / static_cast<float>(std::max(16, config.tileSize));
    const float feature_factor = (config.useWGMMA ? 0.86f : 1.0f) * (config.useTMA ? 0.92f : 1.0f);
    const float memory_factor = workload.isMemoryBound ? 1.10f : 0.95f;
    const float trial_smoothing = 1.0f + 0.03f / static_cast<float>(std::max(1, numTrials));
    return std::max(0.01f, static_cast<float>(work / 1e6) * block_factor * tile_factor *
                               feature_factor * memory_factor * trial_smoothing);
}

} // namespace nerve::gpu::tuning

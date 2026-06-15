
#pragma once

#include "nerve/gpu/tuning_cache.hpp"

#include <cuda_runtime.h>

#include <functional>
#include <vector>

namespace nerve::gpu::tuning
{

struct WorkloadCharacteristics
{
    uint32_t nPoints;
    uint32_t pointDim;
    float sparsityEstimate;
    bool isMemoryBound;
    float arithmeticIntensity;
    size_t workingSetSize;
};

struct GpuHardwareSpecs
{
    int computeCapability;
    int smCount;
    int clockRate;
    int maxThreadsPerSm;
    int sharedMemPerSm;
    float memoryBandwidthGBps;
    float peakTFlops;
    int warpSchedulerCount;
    bool hasTensorCores;
    bool hasAsyncCopy;
    bool hasTMA;
    bool hasWGMMA;
    bool hasFP4;
};

struct PerformancePrediction
{
    float predictedTimeMs;
    float confidence;
    float estimatedOccupancy;
    float estimatedBandwidthUtilization;
    float estimatedComputeUtilization;
    std::string reasoning;
};

class HybridTuner
{
public:
    HybridTuner();

    /// Model-based prediction of optimal configuration
    struct Prediction
    {
        TunedConfig config;
        PerformancePrediction performance;
    };

    Prediction predict(const GpuHardwareSpecs &specs,
                       const WorkloadCharacteristics &workload) const;

    /// Verify prediction with micro-benchmarks
    TunedConfig verify(const Prediction &prediction, const GpuHardwareSpecs &specs,
                       const WorkloadCharacteristics &workload, int numTrials = 3);

    /// Main tuning entry point - hybrid approach
    TunedConfig tune(const GpuHardwareSpecs &specs, const WorkloadCharacteristics &workload,
                     int numTrials = 3, float confidenceThreshold = 0.95f);

    /// Set custom benchmark function (for testing)
    void setBenchmarkFunction(
        std::function<float(const TunedConfig &, const WorkloadCharacteristics &)> func);

private:
    // Analytical models
    float predictMemoryTime(const TunedConfig &config, const GpuHardwareSpecs &specs,
                            const WorkloadCharacteristics &workload) const;

    float predictComputeTime(const TunedConfig &config, const GpuHardwareSpecs &specs,
                             const WorkloadCharacteristics &workload) const;

    float predictOverheadTime(const TunedConfig &config, const GpuHardwareSpecs &specs) const;

    float calculateOccupancy(const TunedConfig &config, const GpuHardwareSpecs &specs) const;

    float calculateBandwidthUtilization(const TunedConfig &config, const GpuHardwareSpecs &specs,
                                        const WorkloadCharacteristics &workload) const;

    // Optimal parameter selection
    int selectOptimalTileSize(const GpuHardwareSpecs &specs,
                              const WorkloadCharacteristics &workload) const;

    int selectOptimalBlockSize(const GpuHardwareSpecs &specs,
                               const WorkloadCharacteristics &workload) const;

    int selectOptimalClusterSize(const GpuHardwareSpecs &specs,
                                 const WorkloadCharacteristics &workload) const;

    int selectOptimalNumStages(const GpuHardwareSpecs &specs, bool isMemoryBound) const;

    // Architecture-specific tuning
    TunedConfig tuneForHopper(const GpuHardwareSpecs &specs,
                              const WorkloadCharacteristics &workload) const;

    TunedConfig tuneForBlackwell(const GpuHardwareSpecs &specs,
                                 const WorkloadCharacteristics &workload) const;

    // Micro-benchmarking
    float runMicroBenchmark(const TunedConfig &config, const WorkloadCharacteristics &workload,
                            int numTrials);

    std::function<float(const TunedConfig &, const WorkloadCharacteristics &)> benchmarkFunc_;
    bool useExternalBenchmark_ = false;
};

GpuHardwareSpecs detectHardwareSpecs(int deviceId = 0);

WorkloadCharacteristics analyzeWorkload(uint32_t nPoints, uint32_t pointDim,
                                        float sparsityEstimate = 0.5f);

struct TuningReport
{
    std::string gpuName;
    std::string architecture;
    WorkloadCharacteristics workload;
    TunedConfig finalConfig;
    float totalTuningTimeMs;
    int numPredictions;
    int numBenchmarks;
    float bestPredictedTime;
    float bestMeasuredTime;
    float speedupVsBaseline;
};

class TuningReporter
{
public:
    static void report(const TuningReport &report);
    static void reportJson(const TuningReport &report, const std::string &path);
};

} // namespace nerve::gpu::tuning

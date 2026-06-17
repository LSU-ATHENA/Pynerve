
#include "nerve/gpu/hybrid_tuning.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#ifdef NERVE_HAS_CUDA

namespace
{

using nerve::gpu::tuning::analyzeWorkload;
using nerve::gpu::tuning::detectHardwareSpecs;
using nerve::gpu::tuning::GpuHardwareSpecs;
using nerve::gpu::tuning::HybridTuner;
using nerve::gpu::tuning::TunedConfig;
using nerve::gpu::tuning::WorkloadCharacteristics;

GpuHardwareSpecs make_test_specs()
{
    GpuHardwareSpecs specs;
    specs.computeCapability = 90;
    specs.smCount = 132;
    specs.clockRate = 1980;
    specs.maxThreadsPerSm = 2048;
    specs.sharedMemPerSm = 128 * 1024;
    specs.memoryBandwidthGBps = 2000.0f;
    specs.peakTFlops = 60.0f;
    specs.warpSchedulerCount = 4;
    specs.hasTensorCores = true;
    specs.hasAsyncCopy = true;
    specs.hasTMA = true;
    specs.hasWGMMA = true;
    specs.hasFP4 = false;
    return specs;
}

WorkloadCharacteristics make_test_workload()
{
    WorkloadCharacteristics wl;
    wl.nPoints = 10000;
    wl.pointDim = 32;
    wl.sparsityEstimate = 0.5f;
    wl.isMemoryBound = true;
    wl.arithmeticIntensity = 0.5f;
    wl.workingSetSize = 10000ULL * 32ULL * sizeof(float) * 2;
    return wl;
}

bool check_hybrid_tuner_construction()
{
    HybridTuner tuner;
    return true;
}

bool check_hybrid_tuner_predict()
{
    HybridTuner tuner;
    auto specs = make_test_specs();
    auto wl = make_test_workload();
    auto pred = tuner.predict(specs, wl);
    if (pred.config.blockSize <= 0)
        return false;
    if (pred.config.tileSize <= 0)
        return false;
    if (pred.performance.predictedTimeMs <= 0.0f)
        return false;
    if (pred.performance.confidence <= 0.0f || pred.performance.confidence > 1.0f)
        return false;
    if (pred.performance.estimatedOccupancy <= 0.0f || pred.performance.estimatedOccupancy > 1.0f)
        return false;
    if (pred.performance.estimatedBandwidthUtilization <= 0.0f ||
        pred.performance.estimatedBandwidthUtilization > 1.0f)
        return false;
    return true;
}

bool check_hybrid_tuner_tune()
{
    HybridTuner tuner;
    auto specs = make_test_specs();
    auto wl = make_test_workload();
    auto config = tuner.tune(specs, wl, 1, 0.95f);
    if (config.blockSize <= 0)
        return false;
    if (config.tileSize <= 0)
        return false;
    return true;
}

bool check_hybrid_tuner_verify()
{
    HybridTuner tuner;
    auto specs = make_test_specs();
    auto wl = make_test_workload();
    auto pred = tuner.predict(specs, wl);
    auto verified = tuner.verify(pred, specs, wl, 1);
    if (verified.blockSize <= 0)
        return false;
    if (verified.tileSize <= 0)
        return false;
    return true;
}

bool check_hybrid_tuner_set_benchmark()
{
    HybridTuner tuner;
    tuner.setBenchmarkFunction(
        [](const TunedConfig &, const WorkloadCharacteristics &) -> float { return 1.0f; });
    auto specs = make_test_specs();
    auto wl = make_test_workload();
    auto config = tuner.tune(specs, wl);
    if (config.blockSize <= 0)
        return false;
    return true;
}

bool check_analyze_workload()
{
    auto wl = analyzeWorkload(1000, 3, 0.3f);
    if (wl.nPoints != 1000)
        return false;
    if (wl.pointDim != 3)
        return false;
    if (std::abs(wl.sparsityEstimate - 0.3f) > 1e-6f)
        return false;
    return true;
}

bool check_detect_hardware_specs()
{
    auto specs = detectHardwareSpecs(0);
    static_cast<void>(specs);
    return true;
}

bool check_tune_for_hopper()
{
    HybridTuner tuner;
    auto specs = make_test_specs();
    auto wl = make_test_workload();
    auto config = tuner.tune(specs, wl);
    if (config.useWGMMA != specs.hasWGMMA)
        return false;
    if (config.useTMA != specs.hasTMA)
        return false;
    return true;
}

bool check_tune_for_blackwell()
{
    HybridTuner tuner;
    auto specs = make_test_specs();
    specs.computeCapability = 100;
    specs.hasFP4 = false;
    auto wl = make_test_workload();
    auto config = tuner.tune(specs, wl);
    if (config.blockSize <= 0)
        return false;
    if (config.tileSize < 64)
        return false;
    return true;
}

bool check_tune_with_low_confidence()
{
    HybridTuner tuner;
    auto specs = make_test_specs();
    auto wl = make_test_workload();
    tuner.setBenchmarkFunction(
        [](const TunedConfig &, const WorkloadCharacteristics &) -> float { return 0.5f; });
    auto config = tuner.tune(specs, wl, 3, 0.99f);
    if (config.blockSize <= 0)
        return false;
    if (config.measuredTime <= 0.0f)
        return false;
    return true;
}

} // namespace

int main()
{
    if (!check_hybrid_tuner_construction())
    {
        std::cerr << "FAIL: hybrid tuner construction\n";
        return 1;
    }
    if (!check_hybrid_tuner_predict())
    {
        std::cerr << "FAIL: hybrid tuner predict\n";
        return 1;
    }
    if (!check_hybrid_tuner_tune())
    {
        std::cerr << "FAIL: hybrid tuner tune\n";
        return 1;
    }
    if (!check_hybrid_tuner_verify())
    {
        std::cerr << "FAIL: hybrid tuner verify\n";
        return 1;
    }
    if (!check_hybrid_tuner_set_benchmark())
    {
        std::cerr << "FAIL: hybrid tuner set benchmark\n";
        return 1;
    }
    if (!check_analyze_workload())
    {
        std::cerr << "FAIL: analyze workload\n";
        return 1;
    }
    if (!check_detect_hardware_specs())
    {
        std::cerr << "FAIL: detect hardware specs\n";
        return 1;
    }
    if (!check_tune_for_hopper())
    {
        std::cerr << "FAIL: tune for hopper\n";
        return 1;
    }
    if (!check_tune_for_blackwell())
    {
        std::cerr << "FAIL: tune for blackwell\n";
        return 1;
    }
    if (!check_tune_with_low_confidence())
    {
        std::cerr << "FAIL: tune with low confidence\n";
        return 1;
    }
    return 0;
}

#else
int main() { return 0; }
#endif
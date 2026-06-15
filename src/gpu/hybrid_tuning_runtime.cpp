#include "nerve/gpu/hybrid_tuning.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string_view>

namespace nerve::gpu::tuning
{
namespace
{

[[nodiscard]] int estimateFp32CoresPerSm(int compute_capability)
{
    if (compute_capability >= 89)
    {
        return 128;
    }
    if (compute_capability >= 70)
    {
        return 64;
    }
    return 32;
}

std::string escapeJson(std::string_view value)
{
    std::string escaped;
    escaped.reserve(value.size());
    for (char ch : value)
    {
        switch (ch)
        {
            case '\\':
            case '"':
                escaped.push_back('\\');
                escaped.push_back(ch);
                break;
            case '\n':
                escaped += "\\n";
                break;
            case '\r':
                escaped += "\\r";
                break;
            case '\t':
                escaped += "\\t";
                break;
            default:
                escaped.push_back(ch);
                break;
        }
    }
    return escaped;
}

} // namespace

GpuHardwareSpecs detectHardwareSpecs(int deviceId)
{
    cudaDeviceProp prop{};
    GpuHardwareSpecs specs{};
    if (cudaGetDeviceProperties(&prop, deviceId) != cudaSuccess)
    {
        return specs;
    }
    specs.computeCapability = prop.major * 10 + prop.minor;
    specs.smCount = prop.multiProcessorCount;
#if CUDART_VERSION >= 13000
    {
        int clock_khz = 0;
        cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, deviceId);
        specs.clockRate = clock_khz;
    }
#else
    specs.clockRate = prop.clockRate;
#endif
    specs.maxThreadsPerSm = std::max(1, prop.maxThreadsPerBlock);
    specs.sharedMemPerSm = static_cast<size_t>(0);
    specs.memoryBandwidthGBps = (specs.computeCapability >= 100)  ? 3500.0f
                                : (specs.computeCapability >= 90) ? 2500.0f
                                : (specs.computeCapability >= 80) ? 1550.0f
                                : (specs.computeCapability >= 70) ? 900.0f
                                                                  : 350.0f;
    const int fp32_cores_per_sm = estimateFp32CoresPerSm(specs.computeCapability);
    const double core_clock_ghz = std::clamp(
        1.2 + 0.02 * static_cast<double>(std::max(0, specs.computeCapability - 70)), 1.0, 2.5);
    const double peak_tflops = 2.0 * static_cast<double>(fp32_cores_per_sm) *
                               static_cast<double>(specs.smCount) * core_clock_ghz / 1000.0;
    specs.peakTFlops = std::max(0.1f, static_cast<float>(peak_tflops));
    specs.warpSchedulerCount = 4;
    specs.hasTensorCores = specs.computeCapability >= 70;
    specs.hasAsyncCopy = specs.computeCapability >= 80;
    specs.hasTMA = specs.computeCapability >= 90;
    specs.hasWGMMA = specs.computeCapability >= 90;
    specs.hasFP4 = specs.computeCapability >= 100;
    return specs;
}

WorkloadCharacteristics analyzeWorkload(uint32_t nPoints, uint32_t pointDim, float sparsityEstimate)
{
    WorkloadCharacteristics workload{};
    workload.nPoints = nPoints;
    workload.pointDim = pointDim;
    workload.sparsityEstimate = std::clamp(sparsityEstimate, 0.0f, 1.0f);

    const double dense_flops = static_cast<double>(nPoints) * static_cast<double>(nPoints) *
                               static_cast<double>(pointDim) * 2.0;
    const double payload_bytes =
        static_cast<double>(nPoints) * static_cast<double>(pointDim) * sizeof(float);
    const double distance_bytes = static_cast<double>(nPoints) * static_cast<double>(nPoints) *
                                  sizeof(float) * (1.0 - 0.5 * workload.sparsityEstimate);

    workload.workingSetSize = static_cast<size_t>(payload_bytes + distance_bytes);
    workload.arithmeticIntensity =
        static_cast<float>(dense_flops / std::max(1.0, payload_bytes + distance_bytes));
    workload.isMemoryBound = workload.arithmeticIntensity < 16.0f;
    return workload;
}

void TuningReporter::report(const TuningReport &report)
{
    std::cout << "Nerve GPU Tuning Report\n";
    std::cout << "  GPU: " << report.gpuName << " (" << report.architecture << ")\n";
    std::cout << "  Workload: n=" << report.workload.nPoints << ", dim=" << report.workload.pointDim
              << ", sparsity=" << report.workload.sparsityEstimate << "\n";
    std::cout << "  Final config: block=" << report.finalConfig.blockSize
              << ", tile=" << report.finalConfig.tileSize
              << ", cluster=" << report.finalConfig.clusterSize
              << ", stages=" << report.finalConfig.numStages << "\n";
    std::cout << "  Timing: predicted=" << report.bestPredictedTime
              << "ms, measured=" << report.bestMeasuredTime
              << "ms, speedup=" << report.speedupVsBaseline << "x\n";
}

void TuningReporter::reportJson(const TuningReport &report, const std::string &path)
{
    std::ofstream out(path);
    if (!out)
    {
        return;
    }

    out << "{\n";
    out << "  \"gpu_name\": \"" << escapeJson(report.gpuName) << "\",\n";
    out << "  \"architecture\": \"" << escapeJson(report.architecture) << "\",\n";
    out << "  \"workload\": {\n";
    out << "    \"n_points\": " << report.workload.nPoints << ",\n";
    out << "    \"point_dim\": " << report.workload.pointDim << ",\n";
    out << "    \"sparsity_estimate\": " << report.workload.sparsityEstimate << "\n";
    out << "  },\n";
    out << "  \"final_config\": {\n";
    out << "    \"block_size\": " << report.finalConfig.blockSize << ",\n";
    out << "    \"tile_size\": " << report.finalConfig.tileSize << ",\n";
    out << "    \"cluster_size\": " << report.finalConfig.clusterSize << ",\n";
    out << "    \"num_stages\": " << report.finalConfig.numStages << "\n";
    out << "  },\n";
    out << "  \"best_predicted_time_ms\": " << report.bestPredictedTime << ",\n";
    out << "  \"best_measured_time_ms\": " << report.bestMeasuredTime << ",\n";
    out << "  \"speedup_vs_baseline\": " << report.speedupVsBaseline << "\n";
    out << "}\n";
}

} // namespace nerve::gpu::tuning


#pragma once

#include "nerve/common/accelerated_types.hpp"
#include "nerve/core/buffer_view.hpp"
#include "nerve/persistence/accelerated/accelerated_interface.hpp"

#include <mutex>
#include <vector>

namespace nerve::persistence::accelerated::detail
{

constexpr std::size_t kMaxStoredMetrics = 256;
constexpr std::size_t kMinGpuChunkElements = 1024;

double bytesToMb(std::size_t bytes);
double estimateProblemOps(std::size_t n_points, std::size_t point_dim, std::size_t max_dim);
::nerve::common::VRConfig toBaseFastVrConfig(const ::nerve::common::VRConfig &config);
SystemCapabilities detectSystemCapabilitiesImpl();

struct GlobalPerformanceState
{
    std::mutex mutex;
    AcceleratedPerformanceStats stats;
    double cpu_ops_per_ms = 0.0;
    double gpu_bytes_per_ms = 0.0;
    std::size_t throughput_samples = 0;
};

inline GlobalPerformanceState &globalPerformanceState()
{
    static GlobalPerformanceState state;
    return state;
}

struct StagePointsResult
{
    std::vector<double> staged_points;
    double gpu_time_ms = 0.0;
    std::size_t bytes_moved = 0;
    bool graph_phase_executed = false;
    double graph_phase_ops = 0.0;
    double graph_phase_edge_density = 0.0;
    std::size_t graph_phase_pairs_evaluated = 0;
    std::size_t graph_phase_sampled_points = 0;
    KernelDiagnosticsCounters kernel_diagnostics;
};

errors::ErrorResult<StagePointsResult> stagePointsOnGpu(core::BufferView<const double> points,
                                                        std::size_t point_dim, double max_radius,
                                                        std::size_t chunk_elements);

void recordGlobalMetric(const PerformanceMetrics &metric, double memory_usage_mb, bool gpu_used,
                        bool hybrid_used);

errors::ErrorResult<std::unique_ptr<GPUAccelerationManager>>
makeDefaultGpuManager(const VRConfig &config);
errors::ErrorResult<std::unique_ptr<PerformanceOptimizer>>
makeDefaultPerformanceOptimizer(const VRConfig &config);
errors::ErrorResult<std::unique_ptr<AcceleratedVREngine>>
makeDefaultAcceleratedEngine(const VRConfig &config);

} // namespace nerve::persistence::accelerated::detail

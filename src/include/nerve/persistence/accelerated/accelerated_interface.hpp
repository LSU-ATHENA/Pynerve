
#pragma once

#include "nerve/common/accelerated_types.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core_types.hpp"
#include "nerve/errors/errors.hpp"
#include "nerve/persistence/core/core_types.hpp"
#include "nerve/types.hpp"

#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace nerve::persistence::accelerated
{

// Import all common types via using declarations
using ::nerve::common::AcceleratedPerformanceStats;
using ::nerve::common::AccelerationConfig;
using ::nerve::common::AccelerationMode;
using ::nerve::common::AlgorithmType;
using ::nerve::common::ApproximationLevel;
using ::nerve::common::DebugConfig;
using ::nerve::common::DeviceInfo;
using ::nerve::common::ExecutionMode;
using ::nerve::common::KernelDiagnosticsCounters;
using ::nerve::common::MemoryConfig;
using ::nerve::common::PerformanceMetrics;
using ::nerve::common::ProblemCharacteristics;
using ::nerve::common::ProblemType;
using ::nerve::common::Strategy;
using ::nerve::common::SystemCapabilities;
using ::nerve::common::VRConfig;

// Classes remain defined in this namespace (they use common types)

class GPUAccelerationManager
{
public:
    static SystemCapabilities detectSystemCapabilities();
    static AccelerationMode recommendAccelerationMode(const ProblemCharacteristics &problem,
                                                      const SystemCapabilities &capabilities);
    static bool isGpuRuntimeAvailable();
    static bool is_gpu_runtime_available() { return isGpuRuntimeAvailable(); }
    static errors::ErrorResult<std::unique_ptr<GPUAccelerationManager>>
    create(const VRConfig &config);

    virtual ~GPUAccelerationManager() = default;
    virtual void updateConfig(const VRConfig &config) = 0;
    virtual bool isAvailable() const = 0;
    virtual DeviceInfo getGpuInfo() const = 0;
};

// Global performance state for tracking metrics across operations
struct GlobalPerformanceState
{
    mutable std::mutex mutex;
    AcceleratedPerformanceStats stats;
    double cpu_ops_per_ms = 0.0;
    double gpu_bytes_per_ms = 0.0;
    std::vector<double> throughput_samples;
    double total_time_ms = 0.0;
    double gpu_time_ms = 0.0;
    double cpu_time_ms = 0.0;
    size_t problems_processed = 0;
};

// Result of GPU transfer operations
struct GpuTransferResult
{
    bool success = false;
    size_t bytes_transferred = 0;
    double transfer_time_ms = 0.0;
};

// Global performance state instance
inline GlobalPerformanceState &globalPerformanceState()
{
    static GlobalPerformanceState instance;
    return instance;
}

// Maximum stored metrics constant
inline constexpr size_t kMaxStoredMetrics = 1000;

class PerformanceOptimizer
{
public:
    struct OptimizationParams
    {
        Size block_size = 256;
        Size grid_size = 0;
        Size shared_memory_size = 0;
        bool use_streaming = false;
        Size streaming_chunk_size = 0;
        double memory_efficiency_threshold = 0.85;
    };

    static errors::ErrorResult<std::unique_ptr<PerformanceOptimizer>>
    create(const VRConfig &config);

    virtual ~PerformanceOptimizer() = default;
    virtual void updateConfig(const VRConfig &config) = 0;
    virtual errors::ErrorResult<OptimizationParams>
    getOptimalParameters(size_t n_points, size_t point_dim, double max_radius) = 0;
    virtual errors::ErrorResult<void> optimizeForCurrentSystem() = 0;
    virtual errors::ErrorResult<void> autoTuneParameters() = 0;
};

class PerformanceMonitor
{
public:
    PerformanceMonitor() = default;
    ~PerformanceMonitor() = default;
    void startMonitoring(const std::string &operation);
    void endMonitoring(const std::string &operation);
    void recordMetrics(const std::string &operation, const PerformanceMetrics &metrics);
    AcceleratedPerformanceStats getAggregatedStats() const;
    void reset();

private:
    std::unordered_map<std::string, PerformanceMetrics> metrics_;
    std::unordered_map<std::string, std::chrono::steady_clock::time_point> start_times_;
};

class AcceleratedVREngine
{
public:
    static errors::ErrorResult<std::unique_ptr<AcceleratedVREngine>> create(const VRConfig &config);
    virtual ~AcceleratedVREngine() = default;
    virtual errors::ErrorResult<std::vector<Pair>>
    computeVrPersistence(core::BufferView<const double>points, Size point_dim,
                         const VRConfig &config) = 0;
    virtual AcceleratedPerformanceStats getPerformanceStats() const = 0;
    virtual void updateConfig(const VRConfig &config) = 0;
    virtual VRConfig getConfig() const = 0;
    virtual bool isAvailable() const = 0;
    virtual DeviceInfo getGpuInfo() const = 0;
    virtual errors::ErrorResult<void> optimizePerformance() = 0;
    virtual errors::ErrorResult<void> autoTuneParameters() = 0;
};

namespace factory
{
errors::ErrorResult<std::unique_ptr<AcceleratedVREngine>>
createAccelerationRuntimeEngine(size_t n_points = 0, size_t point_dim = 3, double max_radius = 1.0);
errors::ErrorResult<std::unique_ptr<AcceleratedVREngine>>
createProductionEngine(const VRConfig &config);
errors::ErrorResult<std::unique_ptr<AcceleratedVREngine>> createDebugEngine(const VRConfig &config);
errors::ErrorResult<std::unique_ptr<AcceleratedVREngine>>
createBenchmarkEngine(const VRConfig &config);
} // namespace factory

namespace utils
{
VRConfig createOptimalConfigForProblem(size_t n_points, size_t point_dim, double max_radius,
                                       const VRConfig &base_config);
SystemCapabilities detectSystemCapabilities();
AccelerationMode recommendAccelerationMode(const ProblemCharacteristics &problem,
                                           const SystemCapabilities &capabilities);
errors::ErrorResult<void> validateAccelerationConfig(const VRConfig &config);
AcceleratedPerformanceStats estimatePerformance(size_t n_points, size_t point_dim,
                                                double max_radius, const VRConfig &config);
bool isAccelerationBeneficial(size_t n_points, size_t point_dim, double max_radius);
size_t estimateMemoryRequirements(size_t n_points, size_t point_dim, size_t max_dim,
                                  const VRConfig &config);
} // namespace utils

errors::ErrorResult<std::vector<Pair>>
computeVrPersistenceAccelerated(core::BufferView<const double>points, Size point_dim,
                                const VRConfig &config);
errors::ErrorResult<std::vector<Pair>>
computeVrPersistenceFast(core::BufferView<const double>points, Size point_dim,
                         const VRConfig &config);
VRConfig createOptimalConfig(core::BufferView<const double>points, Size point_dim,
                             const VRConfig &base_config);

inline errors::ErrorResult<std::vector<Pair>>
compute_vr_persistence_accelerated(core::BufferView<const double>points, Size point_dim,
                                   const VRConfig &config)
{
    return computeVrPersistenceAccelerated(points, point_dim, config);
}

inline errors::ErrorResult<std::vector<Pair>>
compute_vr_persistence_fast(core::BufferView<const double>points, Size point_dim,
                            const VRConfig &config)
{
    return computeVrPersistenceFast(points, point_dim, config);
}

inline VRConfig create_optimal_config(core::BufferView<const double>points, Size point_dim,
                                      const VRConfig &base_config)
{
    return createOptimalConfig(points, point_dim, base_config);
}

namespace performance
{
AcceleratedPerformanceStats getCurrentPerformanceStats();
errors::ErrorResult<void> exportPerformanceReport(const std::string &filename,
                                                  const AcceleratedPerformanceStats &stats);

inline AcceleratedPerformanceStats get_current_performance_stats()
{
    return getCurrentPerformanceStats();
}

inline errors::ErrorResult<void> export_performance_report(const std::string &filename,
                                                           const AcceleratedPerformanceStats &stats)
{
    return exportPerformanceReport(filename, stats);
}
} // namespace performance

namespace accelerated
{
errors::ErrorResult<std::vector<Pair>>
computeVrPersistenceAccelerated(core::BufferView<const double>points, Size point_dim,
                                const VRConfig &config);
errors::ErrorResult<std::vector<Pair>>
computeVrPersistenceFast(core::BufferView<const double>points, Size point_dim,
                         const VRConfig &config);
VRConfig createOptimalConfig(core::BufferView<const double>points, Size point_dim,
                             const VRConfig &base_config);
} // namespace accelerated

} // namespace nerve::persistence::accelerated

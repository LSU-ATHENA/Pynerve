
#pragma once

#include "nerve/common/accelerated_types.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core_types.hpp"
#include "nerve/errors/errors.hpp"
#include "nerve/persistence/accelerated/accelerated_interface.hpp"
#include "nerve/persistence/core/core_types.hpp"
#include "nerve/persistence/detail/fast_vr_accelerated_detail.hpp"

#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace nerve::persistence
{

// Bring types from common namespace to persistence namespace
using ::nerve::common::AcceleratedPerformanceStats;
using ::nerve::common::AccelerationMode;
using ::nerve::common::AlgorithmType;
using ::nerve::common::ApproximationLevel;
using ::nerve::common::DeviceInfo;
using ::nerve::common::ExecutionMode;
using ::nerve::common::KernelDiagnosticsCounters;
using ::nerve::common::PerformanceMetrics;
using ::nerve::common::ProblemCharacteristics;
using ::nerve::common::SystemCapabilities;
using ::nerve::common::VRAlgorithmSelection;
using ::nerve::common::VRConfig;

// Accelerated VR types - defined in accelerated_interface.hpp
namespace accelerated
{

// Types from accelerated_interface.hpp are already declared in this namespace.

struct OptimizationParams
{
    size_t block_size = 256;
    size_t grid_size = 0;
    size_t shared_memory_size = 0;
    bool use_streaming = false;
    size_t streaming_chunk_size = 0;
    double memory_efficiency_threshold = 0.9;
};

// Classes from accelerated_interface.hpp are already declared in this namespace.

namespace detail
{
// All detail namespace functions and types are now in fast_vr_accelerated_detail.hpp
// which is included above

// Additional inline helper functions needed by vr_fast_api.cpp
// Note: to_base_fast_vr_config defined after VRConfig
::nerve::persistence::VRConfig to_base_fast_vr_config(const ::nerve::persistence::VRConfig &config);
} // namespace detail

// Utils namespace
namespace utils
{
// Note: createOptimalConfigForProblem defined after VRConfig
::nerve::persistence::VRConfig
createOptimalConfigForProblem(size_t n_points, size_t point_dim, double max_radius,
                              const ::nerve::persistence::VRConfig &base_config);
AcceleratedPerformanceStats estimatePerformance(size_t n_points, size_t point_dim,
                                                double max_radius,
                                                const ::nerve::persistence::VRConfig &config);
bool isAccelerationBeneficial(size_t n_points, size_t point_dim, double max_radius);
SystemCapabilities detectSystemCapabilities();
AccelerationMode recommendAccelerationMode(const ProblemCharacteristics &problem,
                                           const SystemCapabilities &capabilities);
size_t estimateMemoryRequirements(size_t n_points, size_t point_dim, size_t max_dim,
                                  const VRConfig &config);
// validateAccelerationConfig is defined after VRConfig
errors::ErrorResult<void> validateAccelerationConfig(const ::nerve::persistence::VRConfig &config);
} // namespace utils

// Performance namespace
namespace performance
{
// Declarations only - implementations in vr_fast_api.cpp
AcceleratedPerformanceStats getCurrentPerformanceStats();
errors::ErrorResult<void> exportPerformanceReport(const std::string &filename,
                                                  const AcceleratedPerformanceStats &stats);
} // namespace performance

// Make VRConfig available in accelerated namespace (same type from common)
using ::nerve::common::VRConfig;

} // namespace accelerated

// Inline helper implementations

namespace accelerated::detail
{
inline VRConfig to_base_fast_vr_config(const VRConfig &config)
{
    return config; // Identity function - already base config
}
} // namespace accelerated::detail

// Function declarations
std::vector<Pair> computeVrPersistenceFast(core::BufferView<const double> points, Size point_dim,
                                           const VRConfig &config);

errors::ErrorResult<std::vector<Pair>>
computeVrPersistenceFastResult(core::BufferView<const double> points, Size point_dim,
                               const VRConfig &config);

inline std::vector<Pair> computeVrPersistence(const double *points, Size num_points, Size point_dim,
                                              const VRConfig &config)
{
    core::BufferView<const double> view(points, num_points * point_dim);
    return computeVrPersistenceFast(view, point_dim, config);
}

bool isAdaptiveAccelerationAvailable();
bool is_cuda_available();
errors::ErrorResult<std::string> getAdaptiveAccelerationCapabilities();
VRConfig getOptimalFastvrConfig(Size num_points, Size point_dim);

} // namespace nerve::persistence

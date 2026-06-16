#pragma once

#include "nerve/errors/errors.hpp"
#include "nerve/types.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace nerve::common
{

enum class AccelerationMode
{
    CPU_ONLY,
    GPU_ONLY,
    HYBRID_AUTO,
    HYBRID_GPU_PREFERRED,
    HYBRID_CPU_PREFERRED
};

enum class ExecutionMode
{
    CPU_ONLY,
    GPU_ONLY,
    HYBRID_GPU_PREFERRED,
    HYBRID_CPU_PREFERRED,
    HYBRID_AUTO
};

enum class ProblemType
{
    SMALL,
    MEDIUM,
    LARGE,
    HUGE
};

enum class AlgorithmType
{
    STANDARD_CPU,
    GPU_ACCELERATED,
    HYBRID,
    LOCKFREE_MULTICORE
};

enum class ApproximationLevel
{
    EXACT,
    HIGH_PRECISION,
    MEDIUM_PRECISION,
    LOW_PRECISION,
    VERY_FAST
};

enum class VRAlgorithmSelection
{
    AUTO,           // Automatically select optimal algorithm
    FAST_SIMD,      // Small exact fast path; uses SIMD kernels when available
    MEDIUM_HYBRID,  // Medium-scale tiled/parallel path for 1K-10K points
    LARGE_WITNESS,  // Witness complex approximation for >10K points
    EXACT_STANDARD, // Standard exact computation (exact path)
    ACCELERATED     // Combined edge-collapse, union-find, and lock-free paths
};
using VRAlgorithm = VRAlgorithmSelection;

// Structs (in dependency order - no forward declarations needed)

struct KernelDiagnosticsCounters
{
    std::size_t dropped_invalid_distances = 0;
    std::size_t invalid_distance_inputs = 0;
    std::size_t dimension_tiles_processed = 0;
    std::size_t pivot_hits = 0;
};

struct DeviceInfo
{
    std::string name;
    int device_id = 0;
    Size total_memory = 0;
    Size available_memory = 0;
    int compute_capability_major = 0;
    int compute_capability_minor = 0;
    double compute_capability = 0.0;
    int multiprocessor_count = 0;
};

struct SystemCapabilities
{
    bool cuda_available = false;
    DeviceInfo gpu_info;
    Size available_memory = 0;
    double compute_capability = 0.0;
    std::vector<std::string> supported_features;
    Size num_multiprocessors = 0;
    Size max_threads_per_block = 0;
    Size shared_memory_per_block = 0;
};

struct ProblemCharacteristics
{
    Size estimated_n_points = 0;
    Size point_dim = 0;
    double max_radius = 0.0;
    Size max_dim = 0;
    ProblemType problem_type = ProblemType::MEDIUM;
    double estimated_density = 0.1;
    bool is_high_dimensional = false;
    bool is_sparse = false;
};

struct AccelerationConfig
{
    AccelerationMode mode = AccelerationMode::CPU_ONLY;
    double gpu_work_ratio = 0.0;
    Size threshold = 1000;
    bool enable_streaming = true;
    bool enable_apparent_pairs = true;
    Size gpu_memory_limit = 0;
    bool enable_memory_optimization = true;
    double memory_pressure_threshold = 0.8;
    bool enable_profiling = false;
    std::string log_level = "info";
};

struct MemoryConfig
{
    Size memory_limit = 0;
    bool enable_memory_pool = true;
    bool enable_fragmentation_prevention = true;
    double memory_efficiency_target = 0.85;
    bool enable_memory_monitoring = true;
};

struct DebugConfig
{
    bool enable_logging = false;
    bool enable_profiling = false;
    bool enable_memory_tracking = false;
    bool enable_validation = false;
    std::string log_file = "";
    std::string log_level = "info";
};

struct Strategy
{
    ExecutionMode mode = ExecutionMode::CPU_ONLY;
    double gpu_work_ratio = 0.0;
    Size threshold = 1000;
    bool enable_streaming = true;
    Size block_size = 256;
    Size grid_size = 0;
    Size shared_memory_size = 0;
    Size streaming_chunk_size = 0;
};

struct PerformanceMetrics
{
    double total_time_ms = 0.0;
    double gpu_time_ms = 0.0;
    double cpu_time_ms = 0.0;
    Size problem_size = 0;
    Size point_dim = 0;
    double max_radius = 0.0;
    Size max_dim = 0;
    ExecutionMode execution_mode = ExecutionMode::CPU_ONLY;
    double gpu_work_ratio = 0.0;
    bool gpu_available = false;
    bool success = false;
    Size result_size = 0;
    errors::ErrorCode error_code = errors::ErrorCode::SUCCESS;
    std::string error_message;
    double problem_ops = 0.0;
    double gpu_bytes = 0.0;
    bool gpu_compute_stage_executed = false;
    double gpu_stage_ops = 0.0;
    std::string gpu_stage_details;
    KernelDiagnosticsCounters kernel_diagnostics;
    std::chrono::system_clock::time_point timestamp;

    double getSpeedup() const
    {
        if (!std::isfinite(cpu_time_ms) || !std::isfinite(gpu_time_ms) || cpu_time_ms < 0.0 ||
            gpu_time_ms < 0.0)
        {
            return 0.0;
        }
        return (cpu_time_ms > 0.0 && gpu_time_ms > 0.0) ? (cpu_time_ms / gpu_time_ms) : 1.0;
    }

    double getEfficiency() const
    {
        if (!std::isfinite(total_time_ms) || total_time_ms <= 0.0)
        {
            return 0.0;
        }
        const double expected_time = static_cast<double>(problem_size) * 0.001;
        if (!std::isfinite(expected_time))
        {
            return 0.0;
        }
        return std::min(1.0, expected_time / total_time_ms);
    }
};

struct AcceleratedPerformanceStats
{
    double total_time_ms = 0.0;
    double gpu_time_ms = 0.0;
    double cpu_time_ms = 0.0;
    double memory_usage_mb = 0.0;
    double gpu_utilization = 0.0;
    double speedup = 1.0;
    Size problems_processed = 0;
    double average_speedup = 0.0;
    double peak_memory_usage_mb = 0.0;
    bool gpu_used = false;
    bool hybrid_used = false;
    double gpu_stage_ops = 0.0;
    KernelDiagnosticsCounters kernel_diagnostics;
    std::vector<PerformanceMetrics> detailed_metrics;

    double getOverallEfficiency() const
    {
        const bool valid_numeric =
            std::isfinite(total_time_ms) && total_time_ms >= 0.0 && std::isfinite(gpu_time_ms) &&
            gpu_time_ms >= 0.0 && std::isfinite(cpu_time_ms) && cpu_time_ms >= 0.0 &&
            std::isfinite(memory_usage_mb) && memory_usage_mb >= 0.0 &&
            std::isfinite(gpu_utilization) && gpu_utilization >= 0.0 && std::isfinite(speedup) &&
            speedup >= 0.0 && std::isfinite(average_speedup) && average_speedup >= 0.0 &&
            std::isfinite(peak_memory_usage_mb) && peak_memory_usage_mb >= 0.0 &&
            std::isfinite(gpu_stage_ops) && gpu_stage_ops >= 0.0;
        if (!valid_numeric || total_time_ms == 0.0)
        {
            return 0.0;
        }
        const double time_efficiency = speedup > 1.0 ? std::min(1.0, speedup / 10.0) : 1.0;
        const double memory_efficiency =
            memory_usage_mb > 0.0 ? std::min(1.0, 4096.0 / memory_usage_mb) : 1.0;
        return (time_efficiency + memory_efficiency) * 0.5;
    }

    std::string getPerformanceGrade() const
    {
        const double efficiency = getOverallEfficiency();
        if (efficiency >= 0.9)
            return "A";
        if (efficiency >= 0.8)
            return "B";
        if (efficiency >= 0.7)
            return "C";
        if (efficiency >= 0.6)
            return "D";
        return "F";
    }
};

// Main Configuration Struct

struct VRConfig
{
    // Basic VR parameters
    Size max_dim = 2;
    double max_radius = 1.0;
    Size num_threads = 0;

    // Algorithm selection (uses global VRAlgorithmSelection enum)
    VRAlgorithmSelection algorithm = VRAlgorithmSelection::AUTO;

    // Acceleration settings
    AccelerationConfig acceleration;

    bool use_acceleration = false;

    // Accelerated acceleration flags
    bool use_accelerated_runtime = false;
    bool auto_detect_accelerated_runtime = false;
    bool use_adaptive_acceleration = false;
    bool auto_detect_adaptive_acceleration = false;
    bool enable_matrix_multiplication = false;
    bool enable_sparsification = false;
    bool enable_lockfree_multicore = false;
    bool enable_approximation = false;

    // Performance optimization
    double performance_target_ms = 100.0;
    double memory_limit_mb = 8192.0;
    double min_precision = 1e-10;
    double time_budget_ms = 0.0;

    // GPU-specific options
    bool prefer_gpu = false;
    bool prefer_multicore = false;
    bool enable_tensor_cores = true;
    bool enable_mixed_precision = false;

    // Approximation settings
    ApproximationLevel approximation_level = ApproximationLevel::EXACT;
    double approximation_error_budget = 1e-6;
    double approximation_error_tolerance = 1e-6;

    // Streaming settings
    bool enable_incremental_updates = false;
    Size chunk_size = 1024;
    Size window_size = 10000;

    // Memory management
    bool enable_memory_pooling = true;
    Size memory_pool_size_mb = 1024;
    bool enable_sparse_matrices = true;
    double sparsity_threshold = 0.1;

    // Concurrency settings
    bool enable_work_stealing = false;
    Size max_concurrent_tasks = 0;
    bool enable_numa_awareness = false;

    // Methods
    errors::ErrorResult<void> validate() const
    {
        if (max_dim == 0)
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E50_PH_ABORT);
        }
        if (!std::isfinite(max_radius) || max_radius <= 0.0)
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E50_PH_ABORT);
        }
        if (!std::isfinite(acceleration.gpu_work_ratio) || acceleration.gpu_work_ratio < 0.0 ||
            acceleration.gpu_work_ratio > 1.0)
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E50_PH_ABORT);
        }
        if (!std::isfinite(acceleration.memory_pressure_threshold) ||
            acceleration.memory_pressure_threshold < 0.0 ||
            acceleration.memory_pressure_threshold > 1.0)
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E50_PH_ABORT);
        }
        return errors::ErrorResult<void>::success();
    }

    VRConfig getEffectiveConfig() const
    {
        VRConfig effective = *this;
        if (effective.max_dim == 0)
        {
            effective.max_dim = 1;
        }
        // num_threads and other auto-detect logic is handled at runtime
        return effective;
    }
};

// Forward declarations for classes (defined in accelerated_interface.hpp)

class GPUAccelerationManager;
class PerformanceOptimizer;
class PerformanceMonitor;
class AcceleratedVREngine;

} // namespace nerve::common

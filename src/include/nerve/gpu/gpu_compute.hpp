
#pragma once

#include "nerve/errors/errors.hpp"
#include "nerve/runtime/hardware_probe.hpp"

#include <chrono>
#include <cstddef>
#include <limits>
#include <new>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef __CUDACC__
#include "nerve/gpu/device_array.hpp"
#include "nerve/gpu/kernel_launcher.hpp"
#endif

namespace nerve::gpu
{

#ifdef __CUDACC__

inline errors::ErrorResult<void> initialize()
{
    Config config;
    ComputeManager::getInstance().initialize(config);
    if (!ComputeManager::getInstance().isAvailable())
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E10_GPU_OOM);
    }
    return errors::ErrorResult<void>::success();
}

inline errors::ErrorResult<void> initialize(const Config &config)
{
    ComputeManager::getInstance().initialize(config);
    if (!ComputeManager::getInstance().isAvailable())
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E10_GPU_OOM);
    }
    return errors::ErrorResult<void>::success();
}

inline void shutdown()
{
    ComputeManager::getInstance().shutdown();
}

inline bool isAvailable()
{
    return ComputeManager::getInstance().isAvailable();
}

inline Config getConfig()
{
    return ComputeManager::getInstance().getConfig();
}

inline void setConfig(const Config &config)
{
    ComputeManager::getInstance().setConfig(config);
}

inline void enableOperation(OperationType opType, bool enabled = true)
{
    auto config = ComputeManager::getInstance().getConfig();
    config.enabled[opType] = enabled;
    ComputeManager::getInstance().setConfig(config);
}

inline void disableAll()
{
    auto config = ComputeManager::getInstance().getConfig();
    config.enable_gpu = false;
    ComputeManager::getInstance().setConfig(config);
}

inline ComputeManager::Stats getStats()
{
    return ComputeManager::getInstance().getStats();
}

inline void clearPerformanceHistory()
{
    ComputeManager::getInstance().clearPerformanceHistory();
}

inline void clearStats()
{
    ComputeManager::getInstance().clearStats();
}

#else

enum class OperationType
{
    kDistanceMatrix,
    kLaplacianConstruction,
    kBoundaryReduction,
    kCohomologyReduction,
    kSimplexBoundary,
    kDiagramDistance,
    kColumnSymmetricDifference,
    kStreamingPersistence,
    kEigenvalueComputation,
    kCount
};

struct Stats
{
    size_t total_operations = 0;
    size_t gpu_operations = 0;
    double average_speedup = 1.0;
    double total_time_saved_ms = 0.0;
    size_t initialize_attempts = 0;
    bool runtime_gpu_detected = false;
    bool gpu_enabled = false;
};

struct Config
{
    bool enable_gpu = true;
    size_t threshold = 1000;
    bool auto_select = true;
    bool enable_streaming = true;
    size_t chunk_size = 4096;
    std::unordered_map<OperationType, bool> enabled;
    std::unordered_map<OperationType, size_t> thresholds;
    size_t max_memory_mb = 0;
    size_t num_streams = 4;
    bool tensor_cores = true;
    bool mixed_precision = false;
    bool profiling = false;

    Config()
    {
        for (int i = 0; i < static_cast<int>(OperationType::kCount); ++i)
        {
            enabled[static_cast<OperationType>(i)] = true;
            thresholds[static_cast<OperationType>(i)] = threshold;
        }
    }
};

namespace detail
{
struct HostGpuState
{
    Config config{};
    Stats stats{};
    size_t initialize_attempts = 0;
    bool runtime_gpu_detected = false;
    bool gpu_enabled = false;
};

inline HostGpuState &hostGpuState()
{
    static HostGpuState state;
    return state;
}

inline bool detectRuntimeGpu()
{
    const runtime::HardwareSnapshot snapshot = runtime::collectHardwareSnapshot();
    return runtime::has_cuda_gpu(snapshot);
}

inline void resetPerformanceCounters(Stats *stats)
{
    stats->total_operations = 0;
    stats->gpu_operations = 0;
    stats->average_speedup = 1.0;
    stats->total_time_saved_ms = 0.0;
}
} // namespace detail

inline errors::ErrorResult<void> initialize()
{
    auto &state = detail::hostGpuState();
    state.initialize_attempts += 1;
    state.stats.initialize_attempts = state.initialize_attempts;
    state.config = Config{};
    state.runtime_gpu_detected = detail::detectRuntimeGpu();
    state.gpu_enabled = state.config.enable_gpu && state.runtime_gpu_detected;
    state.stats.runtime_gpu_detected = state.runtime_gpu_detected;
    state.stats.gpu_enabled = state.gpu_enabled;
    if (!state.gpu_enabled)
    {
        return errors::ErrorResult<void>::error(
            errors::ErrorCode::E11_GPU_LAUNCH_FAIL,
            "GPU runtime inactive in host-only compilation mode");
    }
    return errors::ErrorResult<void>::success();
}

inline errors::ErrorResult<void> initialize(const Config &config)
{
    auto &state = detail::hostGpuState();
    state.initialize_attempts += 1;
    state.stats.initialize_attempts = state.initialize_attempts;
    state.config = config;
    state.runtime_gpu_detected = detail::detectRuntimeGpu();
    state.gpu_enabled = config.enable_gpu && state.runtime_gpu_detected;
    state.stats.runtime_gpu_detected = state.runtime_gpu_detected;
    state.stats.gpu_enabled = state.gpu_enabled;
    if (!state.gpu_enabled)
    {
        return errors::ErrorResult<void>::error(
            errors::ErrorCode::E11_GPU_LAUNCH_FAIL,
            "GPU runtime inactive in host-only compilation mode");
    }
    return errors::ErrorResult<void>::success();
}

inline void shutdown()
{
    auto &state = detail::hostGpuState();
    state.gpu_enabled = false;
    state.stats.gpu_enabled = false;
}

inline bool isAvailable()
{
    const auto &state = detail::hostGpuState();
    return state.gpu_enabled && state.runtime_gpu_detected;
}

inline Config getConfig()
{
    return detail::hostGpuState().config;
}

inline void setConfig(const Config &config)
{
    auto &state = detail::hostGpuState();
    state.config = config;
    state.runtime_gpu_detected = detail::detectRuntimeGpu();
    state.gpu_enabled = config.enable_gpu && state.runtime_gpu_detected;
    state.stats.runtime_gpu_detected = state.runtime_gpu_detected;
    state.stats.gpu_enabled = state.gpu_enabled;
}

inline void enableOperation(OperationType opType, bool enabled = true)
{
    detail::hostGpuState().config.enabled[opType] = enabled;
}

inline void disableAll()
{
    auto &state = detail::hostGpuState();
    state.config.enable_gpu = false;
    state.gpu_enabled = false;
}

inline Stats getStats()
{
    const auto &state = detail::hostGpuState();
    Stats stats = state.stats;
    stats.initialize_attempts = state.initialize_attempts;
    stats.runtime_gpu_detected = state.runtime_gpu_detected;
    stats.gpu_enabled = state.gpu_enabled;
    return stats;
}

inline void clearPerformanceHistory()
{
    detail::resetPerformanceCounters(&detail::hostGpuState().stats);
}

inline void clearStats()
{
    auto &state = detail::hostGpuState();
    state.stats = Stats{};
    state.initialize_attempts = 0;
    state.stats.runtime_gpu_detected = state.runtime_gpu_detected;
    state.stats.gpu_enabled = state.gpu_enabled;
}

#endif // __CUDACC__

namespace enable
{
namespace detail
{
inline void setOperationEnabled(OperationType operation, bool enabled)
{
    enableOperation(operation, enabled);
}
} // namespace detail

inline void distanceMatrix(bool enabled = true)
{
    detail::setOperationEnabled(OperationType::kDistanceMatrix, enabled);
}
inline void laplacian(bool enabled = true)
{
    detail::setOperationEnabled(OperationType::kLaplacianConstruction, enabled);
}
inline void reduction(bool enabled = true)
{
    detail::setOperationEnabled(OperationType::kBoundaryReduction, enabled);
}
inline void cohomology(bool enabled = true)
{
    detail::setOperationEnabled(OperationType::kCohomologyReduction, enabled);
}
inline void simplexOperations(bool enabled = true)
{
    detail::setOperationEnabled(OperationType::kSimplexBoundary, enabled);
}
inline void diagramDistances(bool enabled = true)
{
    detail::setOperationEnabled(OperationType::kDiagramDistance, enabled);
}
inline void streaming(bool enabled = true)
{
    detail::setOperationEnabled(OperationType::kStreamingPersistence, enabled);
}
inline void complexConstruction(bool enabled = true)
{
    detail::setOperationEnabled(OperationType::kSimplexBoundary, enabled);
}
inline void hungarian(bool enabled = true)
{
    detail::setOperationEnabled(OperationType::kDiagramDistance, enabled);
}
inline void columnOps(bool enabled = true)
{
    detail::setOperationEnabled(OperationType::kColumnSymmetricDifference, enabled);
}
inline void eigenvalue(bool enabled = true)
{
    detail::setOperationEnabled(OperationType::kEigenvalueComputation, enabled);
}
} // namespace enable

#ifdef __CUDACC__

inline errors::ErrorResult<void>
computeDistanceMatrix(const double *points, size_t n_points, size_t point_dim,
                      double *out_distances,
                      double max_radius_sq = std::numeric_limits<double>::infinity())
{
    (void)max_radius_sq;
    if ((n_points != 0 && points == nullptr) || out_distances == nullptr || point_dim == 0)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E51_PH_INPUT,
                                                "invalid distance matrix buffer");
    }
    if (n_points != 0 && point_dim > std::numeric_limits<size_t>::max() / n_points)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E41_RESOURCE_LIMIT,
                                                "distance point buffer size overflow");
    }
    if (n_points != 0 && n_points > std::numeric_limits<size_t>::max() / n_points)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E41_RESOURCE_LIMIT,
                                                "distance matrix size overflow");
    }

    try
    {
        std::vector<std::vector<double>> point_vectors(n_points, std::vector<double>(point_dim));
        for (size_t i = 0; i < n_points; ++i)
        {
            for (size_t d = 0; d < point_dim; ++d)
            {
                point_vectors[i][d] = points[i * point_dim + d];
            }
        }

        std::vector<std::vector<double>> distances;
        auto result = ComputeManager::getInstance().computeDistanceMatrix(point_vectors, distances);
        if (!result.isSuccess())
        {
            return result;
        }
        for (size_t i = 0; i < n_points; ++i)
        {
            for (size_t j = 0; j < n_points; ++j)
            {
                out_distances[i * n_points + j] = distances[i][j];
            }
        }
    }
    catch (const std::bad_alloc &)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E10_GPU_OOM,
                                                "distance matrix wrapper allocation failed");
    }

    return errors::ErrorResult<void>::success();
}

inline errors::ErrorResult<void>
computeDiagramCostMatrix(const persistence::Diagram &diagram1, const persistence::Diagram &diagram2,
                         std::vector<std::vector<double>> &out_cost_matrix)
{
    return ComputeManager::getInstance().computeDiagramCostMatrix(diagram1, diagram2,
                                                                  out_cost_matrix);
}

inline errors::ErrorResult<void> computeReduction(persistence::Reducer &reducer,
                                                  const algebra::BoundaryMatrix &boundary_matrix)
{
    return ComputeManager::getInstance().computeReduction(reducer, boundary_matrix);
}

inline errors::ErrorResult<void>
accelerateCohomology(persistence::Reducer &reducer, const algebra::BoundaryMatrix &boundary_matrix)
{
    return ComputeManager::getInstance().computeCohomology(reducer, boundary_matrix);
}

#endif // __CUDACC__

#ifdef __CUDACC__

class GPUContext
{
public:
    GPUContext()
        : initialized_(false)
    {}

    explicit GPUContext(const Config &config)
    {
        auto result = gpu::initialize(config);
        initialized_ = result.isSuccess();
    }

    ~GPUContext()
    {
        if (initialized_)
        {
            gpu::shutdown();
        }
    }

    bool isInitialized() const { return initialized_; }

    errors::ErrorResult<void> initialize(const Config &config = Config())
    {
        if (initialized_)
        {
            return errors::ErrorResult<void>::success();
        }
        auto result = gpu::initialize(config);
        initialized_ = result.isSuccess();
        return result;
    }

private:
    bool initialized_;
};

class ProfileScope
{
public:
    explicit ProfileScope(const std::string &operation_name)
        : operation_name_(operation_name)
        , start_(std::chrono::steady_clock::now())
    {}

    ~ProfileScope()
    {
        auto end = std::chrono::steady_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start_).count();

        PerformanceProfile profile;
        profile.operation_name = operation_name_;
        profile.gpu_time_ms = elapsed_ms;
        profile.timestamp = std::chrono::system_clock::now();

        ComputeManager::getInstance().recordPerformance(profile);
    }

private:
    std::string operation_name_;
    std::chrono::steady_clock::time_point start_;
};

#else

class GPUContext
{
public:
    GPUContext()
        : initialized_(false)
    {}

    explicit GPUContext(const Config &config)
    {
        auto result = gpu::initialize(config);
        initialized_ = result.isSuccess();
    }

    [[nodiscard]] bool isInitialized() const { return initialized_; }

    errors::ErrorResult<void> initialize(const Config &config = Config())
    {
        if (initialized_)
        {
            return errors::ErrorResult<void>::success();
        }
        auto result = gpu::initialize(config);
        initialized_ = result.isSuccess();
        return result;
    }

private:
    bool initialized_;
};

class ProfileScope
{
public:
    explicit ProfileScope(const std::string &) {}
};

#endif // __CUDACC__

} // namespace nerve::gpu

#define NERVE_GPU_INIT() nerve::gpu::initialize()

#define NERVE_GPU_INIT_CONFIG(config) nerve::gpu::initialize(config)

#define NERVE_GPU_SHUTDOWN() nerve::gpu::shutdown()

#define NERVE_GPU_AVAILABLE() nerve::gpu::isAvailable()

#define NERVE_GPU_ENABLE(op) nerve::gpu::enable::op(true)

#define NERVE_GPU_DISABLE(op) nerve::gpu::enable::op(false)

#define NERVE_GPU_PROFILE(name) nerve::gpu::ProfileScope _gpu_profile_##__LINE__(name)

#define NERVE_GPU_CONTEXT(name) nerve::gpu::GPUContext name

#define NERVE_GPU_AUTO nerve::gpu::GPUContext _gpu_auto_context

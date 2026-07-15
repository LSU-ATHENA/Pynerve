
#include "nerve/persistence/detail/fast_vr_accelerated_detail.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"

#include <algorithm>
#include <chrono>
#include <string>
#include <utility>

namespace nerve::persistence::accelerated::detail
{
namespace
{

ExecutionMode toExecutionMode(AccelerationMode mode)
{
    switch (mode)
    {
        case AccelerationMode::CPU_ONLY:
            return ExecutionMode::CPU_ONLY;
        case AccelerationMode::GPU_ONLY:
            return ExecutionMode::GPU_ONLY;
        case AccelerationMode::HYBRID_GPU_PREFERRED:
            return ExecutionMode::HYBRID_GPU_PREFERRED;
        case AccelerationMode::HYBRID_CPU_PREFERRED:
            return ExecutionMode::HYBRID_CPU_PREFERRED;
        case AccelerationMode::HYBRID_AUTO:
        default:
            return ExecutionMode::HYBRID_AUTO;
    }
}

class DefaultGpuAccelerationManager final : public GPUAccelerationManager
{
public:
    explicit DefaultGpuAccelerationManager(const VRConfig &config)
        : config_(config)
        , capabilities_(detectSystemCapabilitiesImpl())
    {}

    void updateConfig(const VRConfig &config) override
    {
        config_ = config;
        capabilities_ = detectSystemCapabilitiesImpl();
    }

    bool isAvailable() const override { return capabilities_.cuda_available; }

    DeviceInfo getGpuInfo() const override { return capabilities_.gpu_info; }

private:
    VRConfig config_;
    SystemCapabilities capabilities_;
};

class DefaultPerformanceOptimizer final : public PerformanceOptimizer
{
public:
    explicit DefaultPerformanceOptimizer(const VRConfig &config)
        : config_(config)
    {}

    void updateConfig(const VRConfig &config) override
    {
        std::lock_guard<std::mutex> lock(mutex_);
        config_ = config;
        tuned_ = false;
    }

    errors::ErrorResult<OptimizationParams> getOptimalParameters(size_t n_points, size_t point_dim,
                                                                 double max_radius) override
    {
        std::lock_guard<std::mutex> lock(mutex_);
        OptimizationParams params;
        if (tuned_)
        {
            params = tuned_params_;
        }

        if (params.block_size == 0)
        {
            params.block_size = n_points >= 16384 ? 512 : 256;
        }
        params.grid_size = (n_points + params.block_size - 1) / params.block_size;
        params.shared_memory_size = (point_dim <= 8) ? 16 * 1024 : 32 * 1024;
        if (!tuned_)
        {
            params.use_streaming =
                config_.acceleration.enable_streaming && n_points >= config_.acceleration.threshold;
            params.streaming_chunk_size =
                params.use_streaming ? std::max<std::size_t>(4096, point_dim * 2048) : 0;
            params.memory_efficiency_threshold = std::clamp(max_radius, 0.5, 0.95);
        }
        else
        {
            params.use_streaming =
                params.use_streaming || n_points >= config_.acceleration.threshold;
            if (params.use_streaming && params.streaming_chunk_size == 0)
            {
                params.streaming_chunk_size = std::max<std::size_t>(4096, point_dim * 1024);
            }
            params.memory_efficiency_threshold =
                std::clamp(params.memory_efficiency_threshold, 0.2, 0.99);
        }
        return errors::ErrorResult<OptimizationParams>::success(OptimizationParams(params));
    }

    errors::ErrorResult<void> optimizeForCurrentSystem() override
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const SystemCapabilities capabilities = detectSystemCapabilitiesImpl();

        tuned_params_.block_size =
            capabilities.cuda_available
                ? std::max<std::size_t>(
                      128, std::min<std::size_t>(512, capabilities.max_threads_per_block))
                : 256;
        tuned_params_.shared_memory_size =
            capabilities.cuda_available
                ? std::min<std::size_t>(48ULL * 1024ULL, capabilities.shared_memory_per_block)
                : 0;
        tuned_params_.grid_size = 0;

        std::size_t reference_memory = 0;
        if (capabilities.available_memory > 0)
        {
            reference_memory = capabilities.available_memory;
        }

        tuned_params_.use_streaming = config_.acceleration.enable_streaming;
        if (reference_memory > 0)
        {
            tuned_params_.use_streaming =
                tuned_params_.use_streaming || reference_memory < (512ULL * 1024ULL * 1024ULL);
            tuned_params_.streaming_chunk_size =
                tuned_params_.use_streaming
                    ? std::max<std::size_t>(kMinGpuChunkElements,
                                            reference_memory / (64 * sizeof(double)))
                    : 0;
        }
        else
        {
            tuned_params_.streaming_chunk_size =
                tuned_params_.use_streaming ? kMinGpuChunkElements : 0;
        }
        tuned_params_.memory_efficiency_threshold =
            std::clamp(config_.acceleration.memory_pressure_threshold, 0.2, 0.99);
        tuned_ = true;
        return errors::ErrorResult<void>::success();
    }

    errors::ErrorResult<void> autoTuneParameters() override
    {
        if (!tuned_)
        {
            const auto accelerated = optimizeForCurrentSystem();
            if (accelerated.isError())
            {
                return accelerated;
            }
        }

        std::lock_guard<std::mutex> lock(mutex_);
        auto &global = globalPerformanceState();
        std::lock_guard<std::mutex> perfLock(global.mutex);
        if (!global.stats.detailed_metrics.empty())
        {
            const PerformanceMetrics &latest = global.stats.detailed_metrics.back();
            if (latest.gpu_time_ms > latest.cpu_time_ms &&
                tuned_params_.streaming_chunk_size > kMinGpuChunkElements)
            {
                tuned_params_.streaming_chunk_size = std::max<std::size_t>(
                    kMinGpuChunkElements, tuned_params_.streaming_chunk_size / 2);
            }
            else if (latest.gpu_time_ms > 0.0)
            {
                tuned_params_.streaming_chunk_size = std::min<std::size_t>(
                    tuned_params_.streaming_chunk_size * 2, static_cast<std::size_t>(1u << 20));
            }
        }
        return errors::ErrorResult<void>::success();
    }

private:
    VRConfig config_;
    OptimizationParams tuned_params_{};
    bool tuned_ = false;
    std::mutex mutex_;
};

class AcceleratedVREngineImpl final : public AcceleratedVREngine
{
public:
    explicit AcceleratedVREngineImpl(const VRConfig &config)
        : config_(config.getEffectiveConfig())
    {
        auto manager = makeDefaultGpuManager(config_);
        if (manager.isSuccess())
        {
            gpu_manager_ = std::move(manager.value());
        }
        auto optimizer = makeDefaultPerformanceOptimizer(config_);
        if (optimizer.isSuccess())
        {
            optimizer_ = std::move(optimizer.value());
        }
    }

    errors::ErrorResult<std::vector<Pair>>
    computeVrPersistence(core::BufferView<const double> points, Size point_dim,
                         const VRConfig &config) override
    {
        VRConfig effective = config.getEffectiveConfig();
        const auto validation = effective.validate();
        if (validation.isError())
        {
            return errors::ErrorResult<std::vector<Pair>>::error(validation.errorCode());
        }
        if (point_dim == 0 || points.empty() || (points.size() % point_dim) != 0)
        {
            return errors::ErrorResult<std::vector<Pair>>::error(errors::ErrorCode::E50_PH_ABORT);
        }

        const std::size_t n_points = points.size() / point_dim;
        ExecutionMode requested_mode = toExecutionMode(effective.acceleration.mode);
        const bool gpu_available = isAvailable();
        if (requested_mode == ExecutionMode::GPU_ONLY && !gpu_available)
        {
            return errors::ErrorResult<std::vector<Pair>>::error(errors::ErrorCode::E10_GPU_OOM);
        }
        if (requested_mode == ExecutionMode::HYBRID_AUTO)
        {
            requested_mode =
                gpu_available ? ExecutionMode::HYBRID_GPU_PREFERRED : ExecutionMode::CPU_ONLY;
        }

        monitor_.startMonitoring("computeVrPersistence");
        const auto total_start = std::chrono::steady_clock::now();

        std::size_t gpu_chunk_elements = 0;
        if (optimizer_)
        {
            auto params_result =
                optimizer_->getOptimalParameters(n_points, point_dim, effective.max_radius);
            if (params_result.isSuccess())
            {
                const auto &params = params_result.value();
                if (params.use_streaming && params.streaming_chunk_size > 0)
                {
                    gpu_chunk_elements = params.streaming_chunk_size;
                }
            }
        }

        bool gpu_stage_used = false;
        bool gpu_compute_stage_executed = false;
        std::size_t gpu_bytes_moved = 0;
        double gpu_stage_time_ms = 0.0;
        double gpu_stage_ops = 0.0;
        std::string gpu_stage_details;
        KernelDiagnosticsCounters kernel_diagnostics{};
        std::vector<double> staged_points;

        const bool try_gpu_stage =
            gpu_available && (requested_mode == ExecutionMode::GPU_ONLY ||
                              requested_mode == ExecutionMode::HYBRID_GPU_PREFERRED ||
                              requested_mode == ExecutionMode::HYBRID_AUTO ||
                              (requested_mode == ExecutionMode::HYBRID_CPU_PREFERRED &&
                               n_points >= effective.acceleration.threshold));

        core::BufferView<const double> computePoints(points.data(), points.size());
        if (try_gpu_stage)
        {
            auto stage_result =
                stagePointsOnGpu(points, point_dim, effective.max_radius, gpu_chunk_elements);
            if (stage_result.isSuccess())
            {
                auto transfer = std::move(stage_result.value());
                gpu_stage_time_ms = transfer.gpu_time_ms;
                gpu_bytes_moved = transfer.bytes_moved;
                gpu_compute_stage_executed = transfer.graph_phase_executed;
                gpu_stage_ops = transfer.graph_phase_ops;
                kernel_diagnostics = transfer.kernel_diagnostics;
                if (transfer.graph_phase_executed)
                {
                    gpu_stage_details =
                        "sampled_radius_graph:pairs=" +
                        std::to_string(transfer.graph_phase_pairs_evaluated) +
                        ",points=" + std::to_string(transfer.graph_phase_sampled_points) +
                        ",tiles=" +
                        std::to_string(transfer.kernel_diagnostics.dimension_tiles_processed) +
                        ",pivots=" + std::to_string(transfer.kernel_diagnostics.pivot_hits);
                }
                else
                {
                    gpu_stage_details = "transfer_only";
                }
                staged_points = std::move(transfer.staged_points);
                computePoints =
                    core::BufferView<const double>(staged_points.data(), staged_points.size());
                gpu_stage_used = true;
            }
            else if (requested_mode == ExecutionMode::GPU_ONLY)
            {
                monitor_.endMonitoring("computeVrPersistence");
                return errors::ErrorResult<std::vector<Pair>>::error(stage_result.errorCode());
            }
        }

        const auto cpu_start = std::chrono::steady_clock::now();
        const auto base_config = to_base_fast_vr_config(effective);
        auto pairs =
            ::nerve::persistence::computeVrPersistenceFast(computePoints, point_dim, base_config);
        const auto cpu_end = std::chrono::steady_clock::now();
        const auto total_end = std::chrono::steady_clock::now();
        monitor_.endMonitoring("computeVrPersistence");

        const double total_ms =
            std::chrono::duration<double, std::milli>(total_end - total_start).count();
        const double cpu_ms =
            std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
        ExecutionMode actual_mode = ExecutionMode::CPU_ONLY;
        if (gpu_stage_used)
        {
            if (requested_mode == ExecutionMode::GPU_ONLY)
            {
                actual_mode = ExecutionMode::GPU_ONLY;
            }
            else if (requested_mode == ExecutionMode::HYBRID_CPU_PREFERRED)
            {
                actual_mode = ExecutionMode::HYBRID_CPU_PREFERRED;
            }
            else
            {
                actual_mode = ExecutionMode::HYBRID_GPU_PREFERRED;
            }
        }

        PerformanceMetrics metric;
        metric.total_time_ms = total_ms;
        metric.gpu_time_ms = gpu_stage_time_ms;
        metric.cpu_time_ms = cpu_ms;
        metric.problem_size = n_points;
        metric.point_dim = point_dim;
        metric.max_radius = effective.max_radius;
        metric.max_dim = effective.max_dim;
        metric.execution_mode = actual_mode;
        metric.gpu_work_ratio = effective.acceleration.gpu_work_ratio;
        metric.gpu_available = gpu_available;
        metric.success = true;
        metric.result_size = pairs.size();
        metric.error_code = errors::ErrorCode::SUCCESS;
        metric.timestamp = std::chrono::system_clock::now();
        metric.problem_ops =
            estimateProblemOps(metric.problem_size, metric.point_dim, metric.max_dim);
        metric.gpu_bytes = static_cast<double>(gpu_bytes_moved);
        metric.gpu_compute_stage_executed = gpu_compute_stage_executed;
        metric.gpu_stage_ops = gpu_stage_ops;
        metric.gpu_stage_details = std::move(gpu_stage_details);
        metric.kernel_diagnostics = kernel_diagnostics;
        monitor_.recordMetrics("computeVrPersistence", metric);

        const std::size_t point_bytes = points.size() * sizeof(double);
        const std::size_t pair_bytes = pairs.size() * sizeof(Pair);
        const std::size_t gpu_memory_bytes =
            gpu_stage_used ? std::max<std::size_t>(point_bytes, gpu_chunk_elements * sizeof(double))
                           : 0;
        const double memory_usage_mb = bytesToMb(point_bytes + pair_bytes + gpu_memory_bytes);
        const bool hybrid_mode = actual_mode == ExecutionMode::HYBRID_GPU_PREFERRED ||
                                 actual_mode == ExecutionMode::HYBRID_CPU_PREFERRED ||
                                 actual_mode == ExecutionMode::HYBRID_AUTO;
        recordGlobalMetric(metric, memory_usage_mb, gpu_stage_used, hybrid_mode);
        return errors::ErrorResult<std::vector<Pair>>::success(std::move(pairs));
    }

    AcceleratedPerformanceStats getPerformanceStats() const override
    {
        return monitor_.getAggregatedStats();
    }

    void updateConfig(const VRConfig &config) override
    {
        config_ = config.getEffectiveConfig();
        if (gpu_manager_)
        {
            gpu_manager_->updateConfig(config_);
        }
        if (optimizer_)
        {
            optimizer_->updateConfig(config_);
        }
    }

    VRConfig getConfig() const override { return config_; }
    bool isAvailable() const override { return gpu_manager_ && gpu_manager_->isAvailable(); }
    DeviceInfo getGpuInfo() const override
    {
        return gpu_manager_ ? gpu_manager_->getGpuInfo() : DeviceInfo{};
    }

    errors::ErrorResult<void> optimizePerformance() override
    {
        if (!optimizer_)
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E50_PH_ABORT);
        }
        return optimizer_->optimizeForCurrentSystem();
    }

    errors::ErrorResult<void> autoTuneParameters() override
    {
        if (!optimizer_)
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E50_PH_ABORT);
        }
        return optimizer_->autoTuneParameters();
    }

private:
    VRConfig config_;
    PerformanceMonitor monitor_;
    std::unique_ptr<GPUAccelerationManager> gpu_manager_;
    std::unique_ptr<PerformanceOptimizer> optimizer_;
};

} // namespace

errors::ErrorResult<std::unique_ptr<GPUAccelerationManager>>
makeDefaultGpuManager(const VRConfig &config)
{
    std::unique_ptr<GPUAccelerationManager> manager =
        std::make_unique<DefaultGpuAccelerationManager>(config);
    return errors::ErrorResult<std::unique_ptr<GPUAccelerationManager>>::success(
        std::move(manager));
}

errors::ErrorResult<std::unique_ptr<PerformanceOptimizer>>
makeDefaultPerformanceOptimizer(const VRConfig &config)
{
    std::unique_ptr<PerformanceOptimizer> optimizer =
        std::make_unique<DefaultPerformanceOptimizer>(config);
    return errors::ErrorResult<std::unique_ptr<PerformanceOptimizer>>::success(
        std::move(optimizer));
}

errors::ErrorResult<std::unique_ptr<AcceleratedVREngine>>
makeDefaultAcceleratedEngine(const VRConfig &config)
{
    std::unique_ptr<AcceleratedVREngine> engine = std::make_unique<AcceleratedVREngineImpl>(config);
    return errors::ErrorResult<std::unique_ptr<AcceleratedVREngine>>::success(std::move(engine));
}

} // namespace nerve::persistence::accelerated::detail

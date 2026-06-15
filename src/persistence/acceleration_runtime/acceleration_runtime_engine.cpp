
#include "nerve/persistence/acceleration_runtime/acceleration_runtime_engine.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <utility>

namespace nerve::persistence::acceleration_runtime
{

errors::ErrorResult<std::unique_ptr<AccelerationRuntimeVrEngine>>
AccelerationRuntimeVrEngine::create(const VRConfig &config)
{
    if (config.max_dim == 0)
    {
        return errors::ErrorResult<std::unique_ptr<AccelerationRuntimeVrEngine>>::error(
            errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }
    std::unique_ptr<AccelerationRuntimeVrEngine> engine(new AccelerationRuntimeVrEngine(config));
    return errors::ErrorResult<std::unique_ptr<AccelerationRuntimeVrEngine>>::success(
        std::move(engine));
}

errors::ErrorResult<std::vector<Pair>>
AccelerationRuntimeVrEngine::computeVrPersistence(const std::vector<double> &points,
                                                  std::size_t point_dim, const VRConfig &config)
{
    if (point_dim == 0 || points.empty() || points.size() % point_dim != 0)
    {
        return errors::ErrorResult<std::vector<Pair>>::error(
            errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }

    const auto start = std::chrono::steady_clock::now();
    core::BufferView<const double> pointsView(points.data(), points.size());

    // Call the existing computeVrPersistenceFast function
    std::vector<Pair> pairs;
    try
    {
        pairs = computeVrPersistenceFast(pointsView, point_dim, config);
    }
    catch (const std::exception &e)
    {
        return errors::ErrorResult<std::vector<Pair>>::error(errors::ErrorCode::E50_PH_ABORT);
    }

    const auto end = std::chrono::steady_clock::now();

    // Thread-safe performance statistics update
    {
        std::lock_guard<std::mutex> lock(performance_stats_mutex_);
        performance_stats_.computation_time_ms =
            std::chrono::duration<double, std::milli>(end - start).count();
        performance_stats_.operations_performed = pairs.size();
        performance_stats_.memory_used_bytes = points.size() * sizeof(double);
        performance_stats_.algorithm_used = "cpu_exact";
        performance_stats_.speedup_factor = DEFAULT_SPEEDUP_FACTOR;
        performance_stats_.optimization_details = "deterministic execution";
    }

    return errors::ErrorResult<std::vector<Pair>>::success(std::move(pairs));
}

AccelerationRuntimeStats AccelerationRuntimeVrEngine::getPerformanceStats() const
{
    std::lock_guard<std::mutex> lock(performance_stats_mutex_);
    return performance_stats_;
}

AccelerationRuntimeVrEngine::AccelerationRuntimeVrEngine(const VRConfig &config)
    : config_(config)
    , system_(SystemCapabilities::detectCapabilities())
    , performance_stats_()
{}

errors::ErrorResult<std::unique_ptr<AccelerationRuntimeVrEngine>>
AccelerationRuntimeEngineFactory::createOptimal()
{
    VRConfig config;
    config.use_accelerated_runtime = false;
    config.auto_detect_accelerated_runtime = false;
    config.acceleration.mode = AccelerationMode::CPU_ONLY;
    config.acceleration.gpu_work_ratio = 0.0;
    config.num_threads = SystemCapabilities::getOptimalThreadCount();
    return AccelerationRuntimeVrEngine::create(config);
}

errors::ErrorResult<std::unique_ptr<AccelerationRuntimeVrEngine>>
AccelerationRuntimeEngineFactory::createForUseCase(const std::string &use_case)
{
    // Input validation
    if (use_case.empty())
    {
        return errors::ErrorResult<std::unique_ptr<AccelerationRuntimeVrEngine>>::error(
            errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }

    VRConfig config;
    config.auto_detect_accelerated_runtime = false;
    config.use_accelerated_runtime = false;
    config.acceleration.mode = AccelerationMode::CPU_ONLY;
    config.acceleration.gpu_work_ratio = 0.0;

    if (use_case == "high_accuracy")
    {
        config.enable_approximation = false;
        config.approximation_level = ::nerve::common::ApproximationLevel::EXACT;
    }
    else if (use_case == "high_throughput")
    {
        config.enable_approximation = true;
        config.approximation_level = ::nerve::common::ApproximationLevel::LOW_PRECISION;
    }
    else if (use_case == "streaming")
    {
        config.acceleration.enable_streaming = true;
        config.chunk_size = DEFAULT_CHUNK_SIZE;
    }
    else
    {
        return errors::ErrorResult<std::unique_ptr<AccelerationRuntimeVrEngine>>::error(
            errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }

    return AccelerationRuntimeVrEngine::create(config);
}

errors::ErrorResult<std::unique_ptr<AccelerationRuntimeVrEngine>>
AccelerationRuntimeEngineFactory::createWithConfig(const VRConfig &config)
{
    return AccelerationRuntimeVrEngine::create(config);
}

errors::ErrorResult<bool> AccelerationRuntimeSystemValidator::validateSystem()
{
    const SystemCapabilities capabilities = SystemCapabilities::detectCapabilities();
    const bool valid = capabilities.num_cpu_cores > 0 && capabilities.total_memory > 0;
    if (!valid)
    {
        return errors::ErrorResult<bool>::error(errors::ErrorCode::E41_RESOURCE_LIMIT);
    }
    return errors::ErrorResult<bool>::success(true);
}

errors::ErrorResult<bool> AccelerationRuntimeSystemValidator::validateGpu()
{
    return errors::ErrorResult<bool>::success(SystemCapabilities::is_cuda_available());
}

errors::ErrorResult<bool> AccelerationRuntimeSystemValidator::validateThreading()
{
    const std::size_t thread_count = SystemCapabilities::getOptimalThreadCount();
    return errors::ErrorResult<bool>::success(thread_count > 0);
}

errors::ErrorResult<bool> AccelerationRuntimeSystemValidator::validateMemory()
{
    const SystemCapabilities capabilities = SystemCapabilities::detectCapabilities();
    const bool valid = capabilities.total_memory > 0 && capabilities.available_memory > 0;
    return errors::ErrorResult<bool>::success(static_cast<bool>(valid));
}

errors::ErrorResult<bool> AccelerationRuntimeSystemValidator::validateAll()
{
    const auto system = validateSystem();
    if (system.isError())
    {
        return errors::ErrorResult<bool>::error(system.errorCode());
    }
    const auto threading = validateThreading();
    if (threading.isError())
    {
        return errors::ErrorResult<bool>::error(threading.errorCode());
    }
    const auto memory = validateMemory();
    if (memory.isError())
    {
        return errors::ErrorResult<bool>::error(memory.errorCode());
    }
    return errors::ErrorResult<bool>::success(system.value() && threading.value() &&
                                              memory.value());
}

} // namespace nerve::persistence::acceleration_runtime

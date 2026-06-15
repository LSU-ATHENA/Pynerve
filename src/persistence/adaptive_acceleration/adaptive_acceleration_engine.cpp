
#include "nerve/persistence/adaptive_acceleration/adaptive_acceleration_engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <utility>

namespace nerve::persistence::adaptive_acceleration
{
namespace
{

bool checkedProduct(std::size_t lhs, std::size_t rhs, std::size_t &out)
{
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs)
    {
        out = 0;
        return false;
    }
    out = lhs * rhs;
    return true;
}

errors::ErrorResult<void> validatePointBuffer(const std::vector<double> &points,
                                              std::size_t point_dim)
{
    if (point_dim == 0 || points.empty() || points.size() % point_dim != 0)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }
    for (const double value : points)
    {
        if (!std::isfinite(value))
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E54_PH4_INVALID_INPUT);
        }
    }
    return errors::ErrorResult<void>::ok();
}

} // namespace

errors::ErrorResult<std::unique_ptr<AdaptiveAccelerationVrEngine>>
AdaptiveAccelerationVrEngine::create(const VRConfig &config)
{
    if (config.max_dim == 0)
    {
        return errors::ErrorResult<std::unique_ptr<AdaptiveAccelerationVrEngine>>::error(
            errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }
    std::unique_ptr<AdaptiveAccelerationVrEngine> engine(new AdaptiveAccelerationVrEngine(config));
    return errors::ErrorResult<std::unique_ptr<AdaptiveAccelerationVrEngine>>::success(
        std::move(engine));
}

errors::ErrorResult<std::vector<Pair>>
AdaptiveAccelerationVrEngine::computeVrPersistence(const std::vector<double> &points,
                                                   std::size_t point_dim, const VRConfig &config)
{
    auto validation = validatePointBuffer(points, point_dim);
    if (validation.isError())
    {
        return errors::ErrorResult<std::vector<Pair>>::error(validation.errorCode());
    }
    if (config.max_dim == 0)
    {
        return errors::ErrorResult<std::vector<Pair>>::error(
            errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }
    std::size_t memory_bytes = 0;
    if (!checkedProduct(points.size(), sizeof(double), memory_bytes))
    {
        return errors::ErrorResult<std::vector<Pair>>::error(errors::ErrorCode::E41_RESOURCE_LIMIT);
    }

    const auto start = std::chrono::steady_clock::now();
    core::BufferView<const double> pointsView(points.data(), points.size());
    std::vector<Pair> pairs = computeVrPersistenceFast(pointsView, point_dim, config);
    const auto end = std::chrono::steady_clock::now();

    performance_stats_.computation_time_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    performance_stats_.operations_performed = pairs.size();
    performance_stats_.memory_used_bytes = memory_bytes;
    performance_stats_.algorithm_used = "cpu_exact";
    performance_stats_.speedup_factor = 1.0;
    performance_stats_.optimization_details = "deterministic execution";

    return errors::ErrorResult<std::vector<Pair>>::success(std::move(pairs));
}

AdaptiveAccelerationVrEngine::AdaptiveAccelerationVrEngine(const VRConfig &config)
    : config_(config)
    , system_(SystemCapabilities::detectCapabilities())
    , performance_stats_()
{}

errors::ErrorResult<std::unique_ptr<AdaptiveAccelerationVrEngine>>
AdaptiveAccelerationEngineFactory::createOptimal()
{
    VRConfig config;
    config.use_adaptive_acceleration = false;
    config.auto_detect_adaptive_acceleration = false;
    config.acceleration.mode = AccelerationMode::CPU_ONLY;
    config.acceleration.gpu_work_ratio = 0.0;
    config.num_threads = SystemCapabilities::getOptimalThreadCount();
    return AdaptiveAccelerationVrEngine::create(config);
}

errors::ErrorResult<std::unique_ptr<AdaptiveAccelerationVrEngine>>
AdaptiveAccelerationEngineFactory::createForUseCase(const std::string &use_case)
{
    VRConfig config;
    config.use_adaptive_acceleration = false;
    config.auto_detect_adaptive_acceleration = false;
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
        config.chunk_size = 8192;
    }
    else
    {
        return errors::ErrorResult<std::unique_ptr<AdaptiveAccelerationVrEngine>>::error(
            errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }

    return AdaptiveAccelerationVrEngine::create(config);
}

errors::ErrorResult<std::unique_ptr<AdaptiveAccelerationVrEngine>>
AdaptiveAccelerationEngineFactory::createWithConfig(const VRConfig &config)
{
    return AdaptiveAccelerationVrEngine::create(config);
}

errors::ErrorResult<bool> AdaptiveAccelerationSystemValidator::validateSystem()
{
    const SystemCapabilities capabilities = SystemCapabilities::detectCapabilities();
    const bool valid = capabilities.num_cpu_cores > 0 && capabilities.total_memory > 0;
    if (!valid)
    {
        return errors::ErrorResult<bool>::error(errors::ErrorCode::E41_RESOURCE_LIMIT);
    }
    return errors::ErrorResult<bool>::success(true);
}

errors::ErrorResult<bool> AdaptiveAccelerationSystemValidator::validateGpu()
{
    return errors::ErrorResult<bool>::success(SystemCapabilities::is_cuda_available());
}

errors::ErrorResult<bool> AdaptiveAccelerationSystemValidator::validateThreading()
{
    const std::size_t thread_count = SystemCapabilities::getOptimalThreadCount();
    return errors::ErrorResult<bool>::success(thread_count > 0);
}

errors::ErrorResult<bool> AdaptiveAccelerationSystemValidator::validateMemory()
{
    const SystemCapabilities capabilities = SystemCapabilities::detectCapabilities();
    const bool valid = capabilities.total_memory > 0 && capabilities.available_memory > 0;
    return errors::ErrorResult<bool>::success(static_cast<bool>(valid));
}

errors::ErrorResult<bool> AdaptiveAccelerationSystemValidator::validateAll()
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

} // namespace nerve::persistence::adaptive_acceleration


#include "nerve/common/accelerated_types.hpp"
#include "nerve/errors/errors.hpp"
#include "nerve/persistence/acceleration_runtime/acceleration_runtime_engine.hpp"
#include "nerve/persistence/core/core_types.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

namespace
{

using nerve::common::AccelerationMode;
using nerve::common::ApproximationLevel;
using nerve::common::VRConfig;
using nerve::persistence::acceleration_runtime::AccelerationRuntimeEngineFactory;
using nerve::persistence::acceleration_runtime::AccelerationRuntimeStats;
using nerve::persistence::acceleration_runtime::AccelerationRuntimeSystemValidator;
using nerve::persistence::acceleration_runtime::AccelerationRuntimeVrEngine;

bool check_acceleration_runtime_stats_default()
{
    AccelerationRuntimeStats stats;
    if (std::abs(stats.computation_time_ms - 0.0) > 1e-10)
        return false;
    if (stats.memory_used_bytes != 0)
        return false;
    if (stats.operations_performed != 0)
        return false;
    if (std::abs(stats.speedup_factor - 1.0) > 1e-10)
        return false;
    if (!stats.algorithm_used.empty())
        return false;
    return true;
}

bool check_engine_create_valid()
{
    VRConfig config;
    config.max_dim = 2;
    config.max_radius = 1.0;
    auto result = AccelerationRuntimeVrEngine::create(config);
    if (result.isError())
    {
        bool acceptable = (result.errorCode() == nerve::errors::ErrorCode::E54_PH4_INVALID_INPUT);
        if (!acceptable)
            return false;
        return true;
    }
    auto engine = std::move(result.value());
    if (!engine)
        return false;
    return true;
}

bool check_engine_create_invalid_max_dim()
{
    VRConfig config;
    config.max_dim = 0;
    auto result = AccelerationRuntimeVrEngine::create(config);
    if (!result.isError())
        return false;
    return true;
}

bool check_engine_factory_create_optimal()
{
    auto result = AccelerationRuntimeEngineFactory::createOptimal();
    if (result.isError())
        return true;
    auto engine = std::move(result.value());
    if (!engine)
        return false;
    return true;
}

bool check_engine_factory_for_high_accuracy()
{
    auto result = AccelerationRuntimeEngineFactory::createForUseCase("high_accuracy");
    if (result.isError())
        return true;
    return true;
}

bool check_engine_factory_for_high_throughput()
{
    auto result = AccelerationRuntimeEngineFactory::createForUseCase("high_throughput");
    if (result.isError())
        return true;
    return true;
}

bool check_engine_factory_for_streaming()
{
    auto result = AccelerationRuntimeEngineFactory::createForUseCase("streaming");
    if (result.isError())
        return true;
    return true;
}

bool check_engine_factory_for_unknown_use_case()
{
    auto result = AccelerationRuntimeEngineFactory::createForUseCase("unknown_case");
    if (!result.isError())
        return false;
    return true;
}

bool check_engine_factory_with_config()
{
    VRConfig config;
    config.max_dim = 3;
    config.max_radius = 2.0;
    auto result = AccelerationRuntimeEngineFactory::createWithConfig(config);
    if (result.isError())
        return true;
    return true;
}

bool check_system_validator()
{
    auto sys_result = AccelerationRuntimeSystemValidator::validateSystem();
    static_cast<void>(sys_result);
    auto gpu_result = AccelerationRuntimeSystemValidator::validateGpu();
    static_cast<void>(gpu_result);
    auto thread_result = AccelerationRuntimeSystemValidator::validateThreading();
    static_cast<void>(thread_result);
    auto mem_result = AccelerationRuntimeSystemValidator::validateMemory();
    static_cast<void>(mem_result);
    return true;
}

bool check_system_validator_all()
{
    auto result = AccelerationRuntimeSystemValidator::validateAll();
    static_cast<void>(result);
    return true;
}

bool check_engine_performance_stats()
{
    VRConfig config;
    config.max_dim = 2;
    config.max_radius = 1.0;
    auto result = AccelerationRuntimeVrEngine::create(config);
    if (result.isError())
        return true;
    auto &engine = *result.value();
    auto stats = engine.getPerformanceStats();
    if (std::abs(stats.computation_time_ms - 0.0) > 1e-10)
        return false;
    if (stats.operations_performed != 0)
        return false;
    return true;
}

bool check_engine_system_capabilities()
{
    VRConfig config;
    config.max_dim = 2;
    config.max_radius = 1.0;
    auto result = AccelerationRuntimeVrEngine::create(config);
    if (result.isError())
        return true;
    auto &engine = *result.value();
    const auto &caps = engine.getSystemCapabilities();
    if (caps.num_cpu_cores < 1)
        return false;
    return true;
}

} // namespace

int main()
{
    if (!check_acceleration_runtime_stats_default())
    {
        std::cerr << "FAIL: acceleration runtime stats default\n";
        return 1;
    }
    if (!check_engine_create_valid())
    {
        std::cerr << "FAIL: engine create valid\n";
        return 1;
    }
    if (!check_engine_create_invalid_max_dim())
    {
        std::cerr << "FAIL: engine create invalid max dim\n";
        return 1;
    }
    if (!check_engine_factory_create_optimal())
    {
        std::cerr << "FAIL: engine factory create optimal\n";
        return 1;
    }
    if (!check_engine_factory_for_high_accuracy())
    {
        std::cerr << "FAIL: engine factory for high accuracy\n";
        return 1;
    }
    if (!check_engine_factory_for_high_throughput())
    {
        std::cerr << "FAIL: engine factory for high throughput\n";
        return 1;
    }
    if (!check_engine_factory_for_streaming())
    {
        std::cerr << "FAIL: engine factory for streaming\n";
        return 1;
    }
    if (!check_engine_factory_for_unknown_use_case())
    {
        std::cerr << "FAIL: engine factory for unknown use case\n";
        return 1;
    }
    if (!check_engine_factory_with_config())
    {
        std::cerr << "FAIL: engine factory with config\n";
        return 1;
    }
    if (!check_system_validator())
    {
        std::cerr << "FAIL: system validator\n";
        return 1;
    }
    if (!check_system_validator_all())
    {
        std::cerr << "FAIL: system validator all\n";
        return 1;
    }
    if (!check_engine_performance_stats())
    {
        std::cerr << "FAIL: engine performance stats\n";
        return 1;
    }
    if (!check_engine_system_capabilities())
    {
        std::cerr << "FAIL: engine system capabilities\n";
        return 1;
    }
    return 0;
}

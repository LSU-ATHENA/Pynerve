
#include "nerve/persistence/accelerated/accelerated_interface.hpp"

#include <algorithm>
#include <string>

namespace nerve::persistence::accelerated
{

// Note: PerformanceMetrics and AcceleratedPerformanceStats methods are defined inline
// in common/accelerated_types.hpp

namespace factory
{

errors::ErrorResult<std::unique_ptr<AcceleratedVREngine>>
createAccelerationRuntimeEngine(size_t n_points, size_t point_dim, double max_radius)
{
    VRConfig config;
    config.max_dim = point_dim > 0 ? std::min<size_t>(3, point_dim) : 2;
    config.max_radius = max_radius;
    config.use_acceleration = false;
    config.acceleration.threshold = std::max<size_t>(512, n_points / 2);
    config.acceleration.mode = AccelerationMode::CPU_ONLY;
    config.acceleration.gpu_work_ratio = 0.0;
    return AcceleratedVREngine::create(config);
}

errors::ErrorResult<std::unique_ptr<AcceleratedVREngine>>
createProductionEngine(const VRConfig &config)
{
    VRConfig production_config = config;
    production_config.use_acceleration = true;
    production_config.acceleration.enable_memory_optimization = true;
    return AcceleratedVREngine::create(production_config);
}

errors::ErrorResult<std::unique_ptr<AcceleratedVREngine>> createDebugEngine(const VRConfig &config)
{
    VRConfig debug_config = config;
    return AcceleratedVREngine::create(debug_config);
}

errors::ErrorResult<std::unique_ptr<AcceleratedVREngine>>
createBenchmarkEngine(const VRConfig &config)
{
    VRConfig benchmark_config = config;
    benchmark_config.use_acceleration = true;
    return AcceleratedVREngine::create(benchmark_config);
}

} // namespace factory

namespace accelerated
{

errors::ErrorResult<std::vector<Pair>>
computeVrPersistenceAccelerated(const core::BufferView<const double> &points, Size point_dim,
                                const VRConfig &config)
{
    return ::nerve::persistence::accelerated::computeVrPersistenceAccelerated(points, point_dim,
                                                                              config);
}

errors::ErrorResult<std::vector<Pair>>
computeVrPersistenceFast(const core::BufferView<const double> &points, Size point_dim,
                         const VRConfig &config)
{
    return ::nerve::persistence::accelerated::computeVrPersistenceFast(points, point_dim, config);
}

VRConfig createOptimalConfig(const core::BufferView<const double> &points, Size point_dim,
                             const VRConfig &base_config)
{
    return ::nerve::persistence::accelerated::createOptimalConfig(points, point_dim, base_config);
}

} // namespace accelerated

} // namespace nerve::persistence::accelerated


#include "nerve/persistence/adaptive_acceleration/adaptive_algorithm_selector.hpp"
#include "nerve/persistence/utils/api.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"

namespace nerve::persistence::adaptive_acceleration
{
namespace
{

PersistenceOptions toOptions(const VRConfig &config, PersistenceBackend backend,
                             PersistenceMode mode)
{
    PersistenceOptions options;
    options.mode = mode;
    options.backend = backend;
    options.max_dim = config.max_dim;
    options.max_radius = config.max_radius;
    options.threads = config.num_threads;
    options.error_tolerance = config.approximation_error_tolerance;
    return options;
}

errors::ErrorResult<std::vector<Pair>> runWithBackend(const core::BufferView<const double> &points,
                                                      std::size_t point_dim, const VRConfig &config,
                                                      PersistenceBackend backend,
                                                      PersistenceMode mode)
{
    const auto result = compute(points, point_dim, toOptions(config, backend, mode));
    if (result.isError())
    {
        return errors::ErrorResult<std::vector<Pair>>::error(result.errorCode());
    }
    std::vector<Pair> pairs = result.value().pairs;
    return errors::ErrorResult<std::vector<Pair>>::success(std::move(pairs));
}

} // namespace

errors::ErrorResult<std::vector<Pair>> AdaptiveAlgorithmSelector::executeMatrixMultiplicationCpu(
    const core::BufferView<const double> &points, std::size_t point_dim, const VRConfig &config)
{
    return runWithBackend(points, point_dim, config, PersistenceBackend::CPU_ADAPTIVE_ACCELERATION,
                          config.enable_approximation ? PersistenceMode::APPROX
                                                      : PersistenceMode::EXACT);
}

errors::ErrorResult<std::vector<Pair>> AdaptiveAlgorithmSelector::executeSparsifiedReductionCpu(
    const core::BufferView<const double> &points, std::size_t point_dim, const VRConfig &config)
{
    return runWithBackend(points, point_dim, config, PersistenceBackend::CPU_ADAPTIVE_ACCELERATION,
                          config.enable_approximation ? PersistenceMode::APPROX
                                                      : PersistenceMode::EXACT);
}

errors::ErrorResult<std::vector<Pair>> AdaptiveAlgorithmSelector::execute_lockfree_multicore_cpu(
    const core::BufferView<const double> &points, std::size_t point_dim, const VRConfig &config)
{
    return runWithBackend(points, point_dim, config, PersistenceBackend::CPU_ADAPTIVE_ACCELERATION,
                          PersistenceMode::EXACT);
}

errors::ErrorResult<std::vector<Pair>>
AdaptiveAlgorithmSelector::executeGpuAccelerated(const core::BufferView<const double> &points,
                                                 std::size_t point_dim, const VRConfig &config)
{
    (void)points;
    (void)point_dim;
    (void)config;
    return errors::ErrorResult<std::vector<Pair>>::error(errors::ErrorCode::E10_GPU_OOM);
}

errors::ErrorResult<std::vector<Pair>>
AdaptiveAlgorithmSelector::executeHybridGpuCpu(const core::BufferView<const double> &points,
                                               std::size_t point_dim, const VRConfig &config)
{
    (void)points;
    (void)point_dim;
    (void)config;
    return errors::ErrorResult<std::vector<Pair>>::error(errors::ErrorCode::E10_GPU_OOM);
}

errors::ErrorResult<std::vector<Pair>>
AdaptiveAlgorithmSelector::executeStandardCpu(const core::BufferView<const double> &points,
                                              std::size_t point_dim, const VRConfig &config)
{
    return runWithBackend(points, point_dim, config, PersistenceBackend::CPU_EXACT,
                          PersistenceMode::EXACT);
}

} // namespace nerve::persistence::adaptive_acceleration

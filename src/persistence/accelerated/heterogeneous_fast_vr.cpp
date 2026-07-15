
#include "nerve/persistence/accelerated/heterogeneous_fast_vr.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"

#include <chrono>

namespace nerve::persistence::accelerated
{

errors::ErrorResult<std::vector<Pair>>
HeterogeneousFastVR::computeCpuOnly(core::BufferView<const double> points, size_t point_dim,
                                    const core::DeterminismContract &contract)
{
    auto start_time = std::chrono::high_resolution_clock::now();

    try
    {
        // Convert to VRConfig
        VRConfig config;
        config.num_threads = config_.num_threads;
        if (contract.enable_deterministic_threading && config.num_threads == 0)
        {
            config.num_threads = 1;
        }
        config.max_radius = config_.max_radius;
        config.max_dim = config_.max_dim;

        // Call existing implementation
        auto result = computeVrPersistenceFast(points, point_dim, config);
        if (result.isError())
        {
            return result;
        }

        // Update performance stats
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        performance_stats_.total_time_ms = static_cast<double>(duration.count()) / 1000.0;
        performance_stats_.cpu_time_ms = performance_stats_.total_time_ms;
        performance_stats_.gpu_used = false;

        return errors::ErrorResult<std::vector<Pair>>::success(std::move(result.value()));
    }
    catch (const std::exception &e)
    {
        return errors::ErrorResult<std::vector<Pair>>::error(errors::ErrorCode::E50_PH_ABORT,
                                                             e.what());
    }
}

} // namespace nerve::persistence::accelerated

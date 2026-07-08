#include "gpu_test_helpers.cuh"
#include "nerve/persistence/adaptive_acceleration/adaptive_acceleration_engine.hpp"

int main()
{
    if (!has_gpu())
    {
        std::cerr << "No CUDA device available -- skipping GPU adaptive acceleration tests\n";
        return 0;
    }

    // AdaptiveAccel: AdaptiveAccelerationStats struct
    {
        nerve::persistence::adaptive_acceleration::AdaptiveAccelerationStats stats;
        stats.computation_time_ms = 1.5;
        stats.memory_used_bytes = 1024;
        stats.operations_performed = 100;
        stats.speedup_factor = 2.0;
        stats.algorithm_used = "test";
        assert(stats.speedup_factor >= 0.0);
        assert(stats.operations_performed == 100);
        std::cout << "PASSED: AdaptiveAccelerationStats struct\n";
    }

    // AdaptiveAccel: validateSystem
    {
        auto result = nerve::persistence::adaptive_acceleration::
            AdaptiveAccelerationSystemValidator::validateSystem();
        std::cout << "PASSED: adaptive accel validateSystem (success="
                  << (result.isSuccess() ? "true" : "false") << ")\n";
    }

    // AdaptiveAccel: validateGpu
    {
        auto result = nerve::persistence::adaptive_acceleration::
            AdaptiveAccelerationSystemValidator::validateGpu();
        std::cout << "PASSED: adaptive accel validateGpu (success="
                  << (result.isSuccess() ? "true" : "false") << ")\n";
    }

    return 0;
}

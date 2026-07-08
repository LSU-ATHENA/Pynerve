#include "gpu_test_helpers.cuh"
#include "nerve/runtime/gpu_calibration.hpp"

int main()
{
    if (!has_gpu())
    {
        std::cerr << "No CUDA device available -- skipping GPU calibration tests\n";
        return 0;
    }

    // Calibration: benchmarkCalibration
    {
        auto bench = nerve::runtime::gpu::benchmarkCalibration(100, 10);
        assert(bench.num_samples == 100);
        assert(bench.num_buckets == 10);
        assert(bench.cpu_update_ms >= 0.0);
        assert(bench.gpu_update_ms >= 0.0);
        assert(bench.cpu_query_ms >= 0.0);
        assert(bench.gpu_query_ms >= 0.0);
        assert(bench.speedup_update >= 0.0);
        assert(bench.speedup_query >= 0.0);
        std::cout << "PASSED: benchmarkCalibration (100 samples, 10 buckets)\n";
    }

    return 0;
}

#include "gpu_test_helpers.cuh"
#include "nerve/probabilistic/gpu_probabilistic.hpp"

int main()
{
    if (!has_gpu())
    {
        std::cerr
            << "No CUDA device available -- skipping GPU probabilistic kernel coverage tests\n";
        return 0;
    }

    // Probabilistic: benchmarkProbabilistic
    {
        auto bench = nerve::probabilistic::gpu::benchmarkProbabilistic(100);
        assert(bench.num_samples == 100);
        assert(bench.cpu_time_ms >= 0.0);
        assert(bench.gpu_time_ms >= 0.0);
        assert(bench.accuracy >= 0.0f);
        std::cout << "PASSED: benchmarkProbabilistic (100 samples, speedup=" << bench.speedup
                  << ")\n";
    }

    // Probabilistic: benchmarkProbabilistic larger
    {
        auto bench = nerve::probabilistic::gpu::benchmarkProbabilistic(500);
        assert(bench.num_samples == 500);
        assert(bench.cpu_time_ms >= 0.0);
        assert(bench.gpu_time_ms >= 0.0);
        std::cout << "PASSED: benchmarkProbabilistic (500 samples, speedup=" << bench.speedup
                  << ")\n";
    }

    return 0;
}

#include "gpu_test_helpers.cuh"
#include "nerve/streaming/gpu_streaming.hpp"

int main()
{
    if (!has_gpu())
    {
        std::cerr << "No CUDA device available -- skipping GPU streaming kernel coverage tests\n";
        return 0;
    }

    // Streaming: benchmarkWindowed
    {
        auto bench = nerve::streaming::gpu::benchmarkWindowed(10, 5);
        assert(bench.window_size == 10);
        assert(bench.num_windows == 5);
        assert(bench.cpu_time_ms >= 0.0);
        assert(bench.gpu_time_ms >= 0.0);
        std::cout << "PASSED: benchmarkWindowed (window=10, windows=5)\n";
    }

    return 0;
}

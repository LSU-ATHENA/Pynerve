#include "gpu_test_helpers.cuh"
#include "nerve/encoders/gpu_encoders.hpp"

#if __has_include(<cudnn.h>)
#define NERVE_TEST_HAS_CUDNN 1
#endif

int main()
{
    if (!has_gpu())
    {
        std::cerr << "No CUDA device available -- skipping GPU encoders kernel coverage tests\n";
        return 0;
    }

    // Encoders: benchmarkGPUEncoder
    {
        auto bench = nerve::encoders::gpu::benchmarkGPUEncoder(16, 64);
        assert(bench.batch_size == 16);
        assert(bench.num_features == 64);
        assert(bench.cpu_time_ms >= 0.0);
        assert(bench.gpu_time_ms >= 0.0);
        assert(bench.speedup_gpu >= 0.0);
        std::cout << "PASSED: benchmarkGPUEncoder (batch=16, features=64)\n";
    }

    // Encoders: benchmarkFusedEncoder
    {
        auto bench = nerve::encoders::fusion::benchmarkFusedEncoder(8, 16);
        assert(bench.num_diagrams == 8);
        assert(bench.features_per_diagram == 16);
        assert(bench.unfused_time_ms >= 0.0);
        assert(bench.fused_time_ms >= 0.0);
        std::cout << "PASSED: benchmarkFusedEncoder (8 diagrams, 16 features)\n";
    }

#if NERVE_TEST_HAS_CUDNN
    // Encoders: benchmarkTensorCore
    {
        auto bench = nerve::encoders::tensorcore::benchmarkTensorCore(16, 64, 32);
        assert(bench.batch_size == 16);
        assert(bench.input_dim == 64);
        assert(bench.output_dim == 32);
        assert(bench.fp32_time_ms >= 0.0);
        assert(bench.fp16_time_ms >= 0.0);
        std::cout << "PASSED: benchmarkTensorCore (64->32, batch=16)\n";
    }
#else
    std::cout << "SKIPPED: benchmarkTensorCore (cudnn.h not available)\n";
#endif

    return 0;
}

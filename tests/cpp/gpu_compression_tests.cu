#include "gpu_test_helpers.cuh"
#include "nerve/compression/gpu_autoencoder.hpp"

#if __has_include(<cudnn.h>)
#define NERVE_TEST_HAS_CUDNN 1
#endif

int main()
{
    if (!has_gpu())
    {
        std::cerr << "No CUDA device available -- skipping GPU compression kernel coverage tests\n";
        return 0;
    }

    // Compression: benchmarkAutoencoder
    {
        auto bench = nerve::compression::gpu::benchmarkAutoencoder(64, 16, 8);
        assert(bench.input_dim == 64);
        assert(bench.latent_dim == 16);
        assert(bench.batch_size == 8);
        assert(bench.cpu_encode_ms >= 0.0);
        assert(bench.gpu_encode_ms >= 0.0);
        assert(bench.speedup_encode >= 0.0);
        std::cout << "PASSED: benchmarkAutoencoder (64->16, batch=8)\n";
    }

    // Compression: benchmarkAutoencoder with different config
    {
        auto bench = nerve::compression::gpu::benchmarkAutoencoder(32, 4, 16);
        assert(bench.input_dim == 32);
        assert(bench.latent_dim == 4);
        assert(bench.batch_size == 16);
        assert(bench.cpu_encode_ms >= 0.0);
        std::cout << "PASSED: benchmarkAutoencoder (32->4, batch=16)\n";
    }

#if NERVE_TEST_HAS_CUDNN
    // Compression: benchmarkCuDNNEncoder
    {
        auto bench = nerve::compression::gpu::benchmarkCuDNNEncoder(8, 32, 8);
        assert(bench.batch_size == 8);
        assert(bench.input_dim == 32);
        assert(bench.latent_dim == 8);
        assert(bench.cpu_time_ms >= 0.0);
        assert(bench.cudnn_time_ms >= 0.0);
        std::cout << "PASSED: benchmarkCuDNNEncoder (32->8, batch=8)\n";
    }

    // Compression: benchmarkCuDNNDecoder
    {
        auto bench = nerve::compression::gpu::benchmarkCuDNNDecoder(8, 8, 32);
        assert(bench.batch_size == 8);
        assert(bench.latent_dim == 8);
        assert(bench.output_dim == 32);
        assert(bench.cpu_time_ms >= 0.0);
        assert(bench.cudnn_time_ms >= 0.0);
        std::cout << "PASSED: benchmarkCuDNNDecoder (8->32, batch=8)\n";
    }
#else
    std::cout << "SKIPPED: cuDNN benchmarks (cudnn.h not available)\n";
#endif

    return 0;
}

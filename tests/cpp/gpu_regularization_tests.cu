#include "gpu_test_helpers.cuh"
#include "nerve/regularization/gpu_regularization.hpp"

int main()
{
    if (!has_gpu())
    {
        std::cerr
            << "No CUDA device available -- skipping GPU regularization kernel coverage tests\n";
        return 0;
    }

    // Regularization: benchmarkRegularizer
    {
        auto bench = nerve::regularization::gpu::benchmarkRegularizer(16, 8);
        assert(bench.num_pairs == 16);
        assert(bench.feature_dim == 8);
        assert(bench.cpu_betti_ms >= 0.0);
        assert(bench.gpu_betti_ms >= 0.0);
        assert(bench.speedup_betti >= 0.0);
        assert(bench.speedup_augment >= 0.0);
        std::cout << "PASSED: benchmarkRegularizer (16 pairs, dim=8)\n";
    }

    // Regularization: benchmarkAugmentation
    {
        auto bench = nerve::regularization::gpu::benchmarkAugmentation(32, 16, 4);
        assert(bench.num_samples == 32);
        assert(bench.feature_dim == 16);
        assert(bench.num_augmentations == 4);
        assert(bench.cpu_time_ms >= 0.0);
        assert(bench.gpu_time_ms >= 0.0);
        std::cout << "PASSED: benchmarkAugmentation (32 samples, dim=16, 4 augs)\n";
    }

    // Regularization: benchmarkLoss
    {
        auto bench = nerve::regularization::gpu::benchmarkLoss(4, 32);
        assert(bench.num_betti_dims == 4);
        assert(bench.num_persistence_pairs == 32);
        assert(bench.cpu_time_ms >= 0.0);
        assert(bench.gpu_time_ms >= 0.0);
        std::cout << "PASSED: benchmarkLoss (4 dims, 32 pairs)\n";
    }

    return 0;
}

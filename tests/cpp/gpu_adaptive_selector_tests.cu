#include "gpu_test_helpers.cuh"
#include "nerve/persistence/cuda/gpu_adaptive_selector.hpp"

int main()
{
    if (!has_gpu())
    {
        std::cerr << "No CUDA device available -- skipping GPU adaptive selector tests\n";
        return 0;
    }

    // AdaptiveSelector: SelectorConfig defaults
    {
        nerve::persistence::gpu::SelectorConfig cfg;
        assert(cfg.num_algorithms == 5);
        assert(cfg.feature_dim == 20);
        assert(cfg.hidden_dim == 64);
        std::cout << "PASSED: SelectorConfig defaults\n";
    }

    // AdaptiveSelector: benchmarkSelector
    {
        auto bench = nerve::persistence::gpu::benchmarkSelector(64, 5);
        assert(bench.num_pairs == 64);
        assert(bench.num_algorithms == 5);
        assert(bench.cpu_feature_ms >= 0.0);
        assert(bench.gpu_feature_ms >= 0.0);
        assert(bench.cpu_predict_ms >= 0.0);
        assert(bench.gpu_predict_ms >= 0.0);
        std::cout << "PASSED: benchmarkSelector (64 pairs, 5 algorithms)\n";
    }

    return 0;
}

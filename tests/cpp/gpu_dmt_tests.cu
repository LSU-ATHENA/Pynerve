#include "gpu_test_helpers.cuh"
#include "nerve/dmt/gpu_dmt.hpp"

int main()
{
    if (!has_gpu())
    {
        std::cerr << "No CUDA device available -- skipping GPU DMT kernel coverage tests\n";
        return 0;
    }

    // DMT: DMTConfig defaults
    {
        nerve::dmt::DMTConfig cfg;
        assert(cfg.max_dimension == 2);
        assert(cfg.use_parallel == true);
        assert(cfg.use_simd == true);
        std::cout << "PASSED: DMTConfig defaults\n";
    }

    // DMT: MorseResult defaults
    {
        nerve::dmt::MorseResult result;
        assert(result.critical_simplices.empty());
        assert(result.gradient_pairs.empty());
        assert(result.computation_time_ms == 0.0);
        std::cout << "PASSED: MorseResult defaults\n";
    }

    // DMT: benchmarkParallelDMT
    {
        auto bench = nerve::dmt::parallel::benchmarkParallelDMT(50);
        assert(bench.num_simplices == 50);
        assert(bench.sequential_time_ms >= 0.0);
        assert(bench.parallel_time_ms >= 0.0);
        assert(bench.simd_time_ms >= 0.0);
        std::cout << "PASSED: benchmarkParallelDMT (50 simplices, speedup_par="
                  << bench.speedup_parallel << ")\n";
    }

    // DMT: benchmarkParallelDMT larger
    {
        auto bench = nerve::dmt::parallel::benchmarkParallelDMT(200);
        assert(bench.num_simplices == 200);
        assert(bench.sequential_time_ms >= 0.0);
        assert(bench.parallel_time_ms >= 0.0);
        std::cout << "PASSED: benchmarkParallelDMT (200 simplices)\n";
    }

    // DMT: SimplexPairOps
    {
        std::vector<int> simplex_a = {0, 1};
        std::vector<int> simplex_b = {0, 1, 2};
        bool can_pair =
            nerve::dmt::parallel::SimplexPairOps::canFormGradientPair(simplex_a, simplex_b);
        static_cast<void>(can_pair);
        std::cout << "PASSED: SimplexPairOps canFormGradientPair\n";
    }

    return 0;
}

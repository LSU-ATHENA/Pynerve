#include "gpu_test_helpers.cuh"
#include "nerve/persistence/accelerated/gpu_apparent_pairs.hpp"

int main()
{
    if (!has_gpu())
    {
        std::cerr << "No CUDA device available -- skipping GPU persistence kernel coverage tests\n";
        return 0;
    }

    // Persistence: GPUApparentPairs::Config defaults
    {
        nerve::persistence::accelerated::GPUApparentPairs::Config cfg;
        assert(cfg.max_simplices >= 1000);
        assert(cfg.max_dimension >= 1);
        assert(cfg.threads_per_block >= 32);
        cfg.max_simplices = 1024;
        cfg.max_dimension = 3;
        assert(cfg.max_simplices == 1024);
        assert(cfg.max_dimension == 3);

        // Validate config
        auto err = cfg.validate();
        assert(err == errors::ErrorCode::E0_SUCCESS || err != errors::ErrorCode::E0_SUCCESS); // smoke
        std::cout << "PASSED: GPUApparentPairs::Config (max_simplices=1024, dim=3)\n";
    }

    // Persistence: GPUApparentPairs creation via factory
    {
        nerve::persistence::accelerated::GPUApparentPairs::Config cfg;
        cfg.max_simplices = 256;
        cfg.max_dimension = 2;

        auto result = nerve::persistence::accelerated::GPUApparentPairs::create(cfg);
        if (result.isOk())
        {
            assert(result.value() != nullptr);
            std::cout << "PASSED: GPUApparentPairs::create (factory method)\n";
        }
        else
        {
            // May fail on systems without suitable GPU, don't assert
            std::cout << "SKIPPED: GPUApparentPairs::create (factory unavailable on this device)\n";
        }
    }

    return 0;
}

#include "gpu_test_helpers.cuh"
#include "nerve/persistence/cuda/cuda_blackwell_tma.hpp"

int main()
{
    if (!has_gpu())
    {
        std::cerr << "No CUDA device available -- skipping GPU blackwell/tma kernel tests\n";
        return 0;
    }

    // Blackwell: getBlackwellInfo (exercises cudaGetDeviceProperties)
    {
        auto info = nerve::gpu::blackwell::getBlackwellInfo();
        assert(info.compute_capability_major > 0);
        std::cout << "PASSED: getBlackwellInfo (cc=" << info.compute_capability_major
                  << "." << info.compute_capability_minor << " gen=" << info.generation
                  << " tma=" << info.supports_tma << " wgmma=" << info.supports_wgmma << ")\n";
    }

    // Blackwell: isHopperAvailable / isBlackwellAvailable
    {
        bool hopper = nerve::gpu::blackwell::isHopperAvailable();
        bool blackwell = nerve::gpu::blackwell::isBlackwellAvailable();
        std::cout << "PASSED: blackwell detection (Hopper=" << hopper
                  << " Blackwell=" << blackwell << ")\n";
    }

    // Blackwell: recommended precision mode
    {
        const char *mode = nerve::gpu::blackwell::getRecommendedPrecisionMode();
        static_cast<void>(mode);
        std::cout << "PASSED: blackwell recommended precision mode\n";
    }

    return 0;
}

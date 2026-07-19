#include "gpu_test_helpers.cuh"
#include "nerve/gpu/gpu_capability_core.hpp"

int main()
{
    if (!has_gpu())
    {
        std::cerr << "No CUDA device available -- skipping GPU capability core tests\n";
        return 0;
    }

    // Capability: AdvancedCapabilities::detect (real CUDA calls)
    {
        auto caps = nerve::gpu::advanced::AdvancedCapabilities::detect();
        assert(caps.cuda_available);
        assert(caps.device_count >= 1);
        assert(caps.compute_capability_major > 0);
        assert(caps.sm_count > 0);
        std::cout << "PASSED: AdvancedCapabilities::detect (cc=" << caps.computeCapability()
                  << " sm=" << caps.sm_count << " arch=" << caps.architectureName() << ")\n";
    }

    // Capability: architecture detection
    {
        auto caps = nerve::gpu::advanced::AdvancedCapabilities::detect();
        int cc = caps.computeCapability();
        if (cc >= 110)
            assert(caps.isRubinCpx());
        else if (cc >= 100)
            assert(caps.isBlackwell());
        else if (cc >= 90)
            assert(caps.isHopper());
        std::cout << "PASSED: GPU architecture classification\n";
    }

    // Capability: feature support flags
    {
        auto caps = nerve::gpu::advanced::AdvancedCapabilities::detect();
        std::cout << "PASSED: GPU features (PTX=" << caps.supportsPTXOptimizations()
                  << " WGMMA=" << caps.supportsWGMMA() << " clusters=" << caps.supportsClusters()
                  << " FP4=" << caps.supportsFP4() << ")\n";
    }

    return 0;
}

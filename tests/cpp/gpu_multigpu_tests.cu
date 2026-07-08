#include "gpu_test_helpers.cuh"
#include "nerve/persistence/cuda/cuda_multi_gpu.hpp"

int main()
{
    if (!has_gpu())
    {
        std::cerr << "No CUDA device available -- skipping GPU multi-gpu kernel tests\n";
        return 0;
    }

    // MultiGPU: getMultiGpuInfo (exercises cudaGetDeviceCount)
    {
        auto info = nerve::gpu::multi::getMultiGpuInfo();
        assert(info.num_gpus >= 1);
        assert(info.gpus.size() >= 1);
        std::cout << "PASSED: getMultiGpuInfo (num_gpus=" << info.num_gpus
                  << " nvlink=" << info.nvlink_available << ")\n";
    }

    // MultiGPU: shouldUseMultiGpu
    {
        bool use = nerve::gpu::multi::shouldUseMultiGpu(10000, 3);
        static_cast<void>(use);
        std::cout << "PASSED: shouldUseMultiGpu (10k points, dim=3)=" << use << "\n";
    }

    // MultiGPU: recommendedNumGpus (inline)
    {
        int num = nerve::gpu::multi::recommendedNumGpus(1000, 2);
        assert(num >= 0);
        std::cout << "PASSED: recommendedNumGpus (1k points, dim=2)=" << num << "\n";
    }

    return 0;
}

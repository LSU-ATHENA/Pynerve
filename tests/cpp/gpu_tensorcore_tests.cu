#include "gpu_test_helpers.cuh"
#include "nerve/persistence/cuda/cuda_tensor_core.hpp"

int main()
{
    if (!has_gpu())
    {
        std::cerr << "No CUDA device available -- skipping GPU tensor core kernel tests\n";
        return 0;
    }

    // TensorCore: getTensorCoreInfo (exercises cudaGetDeviceProperties)
    {
        auto info = nerve::gpu::tensorcore::getTensorCoreInfo();
        std::cout << "PASSED: getTensorCoreInfo (available=" << info.available
                  << " gen=" << info.generation << " fp16=" << info.supports_fp16
                  << " bf16=" << info.supports_bf16 << " tf32=" << info.supports_tf32
                  << " fp8=" << info.supports_fp8 << ")\n";
    }

    // TensorCore: areTensorCoresAvailable
    {
        bool avail = nerve::gpu::tensorcore::areTensorCoresAvailable();
        std::cout << "PASSED: areTensorCoresAvailable=" << avail << "\n";
    }

    // TensorCore: recommended precision mode (inline)
    {
        const char *mode = nerve::gpu::tensorcore::getRecommendedPrecisionMode();
        static_cast<void>(mode);
        std::cout << "PASSED: tensor core recommended precision mode\n";
    }

    return 0;
}

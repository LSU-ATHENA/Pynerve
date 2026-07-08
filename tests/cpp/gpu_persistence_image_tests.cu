#include "gpu_test_helpers.cuh"
#include "nerve/gpu/persistence_image.cuh"

#include <vector>

int main()
{
    if (!has_gpu())
    {
        std::cerr << "No CUDA device available -- skipping GPU persistence image kernel tests\n";
        return 0;
    }

    // PersistenceImage: compute kernel smoke
    {
        std::vector<float> births = {0.0f, 0.2f, 0.1f};
        std::vector<float> deaths = {1.0f, 0.8f, 0.9f};
        bool called = false;
        nerve::gpu::persistence_image::compute_persistence_image_gpu(
            births, deaths, 16, 0.1f,
            [&](const std::vector<std::vector<double>> &result) {
                assert(!result.empty());
                called = true;
            });
        assert(called);
        std::cout << "PASSED: persistence image GPU (3 pts, resolution=16)\n";
    }

    return 0;
}

#include "gpu_test_helpers.cuh"
#include "nerve/gpu/bottleneck_distance.cuh"
#include "nerve/gpu/wasserstein_distance.cuh"
#include "nerve/math/persistence_metrics/point2d.hpp"

#include <vector>

int main()
{
    if (!has_gpu())
    {
        std::cerr << "No CUDA device available -- skipping GPU distance kernel coverage tests\n";
        return 0;
    }

    // Bottleneck: compute kernel smoke
    {
        std::vector<nerve::math::Point2D> d1 = {{0.0, 1.0}, {0.2, 0.8}};
        std::vector<nerve::math::Point2D> d2 = {{0.0, 1.0}, {0.2, 0.8}};
        bool called = false;
        nerve::gpu::bottleneck::compute_bottleneck_distance_gpu(d1, d2, [&](double distance) {
            assert(distance >= 0.0);
            called = true;
        });
        assert(called);
        std::cout << "PASSED: bottleneck distance GPU (2 pairs)\n";
    }

    // Wasserstein: sinkhorn kernel smoke
    {
        std::vector<nerve::math::Point2D> d1 = {{0.0, 1.0}};
        std::vector<nerve::math::Point2D> d2 = {{0.0, 1.0}};
        bool called = false;
        nerve::gpu::wasserstein::compute_sinkhorn_distance_gpu(d1, d2, 2.0, 0.01, 50,
                                                               [&](double distance) {
                                                                   assert(distance >= 0.0);
                                                                   called = true;
                                                               });
        assert(called);
        std::cout << "PASSED: sinkhorn distance GPU (1 pair)\n";
    }

    return 0;
}

#include "gpu_test_helpers.cuh"
#include "nerve/gpu/mapper_gpu.cuh"

#include <utility>
#include <vector>

int main()
{
    if (!has_gpu())
    {
        std::cerr << "No CUDA device available -- skipping GPU mapper kernel coverage tests\n";
        return 0;
    }

    // Mapper: density filter kernel smoke
    {
        std::vector<float> points(12); // 4 points in 3D
        for (int i = 0; i < 12; ++i)
            points[i] = static_cast<float>(i);
        bool called = false;
        nerve::gpu::mapper::compute_density_filter_gpu(points, 4, 3, 2,
                                                       [&](std::vector<float> result) {
                                                           assert(!result.empty());
                                                           called = true;
                                                       });
        assert(called);
        std::cout << "PASSED: mapper density filter (4 pts, 3D, k=2)\n";
    }

    // Mapper: eccentricity filter kernel smoke
    {
        std::vector<float> points(8); // 4 points in 2D
        for (int i = 0; i < 8; ++i)
            points[i] = static_cast<float>(i);
        bool called = false;
        nerve::gpu::mapper::compute_eccentricity_filter_gpu(points, 4, 2,
                                                            [&](std::vector<float> result) {
                                                                assert(!result.empty());
                                                                called = true;
                                                            });
        assert(called);
        std::cout << "PASSED: mapper eccentricity filter (4 pts, 2D)\n";
    }

    // Mapper: kmeans clustering kernel smoke
    {
        std::vector<float> points(8); // 4 points in 2D
        for (int i = 0; i < 8; ++i)
            points[i] = static_cast<float>(i);
        bool called = false;
        nerve::gpu::mapper::compute_kmeans_clustering_gpu(points, 4, 2, 2, 10,
                                                          [&](std::vector<int> result) {
                                                              assert(!result.empty());
                                                              called = true;
                                                          });
        assert(called);
        std::cout << "PASSED: mapper kmeans clustering (4 pts, 2D, k=2)\n";
    }

    return 0;
}

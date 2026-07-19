#include "gpu_test_helpers.cuh"
#include "nerve/gpu/distance_fasted.cuh"
#include "nerve/gpu/distance_tedjoin.cuh"
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

    // DistanceFastEd: launch kernel smoke with small point cloud
    {
        std::vector<float> points = {0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.1f, 0.1f};
        int n_points = 3;
        int dim = 3;
        std::vector<float> distances(n_points * n_points, -1.0f);

        nerve::gpu::FastedConfig config;
        config.warp_tile_m = 32;
        config.warp_tile_n = 32;
        config.warp_tile_k = 8;
        config.block_tile_m = 64;
        config.block_tile_n = 64;
        config.pipeline_stages = 1;

        cudaError_t err = nerve::gpu::launchDistanceFastEd(points.data(), n_points, dim,
                                                           distances.data(), n_points, config);
        assert(err == cudaSuccess);
        assert(distances[0] == 0.0f);
        std::cout << "PASSED: distanceFastEd (3 pts, 3D, custom tiles)\n";
    }

    // Tedjoin: launchFp64TensorDistance kernel smoke
    {
        std::vector<double> points = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0};
        int n_points = 2;
        int dim = 3;
        std::vector<double> distances(n_points * n_points, -1.0);

        cudaError_t err = nerve::gpu::tedjoin::launchFp64TensorDistance(
            points.data(), n_points, dim, distances.data(), n_points);
        assert(err == cudaSuccess);
        assert(distances[0] >= 0.0);
        std::cout << "PASSED: tedjoin fp64 tensor distance (2 pts, 3D)\n";
    }

    // Wasserstein: auction distance kernel smoke
    {
        std::vector<nerve::math::Point2D> d1 = {{0.0, 1.0}, {0.1, 0.9}};
        std::vector<nerve::math::Point2D> d2 = {{0.0, 1.0}, {0.1, 0.9}};
        bool called = false;
        nerve::gpu::wasserstein::compute_auction_distance_gpu(d1, d2, 2.0, [&](double distance) {
            assert(distance >= 0.0);
            called = true;
        });
        assert(called);
        std::cout << "PASSED: wasserstein auction distance GPU (2 pairs)\n";
    }

    return 0;
}

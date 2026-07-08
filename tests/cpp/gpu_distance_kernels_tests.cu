#include "gpu_test_helpers.cuh"
#include "nerve/gpu/distance_kernels.hpp"

#include <vector>

int main()
{
    if (!has_gpu())
    {
        std::cerr << "No CUDA device available -- skipping GPU distance kernel coverage tests\n";
        return 0;
    }

    // DistanceKernels: launch_pairwise_distance_radius_f32
    {
        std::vector<float> points = {0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f};
        int n_points = 3;
        int dim = 2;
        std::vector<float> distances(n_points * n_points, -1.0f);
        float radius = 10.0f;

        cudaError_t err = nerve::gpu::launch_pairwise_distance_radius_f32(
            points.data(), n_points, distances.data(), n_points, n_points, dim, radius);
        assert(err == cudaSuccess);
        assert(distances[0] == 0.0f); // self-distance
        std::cout << "PASSED: pairwise_distance_radius_f32 (3 pts, 2D, r=10)\\n";
    }

    // DistanceKernels: launch_pairwise_distance_radius_f64
    {
        std::vector<double> points = {0.0, 0.0, 3.0, 0.0, 0.0, 4.0};
        int n_points = 3;
        int dim = 2;
        std::vector<double> distances(n_points * n_points, -1.0);
        double radius = 10.0;

        cudaError_t err = nerve::gpu::launch_pairwise_distance_radius_f64(
            points.data(), n_points, distances.data(), n_points, n_points, dim, radius);
        assert(err == cudaSuccess);
        assert(distances[0] == 0.0); // self-distance
        std::cout << "PASSED: pairwise_distance_radius_f64 (3 pts, 2D, r=10)\\n";
    }

    return 0;
}

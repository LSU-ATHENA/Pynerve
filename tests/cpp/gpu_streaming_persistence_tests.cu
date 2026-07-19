#include "gpu_test_helpers.cuh"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

// The launch* wrappers are defined in streaming_persistence_cuda.cu under
// namespace nerve::gpu::streaming::kernels.  Include the whole file as a
// single translation unit (same pattern as other GPU tests).
#include "../../src/streaming/gpu/streaming_persistence_cuda.cu"

int main()
{
    if (!has_gpu())
    {
        std::cout << "SKIP: no GPU available" << std::endl;
        return 0;
    }

    // Test: launchAffectedRegionDetection
    //
    // 4 points in 2D: (0,0), (3,4), (6,8), (0,10)
    // New point: (2, 3)  -- should only affect the first point
    // max_radius_sq = 25  (radius 5)
    //
    // Distance from (2,3) to each point:
    //   (0,0):  (2-0)^2 + (3-0)^2  = 4 + 9   = 13  -> affected (13 <= 25)
    //   (3,4):  (2-3)^2 + (3-4)^2  = 1 + 1   = 2   -> affected (2 <= 25)
    //   (6,8):  (2-6)^2 + (3-8)^2  = 16 + 25 = 41  -> NOT affected
    //   (0,10): (2-0)^2 + (3-10)^2 = 4 + 49  = 53  -> NOT affected
    {
        const int n_points = 4;
        const int point_dim = 2;
        const double max_radius_sq = 25.0;

        std::vector<double> h_points = {0.0, 0.0, 3.0, 4.0, 6.0, 8.0, 0.0, 10.0};
        std::vector<double> h_new_point = {2.0, 3.0};
        std::vector<int> h_affected_mask(n_points, -1);

        // Allocate GPU memory
        double *d_points = nullptr;
        double *d_new_point = nullptr;
        int *d_affected_mask = nullptr;

        cudaMalloc(&d_points, n_points * point_dim * sizeof(double));
        cudaMalloc(&d_new_point, point_dim * sizeof(double));
        cudaMalloc(&d_affected_mask, n_points * sizeof(int));

        cudaMemcpy(d_points, h_points.data(), n_points * point_dim * sizeof(double),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_new_point, h_new_point.data(), point_dim * sizeof(double),
                   cudaMemcpyHostToDevice);

        cudaStream_t stream;
        cudaStreamCreate(&stream);

        // Run kernel
        nerve::gpu::streaming::kernels::launchAffectedRegionDetection(
            d_points, d_new_point, d_affected_mask, n_points, point_dim, max_radius_sq, stream);

        cudaStreamSynchronize(stream);

        // Copy results back
        cudaMemcpy(h_affected_mask.data(), d_affected_mask, n_points * sizeof(int),
                   cudaMemcpyDeviceToHost);

        // Verify: points 0 and 1 affected, points 2 and 3 not
        assert(h_affected_mask[0] == 1);
        assert(h_affected_mask[1] == 1);
        assert(h_affected_mask[2] == 0);
        assert(h_affected_mask[3] == 0);

        std::cout << "PASS: launchAffectedRegionDetection correctly identified affected points"
                  << std::endl;

        // Cleanup
        cudaStreamDestroy(stream);
        cudaFree(d_points);
        cudaFree(d_new_point);
        cudaFree(d_affected_mask);
    }

    // Test: launchAffectedRegionDetection null-safety
    // All four launch wrappers should silently return on nullptr inputs
    {
        const int n_points = 1;
        const int point_dim = 1;
        double *d_points = nullptr;
        double *d_new_point = nullptr;
        int *d_mask = nullptr;

        cudaMalloc(&d_points, sizeof(double));
        cudaMalloc(&d_new_point, sizeof(double));
        cudaMalloc(&d_mask, sizeof(int));

        cudaStream_t stream;
        cudaStreamCreate(&stream);

        // Null new_point -- should return silently
        nerve::gpu::streaming::kernels::launchAffectedRegionDetection(
            d_points, nullptr, d_mask, n_points, point_dim, 1.0, stream);

        // Null points -- should return silently
        nerve::gpu::streaming::kernels::launchAffectedRegionDetection(
            nullptr, d_new_point, d_mask, n_points, point_dim, 1.0, stream);

        // Null mask -- should return silently
        nerve::gpu::streaming::kernels::launchAffectedRegionDetection(
            d_points, d_new_point, nullptr, n_points, point_dim, 1.0, stream);

        // Non-finite radius -- should return silently
        nerve::gpu::streaming::kernels::launchAffectedRegionDetection(
            d_points, d_new_point, d_mask, n_points, point_dim, INFINITY, stream);

        // Negative radius -- should return silently
        nerve::gpu::streaming::kernels::launchAffectedRegionDetection(
            d_points, d_new_point, d_mask, n_points, point_dim, -1.0, stream);

        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
        cudaFree(d_points);
        cudaFree(d_new_point);
        cudaFree(d_mask);

        std::cout << "PASS: launchAffectedRegionDetection null-safety checks passed" << std::endl;
    }

    // Test: launchBirthDeathUpdate on a simple simplex
    {
        const int n_points = 3;
        const int n_simplices = 1;
        const int max_simplex_size = 3;

        // Distance matrix (3x3)
        std::vector<double> h_dist = {0.0, 3.0, 4.0, 3.0, 0.0, 5.0, 4.0, 5.0, 0.0};

        // One simplex (triangle) with vertices {0, 1, 2}
        std::vector<int> h_vertices = {0, 1, 2};
        std::vector<int> h_sizes = {3};
        std::vector<int> h_affected = {0};
        std::vector<double> h_birth(n_simplices, -1.0);
        std::vector<double> h_death(n_simplices, -1.0);

        double *d_dist = nullptr;
        int *d_vertices = nullptr;
        int *d_sizes = nullptr;
        int *d_affected = nullptr;
        double *d_birth = nullptr;
        double *d_death = nullptr;

        cudaMalloc(&d_dist, n_points * n_points * sizeof(double));
        cudaMalloc(&d_vertices, n_simplices * max_simplex_size * sizeof(int));
        cudaMalloc(&d_sizes, n_simplices * sizeof(int));
        cudaMalloc(&d_affected, sizeof(int));
        cudaMalloc(&d_birth, n_simplices * sizeof(double));
        cudaMalloc(&d_death, n_simplices * sizeof(double));

        cudaMemcpy(d_dist, h_dist.data(), n_points * n_points * sizeof(double),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_vertices, h_vertices.data(), n_simplices * max_simplex_size * sizeof(int),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_sizes, h_sizes.data(), n_simplices * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_affected, h_affected.data(), sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_birth, h_birth.data(), n_simplices * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_death, h_death.data(), n_simplices * sizeof(double), cudaMemcpyHostToDevice);

        cudaStream_t stream;
        cudaStreamCreate(&stream);

        nerve::gpu::streaming::kernels::launchBirthDeathUpdate(d_affected, 1, d_dist, d_birth,
                                                               d_death, n_points, d_vertices,
                                                               d_sizes, max_simplex_size, stream);

        cudaStreamSynchronize(stream);

        cudaMemcpy(h_birth.data(), d_birth, n_simplices * sizeof(double), cudaMemcpyDeviceToHost);

        // Max edge length of triangle is 5.0
        double expected_birth = 5.0;
        assert(std::fabs(h_birth[0] - expected_birth) < 1e-6);
        (void)expected_birth;

        std::cout << "PASS: launchBirthDeathUpdate computed birth time " << h_birth[0] << std::endl;

        cudaStreamDestroy(stream);
        cudaFree(d_dist);
        cudaFree(d_vertices);
        cudaFree(d_sizes);
        cudaFree(d_affected);
        cudaFree(d_birth);
        cudaFree(d_death);
    }

    std::cout << "PASS: all streaming persistence tests passed" << std::endl;
    return 0;
}

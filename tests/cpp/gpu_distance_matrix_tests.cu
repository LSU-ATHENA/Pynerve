#include "gpu_test_helpers.cuh"

int main()
{
    if (!has_gpu())
    {
        std::cerr
            << "No CUDA device available -- skipping GPU distance matrix kernel coverage tests\n";
        return 0;
    }
    // Distance matrix: verify correctness on known triangle
    {
        CUDADistanceMatrixConfig config;
        config.enable_simd = false;
        config.enable_shared_memory = false;
        config.enable_streaming = false;

        std::vector<double> distances;
        assert(run_distance_matrix_config(kTrianglePoints, kTriangleN, kTriangleDim, 10.0, config,
                                          distances));
        assert(distances.size() == 9);
        assert(approx_equal(distances[0 * kTriangleN + 0], 0.0)); // self
        assert(approx_equal(distances[0 * kTriangleN + 1], 3.0)); // (0,0)-(3,0)
        assert(approx_equal(distances[0 * kTriangleN + 2], 4.0)); // (0,0)-(0,4)
        assert(approx_equal(distances[1 * kTriangleN + 0], 3.0)); // symmetric
        assert(approx_equal(distances[1 * kTriangleN + 1], 0.0));
        assert(approx_equal(distances[1 * kTriangleN + 2], 5.0)); // (3,0)-(0,4)
        assert(approx_equal(distances[2 * kTriangleN + 0], 4.0)); // symmetric
        assert(approx_equal(distances[2 * kTriangleN + 1], 5.0)); // symmetric
        assert(approx_equal(distances[2 * kTriangleN + 2], 0.0));

        std::cout << "PASSED: distance matrix triangle correctness (basic kernel)\n";
    }

    // Distance matrix: verify max_radius filtering (reject edges > radius)
    {
        CUDADistanceMatrixConfig config;
        config.enable_simd = false;
        config.enable_shared_memory = false;
        config.enable_streaming = false;

        std::vector<double> distances;
        // max_radius=4.0 should reject d(1,2)=5.0 but accept d(0,1)=3.0 and d(0,2)=4.0
        assert(run_distance_matrix_config(kTrianglePoints, kTriangleN, kTriangleDim, 4.0, config,
                                          distances));
        assert(approx_equal(distances[0 * kTriangleN + 1], 3.0));
        assert(approx_equal(distances[0 * kTriangleN + 2], 4.0));
        // d(1,2) should be -1.0 (rejected by radius)
        assert(distances[1 * kTriangleN + 2] < 0.0);

        std::cout << "PASSED: distance matrix max_radius filtering\n";
    }

    // Distance matrix: simd kernel determinism (dim=8 to trigger simd path)
    {
        constexpr int n = 4;
        constexpr int dim = 8;
        double points[n * dim];
        for (int i = 0; i < n * dim; ++i)
            points[i] = static_cast<double>(i + 1) * 0.1;

        CUDADistanceMatrixConfig config;
        config.enable_simd = true;
        config.enable_shared_memory = false;
        config.enable_streaming = false;

        std::vector<double> ref;
        assert(run_distance_matrix_config(points, n, dim, 100.0, config, ref));

        for (int run = 1; run < 10; ++run)
        {
            std::vector<double> cur;
            assert(run_distance_matrix_config(points, n, dim, 100.0, config, cur));
            for (size_t i = 0; i < ref.size(); ++i)
                assert(approx_equal(ref[i], cur[i]));
        }

        std::cout << "PASSED: distance matrix simd kernel determinism (10 runs, dim=8)\n";
    }

    // Distance matrix: shared memory kernel determinism (dim=16 to trigger shared path)
    {
        constexpr int n = 4;
        constexpr int dim = 16;
        double points[n * dim];
        for (int i = 0; i < n * dim; ++i)
            points[i] = static_cast<double>(i + 1) * 0.05;

        CUDADistanceMatrixConfig config;
        config.enable_simd = false;
        config.enable_shared_memory = true;
        config.enable_streaming = false;

        std::vector<double> ref;
        assert(run_distance_matrix_config(points, n, dim, 100.0, config, ref));

        for (int run = 1; run < 10; ++run)
        {
            std::vector<double> cur;
            assert(run_distance_matrix_config(points, n, dim, 100.0, config, cur));
            for (size_t i = 0; i < ref.size(); ++i)
                assert(approx_equal(ref[i], cur[i]));
        }

        std::cout << "PASSED: distance matrix shared kernel determinism (10 runs, dim=16)\n";
    }

    // Distance matrix: streaming kernel determinism (with actual streaming path)
    {
        constexpr int n = 4;
        constexpr int dim = 4;
        double points[n * dim];
        for (int i = 0; i < n * dim; ++i)
            points[i] = static_cast<double>(i + 1) * 0.2;

        CUDADistanceMatrixConfig config;
        config.enable_simd = false;
        config.enable_shared_memory = false;
        config.enable_streaming = true;

        // Allocate device memory directly
        size_t points_bytes = n * dim * sizeof(double);
        size_t dist_bytes = n * n * sizeof(double);
        double *d_points = nullptr;
        double *d_distances = nullptr;
        assert(check_cuda(cudaMalloc(&d_points, points_bytes), "cudaMalloc d_points"));
        assert(check_cuda(cudaMalloc(&d_distances, dist_bytes), "cudaMalloc d_distances"));
        cudaMemcpy(d_points, points, points_bytes, cudaMemcpyHostToDevice);

        // First run: full matrix as streaming (offset=0, size=n*n)
        Size total = static_cast<Size>(n) * static_cast<Size>(n);
        auto result =
            cuda_host::launchDistanceMatrixKernel(d_points, d_distances, static_cast<Size>(n),
                                                  static_cast<Size>(dim), 100.0, config, 0, total);
        assert(!result.isError());

        std::vector<double> ref(dist_bytes / sizeof(double));
        cudaMemcpy(ref.data(), d_distances, dist_bytes, cudaMemcpyDeviceToHost);

        // 9 more runs with streaming: verify identical results
        for (int run = 1; run < 10; ++run)
        {
            result = cuda_host::launchDistanceMatrixKernel(
                d_points, d_distances, static_cast<Size>(n), static_cast<Size>(dim), 100.0, config,
                0, total);
            assert(!result.isError());

            std::vector<double> cur(dist_bytes / sizeof(double));
            cudaMemcpy(cur.data(), d_distances, dist_bytes, cudaMemcpyDeviceToHost);
            for (size_t i = 0; i < ref.size(); ++i)
                assert(approx_equal(ref[i], cur[i]));
        }

        cudaFree(d_points);
        cudaFree(d_distances);

        std::cout << "PASSED: distance matrix streaming kernel determinism (10 runs)\n";
    }

    // Distance matrix: 4-point correctness across all kernel paths
    {
        // Known distances: d(0,1)=sqrt(2)~=1.414, d(0,2)=2.0, d(0,3)=3.0
        // d(1,2)=sqrt(2)~=1.414, d(1,3)=sqrt(5)~=2.236, d(2,3)=sqrt(10)~=3.162

        // Test basic kernel
        {
            CUDADistanceMatrixConfig config;
            config.enable_simd = false;
            config.enable_shared_memory = false;
            config.enable_streaming = false;

            std::vector<double> d;
            assert(run_distance_matrix_config(kQuadPoints, kQuadN, kQuadDim, 10.0, config, d));
            assert(approx_equal(d[0 * kQuadN + 0], 0.0));
            assert(approx_equal(d[0 * kQuadN + 1], std::sqrt(2.0)));
            assert(approx_equal(d[0 * kQuadN + 2], 2.0));
            assert(approx_equal(d[0 * kQuadN + 3], 3.0));
            assert(approx_equal(d[1 * kQuadN + 2], std::sqrt(2.0)));
            assert(approx_equal(d[1 * kQuadN + 3], std::sqrt(5.0)));
            assert(approx_equal(d[2 * kQuadN + 3], std::sqrt(10.0)));
        }

        // Test simd kernel (dim=2 is below the 4-dim threshold, so it routes to basic,
        // but we verify the same results via the high-level path that auto-selects)
        {
            CUDADistanceMatrixConfig config;
            config.enable_simd = true;
            config.enable_shared_memory = false;
            config.enable_streaming = false;

            std::vector<double> d;
            assert(run_distance_matrix_config(kQuadPoints, kQuadN, kQuadDim, 10.0, config, d));
            assert(approx_equal(d[0 * kQuadN + 1], std::sqrt(2.0)));
            assert(approx_equal(d[1 * kQuadN + 3], std::sqrt(5.0)));
        }

        std::cout << "PASSED: distance matrix 4-point correctness (basic + auto-select paths)\n";
    }

    // Distance matrix: exhaustive triangle matrix verification (all 9 cells)
    {
        CUDADistanceMatrixConfig config;
        config.enable_simd = false;
        config.enable_shared_memory = false;
        config.enable_streaming = false;

        std::vector<double> d;
        assert(
            run_distance_matrix_config(kTrianglePoints, kTriangleN, kTriangleDim, 10.0, config, d));

        // Expected distance matrix (3x3, row-major):
        //   [ 0,  3,  4 ]
        //   [ 3,  0,  5 ]
        //   [ 4,  5,  0 ]
        const double expected[] = {0.0, 3.0, 4.0, 3.0, 0.0, 5.0, 4.0, 5.0, 0.0};
        for (int i = 0; i < kTriangleN * kTriangleN; ++i)
            assert(approx_equal(d[static_cast<size_t>(i)], expected[i]));

        std::cout << "PASSED: distance matrix exhaustive triangle (all 9 cells exact)\n";
    }

    // Distance matrix: symmetry property verification
    {
        constexpr int n = 5;
        constexpr int dim = 2;
        double points[n * dim];
        for (int i = 0; i < n; ++i)
        {
            points[i * dim + 0] = static_cast<double>(i) * 2.0;
            points[i * dim + 1] = static_cast<double>(i % 3) * 1.5;
        }

        CUDADistanceMatrixConfig config;
        config.enable_simd = false;
        config.enable_shared_memory = false;
        config.enable_streaming = false;

        std::vector<double> d;
        assert(run_distance_matrix_config(points, n, dim, 100.0, config, d));

        // Verify symmetry: d[i][j] == d[j][i] for all i, j
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                assert(approx_equal(d[i * n + j], d[j * n + i]));
            }
            // Verify self-distance is zero
            assert(approx_equal(d[i * n + i], 0.0));
        }

        std::cout << "PASSED: distance matrix symmetry (5x5, all pairs)\n";
    }

    // Distance matrix: 1D points (collinear) correctness
    {
        // Points at x = 1, 4, 9 on a line
        // Distances: d(0,1)=3, d(0,2)=8, d(1,2)=5
        const double line_points[] = {1.0, 4.0, 9.0};
        constexpr int n = 3;
        constexpr int dim = 1;

        CUDADistanceMatrixConfig config;
        config.enable_simd = false;
        config.enable_shared_memory = false;
        config.enable_streaming = false;

        std::vector<double> d;
        assert(run_distance_matrix_config(line_points, n, dim, 100.0, config, d));
        assert(approx_equal(d[0 * n + 0], 0.0));
        assert(approx_equal(d[0 * n + 1], 3.0));
        assert(approx_equal(d[0 * n + 2], 8.0));
        assert(approx_equal(d[1 * n + 0], 3.0));
        assert(approx_equal(d[1 * n + 1], 0.0));
        assert(approx_equal(d[1 * n + 2], 5.0));
        assert(approx_equal(d[2 * n + 0], 8.0));
        assert(approx_equal(d[2 * n + 1], 5.0));
        assert(approx_equal(d[2 * n + 2], 0.0));

        std::cout << "PASSED: distance matrix 1D collinear points (all 9 cells)\n";
    }

    // Distance matrix: 3D unit cube corners correctness
    {
        // 8 corners of a unit cube: all combinations of (0,1) in 3D
        const double cube_points[] = {
            0.0, 0.0, 0.0, // 0
            1.0, 0.0, 0.0, // 1
            0.0, 1.0, 0.0, // 2
            0.0, 0.0, 1.0, // 3
            1.0, 1.0, 0.0, // 4
            1.0, 0.0, 1.0, // 5
            0.0, 1.0, 1.0, // 6
            1.0, 1.0, 1.0, // 7
        };
        constexpr int n = 8;
        constexpr int dim = 3;

        CUDADistanceMatrixConfig config;
        config.enable_simd = false;
        config.enable_shared_memory = false;
        config.enable_streaming = false;

        std::vector<double> d;
        assert(run_distance_matrix_config(cube_points, n, dim, 100.0, config, d));

        // Helper: count differing coordinates between two cube corner indices
        auto hamming = [](int a, int b) {
            int diff = 0;
            for (int bit = 0; bit < dim; ++bit)
                if ((a & (1 << bit)) != (b & (1 << bit)))
                    ++diff;
            return diff;
        };

        // Verify all 64 cells match sqrt(hamming_distance)
        for (int i = 0; i < n; ++i)
        {
            assert(approx_equal(d[i * n + i], 0.0));
            for (int j = i + 1; j < n; ++j)
            {
                double expected = std::sqrt(static_cast<double>(hamming(i, j)));
                assert(approx_equal(d[i * n + j], expected));
                assert(approx_equal(d[j * n + i], expected));
            }
        }

        std::cout << "PASSED: distance matrix 3D unit cube (all 64 cells, sqrt(hamming))\n";
    }

    // Distance matrix: cross-kernel consistency (basic, simd, shared paths)
    {
        // dim=16 triggers both simd (>= 4) and shared (>= 8) kernel paths
        constexpr int n = 4;
        constexpr int dim = 16;
        double points[n * dim];
        for (int i = 0; i < n * dim; ++i)
            points[i] = static_cast<double>((i + 1) * (i % 3 + 1)) * 0.25;

        // Basic kernel
        CUDADistanceMatrixConfig basic_cfg;
        basic_cfg.enable_simd = false;
        basic_cfg.enable_shared_memory = false;
        basic_cfg.enable_streaming = false;
        std::vector<double> basic;
        assert(run_distance_matrix_config(points, n, dim, 100.0, basic_cfg, basic));

        // SIMD kernel (dim >= 4 triggers 4-lane unrolled path)
        CUDADistanceMatrixConfig simd_cfg;
        simd_cfg.enable_simd = true;
        simd_cfg.enable_shared_memory = false;
        simd_cfg.enable_streaming = false;
        std::vector<double> simd;
        assert(run_distance_matrix_config(points, n, dim, 100.0, simd_cfg, simd));

        // Shared memory kernel (dim >= 8 triggers shared tiled path)
        CUDADistanceMatrixConfig shared_cfg;
        shared_cfg.enable_simd = false;
        shared_cfg.enable_shared_memory = true;
        shared_cfg.enable_streaming = false;
        std::vector<double> shared;
        assert(run_distance_matrix_config(points, n, dim, 100.0, shared_cfg, shared));

        // All 3 paths must produce identical results (streaming tested separately)
        for (size_t i = 0; i < basic.size(); ++i)
        {
            assert(approx_equal(basic[i], simd[i]));
            assert(approx_equal(basic[i], shared[i]));
        }

        std::cout << "PASSED: distance matrix cross-kernel consistency (3 paths, all identical)\n";
    }

    // Distance matrix: high-dimensional correctness (dim=16, verify key entries)
    {
        constexpr int n = 3;
        constexpr int dim = 16;
        double points[n * dim];
        // Point 0: all zeros
        for (int d = 0; d < dim; ++d)
            points[0 * dim + d] = 0.0;
        // Point 1: all ones -> distance from 0 is sqrt(16) = 4.0
        for (int d = 0; d < dim; ++d)
            points[1 * dim + d] = 1.0;
        // Point 2: alternating 3 and 0 -> distance from 0 is sqrt(8*9) = sqrt(72) ~= 8.485
        for (int d = 0; d < dim; ++d)
            points[2 * dim + d] = (d % 2 == 0) ? 3.0 : 0.0;

        CUDADistanceMatrixConfig config;
        config.enable_simd = false;
        config.enable_shared_memory = true; // dim=16 triggers shared path
        config.enable_streaming = false;

        std::vector<double> d;
        assert(run_distance_matrix_config(points, n, dim, 100.0, config, d));

        assert(approx_equal(d[0 * n + 0], 0.0));
        assert(approx_equal(d[1 * n + 1], 0.0));
        assert(approx_equal(d[2 * n + 2], 0.0));
        // d(0,1) = sqrt(16 * 1^2) = 4.0
        assert(approx_equal(d[0 * n + 1], 4.0));
        assert(approx_equal(d[1 * n + 0], 4.0));
        // d(0,2) = sqrt(8 * 3^2) = sqrt(72)
        assert(approx_equal(d[0 * n + 2], std::sqrt(72.0)));
        assert(approx_equal(d[2 * n + 0], std::sqrt(72.0)));
        // d(1,2) = sqrt(8*(1-3)^2 + 8*(1-0)^2) = sqrt(8*4 + 8*1) = sqrt(40)
        assert(approx_equal(d[1 * n + 2], std::sqrt(40.0)));
        assert(approx_equal(d[2 * n + 1], std::sqrt(40.0)));

        std::cout << "PASSED: distance matrix high-dim shared kernel (dim=16, exact sqrt values)\n";
    }

    // Distance matrix: all identical points (degenerate -- zero distances)
    {
        constexpr int n = 5;
        constexpr int dim = 3;
        // All 5 points at (0,0,0)
        double points[n * dim] = {};

        CUDADistanceMatrixConfig config;
        config.enable_simd = false;
        config.enable_shared_memory = false;
        config.enable_streaming = false;

        std::vector<double> d;
        assert(run_distance_matrix_config(points, n, dim, 100.0, config, d));

        // Verify all pairwise distances are zero
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                // Self-distances and all pairwise distances must be zero
                assert(approx_equal(d[i * n + j], 0.0, 1e-12));
            }
        }
        // Explicitly check symmetry holds for degenerate case
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j)
                assert(approx_equal(d[i * n + j], d[j * n + i]));

        std::cout << "PASSED: distance matrix all identical points (all zeros, 5x5)\n";
    }

    // Distance matrix: all identical points with simd path (dim=8 to trigger simd)
    {
        constexpr int n = 3;
        constexpr int dim = 8;
        double points[n * dim] = {};
        for (int i = 0; i < n * dim; ++i)
            points[i] = 42.0; // All coordinates identical but non-zero

        CUDADistanceMatrixConfig config;
        config.enable_simd = true;
        config.enable_shared_memory = false;
        config.enable_streaming = false;

        std::vector<double> d;
        assert(run_distance_matrix_config(points, n, dim, 100.0, config, d));

        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                assert(approx_equal(d[i * n + j], 0.0, 1e-12));

        std::cout << "PASSED: distance matrix all identical points simd path (dim=8, all 42.0)\n";
    }

    // Distance matrix: all identical points with shared path (dim=16, high-dim zeros)
    {
        constexpr int n = 4;
        constexpr int dim = 16;
        double points[n * dim] = {};
        // All zeros -- every point at origin in 16D

        CUDADistanceMatrixConfig config;
        config.enable_simd = false;
        config.enable_shared_memory = true;
        config.enable_streaming = false;

        std::vector<double> d;
        assert(run_distance_matrix_config(points, n, dim, 100.0, config, d));

        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                assert(approx_equal(d[i * n + j], 0.0, 1e-12));

        std::cout
            << "PASSED: distance matrix all identical points shared path (dim=16, all zero)\n";
    }

    // Distance matrix: collinear points with duplicate coordinates
    {
        // Points at x = 1, 1, 4, 4, 9 -- duplicates at (1,1) and (4,4)
        const double line_points[] = {1.0, 1.0, 4.0, 4.0, 9.0};
        constexpr int n = 5;
        constexpr int dim = 1;

        CUDADistanceMatrixConfig config;
        config.enable_simd = false;
        config.enable_shared_memory = false;
        config.enable_streaming = false;

        std::vector<double> d;
        assert(run_distance_matrix_config(line_points, n, dim, 100.0, config, d));

        // Self-distances
        for (int i = 0; i < n; ++i)
            assert(approx_equal(d[i * n + i], 0.0));

        // Distance between duplicates must be zero
        assert(approx_equal(d[0 * n + 1], 0.0)); // x=1, x=1
        assert(approx_equal(d[1 * n + 0], 0.0)); // symmetric
        assert(approx_equal(d[2 * n + 3], 0.0)); // x=4, x=4
        assert(approx_equal(d[3 * n + 2], 0.0)); // symmetric

        // Non-duplicate distances
        assert(approx_equal(d[0 * n + 2], 3.0)); // x=1 to x=4
        assert(approx_equal(d[0 * n + 4], 8.0)); // x=1 to x=9
        assert(approx_equal(d[2 * n + 4], 5.0)); // x=4 to x=9

        // Verify all 25 cells are symmetric
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                assert(approx_equal(d[i * n + j], d[j * n + i]));

        std::cout << "PASSED: distance matrix collinear with duplicates (5 points, 2 dup pairs)\n";
    }

    // Distance matrix: raw kernel NaN handling (bypasses API validation)
    {
        constexpr int n = 3;
        constexpr int dim = 2;
        double points[n * dim];
        // Point 0: (0, 0) -- normal
        points[0] = 0.0;
        points[1] = 0.0;
        // Point 1: (NaN, 0) -- NaN in x coordinate
        points[2] = std::numeric_limits<double>::quiet_NaN();
        points[3] = 0.0;
        // Point 2: (3, 4) -- normal
        points[4] = 3.0;
        points[5] = 4.0;

        CUDADistanceMatrixConfig config;
        config.enable_simd = false;
        config.enable_shared_memory = false;
        config.enable_streaming = false;

        std::vector<double> d;
        // The raw kernel path does NOT validate for NaN -- the kernel itself
        // uses isfinite() checks and returns device_inf() for invalid distances.
        assert(run_distance_matrix_config(points, n, dim, 100.0, config, d));

        // The kernel's nan-guard (isfinite check on diffs) converts
        // NaN inputs to device_inf() -- output is always inf, never NaN.
        // d(0,2): both points normal -> should be finite
        assert(std::isfinite(d[0 * n + 2]));
        assert(std::isfinite(d[2 * n + 0]));
        assert(approx_equal(d[0 * n + 2], 5.0)); // (0,0)-(3,4) = 5

        // d(0,1): involves NaN point -> kernel nan-guard produces inf
        assert(std::isinf(d[0 * n + 1]));
        assert(std::isinf(d[1 * n + 0]));

        // d(1,1): self-distance of NaN point -> NaN-NaN=NaN triggers guard -> inf
        assert(std::isinf(d[1 * n + 1]));

        // d(1,2): involves NaN point -> inf
        assert(std::isinf(d[1 * n + 2]));
        assert(std::isinf(d[2 * n + 1]));

        std::cout << "PASSED: distance matrix NaN handling (nan-guard produces inf)\n";
    }

    // Distance matrix: raw kernel Inf handling (bypasses API validation)
    {
        constexpr int n = 3;
        constexpr int dim = 2;
        double points[n * dim];
        // Point 0: (0, 0) -- normal
        points[0] = 0.0;
        points[1] = 0.0;
        // Point 1: (inf, 0) -- Inf in x coordinate
        points[2] = std::numeric_limits<double>::infinity();
        points[3] = 0.0;
        // Point 2: (3, 4) -- normal
        points[4] = 3.0;
        points[5] = 4.0;

        CUDADistanceMatrixConfig config;
        config.enable_simd = false;
        config.enable_shared_memory = false;
        config.enable_streaming = false;

        std::vector<double> d;
        assert(run_distance_matrix_config(points, n, dim, 100.0, config, d));

        // d(0,2): both normal -> should be finite and correct
        assert(std::isfinite(d[0 * n + 2]));
        assert(approx_equal(d[0 * n + 2], 5.0));

        // d(0,1): involves Inf -> should be inf
        assert(std::isinf(d[0 * n + 1]));
        assert(std::isinf(d[1 * n + 0]));

        // d(1,2): involves Inf -> should be inf
        assert(std::isinf(d[1 * n + 2]));
        assert(std::isinf(d[2 * n + 1]));

        std::cout << "PASSED: distance matrix Inf handling (inf-guard produces inf)\n";
    }

    // Cluster: feature availability checks
    {
        bool avail = clusterFeaturesAvailable();
        bool nonportable = nonPortableClustersAvailable();
        // These should return without crashing; availability depends on hardware
        static_cast<void>(avail);
        static_cast<void>(nonportable);

        std::cout << "PASSED: cluster feature availability queries\n";
    }

    // Cluster: AdvancedClusterConfig validation
    {
        AdvancedClusterConfig cfg;
        assert(cfg.totalClusterSize() == cfg.clusterSizeX * cfg.clusterSizeY * cfg.clusterSizeZ);

        // Valid config
        assert(cfg.isValid(90));

        // Invalid: negative size
        AdvancedClusterConfig bad_size;
        bad_size.clusterSizeX = -1;
        assert(!bad_size.isValid(90));

        // Invalid: total > CLUSTER_MAX_SIZE
        AdvancedClusterConfig bad_total;
        bad_total.clusterSizeX = 16;
        bad_total.clusterSizeY = 2;
        assert(!bad_total.isValid(90));

        // Invalid: zero shared memory
        AdvancedClusterConfig bad_smem;
        bad_smem.sharedMemPerBlock = 0;
        assert(!bad_smem.isValid(90));

        // Invalid: zero pipeline stages
        AdvancedClusterConfig bad_stages;
        bad_stages.numPipelineStages = 0;
        assert(!bad_stages.isValid(90));

        std::cout << "PASSED: cluster config validation (5 cases)\n";
    }

    // Cluster: distributed shared memory calculation
    {
        AdvancedClusterConfig cfg;
        cfg.clusterSizeX = 4;
        cfg.clusterSizeY = 2;
        cfg.clusterSizeZ = 1;
        cfg.sharedMemPerBlock = 65536;
        assert(cfg.totalDistributedSmem() == 65536ULL * 8);
        assert(cfg.totalClusterSize() == 8);

        std::cout << "PASSED: cluster distributed shared memory calculation\n";
    }

    // Cluster: benchmark config smoke (returns valid result with 0 points)
    {
        auto result = getOrBenchmarkClusterConfig(0, 0, "smoke_test");
        // Should return default result without crashing
        assert(result.optimalClusterSizeX >= 1);
        assert(result.optimalClusterSizeY >= 1);
        assert(result.optimalClusterSizeZ >= 1);

        std::cout << "PASSED: cluster benchmark config smoke\n";
    }

    // Cluster: benchmark config with small point set
    {
        auto result = benchmarkClusterConfig(16, 3);
        // Should return without crashing
        assert(result.optimalClusterSizeX >= 1);
        assert(!result.blackwellRequired); // Blackwell only needed for SM100+

        std::cout << "PASSED: cluster benchmark with small point set\n";
    }

    // CUDADistanceMatrix class: create and compute
    {
        auto matrix_result = CUDADistanceMatrix::create();
        assert(!matrix_result.isError());
        auto matrix = std::move(matrix_result.value());
        assert(matrix->isAvailable());

        std::vector<double> h_distances(kTriangleN * kTriangleN);
        nerve::core::BufferView<const double> points_view(kTrianglePoints,
                                                          kTriangleN * kTriangleDim);
        nerve::core::BufferView<double> dist_view(h_distances.data(), h_distances.size());

        auto result = matrix->compute(points_view, dist_view, kTriangleDim, 10.0);
        assert(!result.isError());
        assert(approx_equal(h_distances[0 * kTriangleN + 1], 3.0));
        assert(approx_equal(h_distances[1 * kTriangleN + 2], 5.0));

        std::cout << "PASSED: CUDADistanceMatrix create + compute\n";
    }

    // CUDADistanceMatrix: streaming compute
    {
        auto matrix_result = CUDADistanceMatrix::create();
        assert(!matrix_result.isError());
        auto matrix = std::move(matrix_result.value());

        std::vector<double> h_distances(kTriangleN * kTriangleN);
        nerve::core::BufferView<const double> points_view(kTrianglePoints,
                                                          kTriangleN * kTriangleDim);
        nerve::core::BufferView<double> dist_view(h_distances.data(), h_distances.size());

        auto result = matrix->computeStreaming(points_view, dist_view, kTriangleDim, 10.0, 2);
        assert(!result.isError());
        assert(approx_equal(h_distances[0 * kTriangleN + 2], 4.0));

        std::cout << "PASSED: CUDADistanceMatrix streaming compute\n";
    }

    // CUDADistanceMatrix: reject NaN point coordinates with E20_NUM_NAN
    {
        auto matrix_result = CUDADistanceMatrix::create();
        assert(!matrix_result.isError());
        auto matrix = std::move(matrix_result.value());

        // Build 3-point set with one NaN coordinate
        constexpr int n = 3;
        constexpr int dim = 2;
        std::vector<double> bad_points(n * dim);
        bad_points[0] = 0.0;
        bad_points[1] = 0.0;                                      // (0, 0)
        bad_points[2] = std::numeric_limits<double>::quiet_NaN(); // (NaN, 2)
        bad_points[3] = 2.0;
        bad_points[4] = 3.0;
        bad_points[5] = 4.0; // (3, 4)

        std::vector<double> h_distances(static_cast<size_t>(n) * static_cast<size_t>(n));
        nerve::core::BufferView<const double> points_view(bad_points.data(), bad_points.size());
        nerve::core::BufferView<double> dist_view(h_distances.data(), h_distances.size());

        auto result = matrix->compute(points_view, dist_view, dim, 10.0);
        assert(result.isError());
        assert(result.errorCode() == errors::ErrorCode::E20_NUM_NAN);

        std::cout << "PASSED: CUDADistanceMatrix rejects NaN coordinates (E20_NUM_NAN)\n";
    }

    // CUDADistanceMatrix: reject Inf point coordinates with E20_NUM_NAN
    {
        auto matrix_result = CUDADistanceMatrix::create();
        assert(!matrix_result.isError());
        auto matrix = std::move(matrix_result.value());

        constexpr int n = 3;
        constexpr int dim = 2;
        std::vector<double> bad_points(n * dim);
        bad_points[0] = 0.0;
        bad_points[1] = 0.0;                                     // (0, 0)
        bad_points[2] = std::numeric_limits<double>::infinity(); // (inf, 2)
        bad_points[3] = 2.0;
        bad_points[4] = 3.0;
        bad_points[5] = 4.0; // (3, 4)

        std::vector<double> h_distances(static_cast<size_t>(n) * static_cast<size_t>(n));
        nerve::core::BufferView<const double> points_view(bad_points.data(), bad_points.size());
        nerve::core::BufferView<double> dist_view(h_distances.data(), h_distances.size());

        auto result = matrix->compute(points_view, dist_view, dim, 10.0);
        assert(result.isError());
        assert(result.errorCode() == errors::ErrorCode::E20_NUM_NAN);

        std::cout << "PASSED: CUDADistanceMatrix rejects Inf coordinates (E20_NUM_NAN)\n";
    }

    // CUDADistanceMatrix: reject negative infinity with E20_NUM_NAN
    {
        auto matrix_result = CUDADistanceMatrix::create();
        assert(!matrix_result.isError());
        auto matrix = std::move(matrix_result.value());

        constexpr int n = 3;
        constexpr int dim = 2;
        std::vector<double> bad_points(n * dim);
        bad_points[0] = 0.0;
        bad_points[1] = 0.0;                                      // (0, 0)
        bad_points[2] = -std::numeric_limits<double>::infinity(); // (-inf, 2)
        bad_points[3] = 2.0;
        bad_points[4] = 3.0;
        bad_points[5] = 4.0; // (3, 4)

        std::vector<double> h_distances(static_cast<size_t>(n) * static_cast<size_t>(n));
        nerve::core::BufferView<const double> points_view(bad_points.data(), bad_points.size());
        nerve::core::BufferView<double> dist_view(h_distances.data(), h_distances.size());

        auto result = matrix->compute(points_view, dist_view, dim, 10.0);
        assert(result.isError());
        assert(result.errorCode() == errors::ErrorCode::E20_NUM_NAN);

        std::cout << "PASSED: CUDADistanceMatrix rejects -Inf coordinates (E20_NUM_NAN)\n";
    }

    // CUDADistanceMatrix: computeStreaming also rejects NaN with E20_NUM_NAN
    {
        auto matrix_result = CUDADistanceMatrix::create();
        assert(!matrix_result.isError());
        auto matrix = std::move(matrix_result.value());

        constexpr int n = 3;
        constexpr int dim = 2;
        std::vector<double> bad_points(n * dim);
        bad_points[0] = 0.0;
        bad_points[1] = 0.0;
        bad_points[2] = std::numeric_limits<double>::quiet_NaN();
        bad_points[3] = 2.0;
        bad_points[4] = 3.0;
        bad_points[5] = 4.0;

        std::vector<double> h_distances(static_cast<size_t>(n) * static_cast<size_t>(n));
        nerve::core::BufferView<const double> points_view(bad_points.data(), bad_points.size());
        nerve::core::BufferView<double> dist_view(h_distances.data(), h_distances.size());

        auto result = matrix->computeStreaming(points_view, dist_view, dim, 10.0, 2);
        assert(result.isError());
        assert(result.errorCode() == errors::ErrorCode::E20_NUM_NAN);

        std::cout << "PASSED: CUDADistanceMatrix streaming rejects NaN (E20_NUM_NAN)\n";
    }

    // EdgeExtractionStats defaults
    {
        EdgeExtractionStats stats;
        assert(stats.total_time_ms == 0.0);
        assert(stats.edges_extracted == 0);
        assert(stats.getExtractionRate() == 0.0);
        assert(stats.getFilteringRate() == 0.0);

        std::cout << "PASSED: edge extraction stats defaults\n";
    }

    // EdgeExtractionConfig validation
    {
        EdgeExtractionConfig config;
        auto valid = config.validate();
        assert(!valid.isError());

        std::cout << "PASSED: edge extraction config validation\n";
    }

    // Factory: createAcceleratedDistanceMatrix
    {
        auto result = factory::createAcceleratedDistanceMatrix(16, 3, 5.0);
        assert(!result.isError());

        std::cout << "PASSED: distance matrix factory creation\n";
    }

    // Factory: createAcceleratedEdgeExtractor
    {
        auto result = factory::createAcceleratedEdgeExtractor(16, 5.0, 0.1);
        assert(!result.isError());

        std::cout << "PASSED: edge extractor factory creation\n";
    }

    // getOptimalConfig returns valid config (distance matrix)
    {
        Size n = 16;
        Size dim = 3;
        CUDADistanceMatrixConfig base;
        auto config = cuda_host::getOptimalConfig(n, dim, base);
        auto valid = config.validate();
        assert(!valid.isError());
        assert(config.max_block_size > 0);

        std::cout << "PASSED: distance matrix optimal config generation\n";
    }

    // validateLaunchParams
    {
        CUDADistanceMatrixConfig config;
        auto result = cuda_host::validateLaunchParams(16, 3, config);
        assert(!result.isError());

        // Invalid: zero points
        auto bad_result = cuda_host::validateLaunchParams(0, 3, config);
        assert(bad_result.isError());

        std::cout << "PASSED: distance matrix launch parameter validation\n";
    }

    // CUDAEgdeExtractor create + extractEdges
    {
        auto extractor_result = CUDAEgdeExtractor::create();
        assert(!extractor_result.isError());
        auto extractor = std::move(extractor_result.value());

        // Build a small distance matrix on host
        constexpr int n = 3;
        constexpr size_t mat_size = static_cast<size_t>(n) * static_cast<size_t>(n);
        double h_dist[mat_size] = {};
        fill_k3_distance_matrix(h_dist, n, 1.5, 2.0, 0.8);

        std::vector<Edge> h_edges(10);
        nerve::core::BufferView<const double> dist_view(h_dist, mat_size);
        nerve::core::BufferView<Edge> edge_view(h_edges.data(), h_edges.size());

        auto result = extractor->extractEdges(dist_view, edge_view, n, 3.0);
        assert(!result.isError());

        auto &stats = extractor->getPerformanceStats();
        assert(stats.edges_extracted > 0);

        std::cout << "PASSED: CUDAEgdeExtractor create + extractEdges\n";
    }

    // Estimate functions don't crash
    {
        auto mem = utils::estimateMemoryUsage(100, 3);
        assert(mem > 0);

        auto time = utils::estimateComputationTime(100, 3, true);
        assert(time > 0.0);

        auto edge_mem = utils::estimateEdgeExtractionMemoryUsage(100, 5000, EdgeExtractionConfig{});
        assert(edge_mem > 0);

        auto edge_time = utils::estimateEdgeExtractionTime(100, 5000, true);
        assert(edge_time > 0.0);

        std::cout << "PASSED: estimation utility functions\n";
    }

    return 0;
}

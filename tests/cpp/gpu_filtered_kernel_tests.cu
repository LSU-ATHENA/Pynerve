#include "gpu_test_helpers.cuh"

int main()
{
    if (!has_gpu())
    {
        std::cerr << "No CUDA device available -- skipping GPU filtered kernel coverage tests\n";
        return 0;
    }
    // Edge extraction: max_degree enforcement (K5, degree cap at 2)
    {
        constexpr int n = 5;
        constexpr size_t mat_size = static_cast<size_t>(n) * static_cast<size_t>(n);
        double h_dist[mat_size] = {};
        // All edges weight 1.0 -- well within max_radius, so radius doesn't filter.
        // With max_degree=2, each vertex can accept at most 2 incident edges.
        // The kernel uses atomicAdd on degree counters; within a warp, the
        // serialization order is undefined, so the exact set of accepted edges
        // is non-deterministic. We verify invariants only.
        fill_k5_distance_matrix(h_dist, n, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);

        constexpr Size max_edges = 20;
        constexpr double max_radius = 10.0;
        constexpr double min_edge_weight = 0.0;
        constexpr Size max_degree = 2;

        double *d_dist = nullptr;
        Edge *d_edges = nullptr;
        Size *d_count = nullptr;
        Size *d_degree = nullptr;
        assert(check_cuda(cudaMalloc(&d_dist, mat_size * sizeof(double)), "cudaMalloc"));
        assert(check_cuda(cudaMalloc(&d_edges, max_edges * sizeof(Edge)), "cudaMalloc"));
        assert(check_cuda(cudaMalloc(&d_count, sizeof(Size)), "cudaMalloc"));
        assert(check_cuda(cudaMalloc(&d_degree, n * sizeof(Size)), "cudaMalloc"));
        cudaMemcpy(d_dist, h_dist, mat_size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemset(d_count, 0, sizeof(Size));
        cudaMemset(d_degree, 0, n * sizeof(Size));

        cuda_kernels::extractEdgesFilteredKernel(d_dist, d_edges, d_count, n, max_radius,
                                                 max_edges, min_edge_weight, max_degree,
                                                 d_degree);
        assert(check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize"));

        Size edge_count = 0;
        cudaMemcpy(&edge_count, d_count, sizeof(Size), cudaMemcpyDeviceToHost);
        // Theoretical bounds: at least 1 edge (obvious for K5 with degree cap 2),
        // at most floor(n * max_degree / 2) = floor(5*2/2) = 5
        assert(edge_count >= 1);
        assert(edge_count <= 5);

        std::vector<Edge> edges(edge_count);
        cudaMemcpy(edges.data(), d_edges, edge_count * sizeof(Edge), cudaMemcpyDeviceToHost);

        // All output edges must have finite weight 1.0
        for (Size i = 0; i < edge_count; ++i)
        {
            assert(std::isfinite(edges[i].w));
            assert(approx_equal(edges[i].w, 1.0));
        }

        // Verify no vertex exceeds max_degree by counting incident edges
        Size vertex_degree[5] = {0, 0, 0, 0, 0};
        for (Size i = 0; i < edge_count; ++i)
        {
            ++vertex_degree[static_cast<size_t>(edges[i].u)];
            ++vertex_degree[static_cast<size_t>(edges[i].v)];
        }
        for (int v = 0; v < n; ++v)
            assert(vertex_degree[v] <= max_degree);

        cudaFree(d_dist);
        cudaFree(d_edges);
        cudaFree(d_count);
        cudaFree(d_degree);

        std::cout << "PASSED: edge extraction max_degree enforcement (K5, cap=2, "
                  << edge_count << " edges, all degrees <= 2)\n";
    }

    // Edge extraction: max_degree enforcement (K5, degree cap at 1)
    {
        constexpr int n = 5;
        constexpr size_t mat_size = static_cast<size_t>(n) * static_cast<size_t>(n);
        double h_dist[mat_size] = {};
        // All edges weight 1.0, max_degree=1 -- each vertex can accept at most 1 incident edge.
        // Theoretical max: floor(n * max_degree / 2) = floor(5*1/2) = 2 edges.
        fill_k5_distance_matrix(h_dist, n, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);

        constexpr Size max_edges = 20;
        constexpr double max_radius = 10.0;
        constexpr double min_edge_weight = 0.0;
        constexpr Size max_degree = 1;

        double *d_dist = nullptr;
        Edge *d_edges = nullptr;
        Size *d_count = nullptr;
        Size *d_degree = nullptr;
        assert(check_cuda(cudaMalloc(&d_dist, mat_size * sizeof(double)), "cudaMalloc"));
        assert(check_cuda(cudaMalloc(&d_edges, max_edges * sizeof(Edge)), "cudaMalloc"));
        assert(check_cuda(cudaMalloc(&d_count, sizeof(Size)), "cudaMalloc"));
        assert(check_cuda(cudaMalloc(&d_degree, n * sizeof(Size)), "cudaMalloc"));
        cudaMemcpy(d_dist, h_dist, mat_size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemset(d_count, 0, sizeof(Size));
        cudaMemset(d_degree, 0, n * sizeof(Size));

        cuda_kernels::extractEdgesFilteredKernel(d_dist, d_edges, d_count, n, max_radius,
                                                 max_edges, min_edge_weight, max_degree,
                                                 d_degree);
        assert(check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize"));

        Size edge_count = 0;
        cudaMemcpy(&edge_count, d_count, sizeof(Size), cudaMemcpyDeviceToHost);
        // Theoretical bounds: at least 1 edge, at most floor(5*1/2) = 2
        assert(edge_count >= 1);
        assert(edge_count <= 2);

        std::vector<Edge> edges(edge_count);
        cudaMemcpy(edges.data(), d_edges, edge_count * sizeof(Edge), cudaMemcpyDeviceToHost);

        // All output edges must have finite weight 1.0
        for (Size i = 0; i < edge_count; ++i)
        {
            assert(std::isfinite(edges[i].w));
            assert(approx_equal(edges[i].w, 1.0));
        }

        // No vertex may exceed max_degree (1)
        Size vertex_degree[5] = {0, 0, 0, 0, 0};
        for (Size i = 0; i < edge_count; ++i)
        {
            ++vertex_degree[static_cast<size_t>(edges[i].u)];
            ++vertex_degree[static_cast<size_t>(edges[i].v)];
        }
        for (int v = 0; v < n; ++v)
            assert(vertex_degree[v] <= max_degree);

        cudaFree(d_dist);
        cudaFree(d_edges);
        cudaFree(d_count);
        cudaFree(d_degree);

        std::cout << "PASSED: edge extraction max_degree enforcement (K5, cap=1, "
                  << edge_count << " edges, all degrees <= 1)\n";
    }

    // Edge extraction: max_degree enforcement stress test (n=64, 50+ repeated runs)
    {
        constexpr int n = 64;
        constexpr size_t mat_size = static_cast<size_t>(n) * static_cast<size_t>(n);
        double h_dist[mat_size] = {};
        // Fill all 2016 upper-triangular edges with weight 1.0
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j)
                h_dist[i * n + j] = h_dist[j * n + i] = 1.0;

        constexpr Size max_edges = 4096;
        constexpr double max_radius = 10.0;
        constexpr double min_edge_weight = 0.0;
        constexpr Size max_degree = 8;
        constexpr int num_runs = 50;

        double *d_dist = nullptr;
        Edge *d_edges = nullptr;
        Size *d_count = nullptr;
        Size *d_degree = nullptr;
        assert(check_cuda(cudaMalloc(&d_dist, mat_size * sizeof(double)), "cudaMalloc"));
        assert(check_cuda(cudaMalloc(&d_edges, max_edges * sizeof(Edge)), "cudaMalloc"));
        assert(check_cuda(cudaMalloc(&d_count, sizeof(Size)), "cudaMalloc"));
        assert(check_cuda(cudaMalloc(&d_degree, n * sizeof(Size)), "cudaMalloc"));
        cudaMemcpy(d_dist, h_dist, mat_size * sizeof(double), cudaMemcpyHostToDevice);

        // Theoretical max: floor(n * max_degree / 2) = floor(64*8/2) = 256
        constexpr Size theoretical_max = 256;

        for (int run = 0; run < num_runs; ++run)
        {
            cudaMemset(d_count, 0, sizeof(Size));
            cudaMemset(d_degree, 0, n * sizeof(Size));

            cuda_kernels::extractEdgesFilteredKernel(d_dist, d_edges, d_count, n, max_radius,
                                                     max_edges, min_edge_weight, max_degree,
                                                     d_degree);
            assert(check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize"));

            Size edge_count = 0;
            cudaMemcpy(&edge_count, d_count, sizeof(Size), cudaMemcpyDeviceToHost);
            assert(edge_count >= 1);
            assert(edge_count <= theoretical_max);
            static_cast<void>(theoretical_max);

            std::vector<Edge> edges(edge_count);
            cudaMemcpy(edges.data(), d_edges, edge_count * sizeof(Edge), cudaMemcpyDeviceToHost);

            // All output edges must have finite weight 1.0
            for (Size i = 0; i < edge_count; ++i)
            {
                assert(std::isfinite(edges[i].w));
                assert(approx_equal(edges[i].w, 1.0));
            }

            // No vertex may exceed max_degree
            std::vector<Size> vertex_degree(n, 0);
            for (Size i = 0; i < edge_count; ++i)
            {
                ++vertex_degree[static_cast<size_t>(edges[i].u)];
                ++vertex_degree[static_cast<size_t>(edges[i].v)];
            }
            for (int v = 0; v < n; ++v)
                assert(vertex_degree[v] <= max_degree);
        }

        cudaFree(d_dist);
        cudaFree(d_edges);
        cudaFree(d_count);
        cudaFree(d_degree);

        std::cout << "PASSED: edge extraction max_degree stress (n=64, cap=8, "
                  << num_runs << " runs, all degrees <= 8)\n";
    }

    // Edge extraction: max_degree enforcement stress test (n=64, tighter cap=4)
    {
        constexpr int n = 64;
        constexpr size_t mat_size = static_cast<size_t>(n) * static_cast<size_t>(n);
        double h_dist[mat_size] = {};
        // Fill all 2016 upper-triangular edges with weight 1.0
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j)
                h_dist[i * n + j] = h_dist[j * n + i] = 1.0;

        constexpr Size max_edges = 4096;
        constexpr double max_radius = 10.0;
        constexpr double min_edge_weight = 0.0;
        constexpr Size max_degree = 4;
        constexpr int num_runs = 50;

        double *d_dist = nullptr;
        Edge *d_edges = nullptr;
        Size *d_count = nullptr;
        Size *d_degree = nullptr;
        assert(check_cuda(cudaMalloc(&d_dist, mat_size * sizeof(double)), "cudaMalloc"));
        assert(check_cuda(cudaMalloc(&d_edges, max_edges * sizeof(Edge)), "cudaMalloc"));
        assert(check_cuda(cudaMalloc(&d_count, sizeof(Size)), "cudaMalloc"));
        assert(check_cuda(cudaMalloc(&d_degree, n * sizeof(Size)), "cudaMalloc"));
        cudaMemcpy(d_dist, h_dist, mat_size * sizeof(double), cudaMemcpyHostToDevice);

        // Theoretical max: floor(n * max_degree / 2) = floor(64*4/2) = 128
        constexpr Size theoretical_max = 128;

        for (int run = 0; run < num_runs; ++run)
        {
            cudaMemset(d_count, 0, sizeof(Size));
            cudaMemset(d_degree, 0, n * sizeof(Size));

            cuda_kernels::extractEdgesFilteredKernel(d_dist, d_edges, d_count, n, max_radius,
                                                     max_edges, min_edge_weight, max_degree,
                                                     d_degree);
            assert(check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize"));

            Size edge_count = 0;
            cudaMemcpy(&edge_count, d_count, sizeof(Size), cudaMemcpyDeviceToHost);
            assert(edge_count >= 1);
            assert(edge_count <= theoretical_max);
            static_cast<void>(theoretical_max);

            std::vector<Edge> edges(edge_count);
            cudaMemcpy(edges.data(), d_edges, edge_count * sizeof(Edge), cudaMemcpyDeviceToHost);

            // All output edges must have finite weight 1.0
            for (Size i = 0; i < edge_count; ++i)
            {
                assert(std::isfinite(edges[i].w));
                assert(approx_equal(edges[i].w, 1.0));
            }

            // No vertex may exceed max_degree
            std::vector<Size> vertex_degree(n, 0);
            for (Size i = 0; i < edge_count; ++i)
            {
                ++vertex_degree[static_cast<size_t>(edges[i].u)];
                ++vertex_degree[static_cast<size_t>(edges[i].v)];
            }
            for (int v = 0; v < n; ++v)
                assert(vertex_degree[v] <= max_degree);
        }

        cudaFree(d_dist);
        cudaFree(d_edges);
        cudaFree(d_count);
        cudaFree(d_degree);

        std::cout << "PASSED: edge extraction max_degree stress (n=64, cap=4, "
                  << num_runs << " runs, all degrees <= 4)\n";
    }

    // Edge extraction: max_degree enforcement stress test (n=64, absolute limit cap=2)
    {
        constexpr int n = 64;
        constexpr size_t mat_size = static_cast<size_t>(n) * static_cast<size_t>(n);
        double h_dist[mat_size] = {};
        // Fill all 2016 upper-triangular edges with weight 1.0
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j)
                h_dist[i * n + j] = h_dist[j * n + i] = 1.0;

        constexpr Size max_edges = 4096;
        constexpr double max_radius = 10.0;
        constexpr double min_edge_weight = 0.0;
        constexpr Size max_degree = 2;
        constexpr int num_runs = 50;

        double *d_dist = nullptr;
        Edge *d_edges = nullptr;
        Size *d_count = nullptr;
        Size *d_degree = nullptr;
        assert(check_cuda(cudaMalloc(&d_dist, mat_size * sizeof(double)), "cudaMalloc"));
        assert(check_cuda(cudaMalloc(&d_edges, max_edges * sizeof(Edge)), "cudaMalloc"));
        assert(check_cuda(cudaMalloc(&d_count, sizeof(Size)), "cudaMalloc"));
        assert(check_cuda(cudaMalloc(&d_degree, n * sizeof(Size)), "cudaMalloc"));
        cudaMemcpy(d_dist, h_dist, mat_size * sizeof(double), cudaMemcpyHostToDevice);

        // Theoretical max: floor(n * max_degree / 2) = floor(64*2/2) = 64
        constexpr Size theoretical_max = 64;

        for (int run = 0; run < num_runs; ++run)
        {
            cudaMemset(d_count, 0, sizeof(Size));
            cudaMemset(d_degree, 0, n * sizeof(Size));

            cuda_kernels::extractEdgesFilteredKernel(d_dist, d_edges, d_count, n, max_radius,
                                                     max_edges, min_edge_weight, max_degree,
                                                     d_degree);
            assert(check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize"));

            Size edge_count = 0;
            cudaMemcpy(&edge_count, d_count, sizeof(Size), cudaMemcpyDeviceToHost);
            assert(edge_count >= 1);
            assert(edge_count <= theoretical_max);
            static_cast<void>(theoretical_max);

            std::vector<Edge> edges(edge_count);
            cudaMemcpy(edges.data(), d_edges, edge_count * sizeof(Edge), cudaMemcpyDeviceToHost);

            // All output edges must have finite weight 1.0
            for (Size i = 0; i < edge_count; ++i)
            {
                assert(std::isfinite(edges[i].w));
                assert(approx_equal(edges[i].w, 1.0));
            }

            // No vertex may exceed max_degree (2)
            std::vector<Size> vertex_degree(n, 0);
            for (Size i = 0; i < edge_count; ++i)
            {
                ++vertex_degree[static_cast<size_t>(edges[i].u)];
                ++vertex_degree[static_cast<size_t>(edges[i].v)];
            }
            for (int v = 0; v < n; ++v)
                assert(vertex_degree[v] <= max_degree);
        }

        cudaFree(d_dist);
        cudaFree(d_edges);
        cudaFree(d_count);
        cudaFree(d_degree);

        std::cout << "PASSED: edge extraction max_degree stress (n=64, cap=2, "
                  << num_runs << " runs, all degrees <= 2)\n";
    }

    // Edge extraction: max_degree enforcement stress test (n=32, tighter cap=4)
    {
        constexpr int n = 32;
        constexpr size_t mat_size = static_cast<size_t>(n) * static_cast<size_t>(n);
        double h_dist[mat_size] = {};
        // Fill all 496 upper-triangular edges with weight 1.0
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j)
                h_dist[i * n + j] = h_dist[j * n + i] = 1.0;

        constexpr Size max_edges = 2048;
        constexpr double max_radius = 10.0;
        constexpr double min_edge_weight = 0.0;
        constexpr Size max_degree = 4;
        constexpr int num_runs = 100;

        double *d_dist = nullptr;
        Edge *d_edges = nullptr;
        Size *d_count = nullptr;
        Size *d_degree = nullptr;
        assert(check_cuda(cudaMalloc(&d_dist, mat_size * sizeof(double)), "cudaMalloc"));
        assert(check_cuda(cudaMalloc(&d_edges, max_edges * sizeof(Edge)), "cudaMalloc"));
        assert(check_cuda(cudaMalloc(&d_count, sizeof(Size)), "cudaMalloc"));
        assert(check_cuda(cudaMalloc(&d_degree, n * sizeof(Size)), "cudaMalloc"));
        cudaMemcpy(d_dist, h_dist, mat_size * sizeof(double), cudaMemcpyHostToDevice);

        // Theoretical max: floor(n * max_degree / 2) = floor(32*4/2) = 64
        constexpr Size theoretical_max = 64;

        for (int run = 0; run < num_runs; ++run)
        {
            cudaMemset(d_count, 0, sizeof(Size));
            cudaMemset(d_degree, 0, n * sizeof(Size));

            cuda_kernels::extractEdgesFilteredKernel(d_dist, d_edges, d_count, n, max_radius,
                                                     max_edges, min_edge_weight, max_degree,
                                                     d_degree);
            assert(check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize"));

            Size edge_count = 0;
            cudaMemcpy(&edge_count, d_count, sizeof(Size), cudaMemcpyDeviceToHost);
            assert(edge_count >= 1);
            assert(edge_count <= theoretical_max);
            static_cast<void>(theoretical_max);

            std::vector<Edge> edges(edge_count);
            cudaMemcpy(edges.data(), d_edges, edge_count * sizeof(Edge), cudaMemcpyDeviceToHost);

            // All output edges must have finite weight 1.0
            for (Size i = 0; i < edge_count; ++i)
            {
                assert(std::isfinite(edges[i].w));
                assert(approx_equal(edges[i].w, 1.0));
            }

            // No vertex may exceed max_degree
            std::vector<Size> vertex_degree(n, 0);
            for (Size i = 0; i < edge_count; ++i)
            {
                ++vertex_degree[static_cast<size_t>(edges[i].u)];
                ++vertex_degree[static_cast<size_t>(edges[i].v)];
            }
            for (int v = 0; v < n; ++v)
                assert(vertex_degree[v] <= max_degree);
        }

        cudaFree(d_dist);
        cudaFree(d_edges);
        cudaFree(d_count);
        cudaFree(d_degree);

        std::cout << "PASSED: edge extraction max_degree stress (n=32, cap=4, "
                  << num_runs << " runs, all degrees <= 4)\n";
    }

    // Edge extraction: max_degree enforcement stress test (n=32, extremely tight cap=2)
    {
        constexpr int n = 32;
        constexpr size_t mat_size = static_cast<size_t>(n) * static_cast<size_t>(n);
        double h_dist[mat_size] = {};
        // Fill all 496 upper-triangular edges with weight 1.0
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j)
                h_dist[i * n + j] = h_dist[j * n + i] = 1.0;

        constexpr Size max_edges = 2048;
        constexpr double max_radius = 10.0;
        constexpr double min_edge_weight = 0.0;
        constexpr Size max_degree = 2;
        constexpr int num_runs = 100;

        double *d_dist = nullptr;
        Edge *d_edges = nullptr;
        Size *d_count = nullptr;
        Size *d_degree = nullptr;
        assert(check_cuda(cudaMalloc(&d_dist, mat_size * sizeof(double)), "cudaMalloc"));
        assert(check_cuda(cudaMalloc(&d_edges, max_edges * sizeof(Edge)), "cudaMalloc"));
        assert(check_cuda(cudaMalloc(&d_count, sizeof(Size)), "cudaMalloc"));
        assert(check_cuda(cudaMalloc(&d_degree, n * sizeof(Size)), "cudaMalloc"));
        cudaMemcpy(d_dist, h_dist, mat_size * sizeof(double), cudaMemcpyHostToDevice);

        // Theoretical max: floor(n * max_degree / 2) = floor(32*2/2) = 32
        constexpr Size theoretical_max = 32;

        for (int run = 0; run < num_runs; ++run)
        {
            cudaMemset(d_count, 0, sizeof(Size));
            cudaMemset(d_degree, 0, n * sizeof(Size));

            cuda_kernels::extractEdgesFilteredKernel(d_dist, d_edges, d_count, n, max_radius,
                                                     max_edges, min_edge_weight, max_degree,
                                                     d_degree);
            assert(check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize"));

            Size edge_count = 0;
            cudaMemcpy(&edge_count, d_count, sizeof(Size), cudaMemcpyDeviceToHost);
            assert(edge_count >= 1);
            assert(edge_count <= theoretical_max);
            static_cast<void>(theoretical_max);

            std::vector<Edge> edges(edge_count);
            cudaMemcpy(edges.data(), d_edges, edge_count * sizeof(Edge), cudaMemcpyDeviceToHost);

            // All output edges must have finite weight 1.0
            for (Size i = 0; i < edge_count; ++i)
            {
                assert(std::isfinite(edges[i].w));
                assert(approx_equal(edges[i].w, 1.0));
            }

            // No vertex may exceed max_degree (2)
            std::vector<Size> vertex_degree(n, 0);
            for (Size i = 0; i < edge_count; ++i)
            {
                ++vertex_degree[static_cast<size_t>(edges[i].u)];
                ++vertex_degree[static_cast<size_t>(edges[i].v)];
            }
            for (int v = 0; v < n; ++v)
                assert(vertex_degree[v] <= max_degree);
        }

        cudaFree(d_dist);
        cudaFree(d_edges);
        cudaFree(d_count);
        cudaFree(d_degree);

        std::cout << "PASSED: edge extraction max_degree stress (n=32, cap=2, "
                  << num_runs << " runs, all degrees <= 2)\n";
    }

    // Edge extraction: combined min_edge_weight + max_degree filtering (K5)
    {
        constexpr int n = 5;
        constexpr size_t mat_size = static_cast<size_t>(n) * static_cast<size_t>(n);
        double h_dist[mat_size] = {};
        // K5 with mixed weights: three edges below min_edge_weight=0.5,
        // seven edges above. With max_degree=2, each vertex accepts at most 2 edges.
        // Edges below 0.5: (0,1)=0.1, (0,2)=0.2, (3,4)=0.3 -- rejected by min_weight.
        // Edges >= 0.5: (0,3)=0.6, (0,4)=0.7, (1,2)=0.5, (1,3)=0.9,
        //              (1,4)=0.8, (2,3)=1.0, (2,4)=1.2 -- candidates for extraction.
        fill_k5_distance_matrix(h_dist, n,
                                0.1, 0.2, 0.6, 0.7,  // row 0 edges
                                0.5, 0.9, 0.8,         // row 1 edges
                                1.0, 1.2,               // row 2 edges
                                0.3);                   // row 3 edge (3,4)

        constexpr Size max_edges = 20;
        constexpr double max_radius = 10.0;
        constexpr double min_edge_weight = 0.5;
        constexpr Size max_degree = 2;

        double *d_dist = nullptr;
        Edge *d_edges = nullptr;
        Size *d_count = nullptr;
        Size *d_degree = nullptr;
        assert(check_cuda(cudaMalloc(&d_dist, mat_size * sizeof(double)), "cudaMalloc"));
        assert(check_cuda(cudaMalloc(&d_edges, max_edges * sizeof(Edge)), "cudaMalloc"));
        assert(check_cuda(cudaMalloc(&d_count, sizeof(Size)), "cudaMalloc"));
        assert(check_cuda(cudaMalloc(&d_degree, n * sizeof(Size)), "cudaMalloc"));
        cudaMemcpy(d_dist, h_dist, mat_size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemset(d_count, 0, sizeof(Size));
        cudaMemset(d_degree, 0, n * sizeof(Size));

        cuda_kernels::extractEdgesFilteredKernel(d_dist, d_edges, d_count, n, max_radius,
                                                 max_edges, min_edge_weight, max_degree,
                                                 d_degree);
        assert(check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize"));

        Size edge_count = 0;
        cudaMemcpy(&edge_count, d_count, sizeof(Size), cudaMemcpyDeviceToHost);
        // At least 1 edge (obvious), at most floor(5*2/2)=5
        assert(edge_count >= 1);
        assert(edge_count <= 5);

        std::vector<Edge> edges(edge_count);
        cudaMemcpy(edges.data(), d_edges, edge_count * sizeof(Edge), cudaMemcpyDeviceToHost);

        // All output edges must have weight >= min_edge_weight (0.5)
        for (Size i = 0; i < edge_count; ++i)
        {
            assert(std::isfinite(edges[i].w));
            assert(edges[i].w >= min_edge_weight);
        }

        // No vertex may exceed max_degree (2)
        Size vertex_degree[5] = {0, 0, 0, 0, 0};
        for (Size i = 0; i < edge_count; ++i)
        {
            ++vertex_degree[static_cast<size_t>(edges[i].u)];
            ++vertex_degree[static_cast<size_t>(edges[i].v)];
        }
        for (int v = 0; v < n; ++v)
            assert(vertex_degree[v] <= max_degree);

        cudaFree(d_dist);
        cudaFree(d_edges);
        cudaFree(d_count);
        cudaFree(d_degree);

        std::cout << "PASSED: edge extraction combined min_weight + max_degree (K5, "
                  << edge_count << " edges, weight >= 0.5, degrees <= 2)\n";
    }

    // Edge extraction: combined min_edge_weight + max_degree stress (n=32, repeated runs)
    {
        constexpr int n = 32;
        constexpr size_t mat_size = static_cast<size_t>(n) * static_cast<size_t>(n);
        double h_dist[mat_size] = {};
        // Mix: ~1/3 edges at 0.1 (below min_weight=0.5), ~2/3 at 1.0.
        // Pattern: edge (i,j) gets 0.1 if (i+j) % 3 == 0, else 1.0.
        // With max_degree=4, each vertex accepts at most 4 edges among those >= 0.5.
        for (int i = 0; i < n; ++i)
        {
            for (int j = i + 1; j < n; ++j)
            {
                double w = ((i + j) % 3 == 0) ? 0.1 : 1.0;
                h_dist[i * n + j] = h_dist[j * n + i] = w;
            }
        }

        constexpr Size max_edges = 2048;
        constexpr double max_radius = 10.0;
        constexpr double min_edge_weight = 0.5;
        constexpr Size max_degree = 4;
        constexpr int num_runs = 50;

        double *d_dist = nullptr;
        Edge *d_edges = nullptr;
        Size *d_count = nullptr;
        Size *d_degree = nullptr;
        assert(check_cuda(cudaMalloc(&d_dist, mat_size * sizeof(double)), "cudaMalloc"));
        assert(check_cuda(cudaMalloc(&d_edges, max_edges * sizeof(Edge)), "cudaMalloc"));
        assert(check_cuda(cudaMalloc(&d_count, sizeof(Size)), "cudaMalloc"));
        assert(check_cuda(cudaMalloc(&d_degree, n * sizeof(Size)), "cudaMalloc"));
        cudaMemcpy(d_dist, h_dist, mat_size * sizeof(double), cudaMemcpyHostToDevice);

        // Theoretical max: floor(32*4/2) = 64 (all edges >= 0.5)
        constexpr Size theoretical_max = 64;

        for (int run = 0; run < num_runs; ++run)
        {
            cudaMemset(d_count, 0, sizeof(Size));
            cudaMemset(d_degree, 0, n * sizeof(Size));

            cuda_kernels::extractEdgesFilteredKernel(d_dist, d_edges, d_count, n, max_radius,
                                                     max_edges, min_edge_weight, max_degree,
                                                     d_degree);
            assert(check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize"));

            Size edge_count = 0;
            cudaMemcpy(&edge_count, d_count, sizeof(Size), cudaMemcpyDeviceToHost);
            assert(edge_count >= 1);
            assert(edge_count <= theoretical_max);
            static_cast<void>(theoretical_max);

            std::vector<Edge> edges(edge_count);
            cudaMemcpy(edges.data(), d_edges, edge_count * sizeof(Edge), cudaMemcpyDeviceToHost);

            // All output edges must have weight >= min_edge_weight (0.5)
            for (Size i = 0; i < edge_count; ++i)
            {
                assert(std::isfinite(edges[i].w));
                assert(edges[i].w >= min_edge_weight);
            }

            // No vertex may exceed max_degree
            std::vector<Size> vertex_degree(n, 0);
            for (Size i = 0; i < edge_count; ++i)
            {
                ++vertex_degree[static_cast<size_t>(edges[i].u)];
                ++vertex_degree[static_cast<size_t>(edges[i].v)];
            }
            for (int v = 0; v < n; ++v)
                assert(vertex_degree[v] <= max_degree);
        }

        cudaFree(d_dist);
        cudaFree(d_edges);
        cudaFree(d_count);
        cudaFree(d_degree);

        std::cout << "PASSED: edge extraction combined min_weight + max_degree stress (n=32, "
                  << num_runs << " runs, weight >= 0.5, degrees <= 4)\n";
    }

    // Edge extraction: combined min_edge_weight + max_degree stress (n=32, tighter cap=2)
    {
        constexpr int n = 32;
        constexpr size_t mat_size = static_cast<size_t>(n) * static_cast<size_t>(n);
        double h_dist[mat_size] = {};
        // Mix: ~1/3 edges at 0.1 (below min_weight=0.5), ~2/3 at 1.0.
        // With max_degree=2, each vertex accepts at most 2 edges among those >= 0.5.
        for (int i = 0; i < n; ++i)
        {
            for (int j = i + 1; j < n; ++j)
            {
                double w = ((i + j) % 3 == 0) ? 0.1 : 1.0;
                h_dist[i * n + j] = h_dist[j * n + i] = w;
            }
        }

        constexpr Size max_edges = 2048;
        constexpr double max_radius = 10.0;
        constexpr double min_edge_weight = 0.5;
        constexpr Size max_degree = 2;
        constexpr int num_runs = 100;

        double *d_dist = nullptr;
        Edge *d_edges = nullptr;
        Size *d_count = nullptr;
        Size *d_degree = nullptr;
        assert(check_cuda(cudaMalloc(&d_dist, mat_size * sizeof(double)), "cudaMalloc"));
        assert(check_cuda(cudaMalloc(&d_edges, max_edges * sizeof(Edge)), "cudaMalloc"));
        assert(check_cuda(cudaMalloc(&d_count, sizeof(Size)), "cudaMalloc"));
        assert(check_cuda(cudaMalloc(&d_degree, n * sizeof(Size)), "cudaMalloc"));
        cudaMemcpy(d_dist, h_dist, mat_size * sizeof(double), cudaMemcpyHostToDevice);

        // Theoretical max: floor(32*2/2) = 32 (all edges >= 0.5)
        constexpr Size theoretical_max = 32;

        for (int run = 0; run < num_runs; ++run)
        {
            cudaMemset(d_count, 0, sizeof(Size));
            cudaMemset(d_degree, 0, n * sizeof(Size));

            cuda_kernels::extractEdgesFilteredKernel(d_dist, d_edges, d_count, n, max_radius,
                                                     max_edges, min_edge_weight, max_degree,
                                                     d_degree);
            assert(check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize"));

            Size edge_count = 0;
            cudaMemcpy(&edge_count, d_count, sizeof(Size), cudaMemcpyDeviceToHost);
            assert(edge_count >= 1);
            assert(edge_count <= theoretical_max);
            static_cast<void>(theoretical_max);

            std::vector<Edge> edges(edge_count);
            cudaMemcpy(edges.data(), d_edges, edge_count * sizeof(Edge), cudaMemcpyDeviceToHost);

            // All output edges must have weight >= min_edge_weight (0.5)
            for (Size i = 0; i < edge_count; ++i)
            {
                assert(std::isfinite(edges[i].w));
                assert(edges[i].w >= min_edge_weight);
            }

            // No vertex may exceed max_degree (2)
            std::vector<Size> vertex_degree(n, 0);
            for (Size i = 0; i < edge_count; ++i)
            {
                ++vertex_degree[static_cast<size_t>(edges[i].u)];
                ++vertex_degree[static_cast<size_t>(edges[i].v)];
            }
            for (int v = 0; v < n; ++v)
                assert(vertex_degree[v] <= max_degree);
        }

        cudaFree(d_dist);
        cudaFree(d_edges);
        cudaFree(d_count);
        cudaFree(d_degree);

        std::cout << "PASSED: edge extraction combined min_weight + max_degree stress (n=32, "
                  << "cap=2, " << num_runs << " runs, weight >= 0.5, degrees <= 2)\n";
    }

    // Edge extraction: combined min_edge_weight + max_degree stress (n=64, repeated runs)
    {
        constexpr int n = 64;
        constexpr size_t mat_size = static_cast<size_t>(n) * static_cast<size_t>(n);
        double h_dist[mat_size] = {};
        // Mix: ~1/3 edges at 0.1 (below min_weight=0.5), ~2/3 at 1.0.
        // Pattern: edge (i,j) gets 0.1 if (i+j) % 3 == 0, else 1.0.
        // With max_degree=4, each vertex accepts at most 4 edges among those >= 0.5.
        for (int i = 0; i < n; ++i)
        {
            for (int j = i + 1; j < n; ++j)
            {
                double w = ((i + j) % 3 == 0) ? 0.1 : 1.0;
                h_dist[i * n + j] = h_dist[j * n + i] = w;
            }
        }

        constexpr Size max_edges = 4096;
        constexpr double max_radius = 10.0;
        constexpr double min_edge_weight = 0.5;
        constexpr Size max_degree = 4;
        constexpr int num_runs = 50;

        double *d_dist = nullptr;
        Edge *d_edges = nullptr;
        Size *d_count = nullptr;
        Size *d_degree = nullptr;
        assert(check_cuda(cudaMalloc(&d_dist, mat_size * sizeof(double)), "cudaMalloc"));
        assert(check_cuda(cudaMalloc(&d_edges, max_edges * sizeof(Edge)), "cudaMalloc"));
        assert(check_cuda(cudaMalloc(&d_count, sizeof(Size)), "cudaMalloc"));
        assert(check_cuda(cudaMalloc(&d_degree, n * sizeof(Size)), "cudaMalloc"));
        cudaMemcpy(d_dist, h_dist, mat_size * sizeof(double), cudaMemcpyHostToDevice);

        // Theoretical max: floor(64*4/2) = 128 (all edges >= 0.5)
        constexpr Size theoretical_max = 128;

        for (int run = 0; run < num_runs; ++run)
        {
            cudaMemset(d_count, 0, sizeof(Size));
            cudaMemset(d_degree, 0, n * sizeof(Size));

            cuda_kernels::extractEdgesFilteredKernel(d_dist, d_edges, d_count, n, max_radius,
                                                     max_edges, min_edge_weight, max_degree,
                                                     d_degree);
            assert(check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize"));

            Size edge_count = 0;
            cudaMemcpy(&edge_count, d_count, sizeof(Size), cudaMemcpyDeviceToHost);
            assert(edge_count >= 1);
            assert(edge_count <= theoretical_max);
            static_cast<void>(theoretical_max);

            std::vector<Edge> edges(edge_count);
            cudaMemcpy(edges.data(), d_edges, edge_count * sizeof(Edge), cudaMemcpyDeviceToHost);

            // All output edges must have weight >= min_edge_weight (0.5)
            for (Size i = 0; i < edge_count; ++i)
            {
                assert(std::isfinite(edges[i].w));
                assert(edges[i].w >= min_edge_weight);
            }

            // No vertex may exceed max_degree
            std::vector<Size> vertex_degree(n, 0);
            for (Size i = 0; i < edge_count; ++i)
            {
                ++vertex_degree[static_cast<size_t>(edges[i].u)];
                ++vertex_degree[static_cast<size_t>(edges[i].v)];
            }
            for (int v = 0; v < n; ++v)
                assert(vertex_degree[v] <= max_degree);
        }

        cudaFree(d_dist);
        cudaFree(d_edges);
        cudaFree(d_count);
        cudaFree(d_degree);

        std::cout << "PASSED: edge extraction combined min_weight + max_degree stress (n=64, "
                  << num_runs << " runs, weight >= 0.5, degrees <= 4)\n";
    }

    // Edge extraction: combined min_edge_weight + max_degree stress (n=64, absolute limit cap=2)
    {
        constexpr int n = 64;
        constexpr size_t mat_size = static_cast<size_t>(n) * static_cast<size_t>(n);
        double h_dist[mat_size] = {};
        // Mix: ~1/3 edges at 0.1 (below min_weight=0.5), ~2/3 at 1.0.
        // With max_degree=2, each vertex accepts at most 2 edges among those >= 0.5.
        for (int i = 0; i < n; ++i)
        {
            for (int j = i + 1; j < n; ++j)
            {
                double w = ((i + j) % 3 == 0) ? 0.1 : 1.0;
                h_dist[i * n + j] = h_dist[j * n + i] = w;
            }
        }

        constexpr Size max_edges = 4096;
        constexpr double max_radius = 10.0;
        constexpr double min_edge_weight = 0.5;
        constexpr Size max_degree = 2;
        constexpr int num_runs = 50;

        double *d_dist = nullptr;
        Edge *d_edges = nullptr;
        Size *d_count = nullptr;
        Size *d_degree = nullptr;
        assert(check_cuda(cudaMalloc(&d_dist, mat_size * sizeof(double)), "cudaMalloc"));
        assert(check_cuda(cudaMalloc(&d_edges, max_edges * sizeof(Edge)), "cudaMalloc"));
        assert(check_cuda(cudaMalloc(&d_count, sizeof(Size)), "cudaMalloc"));
        assert(check_cuda(cudaMalloc(&d_degree, n * sizeof(Size)), "cudaMalloc"));
        cudaMemcpy(d_dist, h_dist, mat_size * sizeof(double), cudaMemcpyHostToDevice);

        // Theoretical max: floor(64*2/2) = 64
        constexpr Size theoretical_max = 64;

        for (int run = 0; run < num_runs; ++run)
        {
            cudaMemset(d_count, 0, sizeof(Size));
            cudaMemset(d_degree, 0, n * sizeof(Size));

            cuda_kernels::extractEdgesFilteredKernel(d_dist, d_edges, d_count, n, max_radius,
                                                     max_edges, min_edge_weight, max_degree,
                                                     d_degree);
            assert(check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize"));

            Size edge_count = 0;
            cudaMemcpy(&edge_count, d_count, sizeof(Size), cudaMemcpyDeviceToHost);
            assert(edge_count >= 1);
            assert(edge_count <= theoretical_max);
            static_cast<void>(theoretical_max);

            std::vector<Edge> edges(edge_count);
            cudaMemcpy(edges.data(), d_edges, edge_count * sizeof(Edge), cudaMemcpyDeviceToHost);

            // All output edges must have weight >= min_edge_weight (0.5)
            for (Size i = 0; i < edge_count; ++i)
            {
                assert(std::isfinite(edges[i].w));
                assert(edges[i].w >= min_edge_weight);
            }

            // No vertex may exceed max_degree (2)
            std::vector<Size> vertex_degree(n, 0);
            for (Size i = 0; i < edge_count; ++i)
            {
                ++vertex_degree[static_cast<size_t>(edges[i].u)];
                ++vertex_degree[static_cast<size_t>(edges[i].v)];
            }
            for (int v = 0; v < n; ++v)
                assert(vertex_degree[v] <= max_degree);
        }

        cudaFree(d_dist);
        cudaFree(d_edges);
        cudaFree(d_count);
        cudaFree(d_degree);

        std::cout << "PASSED: edge extraction combined min_weight + max_degree stress (n=64, "
                  << "cap=2, " << num_runs << " runs, weight >= 0.5, degrees <= 2)\n";
    }


    return 0;
}

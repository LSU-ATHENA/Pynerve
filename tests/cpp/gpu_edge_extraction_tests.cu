#include "gpu_test_helpers.cuh"

int main()
{
    if (!has_gpu())
    {
        std::cerr
            << "No CUDA device available -- skipping GPU edge extraction kernel coverage tests\n";
        return 0;
    }
    // Edge extraction: basic kernel determinism
    {
        constexpr int n = 5;
        // d(0,1)=1.0, d(0,2)=2.5, d(0,3)=0.5, d(0,4)=4.0
        // d(1,2)=3.0, d(1,3)=2.0, d(1,4)=1.5
        // d(2,3)=1.0, d(2,4)=5.0
        // d(3,4)=0.8
        constexpr size_t mat_size = static_cast<size_t>(n) * static_cast<size_t>(n);
        double h_dist[mat_size] = {};
        fill_k5_distance_matrix(h_dist, n, 1.0, 2.5, 0.5, 4.0, 3.0, 2.0, 1.5, 1.0, 5.0, 0.8);

        constexpr Size max_edges = 20;
        constexpr double max_radius = 3.0;

        // First run: reference
        double *d_dist = nullptr;
        Edge *d_edges = nullptr;
        Size *d_count = nullptr;
        assert(check_cuda(cudaMalloc(&d_dist, mat_size * sizeof(double)), "cudaMalloc d_dist"));
        assert(check_cuda(cudaMalloc(&d_edges, max_edges * sizeof(Edge)), "cudaMalloc d_edges"));
        assert(check_cuda(cudaMalloc(&d_count, sizeof(Size)), "cudaMalloc d_count"));
        cudaMemcpy(d_dist, h_dist, mat_size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemset(d_count, 0, sizeof(Size));

        cuda_kernels::extractEdgesKernel(d_dist, d_edges, d_count, n, max_radius, max_edges);
        assert(check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize"));

        Size ref_count = 0;
        cudaMemcpy(&ref_count, d_count, sizeof(Size), cudaMemcpyDeviceToHost);
        std::vector<Edge> ref_edges(ref_count);
        cudaMemcpy(ref_edges.data(), d_edges, ref_count * sizeof(Edge), cudaMemcpyDeviceToHost);

        // 9 more runs: verify identical edge count
        for (int run = 1; run < 10; ++run)
        {
            cudaMemset(d_count, 0, sizeof(Size));
            cuda_kernels::extractEdgesKernel(d_dist, d_edges, d_count, n, max_radius, max_edges);
            assert(check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize"));

            Size cur_count = 0;
            cudaMemcpy(&cur_count, d_count, sizeof(Size), cudaMemcpyDeviceToHost);
            assert(cur_count == ref_count);
        }

        cudaFree(d_dist);
        cudaFree(d_edges);
        cudaFree(d_count);

        std::cout << "PASSED: edge extraction basic kernel determinism (10 runs)\n";
    }

    // Edge extraction: early termination kernel basic smoke
    {
        constexpr int n = 5;
        constexpr size_t mat_size = static_cast<size_t>(n) * static_cast<size_t>(n);
        double h_dist[mat_size] = {};
        // Same distance matrix as basic kernel determinism test
        fill_k5_distance_matrix(h_dist, n, 1.0, 2.5, 0.5, 4.0, 3.0, 2.0, 1.5, 1.0, 5.0, 0.8);

        constexpr Size max_edges = 20;
        constexpr double max_radius = 3.0;

        double *d_dist = nullptr;
        Edge *d_edges = nullptr;
        Size *d_count = nullptr;
        bool *d_stop = nullptr;
        assert(check_cuda(cudaMalloc(&d_dist, mat_size * sizeof(double)), "cudaMalloc"));
        assert(check_cuda(cudaMalloc(&d_edges, max_edges * sizeof(Edge)), "cudaMalloc"));
        assert(check_cuda(cudaMalloc(&d_count, sizeof(Size)), "cudaMalloc"));
        assert(check_cuda(cudaMalloc(&d_stop, sizeof(bool)), "cudaMalloc"));
        cudaMemcpy(d_dist, h_dist, mat_size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemset(d_count, 0, sizeof(Size));
        cudaMemset(d_stop, 0, sizeof(bool));

        cuda_kernels::extractEdgesEarlyTerminationKernel(d_dist, d_edges, d_count, n, max_radius,
                                                         max_edges, d_stop);
        assert(check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize"));

        Size edge_count = 0;
        cudaMemcpy(&edge_count, d_count, sizeof(Size), cudaMemcpyDeviceToHost);
        assert(edge_count > 0); // should find some edges

        cudaFree(d_dist);
        cudaFree(d_edges);
        cudaFree(d_count);
        cudaFree(d_stop);

        std::cout << "PASSED: edge extraction early termination kernel smoke\n";
    }

    // Edge extraction: shared memory kernel determinism
    {
        constexpr int n = 4;
        constexpr size_t mat_size = static_cast<size_t>(n) * static_cast<size_t>(n);
        double h_dist[mat_size] = {};
        fill_k4_distance_matrix(h_dist, n, 1.5, 2.0, 0.8, 3.5, 2.2, 1.1);

        constexpr Size max_edges = 20;
        constexpr double max_radius = 3.0;

        double *d_dist = nullptr;
        Edge *d_edges = nullptr;
        Size *d_count = nullptr;
        assert(check_cuda(cudaMalloc(&d_dist, mat_size * sizeof(double)), "cudaMalloc"));
        assert(check_cuda(cudaMalloc(&d_edges, max_edges * sizeof(Edge)), "cudaMalloc"));
        assert(check_cuda(cudaMalloc(&d_count, sizeof(Size)), "cudaMalloc"));
        cudaMemcpy(d_dist, h_dist, mat_size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemset(d_count, 0, sizeof(Size));

        cuda_kernels::extractEdgesSharedKernel(d_dist, d_edges, d_count, n, max_radius, max_edges);
        assert(check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize"));

        Size ref_count = 0;
        cudaMemcpy(&ref_count, d_count, sizeof(Size), cudaMemcpyDeviceToHost);

        // 9 more runs
        for (int run = 1; run < 10; ++run)
        {
            cudaMemset(d_count, 0, sizeof(Size));
            cuda_kernels::extractEdgesSharedKernel(d_dist, d_edges, d_count, n, max_radius,
                                                   max_edges);
            assert(check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize"));
            Size cur_count = 0;
            cudaMemcpy(&cur_count, d_count, sizeof(Size), cudaMemcpyDeviceToHost);
            assert(cur_count == ref_count);
        }

        cudaFree(d_dist);
        cudaFree(d_edges);
        cudaFree(d_count);

        std::cout << "PASSED: edge extraction shared kernel determinism (10 runs)\n";
    }

    // Edge extraction: filtered kernel smoke
    {
        constexpr int n = 4;
        constexpr size_t mat_size = static_cast<size_t>(n) * static_cast<size_t>(n);
        double h_dist[mat_size] = {};
        fill_k4_distance_matrix(h_dist, n, 0.5, 1.0, 2.0, 1.5, 0.3, 2.5);

        constexpr Size max_edges = 20;
        constexpr double max_radius = 3.0;
        constexpr double min_edge_weight = 0.4;
        constexpr Size max_degree = 10;

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

        cuda_kernels::extractEdgesFilteredKernel(d_dist, d_edges, d_count, n, max_radius, max_edges,
                                                 min_edge_weight, max_degree, d_degree);
        assert(check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize"));

        Size edge_count = 0;
        cudaMemcpy(&edge_count, d_count, sizeof(Size), cudaMemcpyDeviceToHost);
        // With min_edge_weight=0.4, edges (1,3)=0.3 should be filtered out
        assert(edge_count > 0);

        cudaFree(d_dist);
        cudaFree(d_edges);
        cudaFree(d_count);
        cudaFree(d_degree);

        std::cout << "PASSED: edge extraction filtered kernel smoke\n";
    }

    // Edge extraction: sorted kernel smoke
    {
        constexpr int n = 4;
        constexpr size_t mat_size = static_cast<size_t>(n) * static_cast<size_t>(n);
        double h_dist[mat_size] = {};
        fill_k4_distance_matrix(h_dist, n, 2.0, 1.0, 3.0, 0.5, 1.5, 2.5);

        constexpr Size max_edges = 20;
        constexpr double max_radius = 5.0;

        double *d_dist = nullptr;
        Edge *d_edges = nullptr;
        Size *d_count = nullptr;
        assert(check_cuda(cudaMalloc(&d_dist, mat_size * sizeof(double)), "cudaMalloc"));
        assert(check_cuda(cudaMalloc(&d_edges, max_edges * sizeof(Edge)), "cudaMalloc"));
        assert(check_cuda(cudaMalloc(&d_count, sizeof(Size)), "cudaMalloc"));
        cudaMemcpy(d_dist, h_dist, mat_size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemset(d_count, 0, sizeof(Size));

        cuda_kernels::extractEdgesSortedKernel(d_dist, d_edges, d_count, n, max_radius, max_edges,
                                               true);
        assert(check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize"));

        Size edge_count = 0;
        cudaMemcpy(&edge_count, d_count, sizeof(Size), cudaMemcpyDeviceToHost);
        assert(edge_count == 6); // complete graph K4 = 6 edges

        // Verify edges are sorted by weight
        std::vector<Edge> sorted_edges(edge_count);
        cudaMemcpy(sorted_edges.data(), d_edges, edge_count * sizeof(Edge), cudaMemcpyDeviceToHost);
        for (Size i = 1; i < edge_count; ++i)
            assert(sorted_edges[i - 1].w <= sorted_edges[i].w);

        cudaFree(d_dist);
        cudaFree(d_edges);
        cudaFree(d_count);

        std::cout << "PASSED: edge extraction sorted kernel (6 edges, monotonic weights)\n";
    }

    // Edge extraction: basic kernel correctness on known K3 distances
    {
        constexpr int n = 3;
        constexpr size_t mat_size = static_cast<size_t>(n) * static_cast<size_t>(n);
        double h_dist[mat_size] = {};
        fill_k3_distance_matrix(h_dist, n, 3.0, 4.0, 5.0);

        constexpr Size max_edges = 10;
        constexpr double max_radius = 10.0;

        double *d_dist = nullptr;
        Edge *d_edges = nullptr;
        Size *d_count = nullptr;
        assert(check_cuda(cudaMalloc(&d_dist, mat_size * sizeof(double)), "cudaMalloc"));
        assert(check_cuda(cudaMalloc(&d_edges, max_edges * sizeof(Edge)), "cudaMalloc"));
        assert(check_cuda(cudaMalloc(&d_count, sizeof(Size)), "cudaMalloc"));
        cudaMemcpy(d_dist, h_dist, mat_size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemset(d_count, 0, sizeof(Size));

        cuda_kernels::extractEdgesKernel(d_dist, d_edges, d_count, n, max_radius, max_edges);
        assert(check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize"));

        Size edge_count = 0;
        cudaMemcpy(&edge_count, d_count, sizeof(Size), cudaMemcpyDeviceToHost);
        assert(edge_count == 3); // K3 has 3 edges

        std::vector<Edge> edges(edge_count);
        cudaMemcpy(edges.data(), d_edges, edge_count * sizeof(Edge), cudaMemcpyDeviceToHost);

        // Expected edges (unordered pairs): (0,1,3.0), (0,2,4.0), (1,2,5.0)
        bool found_01 = false, found_02 = false, found_12 = false;
        for (Size i = 0; i < edge_count; ++i)
        {
            int u = edges[i].u;
            int v = edges[i].v;
            double w = edges[i].w;
            if ((u == 0 && v == 1) || (u == 1 && v == 0))
            {
                assert(approx_equal(w, 3.0));
                found_01 = true;
            }
            else if ((u == 0 && v == 2) || (u == 2 && v == 0))
            {
                assert(approx_equal(w, 4.0));
                found_02 = true;
            }
            else if ((u == 1 && v == 2) || (u == 2 && v == 1))
            {
                assert(approx_equal(w, 5.0));
                found_12 = true;
            }
        }
        assert(found_01 && found_02 && found_12);

        cudaFree(d_dist);
        cudaFree(d_edges);
        cudaFree(d_count);

        std::cout << "PASSED: edge extraction basic kernel K3 correctness (3 edges)\n";
    }

    // Edge extraction: basic kernel correctness with max_radius filtering
    {
        constexpr int n = 3;
        constexpr size_t mat_size = static_cast<size_t>(n) * static_cast<size_t>(n);
        double h_dist[mat_size] = {};
        fill_k3_distance_matrix(h_dist, n, 1.0, 2.0, 3.0);

        constexpr Size max_edges = 10;
        constexpr double max_radius = 1.5; // Should reject (0,2)=2.0 and (1,2)=3.0

        double *d_dist = nullptr;
        Edge *d_edges = nullptr;
        Size *d_count = nullptr;
        assert(check_cuda(cudaMalloc(&d_dist, mat_size * sizeof(double)), "cudaMalloc"));
        assert(check_cuda(cudaMalloc(&d_edges, max_edges * sizeof(Edge)), "cudaMalloc"));
        assert(check_cuda(cudaMalloc(&d_count, sizeof(Size)), "cudaMalloc"));
        cudaMemcpy(d_dist, h_dist, mat_size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemset(d_count, 0, sizeof(Size));

        cuda_kernels::extractEdgesKernel(d_dist, d_edges, d_count, n, max_radius, max_edges);
        assert(check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize"));

        Size edge_count = 0;
        cudaMemcpy(&edge_count, d_count, sizeof(Size), cudaMemcpyDeviceToHost);
        assert(edge_count == 1); // Only (0,1)=1.0 within radius

        std::vector<Edge> edges(edge_count);
        cudaMemcpy(edges.data(), d_edges, edge_count * sizeof(Edge), cudaMemcpyDeviceToHost);
        assert(approx_equal(edges[0].w, 1.0));

        cudaFree(d_dist);
        cudaFree(d_edges);
        cudaFree(d_count);

        std::cout << "PASSED: edge extraction max_radius filtering correctness (1 edge)\n";
    }

    // Edge extraction: early termination kernel correctness (all edges, no truncation)
    {
        constexpr int n = 4;
        constexpr size_t mat_size = static_cast<size_t>(n) * static_cast<size_t>(n);
        double h_dist[mat_size] = {};
        fill_k4_distance_matrix(h_dist, n, 0.5, 1.5, 2.5, 1.0, 2.0, 3.0);

        constexpr Size max_edges = 20; // Larger than K4's 6 edges
        constexpr double max_radius = 5.0;

        double *d_dist = nullptr;
        Edge *d_edges = nullptr;
        Size *d_count = nullptr;
        bool *d_stop = nullptr;
        assert(check_cuda(cudaMalloc(&d_dist, mat_size * sizeof(double)), "cudaMalloc"));
        assert(check_cuda(cudaMalloc(&d_edges, max_edges * sizeof(Edge)), "cudaMalloc"));
        assert(check_cuda(cudaMalloc(&d_count, sizeof(Size)), "cudaMalloc"));
        assert(check_cuda(cudaMalloc(&d_stop, sizeof(bool)), "cudaMalloc"));
        cudaMemcpy(d_dist, h_dist, mat_size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemset(d_count, 0, sizeof(Size));
        cudaMemset(d_stop, 0, sizeof(bool));

        cuda_kernels::extractEdgesEarlyTerminationKernel(d_dist, d_edges, d_count, n, max_radius,
                                                         max_edges, d_stop);
        assert(check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize"));

        Size edge_count = 0;
        cudaMemcpy(&edge_count, d_count, sizeof(Size), cudaMemcpyDeviceToHost);
        assert(edge_count == 6); // Full K4 = 6 edges, no early stop needed

        // Verify early termination flag was NOT set
        bool stopped = false;
        cudaMemcpy(&stopped, d_stop, sizeof(bool), cudaMemcpyDeviceToHost);
        assert(!stopped);

        cudaFree(d_dist);
        cudaFree(d_edges);
        cudaFree(d_count);
        cudaFree(d_stop);

        std::cout
            << "PASSED: edge extraction early termination K4 correctness (6 edges, no stop)\n";
    }

    // Edge extraction: early termination kernel truncation at max_edges
    {
        constexpr int n = 4;
        constexpr size_t mat_size = static_cast<size_t>(n) * static_cast<size_t>(n);
        double h_dist[mat_size] = {};
        fill_k4_distance_matrix(h_dist, n, 0.5, 1.5, 2.5, 1.0, 2.0, 3.0);

        constexpr Size max_edges = 3; // Only room for 3 of the 6 K4 edges
        constexpr double max_radius = 5.0;

        double *d_dist = nullptr;
        Edge *d_edges = nullptr;
        Size *d_count = nullptr;
        bool *d_stop = nullptr;
        assert(check_cuda(cudaMalloc(&d_dist, mat_size * sizeof(double)), "cudaMalloc"));
        assert(check_cuda(cudaMalloc(&d_edges, max_edges * sizeof(Edge)), "cudaMalloc"));
        assert(check_cuda(cudaMalloc(&d_count, sizeof(Size)), "cudaMalloc"));
        assert(check_cuda(cudaMalloc(&d_stop, sizeof(bool)), "cudaMalloc"));
        cudaMemcpy(d_dist, h_dist, mat_size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemset(d_count, 0, sizeof(Size));
        cudaMemset(d_stop, 0, sizeof(bool));

        cuda_kernels::extractEdgesEarlyTerminationKernel(d_dist, d_edges, d_count, n, max_radius,
                                                         max_edges, d_stop);
        assert(check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize"));

        Size edge_count = 0;
        cudaMemcpy(&edge_count, d_count, sizeof(Size), cudaMemcpyDeviceToHost);
        assert(edge_count >= max_edges); // At least max_edges were attempted

        // Early termination should have been triggered
        bool stopped = false;
        cudaMemcpy(&stopped, d_stop, sizeof(bool), cudaMemcpyDeviceToHost);
        assert(stopped);

        cudaFree(d_dist);
        cudaFree(d_edges);
        cudaFree(d_count);
        cudaFree(d_stop);

        std::cout << "PASSED: edge extraction early termination truncation (stopped at max=3)\n";
    }

    // Edge extraction: shared kernel K4 correctness
    {
        constexpr int n = 4;
        constexpr size_t mat_size = static_cast<size_t>(n) * static_cast<size_t>(n);
        double h_dist[mat_size] = {};
        fill_k4_distance_matrix(h_dist, n, 3.0, 1.0, 2.0, 2.5, 0.5, 1.5);

        constexpr Size max_edges = 20;
        constexpr double max_radius = 10.0;

        double *d_dist = nullptr;
        Edge *d_edges = nullptr;
        Size *d_count = nullptr;
        assert(check_cuda(cudaMalloc(&d_dist, mat_size * sizeof(double)), "cudaMalloc"));
        assert(check_cuda(cudaMalloc(&d_edges, max_edges * sizeof(Edge)), "cudaMalloc"));
        assert(check_cuda(cudaMalloc(&d_count, sizeof(Size)), "cudaMalloc"));
        cudaMemcpy(d_dist, h_dist, mat_size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemset(d_count, 0, sizeof(Size));

        cuda_kernels::extractEdgesSharedKernel(d_dist, d_edges, d_count, n, max_radius, max_edges);
        assert(check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize"));

        Size edge_count = 0;
        cudaMemcpy(&edge_count, d_count, sizeof(Size), cudaMemcpyDeviceToHost);
        assert(edge_count == 6); // Full K4

        std::vector<Edge> edges(edge_count);
        cudaMemcpy(edges.data(), d_edges, edge_count * sizeof(Edge), cudaMemcpyDeviceToHost);

        // Verify (1,3)=0.5 is present (smallest edge)
        bool found_13 = false;
        for (Size i = 0; i < edge_count; ++i)
        {
            if ((edges[i].u == 1 && edges[i].v == 3) || (edges[i].u == 3 && edges[i].v == 1))
            {
                assert(approx_equal(edges[i].w, 0.5));
                found_13 = true;
            }
        }
        assert(found_13);

        cudaFree(d_dist);
        cudaFree(d_edges);
        cudaFree(d_count);

        std::cout << "PASSED: edge extraction shared kernel K4 correctness (6 edges)\n";
    }

    // Edge extraction: filtered kernel correctness (min_edge_weight enforcement)
    {
        constexpr int n = 4;
        constexpr size_t mat_size = static_cast<size_t>(n) * static_cast<size_t>(n);
        double h_dist[mat_size] = {};
        fill_k4_distance_matrix(h_dist, n, 0.2, 1.0, 0.1, 0.5, 2.0, 1.5);
        // (0,1)=0.2 and (0,3)=0.1 are below min_edge_weight=0.4

        constexpr Size max_edges = 20;
        constexpr double max_radius = 5.0;
        constexpr double min_edge_weight = 0.4;
        constexpr Size max_degree = 10;

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

        cuda_kernels::extractEdgesFilteredKernel(d_dist, d_edges, d_count, n, max_radius, max_edges,
                                                 min_edge_weight, max_degree, d_degree);
        assert(check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize"));

        Size edge_count = 0;
        cudaMemcpy(&edge_count, d_count, sizeof(Size), cudaMemcpyDeviceToHost);
        // 6 K4 edges - 2 below min_weight (0.2, 0.1) = 4 edges
        assert(edge_count == 4);

        std::vector<Edge> edges(edge_count);
        cudaMemcpy(edges.data(), d_edges, edge_count * sizeof(Edge), cudaMemcpyDeviceToHost);

        // Verify no edge has weight < min_edge_weight
        for (Size i = 0; i < edge_count; ++i)
            assert(edges[i].w >= min_edge_weight);

        cudaFree(d_dist);
        cudaFree(d_edges);
        cudaFree(d_count);
        cudaFree(d_degree);

        std::cout << "PASSED: edge extraction filtered kernel correctness (4 of 6 pass filter)\n";
    }

    // Edge extraction: sorted kernel correctness (sorted order + all K4 edges)
    {
        constexpr int n = 4;
        constexpr size_t mat_size = static_cast<size_t>(n) * static_cast<size_t>(n);
        double h_dist[mat_size] = {};
        fill_k4_distance_matrix(h_dist, n, 3.0, 5.0, 1.0, 4.0, 0.5, 2.0);

        constexpr Size max_edges = 20;
        constexpr double max_radius = 10.0;

        double *d_dist = nullptr;
        Edge *d_edges = nullptr;
        Size *d_count = nullptr;
        assert(check_cuda(cudaMalloc(&d_dist, mat_size * sizeof(double)), "cudaMalloc"));
        assert(check_cuda(cudaMalloc(&d_edges, max_edges * sizeof(Edge)), "cudaMalloc"));
        assert(check_cuda(cudaMalloc(&d_count, sizeof(Size)), "cudaMalloc"));
        cudaMemcpy(d_dist, h_dist, mat_size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemset(d_count, 0, sizeof(Size));

        cuda_kernels::extractEdgesSortedKernel(d_dist, d_edges, d_count, n, max_radius, max_edges,
                                               true);
        assert(check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize"));

        Size edge_count = 0;
        cudaMemcpy(&edge_count, d_count, sizeof(Size), cudaMemcpyDeviceToHost);
        assert(edge_count == 6);

        std::vector<Edge> edges(edge_count);
        cudaMemcpy(edges.data(), d_edges, edge_count * sizeof(Edge), cudaMemcpyDeviceToHost);

        // Sorted order: weights should be 0.5, 1.0, 2.0, 3.0, 4.0, 5.0
        const double expected_weights[] = {0.5, 1.0, 2.0, 3.0, 4.0, 5.0};
        for (Size i = 0; i < edge_count; ++i)
            assert(approx_equal(edges[i].w, expected_weights[i]));

        cudaFree(d_dist);
        cudaFree(d_edges);
        cudaFree(d_count);

        std::cout << "PASSED: edge extraction sorted kernel correctness (6 edges, sorted)\n";
    }

    // Edge extraction: CUDAEgdeExtractor class correctness on K3
    {
        auto extractor_result = CUDAEgdeExtractor::create();
        assert(!extractor_result.isError());
        auto extractor = std::move(extractor_result.value());

        constexpr int n = 3;
        constexpr size_t mat_size = static_cast<size_t>(n) * static_cast<size_t>(n);
        double h_dist[mat_size] = {};
        fill_k3_distance_matrix(h_dist, n, 2.0, 3.0, 1.0);

        std::vector<Edge> h_edges(10);
        nerve::core::BufferView<const double> dist_view(h_dist, mat_size);
        nerve::core::BufferView<Edge> edge_view(h_edges.data(), h_edges.size());

        auto result = extractor->extractEdges(dist_view, edge_view, n, 10.0);
        assert(!result.isError());

        auto &stats = extractor->getPerformanceStats();
        assert(stats.edges_extracted == 3);

        // Verify specific edge weights are present
        bool found_1 = false, found_2 = false, found_3 = false;
        for (Size i = 0; i < stats.edges_extracted; ++i)
        {
            double w = h_edges[static_cast<size_t>(i)].w;
            if (approx_equal(w, 1.0))
                found_1 = true;
            else if (approx_equal(w, 2.0))
                found_2 = true;
            else if (approx_equal(w, 3.0))
                found_3 = true;
        }
        assert(found_1 && found_2 && found_3);

        std::cout << "PASSED: CUDAEgdeExtractor K3 correctness (3 edges, all weights verified)\n";
    }

    return 0;
}

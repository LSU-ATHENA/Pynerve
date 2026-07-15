#include "gpu_test_helpers.cuh"

int main()
{
    if (!has_gpu())
    {
        std::cerr << "No CUDA device available -- skipping GPU edge case extraction tests\n";
        return 0;
    }
    // Edge extraction: zero edges when all distances exceed max_radius
    {
        constexpr int n = 4;
        constexpr size_t mat_size = static_cast<size_t>(n) * static_cast<size_t>(n);
        double h_dist[mat_size] = {};
        fill_k4_distance_matrix(h_dist, n, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0);

        constexpr Size max_edges = 20;
        constexpr double max_radius = 1.0; // All edges > 1.0

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
        assert(edge_count == 0);

        cudaFree(d_dist);
        cudaFree(d_edges);
        cudaFree(d_count);

        std::cout << "PASSED: edge extraction zero edges (empty result)\n";
    }

    // Edge extraction: NaN edges in distance matrix (rejected by kernel guard)
    {
        constexpr int n = 4;
        constexpr size_t mat_size = static_cast<size_t>(n) * static_cast<size_t>(n);
        double h_dist[mat_size] = {};
        // K4 with one NaN edge: (0,1)=NaN, others finite
        fill_k4_distance_matrix(h_dist, n, std::numeric_limits<double>::quiet_NaN(), 2.0, 3.0, 4.0,
                                5.0, 6.0);

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

        cuda_kernels::extractEdgesKernel(d_dist, d_edges, d_count, n, max_radius, max_edges);
        assert(check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize"));

        Size edge_count = 0;
        cudaMemcpy(&edge_count, d_count, sizeof(Size), cudaMemcpyDeviceToHost);
        // NaN edge fails d <= max_radius, so 5 of 6 edges extracted
        assert(edge_count == 5);

        std::vector<Edge> edges(edge_count);
        cudaMemcpy(edges.data(), d_edges, edge_count * sizeof(Edge), cudaMemcpyDeviceToHost);

        // No NaN weights in output
        for (Size i = 0; i < edge_count; ++i)
            assert(std::isfinite(edges[i].w));
        // No edge (0,1) should be present
        for (Size i = 0; i < edge_count; ++i)
            assert(!((edges[i].u == 0 && edges[i].v == 1) || (edges[i].u == 1 && edges[i].v == 0)));

        cudaFree(d_dist);
        cudaFree(d_edges);
        cudaFree(d_count);

        std::cout << "PASSED: edge extraction NaN edge rejected (5 of 6 extracted)\n";
    }

    // Edge extraction: Inf edges in distance matrix (rejected by kernel guard)
    {
        constexpr int n = 4;
        constexpr size_t mat_size = static_cast<size_t>(n) * static_cast<size_t>(n);
        double h_dist[mat_size] = {};
        // K4 with two Inf edges: (0,2)=inf, (1,3)=inf
        fill_k4_distance_matrix(h_dist, n, 1.0, std::numeric_limits<double>::infinity(), 3.0, 4.0,
                                std::numeric_limits<double>::infinity(), 6.0);

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

        cuda_kernels::extractEdgesKernel(d_dist, d_edges, d_count, n, max_radius, max_edges);
        assert(check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize"));

        Size edge_count = 0;
        cudaMemcpy(&edge_count, d_count, sizeof(Size), cudaMemcpyDeviceToHost);
        // Two Inf edges rejected, 4 of 6 extracted
        assert(edge_count == 4);

        std::vector<Edge> edges(edge_count);
        cudaMemcpy(edges.data(), d_edges, edge_count * sizeof(Edge), cudaMemcpyDeviceToHost);

        // No Inf weights in output
        for (Size i = 0; i < edge_count; ++i)
            assert(std::isfinite(edges[i].w));
        // Explicitly verify Inf edges (0,2) and (1,3) are absent
        for (Size i = 0; i < edge_count; ++i)
        {
            assert(!((edges[i].u == 0 && edges[i].v == 2) || (edges[i].u == 2 && edges[i].v == 0)));
            assert(!((edges[i].u == 1 && edges[i].v == 3) || (edges[i].u == 3 && edges[i].v == 1)));
        }

        cudaFree(d_dist);
        cudaFree(d_edges);
        cudaFree(d_count);

        std::cout << "PASSED: edge extraction Inf edges rejected (4 of 6 extracted)\n";
    }

    // Edge extraction: all-zero distance matrix (all edges weight 0.0)
    {
        constexpr int n = 4;
        constexpr size_t mat_size = static_cast<size_t>(n) * static_cast<size_t>(n);
        double h_dist[mat_size] = {};
        // All edges = 0.0 -- diagonal already 0, off-diagonal set by helper
        fill_k4_distance_matrix(h_dist, n, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

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

        cuda_kernels::extractEdgesKernel(d_dist, d_edges, d_count, n, max_radius, max_edges);
        assert(check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize"));

        Size edge_count = 0;
        cudaMemcpy(&edge_count, d_count, sizeof(Size), cudaMemcpyDeviceToHost);
        // 0.0 <= max_radius is true, so all 6 K4 edges are extracted
        assert(edge_count == 6);

        std::vector<Edge> edges(edge_count);
        cudaMemcpy(edges.data(), d_edges, edge_count * sizeof(Edge), cudaMemcpyDeviceToHost);

        // All edges must have weight exactly 0.0
        for (Size i = 0; i < edge_count; ++i)
            assert(approx_equal(edges[i].w, 0.0, 1e-12));

        cudaFree(d_dist);
        cudaFree(d_edges);
        cudaFree(d_count);

        std::cout << "PASSED: edge extraction all-zero distance matrix (6 edges, all weight 0.0)\n";
    }

    // Edge extraction: K6 complete graph correctness (15 edges)
    {
        constexpr int n = 6;
        constexpr size_t mat_size = static_cast<size_t>(n) * static_cast<size_t>(n);
        double h_dist[mat_size] = {};
        // Distinct weights 0.1--1.5 so each edge is uniquely identifiable.
        // Edges: (0,1)=0.1, (0,2)=0.2, (0,3)=0.3, (0,4)=0.4, (0,5)=0.5
        //        (1,2)=0.6, (1,3)=0.7, (1,4)=0.8, (1,5)=0.9
        //        (2,3)=1.0, (2,4)=1.1, (2,5)=1.2
        //        (3,4)=1.3, (3,5)=1.4
        //        (4,5)=1.5
        fill_k6_distance_matrix(h_dist, n, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1,
                                1.2, 1.3, 1.4, 1.5);

        constexpr Size max_edges = 30;
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
        assert(edge_count == 15); // K6 has 15 edges

        std::vector<Edge> edges(edge_count);
        cudaMemcpy(edges.data(), d_edges, edge_count * sizeof(Edge), cudaMemcpyDeviceToHost);

        // Spot-check representative edges across the triangular enumeration
        bool found_01 = false, found_05 = false, found_15 = false;
        bool found_23 = false, found_34 = false, found_45 = false;
        for (Size i = 0; i < edge_count; ++i)
        {
            int u = edges[i].u, v = edges[i].v;
            double w = edges[i].w;
            if ((u == 0 && v == 1) || (u == 1 && v == 0))
            {
                assert(approx_equal(w, 0.1));
                found_01 = true;
            }
            else if ((u == 0 && v == 5) || (u == 5 && v == 0))
            {
                assert(approx_equal(w, 0.5));
                found_05 = true;
            }
            else if ((u == 1 && v == 5) || (u == 5 && v == 1))
            {
                assert(approx_equal(w, 0.9));
                found_15 = true;
            }
            else if ((u == 2 && v == 3) || (u == 3 && v == 2))
            {
                assert(approx_equal(w, 1.0));
                found_23 = true;
            }
            else if ((u == 3 && v == 4) || (u == 4 && v == 3))
            {
                assert(approx_equal(w, 1.3));
                found_34 = true;
            }
            else if ((u == 4 && v == 5) || (u == 5 && v == 4))
            {
                assert(approx_equal(w, 1.5));
                found_45 = true;
            }
        }
        assert(found_01 && found_05 && found_15);
        assert(found_23 && found_34 && found_45);

        cudaFree(d_dist);
        cudaFree(d_edges);
        cudaFree(d_count);

        std::cout << "PASSED: edge extraction K6 correctness (15 edges, 6 spot-checked)\n";
    }

    return 0;
}

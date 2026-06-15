#include "nerve/persistence/cuda/cuda_matrix_reduction.hpp"
#include "nerve/persistence/utils/api.hpp"
#include "nerve/types.hpp"

#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace
{

bool check_cuda(cudaError_t code, const char *expression)
{
    if (code == cudaSuccess)
        return true;
    std::cerr << expression << " failed: " << cudaGetErrorString(code) << '\n';
    return false;
}

bool has_gpu()
{
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return err == cudaSuccess && device_count > 0;
}

} // namespace

int main()
{
    if (!has_gpu())
    {
        std::cerr << "No CUDA device available  --  skipping GPU persistence reduction tests\n";
        return 0;
    }

    // Test 1: Simple triangle (3 vertices, 3 edges, 1 triangle) -> 1 H0 pair
    {
        // Construct a small boundary matrix for a triangle
        // Simplices: v0(0), v1(1), v2(2), e01(3), e02(4), e12(5), t(6)
        // Boundaries: v0=[], v1=[], v2=[], e01=[0,1], e02=[0,2], e12=[1,2], t=[3,4,5]
        const int columns_data[] = {
            0, 1,   // column 3: edge 01
            0, 2,   // column 4: edge 02
            1, 2,   // column 5: edge 12
            3, 4, 5 // column 6: triangle
        };
        const nerve::Size column_sizes[] = {0, 0, 0, 2, 2, 2, 3};
        const double weights[] = {0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0};

        auto reduction = nerve::persistence::accelerated::CUDAMatrixReduction::create(
            nerve::persistence::accelerated::MatrixReductionConfig{2, true});
        assert(reduction.isSuccess());

        auto result =
            reduction.value()->compute_reduction(columns_data, column_sizes, weights, 7, 2);
        assert(result.isSuccess());

        const auto &stats = reduction.value()->get_performance_stats();
        assert(stats.columns_processed == 7);
        assert(stats.pairs_created >= 1);
    }

    // Test 2: Square with diagonal (4 vertices, 5 edges, 2 triangles)
    {
        // Vertices: v0(0) at (0,0), v1(1) at (1,0), v2(2) at (0,1), v3(3) at (1,1)
        // Edges: e01(4)=[0,1], e02(5)=[0,2], e13(6)=[1,3], e23(7)=[2,3], e03(8)=[0,3]
        // Triangles: t1(9)=[4,6,8], t2(10)=[5,7,8]
        // Sorted order: v0..v3 (0-3), e*(4-8), t*(9-10)

        const int columns_data[] = {
            0, 1,    // column 4: e01
            0, 2,    // column 5: e02
            1, 3,    // column 6: e13
            2, 3,    // column 7: e23
            0, 3,    // column 8: e03 diagonal
            4, 6, 8, // column 9: t1
            5, 7, 8  // column 10: t2
        };
        const nerve::Size column_sizes[] = {0, 0, 0, 0, 2, 2, 2, 2, 2, 3, 3};
        const double weights[] = {0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.414, 2.0, 2.0};

        auto reduction = nerve::persistence::accelerated::CUDAMatrixReduction::create(
            nerve::persistence::accelerated::MatrixReductionConfig{2, true});
        assert(reduction.isSuccess());

        auto result =
            reduction.value()->compute_reduction(columns_data, column_sizes, weights, 11, 2);
        assert(result.isSuccess());

        const auto &stats = reduction.value()->get_performance_stats();
        assert(stats.columns_processed == 11);
        assert(stats.pairs_created >= 2);
    }

    // Test 3: Compare GPU output vs CPU persistence on a small point cloud
    {
        const double points[] = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5};

        // CPU path
        nerve::persistence::PersistenceOptions cpu_opts;
        cpu_opts.backend = nerve::persistence::PersistenceBackend::CPU_EXACT;
        cpu_opts.max_dim = 1;
        cpu_opts.max_radius = 2.0;

        auto cpu_result = nerve::persistence::compute(
            nerve::core::BufferView<const double>(points, 10), 2, cpu_opts);
        assert(cpu_result.isSuccess());
        assert(!cpu_result.value().pairs.empty());

        // GPU path
        nerve::persistence::PersistenceOptions gpu_opts;
        gpu_opts.backend = nerve::persistence::PersistenceBackend::CUDA_HYBRID;
        gpu_opts.max_dim = 1;
        gpu_opts.max_radius = 2.0;

        auto gpu_result = nerve::persistence::compute(
            nerve::core::BufferView<const double>(points, 10), 2, gpu_opts);
        assert(gpu_result.isSuccess());
        assert(!gpu_result.value().pairs.empty());

        // Both should produce H0 pairs
        assert(cpu_result.value().pairs.size() == gpu_result.value().pairs.size());
    }

    // Test 4: Cohomology variant via persistence API
    {
        const double points[] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
                                 0.0, 1.0, 1.0, 0.0, 0.5, 0.5, 0.2};

        nerve::persistence::PersistenceOptions opts;
        opts.backend = nerve::persistence::PersistenceBackend::CUDA_HYBRID;
        opts.max_dim = 1;
        opts.max_radius = 2.0;

        auto result = nerve::persistence::computePersistenceCohomology(
            nerve::core::BufferView<const double>(points, 15), 3, opts);
        assert(result.isSuccess());
        assert(!result.value().pairs.empty());

        bool has_h0 = false;
        for (const auto &pair : result.value().pairs)
        {
            assert(std::isfinite(pair.birth));
            if (pair.dimension == 0)
                has_h0 = true;
        }
        assert(has_h0);
    }

    // Test 5: Clearing optimization
    {
        nerve::persistence::accelerated::MatrixReductionConfig config;
        config.max_dim = 2;
        config.enable_clearing = true;
        config.enable_performance_monitoring = true;

        auto reduction = nerve::persistence::accelerated::CUDAMatrixReduction::create(config);
        assert(reduction.isSuccess());

        const int columns_data[] = {0, 1, 0, 2, 1, 2, 3, 4, 5};
        const nerve::Size column_sizes[] = {0, 0, 0, 2, 2, 2, 3};
        const double weights[] = {0.0, 0.0, 0.0, 1.0, 1.5, 2.0, 3.0};

        auto result =
            reduction.value()->compute_reduction(columns_data, column_sizes, weights, 7, 2);
        assert(result.isSuccess());

        const auto &stats = reduction.value()->get_performance_stats();
        if (config.enable_performance_monitoring)
        {
            assert(stats.total_time_ms >= 0.0);
        }
        assert(stats.pairs_created >= 1);
    }

    // Test 6: Error path  --  null inputs
    {
        auto reduction = nerve::persistence::accelerated::CUDAMatrixReduction::create();
        assert(reduction.isSuccess());

        auto result = reduction.value()->compute_reduction(nullptr, nullptr, nullptr, 0, 2);
        assert(result.isError());
    }

    // Test 7: Error path  --  zero columns
    {
        auto reduction = nerve::persistence::accelerated::CUDAMatrixReduction::create();
        assert(reduction.isSuccess());

        const double dummy_weight = 0.0;
        auto result = reduction.value()->compute_reduction(nullptr, nullptr, &dummy_weight, 0, 1);
        assert(result.isError());
    }

    // Test 8: Error path  --  negative dimensions
    {
        auto reduction = nerve::persistence::accelerated::CUDAMatrixReduction::create();
        assert(reduction.isSuccess());

        int dummy_col = 0;
        nerve::Size dummy_size = 1;
        double dummy_weight = 0.0;
        auto result = reduction.value()->compute_reduction(&dummy_col, &dummy_size, &dummy_weight,
                                                           1, static_cast<nerve::Size>(-1));
        assert(result.isError());
    }

    return 0;
}

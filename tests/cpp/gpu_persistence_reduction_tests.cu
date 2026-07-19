// GPU persistence reduction tests.
// Tests CUDAMatrixReduction on known small complexes, compares vs CPU
// persistence API, exercises cohomology variant, clearing optimization,
// error paths, dim-3+ boundary matrices, and stress tests.
//
// Label: persistence;reduction;gpu;cuda;regression

#include "nerve/persistence/cuda/cuda_matrix_reduction.hpp"
#include "nerve/persistence/utils/api.hpp"
#include "nerve/types.hpp"

#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
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

void assert_pair_count_positive(nerve::persistence::accelerated::CUDAMatrixReduction &red)
{
    const auto &stats = red.get_performance_stats();
    if (stats.pairs_created < 1)
    {
        std::cerr << "FAIL: expected at least 1 pair, got " << stats.pairs_created << "\n";
        std::exit(1);
    }
    if (stats.columns_processed == 0)
    {
        std::cerr << "FAIL: expected columns_processed > 0\n";
        std::exit(1);
    }
}

} // namespace

int main()
{
    if (!has_gpu())
    {
        std::cerr << "No CUDA device available -- skipping GPU persistence reduction tests\n";
        return 0;
    }

    // Simple triangle (3 vertices, 3 edges, 1 triangle) -> dim-2
    {
        const int columns_data[] = {0, 1, 0, 2, 1, 2, 3, 4, 5};
        const nerve::Size column_sizes[] = {0, 0, 0, 2, 2, 2, 3};
        const double weights[] = {0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0};

        auto reduction = nerve::persistence::accelerated::CUDAMatrixReduction::create(
            nerve::persistence::accelerated::MatrixReductionConfig{2, true});
        assert(reduction.isSuccess());

        auto result =
            reduction.value()->compute_reduction(columns_data, column_sizes, weights, 7, 2);
        assert(result.isSuccess());
        assert_pair_count_positive(*reduction.value());
    }

    // Square with diagonal (4 vertices, 5 edges, 2 triangles) dim-2
    {
        const int columns_data[] = {0, 1, 0, 2, 1, 3, 2, 3, 0, 3, 4, 6, 8, 5, 7, 8};
        const nerve::Size column_sizes[] = {0, 0, 0, 0, 2, 2, 2, 2, 2, 3, 3};
        const double weights[] = {0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.414, 2.0, 2.0};

        auto reduction = nerve::persistence::accelerated::CUDAMatrixReduction::create(
            nerve::persistence::accelerated::MatrixReductionConfig{2, true});
        assert(reduction.isSuccess());

        auto result =
            reduction.value()->compute_reduction(columns_data, column_sizes, weights, 11, 2);
        assert(result.isSuccess());
        assert_pair_count_positive(*reduction.value());
    }

    // Tetrahedron (4 vertices, 6 edges, 4 triangles, 1 tetrahedron) dim-3 boundary matrix
    {
        const int columns_data[] = {0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2,  3,  4,  5,
                                    7, 4, 6, 8, 5, 6, 9, 7, 8, 9, 10, 11, 12, 13};
        const nerve::Size column_sizes[] = {0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4};
        const double weights[] = {0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3};

        auto reduction = nerve::persistence::accelerated::CUDAMatrixReduction::create(
            nerve::persistence::accelerated::MatrixReductionConfig{3, true});
        assert(reduction.isSuccess());
        auto result =
            reduction.value()->compute_reduction(columns_data, column_sizes, weights, 15, 3);
        assert(result.isSuccess());
        assert_pair_count_positive(*reduction.value());
        const auto &stats = reduction.value()->get_performance_stats();
        assert(stats.columns_processed == 15);
        assert(stats.pairs_created >= 3);
    }

    // Octahedron (6 vertices, 12 edges, 8 triangles) dim-2
    {
        const int columns_data[] = {0,  1, 0, 2,  0,  3,  0,  4,  1,  5,  2,  5,  3,  5,  4,  5,
                                    1,  2, 2, 3,  3,  4,  4,  1,  6,  8,  14, 6,  9,  15, 7,  9,
                                    16, 7, 8, 17, 10, 14, 17, 11, 15, 14, 12, 16, 15, 13, 17, 16};

        nerve::Size column_sizes[26];
        double weights[26];
        for (int i = 0; i < 6; ++i)
        {
            column_sizes[i] = 0;
            weights[i] = 0.0;
        }
        for (int i = 6; i < 18; ++i)
        {
            column_sizes[i] = 2;
            weights[i] = 1.0;
        }
        for (int i = 18; i < 26; ++i)
        {
            column_sizes[i] = 3;
            weights[i] = 2.0;
        }

        auto reduction = nerve::persistence::accelerated::CUDAMatrixReduction::create(
            nerve::persistence::accelerated::MatrixReductionConfig{2, true});
        assert(reduction.isSuccess());
        auto result =
            reduction.value()->compute_reduction(columns_data, column_sizes, weights, 26, 2);
        assert(result.isSuccess());
        assert(reduction.value()->get_performance_stats().pairs_created >= 3);
    }

    // Compare GPU output vs CPU persistence on a small point cloud
    {
        const double points[] = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5};

        nerve::persistence::PersistenceOptions cpu_opts;
        cpu_opts.backend = nerve::persistence::PersistenceBackend::CPU_EXACT;
        cpu_opts.max_dim = 1;
        cpu_opts.max_radius = 2.0;

        auto cpu_result = nerve::persistence::compute(
            nerve::core::BufferView<const double>(points, 10), 2, cpu_opts);
        assert(cpu_result.isSuccess());
        assert(!cpu_result.value().pairs.empty());

        nerve::persistence::PersistenceOptions gpu_opts;
        gpu_opts.backend = nerve::persistence::PersistenceBackend::CUDA_HYBRID;
        gpu_opts.max_dim = 1;
        gpu_opts.max_radius = 2.0;

        auto gpu_result = nerve::persistence::compute(
            nerve::core::BufferView<const double>(points, 10), 2, gpu_opts);
        assert(gpu_result.isSuccess());
        assert(!gpu_result.value().pairs.empty());

        if (cpu_result.value().pairs.size() != gpu_result.value().pairs.size())
        {
            std::cerr << "FAIL: CPU produced " << cpu_result.value().pairs.size()
                      << " pairs, GPU produced " << gpu_result.value().pairs.size() << "\n";
            return 1;
        }
    }

    // Cohomology variant via persistence API
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
            if (!std::isfinite(pair.birth))
            {
                std::cerr << "FAIL: non-finite birth value in pair\n";
                return 1;
            }
            if (pair.dimension == 0)
                has_h0 = true;
        }
        if (!has_h0)
        {
            std::cerr << "FAIL: expected at least one H0 pair\n";
            return 1;
        }
    }

    // Clearing optimization
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
            if (stats.total_time_ms < 0.0)
            {
                std::cerr << "FAIL: negative timing\n";
                return 1;
            }
        }
        if (stats.pairs_created < 1)
        {
            std::cerr << "FAIL: clearing test produced no pairs\n";
            return 1;
        }
    }

    // Octahedron clearing: clearing vs non-clearing produce same pairs (dim-2)
    // Uses a larger 26-column octahedron boundary matrix with 8 triangles
    {
        const int columns_data[] = {0,  1, 0, 2,  0,  3,  0,  4,  1,  5,  2,  5,  3,  5,  4,  5,
                                    1,  2, 2, 3,  3,  4,  4,  1,  6,  8,  14, 6,  9,  15, 7,  9,
                                    16, 7, 8, 17, 10, 14, 17, 11, 15, 14, 12, 16, 15, 13, 17, 16};

        nerve::Size column_sizes[26];
        double weights[26];
        for (int i = 0; i < 6; ++i)
        {
            column_sizes[i] = 0;
            weights[i] = 0.0;
        }
        for (int i = 6; i < 18; ++i)
        {
            column_sizes[i] = 2;
            weights[i] = 1.0;
        }
        for (int i = 18; i < 26; ++i)
        {
            column_sizes[i] = 3;
            weights[i] = 2.0;
        }

        nerve::persistence::accelerated::MatrixReductionConfig config_clear;
        config_clear.max_dim = 2;
        config_clear.enable_clearing = true;
        config_clear.enable_performance_monitoring = true;

        auto red_clear = nerve::persistence::accelerated::CUDAMatrixReduction::create(config_clear);
        assert(red_clear.isSuccess());
        auto result_clear =
            red_clear.value()->compute_reduction(columns_data, column_sizes, weights, 26, 2);
        assert(result_clear.isSuccess());
        const auto &stats_clear = red_clear.value()->get_performance_stats();

        nerve::persistence::accelerated::MatrixReductionConfig config_no_clear;
        config_no_clear.max_dim = 2;
        config_no_clear.enable_clearing = false;
        config_no_clear.enable_performance_monitoring = true;

        auto red_no_clear =
            nerve::persistence::accelerated::CUDAMatrixReduction::create(config_no_clear);
        assert(red_no_clear.isSuccess());
        auto result_no_clear =
            red_no_clear.value()->compute_reduction(columns_data, column_sizes, weights, 26, 2);
        assert(result_no_clear.isSuccess());
        const auto &stats_no_clear = red_no_clear.value()->get_performance_stats();

        if (stats_clear.pairs_created != stats_no_clear.pairs_created)
        {
            std::cerr << "FAIL: octahedron clearing pairs=" << stats_clear.pairs_created
                      << " vs non-clearing pairs=" << stats_no_clear.pairs_created << "\n";
            return 1;
        }
        if (stats_clear.columns_processed != 26)
        {
            std::cerr << "FAIL: expected 26 columns with clearing, got "
                      << stats_clear.columns_processed << "\n";
            return 1;
        }
        if (stats_clear.total_time_ms < 0.0)
        {
            std::cerr << "FAIL: negative clearing timing\n";
            return 1;
        }
    }

    // Cubical grid clearing: clearing vs non-clearing across dim-2 and dim-3
    // 8 vertices, 12 edges, 6 faces, 1 cube -> 27 columns total
    {
        const int columns_data[] = {0, 1,  1, 3,  3, 2, 2, 0,  4, 5, 5, 7,  7, 6, 6, 4, 0, 4,
                                    1, 5,  2, 6,  3, 7, 0, 1,  2, 3, 4, 5,  6, 7, 0, 8, 4, 9,
                                    2, 10, 6, 11, 3, 8, 7, 10, 1, 9, 5, 11, 0, 1, 2, 3, 4, 5};

        nerve::Size column_sizes[27];
        double weights[27];
        for (int i = 0; i < 8; ++i)
        {
            column_sizes[i] = 0;
            weights[i] = 0.0;
        }
        for (int i = 8; i < 20; ++i)
        {
            column_sizes[i] = 2;
            weights[i] = 1.0;
        }
        for (int i = 20; i < 26; ++i)
        {
            column_sizes[i] = 4;
            weights[i] = 2.0;
        }
        column_sizes[26] = 6;
        weights[26] = 3.0;

        // dim-2 clearing comparison
        for (int dim = 2; dim <= 3; ++dim)
        {
            nerve::persistence::accelerated::MatrixReductionConfig cfg_clear;
            cfg_clear.max_dim = static_cast<nerve::Size>(dim);
            cfg_clear.enable_clearing = true;
            cfg_clear.enable_performance_monitoring = true;

            auto red_clear =
                nerve::persistence::accelerated::CUDAMatrixReduction::create(cfg_clear);
            assert(red_clear.isSuccess());
            auto res_clear =
                red_clear.value()->compute_reduction(columns_data, column_sizes, weights, 27, dim);
            assert(res_clear.isSuccess());
            const auto &s_clear = red_clear.value()->get_performance_stats();

            nerve::persistence::accelerated::MatrixReductionConfig cfg_no_clear;
            cfg_no_clear.max_dim = static_cast<nerve::Size>(dim);
            cfg_no_clear.enable_clearing = false;
            cfg_no_clear.enable_performance_monitoring = true;

            auto red_no_clear =
                nerve::persistence::accelerated::CUDAMatrixReduction::create(cfg_no_clear);
            assert(red_no_clear.isSuccess());
            auto res_no_clear = red_no_clear.value()->compute_reduction(columns_data, column_sizes,
                                                                        weights, 27, dim);
            assert(res_no_clear.isSuccess());
            const auto &s_no_clear = red_no_clear.value()->get_performance_stats();

            if (s_clear.pairs_created != s_no_clear.pairs_created)
            {
                std::cerr << "FAIL: cubical grid dim-" << dim
                          << " clearing pairs=" << s_clear.pairs_created
                          << " vs non-clearing pairs=" << s_no_clear.pairs_created << "\n";
                return 1;
            }
            if (s_clear.columns_processed != 27)
            {
                std::cerr << "FAIL: cubical grid dim-" << dim << " expected 27 columns, got "
                          << s_clear.columns_processed << "\n";
                return 1;
            }
        }
    }

    // Clearing reproducibility: octahedron with clearing, 5 runs
    {
        const int columns_data[] = {0,  1, 0, 2,  0,  3,  0,  4,  1,  5,  2,  5,  3,  5,  4,  5,
                                    1,  2, 2, 3,  3,  4,  4,  1,  6,  8,  14, 6,  9,  15, 7,  9,
                                    16, 7, 8, 17, 10, 14, 17, 11, 15, 14, 12, 16, 15, 13, 17, 16};

        nerve::Size column_sizes[26];
        double weights[26];
        for (int i = 0; i < 6; ++i)
        {
            column_sizes[i] = 0;
            weights[i] = 0.0;
        }
        for (int i = 6; i < 18; ++i)
        {
            column_sizes[i] = 2;
            weights[i] = 1.0;
        }
        for (int i = 18; i < 26; ++i)
        {
            column_sizes[i] = 3;
            weights[i] = 2.0;
        }

        nerve::persistence::accelerated::MatrixReductionConfig config;
        config.max_dim = 2;
        config.enable_clearing = true;

        int prev_pairs = -1;
        for (int run = 0; run < 5; ++run)
        {
            auto reduction = nerve::persistence::accelerated::CUDAMatrixReduction::create(config);
            assert(reduction.isSuccess());
            auto result =
                reduction.value()->compute_reduction(columns_data, column_sizes, weights, 26, 2);
            assert(result.isSuccess());
            const auto &stats = reduction.value()->get_performance_stats();

            if (run == 0)
            {
                prev_pairs = static_cast<int>(stats.pairs_created);
                if (prev_pairs == 0)
                {
                    std::cerr << "FAIL: clearing reproducibility run 0 produced 0 pairs\n";
                    return 1;
                }
            }
            else if (static_cast<int>(stats.pairs_created) != prev_pairs)
            {
                std::cerr << "FAIL: clearing reproducibility: run " << run << " produced "
                          << stats.pairs_created << " pairs but run 0 produced " << prev_pairs
                          << "\n";
                return 1;
            }
        }
    }

    // Clearing stress test: large random point cloud via persistence API,
    // compare clearing (CUDA_HYBRID) vs CPU reference
    {
        const int n_pts = 25;
        const float threshold = 0.8f;
        const unsigned seed = 4242;

        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        std::vector<double> pts(static_cast<std::size_t>(n_pts * 3));
        for (auto &v : pts)
            v = dist(rng);

        nerve::persistence::PersistenceOptions gpu_opts;
        gpu_opts.backend = nerve::persistence::PersistenceBackend::CUDA_HYBRID;
        gpu_opts.max_dim = 2;
        gpu_opts.max_radius = threshold;

        auto gpu_result = nerve::persistence::compute(
            nerve::core::BufferView<const double>(pts.data(), pts.size()), 3, gpu_opts);
        assert(gpu_result.isSuccess());

        nerve::persistence::PersistenceOptions cpu_opts;
        cpu_opts.backend = nerve::persistence::PersistenceBackend::CPU_EXACT;
        cpu_opts.max_dim = 2;
        cpu_opts.max_radius = threshold;

        auto cpu_result = nerve::persistence::compute(
            nerve::core::BufferView<const double>(pts.data(), pts.size()), 3, cpu_opts);
        assert(cpu_result.isSuccess());

        if (gpu_result.value().pairs.empty())
        {
            std::cerr << "FAIL: clearing stress GPU produced no pairs\n";
            return 1;
        }
        if (cpu_result.value().pairs.empty())
        {
            std::cerr << "FAIL: clearing stress CPU produced no pairs\n";
            return 1;
        }
        if (gpu_result.value().pairs.size() != cpu_result.value().pairs.size())
        {
            std::cerr << "FAIL: clearing stress GPU pairs=" << gpu_result.value().pairs.size()
                      << " vs CPU pairs=" << cpu_result.value().pairs.size() << "\n";
            return 1;
        }
        for (const auto &pair : gpu_result.value().pairs)
        {
            if (!std::isfinite(pair.birth))
            {
                std::cerr << "FAIL: clearing stress non-finite birth\n";
                return 1;
            }
        }
    }

    // Square clearing: clearing vs non-clearing produce same pairs (dim-2)
    // 4 vertices, 5 edges, 2 triangles -> 11 columns
    {
        const int columns_data[] = {0, 1, 0, 2, 1, 3, 2, 3, 0, 3, 4, 6, 8, 5, 7, 8};
        const nerve::Size column_sizes[] = {0, 0, 0, 0, 2, 2, 2, 2, 2, 3, 3};
        const double weights[] = {0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.414, 2.0, 2.0};

        nerve::persistence::accelerated::MatrixReductionConfig config_clear;
        config_clear.max_dim = 2;
        config_clear.enable_clearing = true;

        auto red_clear = nerve::persistence::accelerated::CUDAMatrixReduction::create(config_clear);
        assert(red_clear.isSuccess());
        auto res_clear =
            red_clear.value()->compute_reduction(columns_data, column_sizes, weights, 11, 2);
        assert(res_clear.isSuccess());
        const auto &s_clear = red_clear.value()->get_performance_stats();

        nerve::persistence::accelerated::MatrixReductionConfig config_no_clear;
        config_no_clear.max_dim = 2;
        config_no_clear.enable_clearing = false;

        auto red_no_clear =
            nerve::persistence::accelerated::CUDAMatrixReduction::create(config_no_clear);
        assert(red_no_clear.isSuccess());
        auto res_no_clear =
            red_no_clear.value()->compute_reduction(columns_data, column_sizes, weights, 11, 2);
        assert(res_no_clear.isSuccess());
        const auto &s_no_clear = red_no_clear.value()->get_performance_stats();

        if (s_clear.pairs_created != s_no_clear.pairs_created)
        {
            std::cerr << "FAIL: square clearing pairs=" << s_clear.pairs_created
                      << " vs non-clearing pairs=" << s_no_clear.pairs_created << "\n";
            return 1;
        }
        if (s_clear.columns_processed != 11)
        {
            std::cerr << "FAIL: square expected 11 columns, got " << s_clear.columns_processed
                      << "\n";
            return 1;
        }
    }

    // Tetrahedron clearing: clearing vs non-clearing (dim-3)
    // 4 vertices, 6 edges, 4 triangles, 1 tetrahedron -> 15 columns
    {
        const int columns_data[] = {0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2,  3,  4,  5,
                                    7, 4, 6, 8, 5, 6, 9, 7, 8, 9, 10, 11, 12, 13};
        const nerve::Size column_sizes[] = {0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4};
        const double weights[] = {0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3};

        nerve::persistence::accelerated::MatrixReductionConfig config_clear;
        config_clear.max_dim = 3;
        config_clear.enable_clearing = true;

        auto red_clear = nerve::persistence::accelerated::CUDAMatrixReduction::create(config_clear);
        assert(red_clear.isSuccess());
        auto res_clear =
            red_clear.value()->compute_reduction(columns_data, column_sizes, weights, 15, 3);
        assert(res_clear.isSuccess());
        const auto &s_clear = red_clear.value()->get_performance_stats();

        nerve::persistence::accelerated::MatrixReductionConfig config_no_clear;
        config_no_clear.max_dim = 3;
        config_no_clear.enable_clearing = false;

        auto red_no_clear =
            nerve::persistence::accelerated::CUDAMatrixReduction::create(config_no_clear);
        assert(red_no_clear.isSuccess());
        auto res_no_clear =
            red_no_clear.value()->compute_reduction(columns_data, column_sizes, weights, 15, 3);
        assert(res_no_clear.isSuccess());
        const auto &s_no_clear = red_no_clear.value()->get_performance_stats();

        if (s_clear.pairs_created != s_no_clear.pairs_created)
        {
            std::cerr << "FAIL: tetrahedron dim-3 clearing pairs=" << s_clear.pairs_created
                      << " vs non-clearing pairs=" << s_no_clear.pairs_created << "\n";
            return 1;
        }
        if (s_clear.columns_processed != 15)
        {
            std::cerr << "FAIL: tetrahedron expected 15 columns, got " << s_clear.columns_processed
                      << "\n";
            return 1;
        }
        if (s_clear.pairs_created < 3)
        {
            std::cerr << "FAIL: tetrahedron produced only " << s_clear.pairs_created << " pairs\n";
            return 1;
        }
    }

    // Octahedron multi-dim: run with clearing at dim-2 and dim-3 on same data
    {
        const int columns_data[] = {0,  1, 0, 2,  0,  3,  0,  4,  1,  5,  2,  5,  3,  5,  4,  5,
                                    1,  2, 2, 3,  3,  4,  4,  1,  6,  8,  14, 6,  9,  15, 7,  9,
                                    16, 7, 8, 17, 10, 14, 17, 11, 15, 14, 12, 16, 15, 13, 17, 16};

        nerve::Size column_sizes[26];
        double weights[26];
        for (int i = 0; i < 6; ++i)
        {
            column_sizes[i] = 0;
            weights[i] = 0.0;
        }
        for (int i = 6; i < 18; ++i)
        {
            column_sizes[i] = 2;
            weights[i] = 1.0;
        }
        for (int i = 18; i < 26; ++i)
        {
            column_sizes[i] = 3;
            weights[i] = 2.0;
        }

        for (int dim = 1; dim <= 3; ++dim)
        {
            nerve::persistence::accelerated::MatrixReductionConfig cfg;
            cfg.max_dim = static_cast<nerve::Size>(dim);
            cfg.enable_clearing = true;

            auto red = nerve::persistence::accelerated::CUDAMatrixReduction::create(cfg);
            assert(red.isSuccess());
            auto res = red.value()->compute_reduction(columns_data, column_sizes, weights, 26, dim);
            assert(res.isSuccess());
            const auto &stats = red.value()->get_performance_stats();

            if (stats.columns_processed != 26)
            {
                std::cerr << "FAIL: octahedron multi-dim dim=" << dim
                          << " columns=" << stats.columns_processed << " expected 26\n";
                return 1;
            }
            // dim-1 reduction should produce at most 5 H0 pairs (6 vertices - 1)
            if (dim == 1 && stats.pairs_created > 6)
            {
                std::cerr << "FAIL: octahedron dim-1 pairs=" << stats.pairs_created
                          << " expected <=6\n";
                return 1;
            }
        }
    }

    // Clearing with various max_dim and config combinations
    {
        const int columns_data[] = {0, 1, 0, 2, 1, 2, 3, 4, 5};
        const nerve::Size column_sizes[] = {0, 0, 0, 2, 2, 2, 3};
        const double weights[] = {0.0, 0.0, 0.0, 1.0, 1.5, 2.0, 3.0};

        // configurations with clearing
        struct TestConfig
        {
            nerve::Size max_dim;
            bool enable_clearing;
            bool enable_streaming;
        };

        TestConfig configs[] = {
            {1, true, false}, {2, true, false},  {2, true, true},  {2, false, true},
            {3, true, false}, {3, false, false}, {2, true, false}, // repeat to verify determinism
        };

        int config_idx = -1;
        int prev_pairs = -1;
        for (const auto &tc : configs)
        {
            ++config_idx;

            nerve::persistence::accelerated::MatrixReductionConfig cfg;
            cfg.max_dim = tc.max_dim;
            cfg.enable_clearing = tc.enable_clearing;
            cfg.enable_streaming = tc.enable_streaming;

            auto red = nerve::persistence::accelerated::CUDAMatrixReduction::create(cfg);
            assert(red.isSuccess());
            auto res =
                red.value()->compute_reduction(columns_data, column_sizes, weights, 7, tc.max_dim);
            assert(res.isSuccess());
            const auto &stats = red.value()->get_performance_stats();

            if (stats.columns_processed != 7)
            {
                std::cerr << "FAIL: config " << config_idx << " max_dim=" << tc.max_dim
                          << " clear=" << tc.enable_clearing << " stream=" << tc.enable_streaming
                          << " columns=" << stats.columns_processed << " expected 7\n";
                return 1;
            }

            if (!tc.enable_clearing && !tc.enable_streaming)
            {
                // Non-default config: just verify it works
                if (stats.pairs_created < 1)
                {
                    std::cerr << "FAIL: config " << config_idx
                              << " (no clear, no stream) produced no pairs\n";
                    return 1;
                }
            }

            // Check determinism for identical config (last entry == first with clearing)
            if (config_idx == 6) // last entry
            {
                if (static_cast<int>(stats.pairs_created) != prev_pairs)
                {
                    std::cerr << "FAIL: config determinism: pairs=" << stats.pairs_created
                              << " vs previous " << prev_pairs << " for same config\n";
                    return 1;
                }
            }
            else if (config_idx == 0)
            {
                prev_pairs = static_cast<int>(stats.pairs_created);
            }
        }
    }

    // Performance monitoring stats sanity
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

        if (stats.get_reduction_rate() < 0.0)
        {
            std::cerr << "FAIL: invalid reduction rate\n";
            return 1;
        }
        if (stats.get_efficiency_score() <= 0.0)
        {
            std::cerr << "FAIL: non-positive efficiency score\n";
            return 1;
        }
        if (stats.get_memory_efficiency() <= 0.0 || stats.get_memory_efficiency() > 1.0)
        {
            std::cerr << "FAIL: memory efficiency out of range: " << stats.get_memory_efficiency()
                      << "\n";
            return 1;
        }
        assert(stats.columns_processed > 0);
        assert(stats.total_time_ms >= 0.0);
    }

    // Error path -- null inputs
    {
        auto reduction = nerve::persistence::accelerated::CUDAMatrixReduction::create();
        assert(reduction.isSuccess());

        auto result = reduction.value()->compute_reduction(nullptr, nullptr, nullptr, 0, 2);
        if (result.isSuccess())
        {
            std::cerr << "FAIL: expected error for null inputs\n";
            return 1;
        }
    }

    // Error path -- zero columns
    {
        auto reduction = nerve::persistence::accelerated::CUDAMatrixReduction::create();
        assert(reduction.isSuccess());

        const double dummy_weight = 0.0;
        auto result = reduction.value()->compute_reduction(nullptr, nullptr, &dummy_weight, 0, 1);
        // Zero columns with non-null weights is a valid edge case (nothing to compute)
        assert(result.isSuccess());
    }

    // Error path -- negative dimensions
    {
        auto reduction = nerve::persistence::accelerated::CUDAMatrixReduction::create();
        assert(reduction.isSuccess());

        int dummy_col = 0;
        nerve::Size dummy_size = 1;
        double dummy_weight = 0.0;
        auto result = reduction.value()->compute_reduction(&dummy_col, &dummy_size, &dummy_weight,
                                                           1, static_cast<nerve::Size>(-1));
        if (result.isSuccess())
        {
            std::cerr << "FAIL: expected error for negative dim\n";
            return 1;
        }
    }

    // Stress -- larger random boundary matrix
    {
        const int n_pts = 30;
        const float threshold = 0.6f;
        const unsigned seed = 7777;

        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        std::vector<double> pts(static_cast<std::size_t>(n_pts * 3));
        for (auto &v : pts)
            v = dist(rng);

        nerve::persistence::PersistenceOptions gpu_opts;
        gpu_opts.backend = nerve::persistence::PersistenceBackend::CUDA_HYBRID;
        gpu_opts.max_dim = 2;
        gpu_opts.max_radius = threshold;

        auto gpu_result = nerve::persistence::compute(
            nerve::core::BufferView<const double>(pts.data(), pts.size()), 3, gpu_opts);
        assert(gpu_result.isSuccess());

        nerve::persistence::PersistenceOptions cpu_opts;
        cpu_opts.backend = nerve::persistence::PersistenceBackend::CPU_EXACT;
        cpu_opts.max_dim = 2;
        cpu_opts.max_radius = threshold;

        auto cpu_result = nerve::persistence::compute(
            nerve::core::BufferView<const double>(pts.data(), pts.size()), 3, cpu_opts);
        assert(cpu_result.isSuccess());

        assert(!gpu_result.value().pairs.empty());
        assert(!cpu_result.value().pairs.empty());

        for (const auto &pair : gpu_result.value().pairs)
        {
            if (!std::isfinite(pair.birth))
            {
                std::cerr << "FAIL: non-finite birth in GPU result\n";
                return 1;
            }
        }
    }

    // Dim-3 persistence on a cubical grid (8 vertices, 12 edges, 6 faces, 1 cube)
    {
        const int columns_data[] = {0, 1,  1, 3,  3, 2, 2, 0,  4, 5, 5, 7,  7, 6, 6, 4, 0, 4,
                                    1, 5,  2, 6,  3, 7, 0, 1,  2, 3, 4, 5,  6, 7, 0, 8, 4, 9,
                                    2, 10, 6, 11, 3, 8, 7, 10, 1, 9, 5, 11, 0, 1, 2, 3, 4, 5};

        nerve::Size column_sizes[27];
        double weights[27];
        for (int i = 0; i < 8; ++i)
        {
            column_sizes[i] = 0;
            weights[i] = 0.0;
        }
        for (int i = 8; i < 20; ++i)
        {
            column_sizes[i] = 2;
            weights[i] = 1.0;
        }
        for (int i = 20; i < 26; ++i)
        {
            column_sizes[i] = 4;
            weights[i] = 2.0;
        }
        column_sizes[26] = 6;
        weights[26] = 3.0;

        // dim-2 boundary matrix (triangles x edges)
        {
            auto reduction = nerve::persistence::accelerated::CUDAMatrixReduction::create(
                nerve::persistence::accelerated::MatrixReductionConfig{2, true});
            assert(reduction.isSuccess());
            auto result =
                reduction.value()->compute_reduction(columns_data, column_sizes, weights, 27, 2);
            assert(result.isSuccess());
            const auto &stats = reduction.value()->get_performance_stats();
            assert(stats.columns_processed > 0);
        }

        // dim-3 boundary matrix (tetrahedra x triangles)
        {
            auto reduction = nerve::persistence::accelerated::CUDAMatrixReduction::create(
                nerve::persistence::accelerated::MatrixReductionConfig{3, true});
            assert(reduction.isSuccess());
            auto result =
                reduction.value()->compute_reduction(columns_data, column_sizes, weights, 27, 3);
            assert(result.isSuccess());
            const auto &stats = reduction.value()->get_performance_stats();
            assert(stats.columns_processed > 0);
        }
    }

    // Multiple identical configs (reproducibility check)
    {
        const int columns_data[] = {0, 1, 0, 2, 1, 2, 3, 4, 5};
        const nerve::Size column_sizes[] = {0, 0, 0, 2, 2, 2, 3};
        const double weights[] = {0.0, 0.0, 0.0, 1.0, 1.5, 2.0, 3.0};

        int prev_pairs = -1;
        for (int run = 0; run < 5; ++run)
        {
            auto reduction = nerve::persistence::accelerated::CUDAMatrixReduction::create(
                nerve::persistence::accelerated::MatrixReductionConfig{2, true});
            assert(reduction.isSuccess());
            auto result =
                reduction.value()->compute_reduction(columns_data, column_sizes, weights, 7, 2);
            assert(result.isSuccess());
            const auto &stats = reduction.value()->get_performance_stats();

            if (run == 0)
            {
                prev_pairs = static_cast<int>(stats.pairs_created);
                if (prev_pairs == 0)
                {
                    std::cerr << "FAIL: run 0 produced 0 pairs\n";
                    return 1;
                }
            }
            else if (static_cast<int>(stats.pairs_created) != prev_pairs)
            {
                std::cerr << "FAIL: reproducibility: run " << run << " produced "
                          << stats.pairs_created << " pairs but run 0 produced " << prev_pairs
                          << "\n";
                return 1;
            }
        }
    }

    // Large random point cloud stress test
    {
        const int n_pts = 50;
        const float threshold = 0.5f;
        const unsigned seed = 12345;

        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        std::vector<double> pts(static_cast<std::size_t>(n_pts * 3));
        for (auto &v : pts)
            v = dist(rng);

        {
            nerve::persistence::PersistenceOptions opts;
            opts.backend = nerve::persistence::PersistenceBackend::CUDA_HYBRID;
            opts.max_dim = 0;
            opts.max_radius = threshold;

            auto result = nerve::persistence::compute(
                nerve::core::BufferView<const double>(pts.data(), pts.size()), 3, opts);
            assert(result.isSuccess());
        }
    }

    return 0;
}

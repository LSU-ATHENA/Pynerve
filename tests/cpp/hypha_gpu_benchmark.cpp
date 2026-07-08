// Benchmark: HyphaReducer (GPU boundary scan) vs reduceMatrixLockfree (CPU)
//
// Generates random Vietoris-Rips filtrations at increasing sizes and reports
// wall-clock time and speedup for each reducer.

#include "nerve/algebra/boundary.hpp"
#include "nerve/persistence/reduction/reduction_hypha_ops.hpp"
#include "hypha_test_helpers.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#ifdef NERVE_HAS_CUDA

#include <cuda_runtime.h>

namespace
{

bool has_gpu()
{
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return err == cudaSuccess && device_count > 0;
}



struct PhaseBreakdown
{
    double csc_build_ms = 0.0;
    double clearing_ms = 0.0;
    double submatrix_build_ms = 0.0;
    double gpu_pack_ms = 0.0;
    double gpu_reduction_ms = 0.0;
    double gpu_download_ms = 0.0;
    double overhead_ms = 0.0;
};

struct BenchmarkResult
{
    int num_points;
    nerve::Size num_simplices;
    nerve::Size num_edges;
    nerve::Size num_triangles;
    nerve::Size num_tetrahedra;
    nerve::Size num_columns;
    double hypha_time_ms;
    double lockfree_time_ms;
    double speedup;
    int hypha_pair_count;
    int lockfree_pair_count;
    bool pairs_match;
    PhaseBreakdown phases;
    nerve::persistence::LockfreeProfile lockfree_profile;
};

BenchmarkResult run_benchmark(int num_points, float threshold, unsigned seed, int dim)
{
    BenchmarkResult result{};
    result.num_points = num_points;
    auto points = nerve::test::hypha::random_point_cloud(num_points, seed);
    auto complex = nerve::test::hypha::build_vr_complex(points, threshold);
    if (dim == 3)
        nerve::test::hypha::build_tetrahedra(complex);

    auto all_simplices = complex.getSimplices();
    result.num_simplices = static_cast<nerve::Size>(all_simplices.size());
    for (const auto &s : all_simplices)
    {
        if (s.dimension() == 1) ++result.num_edges;
        if (s.dimension() == 2) ++result.num_triangles;
        if (s.dimension() == 3) ++result.num_tetrahedra;
    }

    nerve::algebra::BoundaryMatrix bm(complex, static_cast<nerve::Size>(dim));
    result.num_columns = bm.cols();
    nerve::Size n_cols = bm.cols();
    if (n_cols == 0)
    {
        result.hypha_time_ms = 0.0;
        result.lockfree_time_ms = 0.0;
        result.speedup = 1.0;
        result.pairs_match = true;
        return result;
    }

    // Warmup
    nerve::persistence::HyphaReducer warmup_hr;
    warmup_hr.compute(bm);

    auto lockfree_boundary = std::vector<std::vector<int>>();
    auto lockfree_filtration = std::vector<double>();
    auto lockfree_row_filtration = std::vector<double>();
    auto lockfree_dims = std::vector<nerve::Dimension>();
    nerve::test::hypha::to_lockfree_format(bm, lockfree_boundary, lockfree_filtration, lockfree_row_filtration, lockfree_dims);
    auto _ = nerve::persistence::reduceMatrixLockfree(
        lockfree_boundary, lockfree_filtration, &lockfree_row_filtration, lockfree_dims,
        nerve::persistence::recommendedThreadCount());

    constexpr int kTrials = 3;
    std::vector<double> hypha_times;
    hypha_times.reserve(kTrials);
    std::vector<nerve::persistence::HyphaPhaseTimings> phase_samples;
    phase_samples.reserve(kTrials);
    for (int t = 0; t < kTrials; ++t)
    {
        nerve::persistence::HyphaReducer hr;
        nerve::persistence::HyphaPhaseTimings pt;
        auto start = std::chrono::high_resolution_clock::now();
        auto pairs = hr.compute(bm, &pt);
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        if (t > 0)
        {
            hypha_times.push_back(ms);
            phase_samples.push_back(pt);
        }
        if (t == kTrials - 1)
        {
            result.hypha_pair_count = static_cast<int>(pairs.size());
            result.pairs_match = !pairs.empty();
        }
    }

    // Aggregate per-phase timings (median across timed trials)
    if (!phase_samples.empty())
    {
        auto median_phase = [&](auto getter) -> double {
            std::vector<double> vals;
            for (auto &s : phase_samples) vals.push_back(getter(s));
            std::sort(vals.begin(), vals.end());
            return vals[vals.size() / 2];
        };
        result.phases.csc_build_ms =
            median_phase([](const auto &s) { return s.csc_build_ms; });
        result.phases.clearing_ms =
            median_phase([](const auto &s) { return s.clearing_ms; });
        result.phases.submatrix_build_ms =
            median_phase([](const auto &s) { return s.submatrix_build_ms; });
        result.phases.gpu_pack_ms =
            median_phase([](const auto &s) { return s.gpu_pack_ms; });
        result.phases.gpu_reduction_ms =
            median_phase([](const auto &s) { return s.gpu_reduction_ms; });
        result.phases.gpu_download_ms =
            median_phase([](const auto &s) { return s.gpu_download_ms; });
        result.phases.overhead_ms =
            median_phase([](const auto &s) { return s.overhead_ms; });
    }

    std::vector<double> lockfree_times;
    lockfree_times.reserve(kTrials);
    for (int t = 0; t < kTrials; ++t)
    {
        nerve::persistence::LockfreeProfile lf_prof;
        auto start = std::chrono::high_resolution_clock::now();
        auto pairs = nerve::persistence::reduceMatrixLockfreeProfiled(
            lockfree_boundary, lockfree_filtration, &lockfree_row_filtration, lockfree_dims,
            nerve::persistence::recommendedThreadCount(), &lf_prof);
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        if (t > 0) lockfree_times.push_back(ms);
        if (t == kTrials - 1)
        {
            result.lockfree_pair_count = static_cast<int>(pairs.size());
            result.pairs_match = result.pairs_match && (result.hypha_pair_count == static_cast<int>(pairs.size()));
            result.lockfree_profile = lf_prof;
        }
    }

    auto median = [](std::vector<double> &v) -> double {
        if (v.empty()) return 0.0;
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    result.hypha_time_ms = median(hypha_times);
    result.lockfree_time_ms = median(lockfree_times);
    result.speedup = (result.lockfree_time_ms > 0.0) ? result.lockfree_time_ms / result.hypha_time_ms : 1.0;
    return result;
}

} // anonymous namespace

int main()
{
    if (!has_gpu())
    {
        std::cerr << "No CUDA device available -- skipping hypha GPU benchmark\n";
        return 0;
    }

    // Write output to both stdout and a result file (for retrieval after CTest)
    auto write = [](const std::string &line) {
        std::cout << line << std::flush;
        FILE *f = fopen("/tmp/hypha_benchmark_results.txt", "a");
        if (f) { fputs(line.c_str(), f); fclose(f); }
    };

    // Clear result file
    FILE *f = fopen("/tmp/hypha_benchmark_results.txt", "w");
    if (f) fclose(f);

    char buf[1024];

    std::snprintf(buf, sizeof(buf), "%s\n", "========================================================================");
    write(buf);
    std::snprintf(buf, sizeof(buf), "%s\n", "  HyphaReducer (GPU) vs reduceMatrixLockfree (CPU) Benchmark");
    write(buf);
    std::snprintf(buf, sizeof(buf), "%s\n", "  Random 3D point clouds, VR filtration (dim<=2, dim=3 with tetrahedra)");
    write(buf);
    std::snprintf(buf, sizeof(buf), "%s\n\n", "========================================================================");
    write(buf);

    auto print_header = [&](int dim) {
        if (dim == 3)
        {
            std::snprintf(buf, sizeof(buf), "  %-6s | %-11s | %-7s | %-11s | %-12s | %-11s | %-14s | %-9s | %-6s | %s\n",
                "Points", "Simplices", "Edges", "Triangles", "Tetrahedra", "Hypha(ms)", "Lockfree(ms)", "Speedup", "Cols", "OK");
            write(buf);
            std::snprintf(buf, sizeof(buf), "  %s\n",
                "-------+-------------+---------+-------------+--------------+-------------+----------------+-----------+--------+-----");
            write(buf);
        }
        else
        {
            std::snprintf(buf, sizeof(buf), "  %-6s | %-11s | %-7s | %-11s | %-11s | %-14s | %-9s | %-6s | %s\n",
                "Points", "Simplices", "Edges", "Triangles", "Hypha(ms)", "Lockfree(ms)", "Speedup", "Pairs", "OK");
            write(buf);
            std::snprintf(buf, sizeof(buf), "  %s\n",
                "-------+-------------+---------+-------------+-------------+----------------+-----------+--------+-----");
            write(buf);
        }
    };

    // Dim-2 benchmarks
    struct BenchConfig { int points; float threshold; unsigned seed; int dim; };
    BenchConfig configs[] = {
        {20,   0.8f,  42,  2},
        {50,   0.6f,  42,  2},
        {100,  0.5f,  42,  2},
        {200,  0.40f, 42,  2},
        {500,  0.28f, 42,  2},
        {800,  0.20f, 42,  2},
        {1200, 0.15f, 42,  2},
        {2000, 0.10f, 42,  2},
        {10,   1.0f,  42,  3},
        {15,   0.9f,  42,  3},
        {20,   0.8f,  42,  3},
        {30,   0.7f,  42,  3},
        {50,   0.5f,  42,  3},
    };

    int prev_dim = 2;
    print_header(prev_dim);

    for (const auto &cfg : configs)
    {
        if (cfg.dim != prev_dim)
        {
            // Dim-3 section header
            std::snprintf(buf, sizeof(buf), "  %s\n", "");
            write(buf);
            std::snprintf(buf, sizeof(buf), "  %s\n", "  Dim-3 (with tetrahedra):");
            write(buf);
            print_header(cfg.dim);
            prev_dim = cfg.dim;
        }

        auto result = run_benchmark(cfg.points, cfg.threshold, cfg.seed, cfg.dim);

        if (result.lockfree_time_ms > 0.0)
        {
            if (cfg.dim == 3)
            {
                std::snprintf(buf, sizeof(buf),
                    "  %6d | %11zu | %7zu | %11zu | %12zu | %11.2f | %14.2f | %9.2fx | %6zu | %s\n",
                    result.num_points,
                    static_cast<std::size_t>(result.num_simplices),
                    static_cast<std::size_t>(result.num_edges),
                    static_cast<std::size_t>(result.num_triangles),
                    static_cast<std::size_t>(result.num_tetrahedra),
                    result.hypha_time_ms,
                    result.lockfree_time_ms,
                    result.speedup,
                    static_cast<std::size_t>(result.num_columns),
                    result.pairs_match ? "OK" : "MISMATCH");
            }
            else
            {
                std::snprintf(buf, sizeof(buf),
                    "  %6d | %11zu | %7zu | %11zu | %11.2f | %14.2f | %9.2fx | %4d/%-d | %s\n",
                    result.num_points,
                    static_cast<std::size_t>(result.num_simplices),
                    static_cast<std::size_t>(result.num_edges),
                    static_cast<std::size_t>(result.num_triangles),
                    result.hypha_time_ms,
                    result.lockfree_time_ms,
                    result.speedup,
                    result.hypha_pair_count, result.lockfree_pair_count,
                    result.pairs_match ? "OK" : "MISMATCH");
            }
        }
        else
        {
            if (cfg.dim == 3)
            {
                std::snprintf(buf, sizeof(buf),
                    "  %6d | %11zu | %7zu | %11zu | %12zu | %11s | %14s | %9s | %6s | %s\n",
                    result.num_points,
                    static_cast<std::size_t>(result.num_simplices),
                    static_cast<std::size_t>(result.num_edges),
                    static_cast<std::size_t>(result.num_triangles),
                    static_cast<std::size_t>(result.num_tetrahedra),
                    "N/A", "N/A", "N/A", "N/A",
                    result.pairs_match ? "OK" : "MISMATCH");
            }
            else
            {
                std::snprintf(buf, sizeof(buf),
                    "  %6d | %11zu | %7zu | %11zu | %11s | %14s | %9s | %6s | %s\n",
                    result.num_points,
                    static_cast<std::size_t>(result.num_simplices),
                    static_cast<std::size_t>(result.num_edges),
                    static_cast<std::size_t>(result.num_triangles),
                    "N/A", "N/A", "N/A", "N/A",
                    result.pairs_match ? "OK" : "MISMATCH");
            }
        }
        write(buf);

        // Phase breakdown
        auto &p = result.phases;
        if (p.csc_build_ms > 0.0 || p.gpu_reduction_ms > 0.0)
        {
            std::snprintf(buf, sizeof(buf),
                "  %46s  %9s: %8.2f ms | clearing: %8.2f ms | submatrix: %8.2f ms | pack: %8.2f ms | GPU red: %8.2f ms | download: %8.2f ms | overhead: %8.2f ms\n",
                "", "CSC build", p.csc_build_ms, p.clearing_ms,
                p.submatrix_build_ms, p.gpu_pack_ms,
                p.gpu_reduction_ms, p.gpu_download_ms, p.overhead_ms);
            write(buf);

            // Lockfree profile breakdown for largest configs only
            if (cfg.points >= 100 && result.lockfree_profile.add_column_calls > 0)
            {
                auto &lp = result.lockfree_profile;
                double total_lf = lp.column_init_ms + lp.coboundary_build_ms +
                    lp.atomics_init_ms + lp.queue_setup_ms + lp.worker_reduction_ms +
                    lp.pair_extract_ms;
                std::snprintf(buf, sizeof(buf),
                    "  %46s  [lockfree profile: %d cols, %zu nnz, %d rows, %d threads]\n",
                    "", lp.num_columns, lp.nnz, lp.num_rows, lp.num_threads);
                write(buf);
                std::snprintf(buf, sizeof(buf),
                    "  %46s  col init: %6.2f ms | coboundary: %6.2f ms | atomics: %6.2f ms | queue: %6.2f ms\n",
                    "", lp.column_init_ms, lp.coboundary_build_ms,
                    lp.atomics_init_ms, lp.queue_setup_ms);
                write(buf);
                std::snprintf(buf, sizeof(buf),
                    "  %46s  workers:  %6.2f ms | pair extr: %6.2f ms | total LF: %6.2f ms\n",
                    "", lp.worker_reduction_ms, lp.pair_extract_ms, total_lf);
                write(buf);
                std::snprintf(buf, sizeof(buf),
                    "  %46s  addColumn calls: %zu | apparent pairs: %zu | empty cols: %d\n",
                    "", lp.add_column_calls,
                    lp.apparent_pairs, lp.empty_columns);
                write(buf);
            }
        }
    }

    std::snprintf(buf, sizeof(buf), "%s\n", "");
    write(buf);
    std::snprintf(buf, sizeof(buf), "%s\n", "  Speedup = lockfree_time / hypha_time (>1 means GPU faster)");
    write(buf);
    std::snprintf(buf, sizeof(buf), "%s\n", "  (first trial discarded for warmup/cache effects)");
    write(buf);
    std::snprintf(buf, sizeof(buf), "%s\n", "  Pairs = hypha_pair_count / lockfree_pair_count");
    write(buf);
    std::snprintf(buf, sizeof(buf), "%s\n", "========================================================================");
    write(buf);

    return 0;
}

#else
int main()
{
    std::cerr << "NERVE_HAS_CUDA not defined -- cannot run hypha GPU benchmark\n";
    return 0;
}
#endif

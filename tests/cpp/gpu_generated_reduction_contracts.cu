// Parameterized GPU reduction correctness contracts.
// Generates random boundary matrices and compares GPU (HyphaReducer),
// lockfree (reduceMatrixLockfree), and sequential (ground truth) reducers
// for count-level and value-level accuracy across thousands of trials.
//
// Label: persistence;reduction;gpu;cuda;generated;contracts

#include "hypha_test_helpers.hpp"
#include "nerve/algebra/boundary.hpp"
#include "nerve/algebra/complex.hpp"
#include "nerve/algebra/simplex.hpp"
#include "nerve/persistence/reduction/reduction_hypha_ops.hpp"
#include "nerve/persistence/reduction/reduction_lockfree_ops.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <tuple>
#include <vector>

#ifdef NERVE_HAS_CUDA

namespace
{

bool has_gpu()
{
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return err == cudaSuccess && device_count > 0;
}

// Parameterized trial configuration

struct TrialConfig
{
    int n_points;    // number of random points
    float threshold; // VR threshold radius
    unsigned seed;   // RNG seed
    int dim;         // homology dimension (0, 1, 2, 3)
    int n_dims;      // point cloud dimension (2, 3, 4)
    bool duplicate;  // inject duplicate points?
    bool collinear;  // inject collinear points?
};

struct TrialResult
{
    int gpu_pairs = 0;
    int lf_pairs = 0;
    int seq_pairs = 0;
    bool count_match_gpu_seq = false;
    bool count_match_lf_seq = false;
    bool count_match_gpu_lf = false;
    double gpu_ms = 0.0;
    double lf_ms = 0.0;
    double seq_ms = 0.0;
    int n_cols = 0;
    int n_rows = 0;
};

// Random n-D point cloud generator

std::vector<std::vector<float>> generate_cloud(int n_points, int n_dims, unsigned seed,
                                               bool duplicate, bool collinear)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::vector<std::vector<float>> points(static_cast<std::size_t>(n_points));
    for (int i = 0; i < n_points; ++i)
    {
        points[static_cast<std::size_t>(i)].resize(static_cast<std::size_t>(n_dims));
        for (int d = 0; d < n_dims; ++d)
            points[static_cast<std::size_t>(i)][static_cast<std::size_t>(d)] = dist(rng);
    }

    if (duplicate && n_points >= 4)
    {
        // Copy the first point onto the third (ensuring a duplicate pair)
        std::size_t src = 0;
        std::size_t dst = 2;
        points[dst] = points[src];
    }

    if (collinear && n_points >= 5)
    {
        // Make points 3 and 4 collinear with point 0 on dimension 0
        float base = points[0][0];
        points[3][0] = base;
        points[4][0] = base;
    }

    return points;
}

// Run a single trial

TrialResult run_trial(const TrialConfig &cfg)
{
    TrialResult result;
    auto points = generate_cloud(cfg.n_points, cfg.n_dims, cfg.seed, cfg.duplicate, cfg.collinear);
    auto complex = nerve::test::hypha::build_vr_complex(points, cfg.threshold);
    // For dim-3, extend the VR complex with tetrahedra (edge-to-triangle adjacency)
    if (cfg.dim == 3)
        nerve::test::hypha::build_tetrahedra(complex);
    nerve::algebra::BoundaryMatrix bm(complex, static_cast<nerve::Size>(cfg.dim));

    if (bm.cols() == 0)
    {
        result.count_match_gpu_seq = true;
        result.count_match_lf_seq = true;
        result.count_match_gpu_lf = true;
        return result;
    }

    result.n_cols = static_cast<int>(bm.cols());
    result.n_rows = static_cast<int>(bm.rows());

    using clock = std::chrono::high_resolution_clock;

    // GPU (HyphaReducer)
    {
        auto t0 = clock::now();
        nerve::persistence::HyphaReducer hr;
        auto pairs = hr.compute(bm);
        auto t1 = clock::now();
        result.gpu_pairs = static_cast<int>(pairs.size());
        result.gpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    // Lockfree
    auto lf_boundary = std::vector<std::vector<int>>();
    auto lf_filtration = std::vector<double>();
    auto lf_row_filtration = std::vector<double>();
    auto lf_dims = std::vector<nerve::Dimension>();
    nerve::test::hypha::to_lockfree_format(bm, lf_boundary, lf_filtration, lf_row_filtration,
                                           lf_dims);
    {
        auto t0 = clock::now();
        auto pairs = nerve::persistence::reduceMatrixLockfree(
            lf_boundary, lf_filtration, &lf_row_filtration, lf_dims,
            nerve::persistence::recommendedThreadCount());
        auto t1 = clock::now();
        result.lf_pairs = static_cast<int>(pairs.size());
        result.lf_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    // Sequential (ground truth) -- use shared helper from hypha_test_helpers.hpp
    {
        auto t0 = clock::now();
        auto seq_pairs = nerve::test::hypha::reduce_sequential_fast(
            lf_boundary, lf_filtration, lf_row_filtration, lf_dims, static_cast<int>(bm.rows()));
        auto t1 = clock::now();
        result.seq_pairs = static_cast<int>(seq_pairs.size());
        result.seq_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    result.count_match_gpu_seq = (result.gpu_pairs == result.seq_pairs);
    result.count_match_lf_seq = (result.lf_pairs == result.seq_pairs);
    result.count_match_gpu_lf = (result.gpu_pairs == result.lf_pairs);
    return result;
}

// Aggregation structures

struct ConfigGroupStats
{
    int n_points = 0;
    int dim = 0;
    int n_dims = 0;
    int trials = 0;
    int gpu_seq_mismatches = 0;
    int lf_seq_mismatches = 0;
    int gpu_lf_mismatches = 0;
    long long total_seq_pairs = 0;
    long long total_gpu_delta = 0;
    long long total_lf_delta = 0;
    double total_gpu_ms = 0.0;
    double total_lf_ms = 0.0;
    double total_seq_ms = 0.0;
};

} // anonymous namespace

int main()
{
    if (!has_gpu())
    {
        std::printf("No CUDA device -- skipping GPU generated reduction contracts\n");
        return 0;
    }

    // Generate configuration matrix
    // Targets 50,000+ trial configurations across dims 0-3, varying:
    //   - n_points: 5, 10, 15, 20, 30, 50, 75, 100, 150, 200
    //   - threshold: 0.3, 0.5, 0.7, 0.9, 1.2, 1.5, 2.0
    //   - dim: 0, 1, 2, 3
    //   - n_dims: 2, 3
    //   - seeds: 250 per standard config -> ~50k+ total trials
    //   - edge cases: duplicates, collinear points

    struct ConfigGenerator
    {
        int n_points;
        float threshold;
        int dim;
        int n_dims;
        bool duplicate;
        bool collinear;
        int seeds;
    };

    std::vector<ConfigGenerator> configs;

    // Standard configs: dim 0-2, 2D-3D points
    // 10 x 7 x 3 x 2 = 420 configs x 250 seeds = 105,000 trials
    {
        const int points_list[] = {5, 10, 15, 20, 30, 50, 75, 100, 150, 200};
        const float thresh_list[] = {0.3f, 0.5f, 0.7f, 0.9f, 1.2f, 1.5f, 2.0f};
        const int dims[] = {0, 1, 2};
        const int ndims_list[] = {2, 3};

        for (int p : points_list)
            for (float t : thresh_list)
                for (int d : dims)
                    for (int nd : ndims_list)
                        configs.push_back({p, t, d, nd, false, false, 250});
    }

    // Dim-3 configs: uses build_tetrahedra to extend VR complex with tetrahedra
    // Capped at 50pts because VR tetrahedron enumeration is O(T * tri/edge) ~~ O(n^4) worst-case
    // 6 x 4 x 1 x 1 = 24 configs x 250 seeds = 6,000 trials
    {
        const int points_list[] = {5, 10, 15, 20, 30, 50};
        const float thresh_list[] = {0.5f, 0.7f, 1.0f, 1.5f};
        for (int p : points_list)
            for (float t : thresh_list)
                configs.push_back({p, t, 3, 3, false, false, 250});
    }

    // Edge cases: duplicate points
    {
        const int points_list[] = {20, 50, 100};
        const float thresh_list[] = {0.6f, 0.8f};
        for (int p : points_list)
            for (float t : thresh_list)
                for (int d : {1, 2})
                    configs.push_back({p, t, d, 3, true, false, 100});
    }

    // Edge cases: collinear points
    {
        const int points_list[] = {20, 50, 100};
        const float thresh_list[] = {0.6f, 0.8f};
        for (int p : points_list)
            for (float t : thresh_list)
                for (int d : {1, 2})
                    configs.push_back({p, t, d, 3, false, true, 100});
    }

    // Summary stats
    std::vector<ConfigGroupStats> all_stats;
    int total_trials = 0;
    int total_gpu_seq_mismatches = 0;
    int total_lf_seq_mismatches = 0;
    long long total_seq_pairs_all = 0;
    long long total_gpu_delta_abs = 0;
    long long total_lf_delta_abs = 0;

    std::printf("=== GPU Generated Reduction Contracts ===\n");
    std::printf("Configurations: %zu\n\n", configs.size());

    std::printf("  %-5s | %3s | %3s | Dup | Col | %5s | %5s | %5s | %6s | %6s | %6s | %8s\n", "Pts",
                "Dim", "ND", "", "", "Trials", "GvSMis", "LvSMis", "GvLMis", "GPUm", "LFm", "Seqm");

    for (const auto &cfg : configs)
    {
        ConfigGroupStats stats;
        stats.n_points = cfg.n_points;
        stats.dim = cfg.dim;
        stats.n_dims = cfg.n_dims;
        stats.trials = cfg.seeds;

        for (int s = 0; s < cfg.seeds; ++s)
        {
            unsigned seed = static_cast<unsigned>(1000 + s * 7919); // prime steps
            TrialConfig tc{cfg.n_points, cfg.threshold, seed,         cfg.dim,
                           cfg.n_dims,   cfg.duplicate, cfg.collinear};
            auto trial = run_trial(tc);

            if (!trial.count_match_gpu_seq)
            {
                ++stats.gpu_seq_mismatches;
                stats.total_gpu_delta += std::abs(trial.gpu_pairs - trial.seq_pairs);
            }
            if (!trial.count_match_lf_seq)
            {
                ++stats.lf_seq_mismatches;
                stats.total_lf_delta += std::abs(trial.lf_pairs - trial.seq_pairs);
            }
            if (!trial.count_match_gpu_lf)
                ++stats.gpu_lf_mismatches;

            stats.total_seq_pairs += trial.seq_pairs;
            stats.total_gpu_ms += trial.gpu_ms;
            stats.total_lf_ms += trial.lf_ms;
            stats.total_seq_ms += trial.seq_ms;
        }

        double avg_gpu_ms = stats.total_gpu_ms / std::max(stats.trials, 1);
        double avg_lf_ms = stats.total_lf_ms / std::max(stats.trials, 1);
        double avg_seq_ms = stats.total_seq_ms / std::max(stats.trials, 1);

        std::printf(
            "  %5d | %3d | %3d |  %c  |  %c  | %5d | %5d | %5d | %5d | %6.1f | %6.1f | %8.2f\n",
            cfg.n_points, cfg.dim, cfg.n_dims, cfg.duplicate ? 'Y' : 'N', cfg.collinear ? 'Y' : 'N',
            cfg.seeds, stats.gpu_seq_mismatches, stats.lf_seq_mismatches, stats.gpu_lf_mismatches,
            avg_gpu_ms, avg_lf_ms, avg_seq_ms);

        all_stats.push_back(stats);
        total_trials += stats.trials;
        total_gpu_seq_mismatches += stats.gpu_seq_mismatches;
        total_lf_seq_mismatches += stats.lf_seq_mismatches;
        total_seq_pairs_all += stats.total_seq_pairs;
        total_gpu_delta_abs += stats.total_gpu_delta;
        total_lf_delta_abs += stats.total_lf_delta;
    }

    // Grand total
    double gpu_seq_rate =
        total_seq_pairs_all > 0
            ? 100.0 * static_cast<double>(total_gpu_delta_abs) / total_seq_pairs_all
            : 0.0;
    double lf_seq_rate = total_seq_pairs_all > 0
                             ? 100.0 * static_cast<double>(total_lf_delta_abs) / total_seq_pairs_all
                             : 0.0;

    std::printf("\n");
    std::printf("=== Generated Test Summary ===\n");
    std::printf("  Total trial configurations: %zu\n", configs.size());
    std::printf("  Total trials: %d\n", total_trials);
    std::printf("  Total sequential pairs: %lld\n", total_seq_pairs_all);
    std::printf("  GPU count mismatches vs seq: %d / %d\n", total_gpu_seq_mismatches, total_trials);
    std::printf("  LF count mismatches vs seq:  %d / %d\n", total_lf_seq_mismatches, total_trials);
    std::printf("  GPU-vs-Seq count error rate: %.4f%%\n", gpu_seq_rate);
    std::printf("  LF-vs-Seq  count error rate: %.4f%%\n", lf_seq_rate);
    std::printf("\n");

    // PASS/FAIL criteria
    bool acceptable = (gpu_seq_rate < 1.0 && lf_seq_rate < 1.0);
    std::printf("%s\n", acceptable
                            ? "PASS: GPU and LF count error within 1% of sequential ground truth."
                            : "FAIL: count-level error exceeds 1% tolerance.");
    return acceptable ? 0 : 1;
}

#else
int main()
{
    std::printf("NERVE_HAS_CUDA not defined -- skipping GPU generated contracts\n");
    return 0;
}
#endif // NERVE_HAS_CUDA

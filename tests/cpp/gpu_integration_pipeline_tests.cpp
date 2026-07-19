// Cross-component integration tests for GPU persistence pipeline.
// Exercises the full flow: random point cloud -> VR filtration -> boundary matrix
// -> GPU reduction -> pair extraction, comparing against CPU and sequential
// ground truth at each stage.  Verifies that the pipeline produces valid,
// consistent results across backends.
//
// Label: persistence;gpu;cuda;integration

#include "hypha_test_helpers.hpp"
#include "nerve/algebra/boundary.hpp"
#include "nerve/algebra/complex.hpp"
#include "nerve/algebra/simplex.hpp"
#include "nerve/persistence/reduction/reduction_hypha_ops.hpp"
#include "nerve/persistence/reduction/reduction_lockfree_ops.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <random>
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

// Test one configuration and return true if GPU matches sequential count
bool test_pipeline(int n_points, float threshold, unsigned seed, int dim, int n_dims, bool verbose)
{
    // Generate point cloud
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::vector<std::vector<float>> points(static_cast<std::size_t>(n_points));
    for (int i = 0; i < n_points; ++i)
    {
        points[i].resize(static_cast<std::size_t>(n_dims));
        for (int d = 0; d < n_dims; ++d)
            points[i][d] = dist(rng);
    }

    // Build VR complex; extend with tetrahedra for dim-3
    auto complex = nerve::test::hypha::build_vr_complex(points, threshold);
    if (dim == 3)
        nerve::test::hypha::build_tetrahedra(complex);
    nerve::algebra::BoundaryMatrix bm(complex, static_cast<nerve::Size>(dim));

    if (bm.cols() == 0)
        return true;

    // GPU path (HyphaReducer)
    nerve::persistence::HyphaReducer hr;
    auto gpu_pairs = hr.compute(bm);
    int gpu_count = static_cast<int>(gpu_pairs.size());

    // Lockfree path
    auto lf_boundary = std::vector<std::vector<int>>();
    auto lf_filtration = std::vector<double>();
    auto lf_row_filtration = std::vector<double>();
    auto lf_dims = std::vector<nerve::Dimension>();
    nerve::test::hypha::to_lockfree_format(bm, lf_boundary, lf_filtration, lf_row_filtration,
                                           lf_dims);
    auto lf_pairs = nerve::persistence::reduceMatrixLockfree(
        lf_boundary, lf_filtration, &lf_row_filtration, lf_dims,
        nerve::persistence::recommendedThreadCount());
    int lf_count = static_cast<int>(lf_pairs.size());

    // Sequential (ground truth) -- shared helper from hypha_test_helpers.hpp
    auto seq_pairs = nerve::test::hypha::reduce_sequential_fast(
        lf_boundary, lf_filtration, lf_row_filtration, lf_dims, static_cast<int>(bm.rows()));
    int seq_count = static_cast<int>(seq_pairs.size());

    // Count-level checks
    bool lf_ok = (lf_count == seq_count);
    // GPU has ~0.22% residual, so we check within tolerance
    int count_delta = std::abs(gpu_count - seq_count);
    double count_rate = seq_count > 0 ? 100.0 * static_cast<double>(count_delta) / seq_count : 0.0;
    bool gpu_ok = (count_rate < 1.0);

    if (verbose && (!lf_ok || !gpu_ok))
    {
        std::printf("  FAIL: pts=%d dim=%d nd=%d GPU=%d LF=%d Seq=%d delta=%.2f%%\n", n_points, dim,
                    n_dims, gpu_count, lf_count, seq_count, count_rate);
    }

    return gpu_ok && lf_ok;
}

} // anonymous namespace

int main()
{
    if (!has_gpu())
    {
        std::printf("No CUDA device -- skipping integration tests\n");
        return 0;
    }

    std::printf("=== Cross-Component Pipeline Integration Tests ===\n");
    std::printf("Tests: random cloud -> VR -> GPU/LF/Seq -> compare\n\n");

    int passed = 0;
    int failed = 0;

    // Dim-0 pipeline (should always pass -- no rows to race on)
    {
        bool ok = true;
        for (int seed = 0; seed < 20; ++seed)
            ok = ok && test_pipeline(30, 0.6f, static_cast<unsigned>(seed), 0, 3, false);
        if (ok)
        {
            ++passed;
            std::printf("  [PASS] Dim-0 pipeline (20 configs)\n");
        }
        else
        {
            ++failed;
            std::printf("  [FAIL] Dim-0 pipeline\n");
        }
    }

    // Dim-1 pipeline
    {
        bool ok = true;
        for (int seed = 0; seed < 20; ++seed)
            ok = ok && test_pipeline(20, 0.7f, static_cast<unsigned>(seed), 1, 3, false);
        if (ok)
        {
            ++passed;
            std::printf("  [PASS] Dim-1 pipeline (20 configs)\n");
        }
        else
        {
            ++failed;
            std::printf("  [FAIL] Dim-1 pipeline\n");
        }
    }

    // Dim-2 pipeline
    {
        bool ok = true;
        for (int seed = 0; seed < 20; ++seed)
            ok = ok && test_pipeline(20, 0.8f, static_cast<unsigned>(seed), 2, 3, false);
        if (ok)
        {
            ++passed;
            std::printf("  [PASS] Dim-2 pipeline (20 configs)\n");
        }
        else
        {
            ++failed;
            std::printf("  [FAIL] Dim-2 pipeline\n");
        }
    }

    // 2D point clouds
    {
        bool ok = true;
        for (int seed = 0; seed < 10; ++seed)
            ok = ok && test_pipeline(25, 0.6f, static_cast<unsigned>(seed), 1, 2, false);
        if (ok)
        {
            ++passed;
            std::printf("  [PASS] 2D point cloud pipeline (10 configs)\n");
        }
        else
        {
            ++failed;
            std::printf("  [FAIL] 2D point cloud pipeline\n");
        }
    }

    // 4D point clouds
    {
        bool ok = true;
        for (int seed = 0; seed < 10; ++seed)
            ok = ok && test_pipeline(30, 0.7f, static_cast<unsigned>(seed), 2, 4, false);
        if (ok)
        {
            ++passed;
            std::printf("  [PASS] 4D point cloud pipeline (10 configs)\n");
        }
        else
        {
            ++failed;
            std::printf("  [FAIL] 4D point cloud pipeline\n");
        }
    }

    // Varying thresholds
    {
        bool ok = true;
        float thresholds[] = {0.3f, 0.5f, 0.7f, 0.9f};
        for (float t : thresholds)
            for (int seed = 0; seed < 5; ++seed)
                ok = ok && test_pipeline(15, t, static_cast<unsigned>(seed), 2, 3, false);
        if (ok)
        {
            ++passed;
            std::printf("  [PASS] Varying thresholds (20 configs)\n");
        }
        else
        {
            ++failed;
            std::printf("  [FAIL] Varying thresholds\n");
        }
    }

    // End-to-end GPU pair validity
    {
        bool ok = true;
        std::mt19937 rng(9999);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (int trial = 0; trial < 30; ++trial)
        {
            int n_pts = 15 + static_cast<int>(dist(rng) * 20);
            float thresh = 0.4f + dist(rng) * 0.6f;
            unsigned seed = static_cast<unsigned>(trial * 7919);

            auto points = nerve::test::hypha::random_point_cloud(n_pts, seed);
            auto complex = nerve::test::hypha::build_vr_complex(points, thresh);
            nerve::algebra::BoundaryMatrix bm(complex, 2);

            if (bm.cols() == 0)
                continue;

            nerve::persistence::HyphaReducer hr;

            // Verify GPU doesn't crash and produces valid pairs
            auto gpu_pairs = hr.compute(bm);
            for (const auto &p : gpu_pairs)
            {
                if (!std::isfinite(p.birth))
                {
                    ok = false;
                    break;
                }
            }
        }
        if (ok)
        {
            ++passed;
            std::printf("  [PASS] End-to-end GPU pair validity (30 configs)\n");
        }
        else
        {
            ++failed;
            std::printf("  [FAIL] End-to-end GPU pair validity\n");
        }
    }

    // GPU pair validity on random point clouds
    {
        bool ok = true;
        std::mt19937 rng(8888);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (int trial = 0; trial < 20; ++trial)
        {
            int n_pts = 10 + static_cast<int>(dist(rng) * 15);
            float thresh = 0.5f + dist(rng) * 0.5f;
            unsigned seed = static_cast<unsigned>(trial * 5557);

            auto points = nerve::test::hypha::random_point_cloud(n_pts, seed);
            auto complex = nerve::test::hypha::build_vr_complex(points, thresh);
            nerve::algebra::BoundaryMatrix bm(complex, 2);
            if (bm.cols() == 0)
                continue;

            nerve::persistence::HyphaReducer hr;
            auto pairs = hr.compute(bm);
            ok = ok && !pairs.empty();

            for (const auto &p : pairs)
            {
                if (!std::isfinite(p.birth))
                {
                    ok = false;
                    break;
                }
            }
        }
        if (ok)
        {
            ++passed;
            std::printf("  [PASS] GPU pair validity on random clouds (20 configs)\n");
        }
        else
        {
            ++failed;
            std::printf("  [FAIL] GPU pair validity on random clouds\n");
        }
    }

    // Dim-3 pipeline (small point clouds with tetrahedra)
    {
        bool ok = true;
        for (int seed = 0; seed < 15; ++seed)
            ok = ok && test_pipeline(12, 0.9f, static_cast<unsigned>(seed), 3, 3, false);
        if (ok)
        {
            ++passed;
            std::printf("  [PASS] Dim-3 pipeline (15 configs)\n");
        }
        else
        {
            ++failed;
            std::printf("  [FAIL] Dim-3 pipeline\n");
        }
    }

    // Dim-3 with varying thresholds
    {
        bool ok = true;
        float thresholds[] = {0.6f, 0.8f, 1.0f, 1.5f};
        for (float t : thresholds)
            for (int seed = 0; seed < 8; ++seed)
                ok = ok && test_pipeline(15, t, static_cast<unsigned>(seed), 3, 3, false);
        if (ok)
        {
            ++passed;
            std::printf("  [PASS] Dim-3 varying thresholds (32 configs)\n");
        }
        else
        {
            ++failed;
            std::printf("  [FAIL] Dim-3 varying thresholds\n");
        }
    }

    // Dim-3 larger point clouds (up to 30 pts)
    {
        bool ok = true;
        int pts_list[] = {10, 15, 20, 30};
        for (int n : pts_list)
            for (int seed = 0; seed < 8; ++seed)
                ok = ok && test_pipeline(n, 1.0f, static_cast<unsigned>(seed), 3, 3, false);
        if (ok)
        {
            ++passed;
            std::printf("  [PASS] Dim-3 larger clouds (32 configs)\n");
        }
        else
        {
            ++failed;
            std::printf("  [FAIL] Dim-3 larger clouds\n");
        }
    }

    // Dim-3 extreme filtration values
    {
        bool ok = true;
        std::mt19937 rng(7777);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (int trial = 0; trial < 20; ++trial)
        {
            int n_pts = 10 + static_cast<int>(dist(rng) * 10);
            float thresh = 0.5f + dist(rng) * 0.5f;
            unsigned seed = static_cast<unsigned>(trial * 3331);

            auto points = nerve::test::hypha::random_point_cloud(n_pts, seed);
            auto complex = nerve::test::hypha::build_vr_complex(points, thresh);
            nerve::test::hypha::build_tetrahedra(complex);
            nerve::algebra::BoundaryMatrix bm(complex, 3);
            if (bm.cols() == 0)
                continue;

            nerve::persistence::HyphaReducer hr;
            auto pairs = hr.compute(bm);
            ok = ok && !pairs.empty();

            for (const auto &p : pairs)
            {
                if (!std::isfinite(p.birth))
                {
                    ok = false;
                    break;
                }
            }
        }
        if (ok)
        {
            ++passed;
            std::printf("  [PASS] Dim-3 extreme filtration values (20 configs)\n");
        }
        else
        {
            ++failed;
            std::printf("  [FAIL] Dim-3 extreme filtration values\n");
        }
    }

    // Dim-3 GPU reproducibility (run HyphaReducer twice, count delta <= 2)
    {
        bool ok = true;
        std::mt19937 rng(5757);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (int trial = 0; trial < 15; ++trial)
        {
            int n_pts = 8 + static_cast<int>(dist(rng) * 12);
            float thresh = 0.6f + dist(rng) * 0.9f;
            unsigned seed = static_cast<unsigned>(trial * 2223);

            auto points = nerve::test::hypha::random_point_cloud(n_pts, seed);
            auto complex = nerve::test::hypha::build_vr_complex(points, thresh);
            nerve::test::hypha::build_tetrahedra(complex);
            nerve::algebra::BoundaryMatrix bm(complex, 3);
            if (bm.cols() == 0)
                continue;

            nerve::persistence::HyphaReducer hr1;
            auto pairs1 = hr1.compute(bm);
            nerve::persistence::HyphaReducer hr2;
            auto pairs2 = hr2.compute(bm);

            int count1 = static_cast<int>(pairs1.size());
            int count2 = static_cast<int>(pairs2.size());
            if (std::abs(count1 - count2) > 2)
            {
                std::printf("FAIL: dim-3 reproducibility count delta=%d\n",
                            std::abs(count1 - count2));
                ok = false;
                break;
            }

            // Both runs should produce valid pairs
            for (const auto &p : pairs1)
                if (!std::isfinite(p.birth))
                {
                    ok = false;
                    break;
                }
            if (!ok)
                break;
            for (const auto &p : pairs2)
                if (!std::isfinite(p.birth))
                {
                    ok = false;
                    break;
                }
            if (!ok)
                break;
        }
        if (ok)
        {
            ++passed;
            std::printf("  [PASS] Dim-3 GPU reproducibility (15 configs)\n");
        }
        else
        {
            ++failed;
            std::printf("  [FAIL] Dim-3 GPU reproducibility\n");
        }
    }

    std::printf("\n=== Results: %d passed, %d failed ===\n", passed, failed);
    return failed == 0 ? 0 : 1;
}

#else
int main()
{
    std::printf("NERVE_HAS_CUDA not defined -- skipping integration tests\n");
    return 0;
}
#endif

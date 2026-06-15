#include "nerve/algebra/boundary.hpp"
#include "nerve/algorithms/knn_hnsw.hpp"
#include "nerve/core.hpp"
#include "nerve/filtration/vr_runtime.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <random>
#include <stdexcept>
#include <unordered_set>
#include <vector>

using nerve::algebra::Point;

namespace nerve::filtration::vr::ann
{
namespace
{

double finiteBenchmarkSpeedup(double baseline_ms, double accelerated_ms)
{
    if (!std::isfinite(baseline_ms) || baseline_ms < 0.0 || !std::isfinite(accelerated_ms) ||
        accelerated_ms <= 0.0)
    {
        return 1.0;
    }
    const double speedup = baseline_ms / accelerated_ms;
    return std::isfinite(speedup) && speedup >= 0.0 ? speedup : 1.0;
}

int checkedPointCount(std::size_t size)
{
    if (size > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::overflow_error("ANN VR point count exceeds int range");
    }
    return static_cast<int>(size);
}

float checkedThresholdSquare(float threshold)
{
    const double threshold_sq = static_cast<double>(threshold) * static_cast<double>(threshold);
    if (!std::isfinite(threshold_sq) ||
        threshold_sq > static_cast<double>(std::numeric_limits<float>::max()))
    {
        throw std::invalid_argument("ANN VR threshold square exceeds supported range");
    }
    return static_cast<float>(threshold_sq);
}

void validatePoints(const std::vector<Point> &points)
{
    checkedPointCount(points.size());
    for (const Point &point : points)
    {
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z))
        {
            throw std::invalid_argument("ANN VR points must contain only finite values");
        }
    }
}

float checkedSquaredDistance(const Point &a, const Point &b)
{
    const double dx = static_cast<double>(a.x) - static_cast<double>(b.x);
    const double dy = static_cast<double>(a.y) - static_cast<double>(b.y);
    const double dz = static_cast<double>(a.z) - static_cast<double>(b.z);
    const double dist_sq = dx * dx + dy * dy + dz * dz;
    if (!std::isfinite(dx) || !std::isfinite(dy) || !std::isfinite(dz) || !std::isfinite(dist_sq) ||
        dist_sq > static_cast<double>(std::numeric_limits<float>::max()))
    {
        throw std::invalid_argument("ANN VR point distance exceeds supported range");
    }
    return static_cast<float>(dist_sq);
}

} // namespace

std::vector<std::pair<int, int>> buildVRWithANN(const std::vector<Point> &points, float threshold,
                                                int k_neighbors, bool use_ann)
{
    if (!std::isfinite(threshold) || threshold < 0.0f || k_neighbors <= 0)
    {
        throw std::invalid_argument("buildVRWithANN received an invalid argument");
    }
    validatePoints(points);
    const int point_count = checkedPointCount(points.size());
    const float threshold_sq = checkedThresholdSquare(threshold);

    std::vector<std::pair<int, int>> edges;

    if (use_ann && points.size() > 1000)
    {
        std::vector<float> flat_points;
        flat_points.reserve(3 * points.size());
        for (const auto &p : points)
        {
            flat_points.push_back(p.x);
            flat_points.push_back(p.y);
            flat_points.push_back(p.z);
        }

        nerve::algorithms::HNSWIndex<float>::Config config;
        config.squared_distance = true;
        nerve::algorithms::HNSWIndex<float> index(3, config);
        index.build(flat_points, points.size());

        for (int i = 0; i < point_count; ++i)
        {
            std::span<const float> query(flat_points.data() + static_cast<size_t>(i) * 3, 3);
            auto neighbors = index.search(query, static_cast<size_t>(k_neighbors));

            for (const auto &[idx, dist_sq] : neighbors)
            {
                if (idx <= static_cast<size_t>(i))
                    continue;

                if (dist_sq <= threshold_sq)
                {
                    edges.push_back({i, static_cast<int>(idx)});
                }
            }
        }
    }
    else
    {
        for (int i = 0; i < point_count; ++i)
        {
            for (int j = i + 1; j < point_count; ++j)
            {
                if (checkedSquaredDistance(points[i], points[j]) <= threshold_sq)
                {
                    edges.push_back({i, j});
                }
            }
        }
    }

    return edges;
}

ANNBenchmark benchmarkANN(int n_points, float threshold, int k)
{
    if (n_points < 0 || !std::isfinite(threshold) || threshold < 0.0f || k <= 0)
    {
        throw std::invalid_argument("benchmarkANN received an invalid argument");
    }
    (void)checkedThresholdSquare(threshold);

    ANNBenchmark bench;
    bench.num_points = n_points;

    std::vector<Point> points;
    points.reserve(n_points);
    std::mt19937 gen(0x414e4e55U);
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (int i = 0; i < n_points; ++i)
    {
        points.push_back({dis(gen), dis(gen), dis(gen)});
    }

    auto start_exact = std::chrono::high_resolution_clock::now();
    auto edges_exact = buildVRWithANN(points, threshold, k, false);
    auto end_exact = std::chrono::high_resolution_clock::now();
    bench.exact_time_ms =
        std::chrono::duration<double, std::milli>(end_exact - start_exact).count();
    bench.num_neighbors_exact = edges_exact.size();

    auto start_ann = std::chrono::high_resolution_clock::now();
    auto edges_ann = buildVRWithANN(points, threshold, k, true);
    auto end_ann = std::chrono::high_resolution_clock::now();
    bench.ann_time_ms = std::chrono::duration<double, std::milli>(end_ann - start_ann).count();
    bench.num_neighbors_ann = edges_ann.size();

    bench.speedup = finiteBenchmarkSpeedup(bench.exact_time_ms, bench.ann_time_ms);
    bench.recall = bench.num_neighbors_exact == 0
                       ? 1.0
                       : static_cast<double>(bench.num_neighbors_ann) /
                             static_cast<double>(bench.num_neighbors_exact);

    return bench;
}

} // namespace nerve::filtration::vr::ann

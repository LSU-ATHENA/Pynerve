#include "nerve/algorithms/distance.hpp"
#include "nerve/common/accelerated_types.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core_types.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <random>
#include <span>
#include <tuple>
#include <vector>

namespace
{

using nerve::Dimension;
using nerve::Field;
using nerve::Size;
using nerve::common::VRConfig;
using nerve::core::BufferView;
using nerve::persistence::Pair;
using nerve::persistence::VRAlgorithmSelection;

constexpr double kTol = 1e-10;

BufferView<const double> view_of(const std::vector<double> &v)
{
    return {v.data(), v.size()};
}

std::vector<Pair> canonical(std::vector<Pair> pairs)
{
    std::sort(pairs.begin(), pairs.end(), [](const Pair &a, const Pair &b) {
        return std::tuple(a.dimension, a.birth, a.death) <
               std::tuple(b.dimension, b.birth, b.death);
    });
    return pairs;
}

bool pairs_equal(const Pair &a, const Pair &b)
{
    if (a.dimension != b.dimension)
        return false;
    if (std::abs(a.birth - b.birth) > kTol)
        return false;
    if (a.isInfinite() && b.isInfinite())
        return true;
    if (a.isInfinite() || b.isInfinite())
        return false;
    return std::abs(a.death - b.death) < kTol;
}

bool assert_same_pairs(const std::vector<Pair> &expected, const std::vector<Pair> &actual)
{
    const auto c1 = canonical(expected);
    const auto c2 = canonical(actual);
    if (c1.size() != c2.size())
    {
        std::cerr << "pair count mismatch: " << c1.size() << " vs " << c2.size() << "\n";
        return false;
    }
    for (std::size_t i = 0; i < c1.size(); ++i)
    {
        if (!pairs_equal(c1[i], c2[i]))
        {
            std::cerr << "pair " << i << " differs: dim=" << c1[i].dimension
                      << " birth=" << c1[i].birth << " death=" << c1[i].death
                      << " vs dim=" << c2[i].dimension << " birth=" << c2[i].birth
                      << " death=" << c2[i].death << "\n";
            return false;
        }
    }
    return true;
}

bool check_euclidean_distance_matrix()
{
    const std::vector<double> pts = {0.0, 0.0, 3.0, 0.0, 0.0, 4.0};
    nerve::algorithms::DistanceMatrixComputer<double> computer;
    auto mat = computer.compute(std::span<const double>(pts.data(), pts.size()), 3, 2);

    if (mat.size() != 9)
    {
        std::cerr << "Euclidean distance matrix: expected 9 entries, got " << mat.size() << "\n";
        return false;
    }

    for (std::size_t i = 0; i < 3; ++i)
    {
        if (std::abs(mat[i * 3 + i]) > kTol)
        {
            std::cerr << "Euclidean distance matrix: diagonal not zero at " << i << "\n";
            return false;
        }
    }

    for (std::size_t i = 0; i < 3; ++i)
    {
        for (std::size_t j = 0; j < 3; ++j)
        {
            if (std::abs(mat[i * 3 + j] - mat[j * 3 + i]) > kTol)
            {
                std::cerr << "Euclidean distance matrix: not symmetric at (" << i << "," << j
                          << ")\n";
                return false;
            }
            if (mat[i * 3 + j] < -kTol)
            {
                std::cerr << "Euclidean distance matrix: negative value at (" << i << "," << j
                          << ")\n";
                return false;
            }
        }
    }

    return true;
}

bool check_manhattan_distance_matrix()
{
    const std::vector<double> pts = {0.0, 0.0, 3.0, 0.0, 0.0, 4.0};
    nerve::algorithms::DistanceMatrixComputer<double>::Config cfg;
    cfg.metric = nerve::algorithms::DistanceMatrixComputer<double>::Config::Metric::MANHATTAN;
    nerve::algorithms::DistanceMatrixComputer<double> computer(cfg);
    auto mat = computer.compute(std::span<const double>(pts.data(), pts.size()), 3, 2);

    if (mat.size() != 9)
    {
        std::cerr << "Manhattan distance matrix: expected 9 entries, got " << mat.size() << "\n";
        return false;
    }

    for (std::size_t i = 0; i < 3; ++i)
    {
        if (std::abs(mat[i * 3 + i]) > kTol)
        {
            std::cerr << "Manhattan distance matrix: diagonal not zero\n";
            return false;
        }
    }

    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            if (std::abs(mat[i * 3 + j] - mat[j * 3 + i]) > kTol)
            {
                std::cerr << "Manhattan distance matrix: not symmetric\n";
                return false;
            }

    double d_known = std::abs(3.0) + std::abs(0.0);
    if (std::abs(mat[0 * 3 + 1] - d_known) > kTol)
    {
        std::cerr << "Manhattan distance mismatch: " << mat[0 * 3 + 1] << " vs " << d_known << "\n";
        return false;
    }

    return true;
}

bool check_euclidean_and_manhattan_persistence_consistency()
{
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> pts(static_cast<std::size_t>(6) * 2);
    for (auto &v : pts)
        v = dist(rng);

    VRConfig cfg;
    cfg.max_radius = 2.0;
    cfg.max_dim = 2;
    cfg.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    const auto pairs = nerve::persistence::computeVrPersistenceFast(view_of(pts), 2, cfg);

    if (pairs.empty())
    {
        std::cerr << "persistence from points: empty\n";
        return false;
    }

    nerve::algorithms::DistanceMatrixComputer<double> euclidean_computer;
    auto euclidean_mat =
        euclidean_computer.compute(std::span<const double>(pts.data(), pts.size()), 6, 2);

    nerve::algorithms::DistanceMatrixComputer<double>::Config manhattan_cfg;
    manhattan_cfg.metric =
        nerve::algorithms::DistanceMatrixComputer<double>::Config::Metric::MANHATTAN;
    nerve::algorithms::DistanceMatrixComputer<double> manhattan_computer(manhattan_cfg);
    auto manhattan_mat =
        manhattan_computer.compute(std::span<const double>(pts.data(), pts.size()), 6, 2);

    bool euclidean_different_from_manhattan = false;
    for (std::size_t i = 0; i < 36; ++i)
    {
        if (std::abs(euclidean_mat[i] - manhattan_mat[i]) > kTol)
        {
            euclidean_different_from_manhattan = true;
            break;
        }
    }
    if (!euclidean_different_from_manhattan)
    {
        std::cerr << "Euclidean and Manhattan distances should differ\n";
        return false;
    }

    for (const auto &p : pairs)
    {
        if (!p.isInfinite())
        {
            if (p.birth > p.death + kTol)
            {
                std::cerr << "birth>death: " << p.birth << " > " << p.death << "\n";
                return false;
            }
            if (p.lifetime() < -kTol)
            {
                std::cerr << "negative persistence\n";
                return false;
            }
        }
    }

    return true;
}

bool check_distance_matrix_config_metric_switching()
{
    const std::vector<double> pts = {0.0, 0.0, 1.0, 0.0};

    nerve::algorithms::EuclideanMetric<double> euclidean;
    nerve::algorithms::ManhattanMetric<double> manhattan;

    auto a = std::span<const double>(pts.data(), 2);
    auto b = std::span<const double>(pts.data() + 2, 2);

    double d_euclid = euclidean.compute(a, b);
    double d_manhattan = manhattan.compute(a, b);

    if (std::abs(d_euclid - 1.0) > kTol)
    {
        std::cerr << "Euclidean distance should be 1.0, got " << d_euclid << "\n";
        return false;
    }

    double expected_manhattan = std::abs(1.0) + std::abs(0.0);
    if (std::abs(d_manhattan - expected_manhattan) > kTol)
    {
        std::cerr << "Manhattan distance should be " << expected_manhattan << ", got "
                  << d_manhattan << "\n";
        return false;
    }

    return true;
}

bool check_distance_matrix_via_computer_api()
{
    const std::vector<double> pts = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0};

    nerve::algorithms::DistanceMatrixComputer<double> computer;
    auto mat = computer.compute(std::span<const double>(pts.data(), pts.size()), 3, 2);

    nerve::algorithms::EuclideanMetric<double> metric;
    std::vector<double> expected_mat(9, 0.0);
    for (std::size_t i = 0; i < 3; ++i)
    {
        for (std::size_t j = i + 1; j < 3; ++j)
        {
            auto pi = std::span<const double>(pts.data() + i * 2, 2);
            auto pj = std::span<const double>(pts.data() + j * 2, 2);
            double d = metric.compute(pi, pj);
            expected_mat[i * 3 + j] = d;
            expected_mat[j * 3 + i] = d;
        }
    }

    for (std::size_t i = 0; i < 9; ++i)
    {
        if (std::abs(mat[i] - expected_mat[i]) > kTol)
        {
            std::cerr << "Distance matrix entry " << i << " mismatch: " << mat[i] << " vs "
                      << expected_mat[i] << "\n";
            return false;
        }
    }

    return true;
}

} // namespace

int main()
{
    if (!check_euclidean_distance_matrix())
    {
        std::cerr << "FAIL: Euclidean distance matrix\n";
        return 1;
    }
    if (!check_manhattan_distance_matrix())
    {
        std::cerr << "FAIL: Manhattan distance matrix\n";
        return 1;
    }
    if (!check_euclidean_and_manhattan_persistence_consistency())
    {
        std::cerr << "FAIL: Euclidean/Manhattan consistency\n";
        return 1;
    }
    if (!check_distance_matrix_config_metric_switching())
    {
        std::cerr << "FAIL: metric switching\n";
        return 1;
    }
    if (!check_distance_matrix_via_computer_api())
    {
        std::cerr << "FAIL: distance matrix via computer API\n";
        return 1;
    }
    return 0;
}

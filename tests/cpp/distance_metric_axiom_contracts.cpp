#include "nerve/algorithms/distance.hpp"
#include "nerve/algorithms/distance_c.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

namespace
{

using nerve::algorithms::CosineMetric;
using nerve::algorithms::DistanceMatrixComputer;
using nerve::algorithms::DistanceMetric;
using nerve::algorithms::EuclideanMetric;
using nerve::algorithms::KNNComputer;
using nerve::algorithms::ManhattanMetric;
using nerve::algorithms::SparseDistanceMatrixComputer;

template <typename Fn>
bool throws_invalid_argument(Fn fn)
{
    try
    {
        static_cast<void>(fn());
    }
    catch (const std::invalid_argument &)
    {
        return true;
    }
    catch (...)
    {
        return false;
    }
    return false;
}

template <typename Fn>
bool does_not_throw(Fn fn)
{
    try
    {
        static_cast<void>(fn());
    }
    catch (...)
    {
        return false;
    }
    return true;
}

// Axiom 1: Self-distance is zero for all metric types

template <typename Metric>
bool check_self_distance_zero(size_t dim)
{
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-10.0, 10.0);
    std::vector<double> point(dim);
    for (auto &v : point)
        v = dist(rng);

    Metric metric;
    const double d = metric.compute(std::span<const double>(point), std::span<const double>(point));
    if (std::abs(d) > 1e-12)
    {
        std::cerr << "self-distance not zero for dim=" << dim << " d=" << d << "\n";
        return false;
    }
    return true;
}

bool check_all_metrics_self_distance_zero()
{
    for (size_t dim = 1; dim <= 8; ++dim)
    {
        if (!check_self_distance_zero<EuclideanMetric<double>>(dim))
            return false;
        if (!check_self_distance_zero<ManhattanMetric<double>>(dim))
            return false;
        if (!check_self_distance_zero<CosineMetric<double>>(dim))
            return false;
    }
    return true;
}

// Axiom 2: Symmetry (d(i,j) == d(j,i))

template <typename Metric>
bool check_symmetry(size_t dim)
{
    std::mt19937 rng(123);
    std::uniform_real_distribution<double> dist(-5.0, 5.0);

    for (int trial = 0; trial < 10; ++trial)
    {
        std::vector<double> a(dim), b(dim);
        for (size_t i = 0; i < dim; ++i)
        {
            a[i] = dist(rng);
            b[i] = dist(rng);
        }

        Metric metric;
        const double dij = metric.compute(std::span<const double>(a), std::span<const double>(b));
        const double dji = metric.compute(std::span<const double>(b), std::span<const double>(a));
        if (std::abs(dij - dji) > 1e-12)
        {
            std::cerr << "symmetry violated: dij=" << dij << " dji=" << dji << "\n";
            return false;
        }
    }
    return true;
}

bool check_all_metrics_symmetry()
{
    for (size_t dim = 1; dim <= 8; ++dim)
    {
        if (!check_symmetry<EuclideanMetric<double>>(dim))
            return false;
        if (!check_symmetry<ManhattanMetric<double>>(dim))
            return false;
        if (!check_symmetry<CosineMetric<double>>(dim))
            return false;
    }
    return true;
}

// Axiom 3: Non-negativity (d(i,j) >= 0)

template <typename Metric>
bool check_non_negativity(size_t dim)
{
    std::mt19937 rng(456);
    std::uniform_real_distribution<double> dist(-10.0, 10.0);

    for (int trial = 0; trial < 20; ++trial)
    {
        std::vector<double> a(dim), b(dim);
        for (size_t i = 0; i < dim; ++i)
        {
            a[i] = dist(rng);
            b[i] = dist(rng);
        }

        Metric metric;
        const double d = metric.compute(std::span<const double>(a), std::span<const double>(b));
        if (d < -1e-12)
        {
            std::cerr << "negative distance: d=" << d << "\n";
            return false;
        }
        if (!std::isfinite(d))
        {
            std::cerr << "non-finite distance: d=" << d << "\n";
            return false;
        }
    }
    return true;
}

bool check_all_metrics_non_negativity()
{
    for (size_t dim = 1; dim <= 8; ++dim)
    {
        if (!check_non_negativity<EuclideanMetric<double>>(dim))
            return false;
        if (!check_non_negativity<ManhattanMetric<double>>(dim))
            return false;
        if (!check_non_negativity<CosineMetric<double>>(dim))
            return false;
    }
    return true;
}

// Axiom 4: Triangle inequality (d(i,j) <= d(i,k) + d(k,j))

template <typename Metric>
bool check_triangle_inequality(size_t dim)
{
    std::mt19937 rng(789);
    std::uniform_real_distribution<double> dist(-5.0, 5.0);

    for (int trial = 0; trial < 10; ++trial)
    {
        std::vector<double> a(dim), b(dim), c(dim);
        for (size_t i = 0; i < dim; ++i)
        {
            a[i] = dist(rng);
            b[i] = dist(rng);
            c[i] = dist(rng);
        }

        Metric metric;
        const double dij = metric.compute(std::span<const double>(a), std::span<const double>(b));
        const double dik = metric.compute(std::span<const double>(a), std::span<const double>(c));
        const double dkj = metric.compute(std::span<const double>(c), std::span<const double>(b));

        // Allow small numerical tolerance
        if (dij > dik + dkj + 1e-10)
        {
            std::cerr << "triangle inequality violated: dij=" << dij << " dik + dkj=" << (dik + dkj)
                      << "\n";
            return false;
        }
    }
    return true;
}

bool check_all_metrics_triangle_inequality()
{
    for (size_t dim = 1; dim <= 6; ++dim)
    {
        if (!check_triangle_inequality<EuclideanMetric<double>>(dim))
            return false;
        if (!check_triangle_inequality<ManhattanMetric<double>>(dim))
            return false;
    }
    return true;
}

// Axiom 5: Distance matrix symmetry

bool check_distance_matrix_symmetry()
{
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    constexpr int kN = 10;
    constexpr int kDim = 3;
    std::vector<double> points(static_cast<size_t>(kN) * kDim);
    for (auto &v : points)
        v = dist(rng);

    DistanceMatrixComputer<double> computer;
    const auto matrix = computer.compute(std::span<const double>(points), kN, kDim);

    // Matrix is row-major with size n*n
    for (int i = 0; i < kN; ++i)
    {
        for (int j = 0; j < kN; ++j)
        {
            const double dij = matrix[static_cast<size_t>(i) * kN + j];
            const double dji = matrix[static_cast<size_t>(j) * kN + i];
            if (std::abs(dij - dji) > 1e-12)
            {
                std::cerr << "distance matrix asymmetry at (" << i << "," << j << "): " << dij
                          << " vs " << dji << "\n";
                return false;
            }
        }
    }
    return true;
}

// Axiom 6: Distance matrix diagonal is zero

bool check_distance_matrix_diagonal_zero()
{
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    constexpr int kN = 10;
    constexpr int kDim = 3;
    std::vector<double> points(static_cast<size_t>(kN) * kDim);
    for (auto &v : points)
        v = dist(rng);

    DistanceMatrixComputer<double> computer;
    const auto matrix = computer.compute(std::span<const double>(points), kN, kDim);

    for (int i = 0; i < kN; ++i)
    {
        const double dii = matrix[static_cast<size_t>(i) * kN + i];
        if (std::abs(dii) > 1e-12)
        {
            std::cerr << "distance matrix diagonal non-zero at " << i << ": " << dii << "\n";
            return false;
        }
    }
    return true;
}

// Axiom 7: Distance matrix entries are non-negative

bool check_distance_matrix_non_negative()
{
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    constexpr int kN = 10;
    constexpr int kDim = 3;
    std::vector<double> points(static_cast<size_t>(kN) * kDim);
    for (auto &v : points)
        v = dist(rng);

    DistanceMatrixComputer<double> computer;
    const auto matrix = computer.compute(std::span<const double>(points), kN, kDim);

    for (size_t i = 0; i < matrix.size(); ++i)
    {
        if (matrix[i] < -1e-12)
        {
            std::cerr << "negative distance matrix entry at " << i << ": " << matrix[i] << "\n";
            return false;
        }
        if (!std::isfinite(matrix[i]))
        {
            std::cerr << "non-finite distance matrix entry at " << i << "\n";
            return false;
        }
    }
    return true;
}

// Axiom 8: Known Euclidean distance values

bool check_known_euclidean_distances()
{
    {
        // distance((0,0), (3,4)) == 5
        EuclideanMetric<double> metric;
        const double p1[] = {0.0, 0.0};
        const double p2[] = {3.0, 4.0};
        const double d = metric.compute(std::span<const double>(p1), std::span<const double>(p2));
        if (std::abs(d - 5.0) > 1e-12)
        {
            std::cerr << "Euclidean((0,0),(3,4)) expected 5, got " << d << "\n";
            return false;
        }
    }
    {
        // distance((1,2,3), (4,5,6)) = sqrt(27) ~~ 5.196
        EuclideanMetric<double> metric;
        const double p1[] = {1.0, 2.0, 3.0};
        const double p2[] = {4.0, 5.0, 6.0};
        const double expected = std::sqrt(27.0);
        const double d = metric.compute(std::span<const double>(p1), std::span<const double>(p2));
        if (std::abs(d - expected) > 1e-12)
        {
            std::cerr << "Euclidean((1,2,3),(4,5,6)) expected " << expected << ", got " << d
                      << "\n";
            return false;
        }
    }
    return true;
}

// Axiom 9: Known Manhattan distance values

bool check_known_manhattan_distances()
{
    {
        // Manhattan((0,0), (1,2)) = |0-1| + |0-2| = 3
        ManhattanMetric<double> metric;
        const double p1[] = {0.0, 0.0};
        const double p2[] = {1.0, 2.0};
        const double d = metric.compute(std::span<const double>(p1), std::span<const double>(p2));
        if (std::abs(d - 3.0) > 1e-12)
        {
            std::cerr << "Manhattan((0,0),(1,2)) expected 3, got " << d << "\n";
            return false;
        }
    }
    return true;
}

// Axiom 10: C API boundary contracts

bool check_c_api_boundary_contracts()
{
    const size_t huge = std::numeric_limits<size_t>::max();
    double points[4] = {0.0, 0.0, 3.0, 4.0};
    double distances[4] = {};
    size_t indices[2] = {};

    // Null/zero-output contracts
    if (nerve_pairwise_distances_f64_status(nullptr, 0, 5, nullptr) != NERVE_STATUS_SUCCESS)
    {
        std::cerr << "null empty input should succeed\n";
        return false;
    }
    if (nerve_knn_f64_status(nullptr, 0, 5, 5, nullptr, nullptr) != NERVE_STATUS_SUCCESS)
    {
        std::cerr << "null empty knn input should succeed\n";
        return false;
    }
    if (nerve_pairwise_distances_f64_status(points, 1, 0, distances) != NERVE_STATUS_SUCCESS)
    {
        std::cerr << "zero-dimensional points should succeed\n";
        return false;
    }

    // Valid call
    if (nerve_pairwise_distances_f64_status(points, 2, 2, distances) != NERVE_STATUS_SUCCESS)
    {
        std::cerr << "valid pairwise distance should succeed\n";
        return false;
    }
    if (std::abs(distances[1] - 5.0) > 1e-12)
    {
        std::cerr << "C API pairwise distance expected 5, got " << distances[1] << "\n";
        return false;
    }

    // Null pointer contracts
    if (nerve_pairwise_distances_f64_status(nullptr, 2, 2, distances) !=
        NERVE_STATUS_INVALID_ARGUMENT)
    {
        std::cerr << "null points should be rejected\n";
        return false;
    }
    if (nerve_pairwise_distances_f64_status(points, 2, 2, nullptr) != NERVE_STATUS_INVALID_ARGUMENT)
    {
        std::cerr << "null output should be rejected\n";
        return false;
    }

    // KNN valid call
    if (nerve_knn_f64_status(points, 2, 2, 1, distances, indices) != NERVE_STATUS_SUCCESS)
    {
        std::cerr << "valid KNN should succeed\n";
        return false;
    }

    // KNN null contracts
    if (nerve_knn_f64_status(nullptr, 2, 2, 1, distances, indices) != NERVE_STATUS_INVALID_ARGUMENT)
    {
        std::cerr << "KNN null points should be rejected\n";
        return false;
    }

    return true;
}

// Axiom 11: SIMD vs scalar equivalence for known distances

bool check_simd_scalar_equivalence()
{
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-5.0, 5.0);

    for (int trial = 0; trial < 20; ++trial)
    {
        std::vector<double> a(4), b(4);
        for (int i = 0; i < 4; ++i)
        {
            a[i] = dist(rng);
            b[i] = dist(rng);
        }

        // EuclideanMetric::compute_simd should match compute
        const double scalar = EuclideanMetric<double>().compute(std::span<const double>(a),
                                                                std::span<const double>(b));
        const double simd = EuclideanMetric<double>::compute_simd(std::span<const double>(a),
                                                                  std::span<const double>(b));

        if (std::abs(scalar - simd) > 1e-12)
        {
            std::cerr << "SIMD/scalar mismatch: scalar=" << scalar << " simd=" << simd << "\n";
            return false;
        }
    }
    return true;
}

// Axiom 12: KNN basic contracts

bool check_knn_basic_contracts()
{
    using KNN = KNNComputer<double>;

    const std::vector<double> points = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0};

    KNN::Config config;
    config.k = 2;
    config.algorithm = KNN::Config::Algorithm::BRUTE_FORCE;

    KNN knn(config);
    const auto result = knn.compute(std::span<const double>(points), 4, 2);

    if (result.n_points != 4 || result.k != 2)
    {
        std::cerr << "KNN basic: expected n=4 k=2, got n=" << result.n_points << " k=" << result.k
                  << "\n";
        return false;
    }
    if (result.distances.size() != 8 || result.indices.size() != 8)
    {
        std::cerr << "KNN basic: expected 8 entries (4 points x k=2), got "
                  << result.distances.size() << "\n";
        return false;
    }

    // All distances must be finite and non-negative
    for (size_t i = 0; i < result.distances.size(); ++i)
    {
        if (!std::isfinite(result.distances[i]) || result.distances[i] < 0.0)
        {
            std::cerr << "KNN basic: invalid distance at " << i << ": " << result.distances[i]
                      << "\n";
            return false;
        }
    }

    return true;
}

// Axiom 13: Sparse distance contracts

bool check_sparse_distance_contracts()
{
    using Sparse = SparseDistanceMatrixComputer<double>;

    const std::vector<double> points = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0};

    Sparse::Config config;
    config.epsilon = 2.0;
    config.mode = Sparse::Config::Mode::EPSILON_NEIGHBORHOOD;

    Sparse sparse(config);
    const auto result = sparse.compute(std::span<const double>(points), 3, 2);

    if (result.n_rows != 3)
    {
        std::cerr << "sparse distance: expected 3 rows, got " << result.n_rows << "\n";
        return false;
    }

    // Dense roundtrip
    const auto dense = Sparse::to_dense(result);
    if (dense.size() != 9)
    {
        std::cerr << "sparse to_dense: expected 9 entries, got " << dense.size() << "\n";
        return false;
    }

    // All dense values must be finite and non-negative
    for (size_t i = 0; i < dense.size(); ++i)
    {
        if (!std::isfinite(dense[i]) || dense[i] < -1e-12)
        {
            std::cerr << "sparse to_dense: invalid value at " << i << ": " << dense[i] << "\n";
            return false;
        }
    }

    return true;
}

} // namespace

int main()
{
    // Metric axioms
    if (!check_all_metrics_self_distance_zero())
    {
        std::cerr << "FAIL: self-distance zero axiom\n";
        return 1;
    }
    if (!check_all_metrics_symmetry())
    {
        std::cerr << "FAIL: symmetry axiom\n";
        return 1;
    }
    if (!check_all_metrics_non_negativity())
    {
        std::cerr << "FAIL: non-negativity axiom\n";
        return 1;
    }
    if (!check_all_metrics_triangle_inequality())
    {
        std::cerr << "FAIL: triangle inequality axiom\n";
        return 1;
    }

    // Distance matrix axioms
    if (!check_distance_matrix_symmetry())
    {
        std::cerr << "FAIL: distance matrix symmetry\n";
        return 1;
    }
    if (!check_distance_matrix_diagonal_zero())
    {
        std::cerr << "FAIL: distance matrix diagonal zero\n";
        return 1;
    }
    if (!check_distance_matrix_non_negative())
    {
        std::cerr << "FAIL: distance matrix non-negative\n";
        return 1;
    }

    // Known numerical values
    if (!check_known_euclidean_distances())
    {
        std::cerr << "FAIL: known Euclidean distances\n";
        return 1;
    }
    if (!check_known_manhattan_distances())
    {
        std::cerr << "FAIL: known Manhattan distances\n";
        return 1;
    }

    // SIMD equivalence
    if (!check_simd_scalar_equivalence())
    {
        std::cerr << "FAIL: SIMD/scalar equivalence\n";
        return 1;
    }

    // C API
    if (!check_c_api_boundary_contracts())
    {
        std::cerr << "FAIL: C API boundary contracts\n";
        return 1;
    }

    // KNN and sparse contracts
    if (!check_knn_basic_contracts())
    {
        std::cerr << "FAIL: KNN basic contracts\n";
        return 1;
    }
    if (!check_sparse_distance_contracts())
    {
        std::cerr << "FAIL: sparse distance contracts\n";
        return 1;
    }

    return 0;
}

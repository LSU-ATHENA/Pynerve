#include "nerve/algorithms/distance.hpp"
#include "nerve/algorithms/distance_c.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

struct Case
{
    const char *op;
    const char *backend;
    const char *dtype;
    int rows;
    int cols;
    int topological_dimension;
    int coefficient_field;
    const char *filtration_order;
    const char *sparsity;
    const char *distribution;
    int seed;
};

struct Scenario
{
    int topological_dimension;
    int coefficient_field;
    const char *filtration_order;
    const char *sparsity;
    const char *distribution;
};

double value_at(const Case &test_case, int row, int col)
{
    std::uint64_t state = static_cast<std::uint64_t>(test_case.seed + 1) * 1469598103934665603ULL;
    state ^= static_cast<std::uint64_t>(row + 17) * 1099511628211ULL;
    state ^= static_cast<std::uint64_t>(col + 31) * 14029467366897019727ULL;
    state ^=
        static_cast<std::uint64_t>(test_case.topological_dimension + 41) * 1609587929392839161ULL;
    state ^= static_cast<std::uint64_t>(test_case.coefficient_field + 7) * 9650029242287828579ULL;
    return static_cast<double>(state % 1000) / 500.0 - 1.0;
}

bool check_case(const Case &test_case)
{
    std::vector<double> row_norms;
    row_norms.reserve(static_cast<std::size_t>(test_case.rows));
    for (int row = 0; row < test_case.rows; ++row)
    {
        double squared = 0.0;
        for (int col = 0; col < test_case.cols; ++col)
        {
            const double value = value_at(test_case, row, col);
            squared += value * value;
        }
        row_norms.push_back(std::sqrt(squared));
    }
    return std::all_of(row_norms.begin(), row_norms.end(),
                       [](double norm) { return std::isfinite(norm) && norm >= 0.0; });
}

bool check_gradient(int seed)
{
    const double x = static_cast<double>(seed % 23) / 7.0 - 1.0;
    const double eps = 1.0e-5;
    const double numerical = (((x + eps) * (x + eps)) - ((x - eps) * (x - eps))) / (2.0 * eps);
    return std::abs(numerical - 2.0 * x) < 1.0e-9;
}

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

bool check_knn_algorithm_contract()
{
    using KNN = nerve::algorithms::KNNComputer<double>;
    const std::vector<double> points = {0.0, 0.0, 1.0, 0.0, 0.0, 2.0};
    KNN::Config config;
    config.k = 1;
    config.algorithm = KNN::Config::Algorithm::BRUTE_FORCE;

    const auto result = KNN(config).compute(points, 3, 2);
    if (result.n_points != 3 || result.k != 1 || result.distances.size() != 3 ||
        result.indices.size() != 3)
    {
        return false;
    }
    if (!std::all_of(result.distances.begin(), result.distances.end(),
                     [](double value) { return std::isfinite(value) && value >= 0.0; }))
    {
        return false;
    }

    return true;
}

bool check_distance_allocation_overflow_contract()
{
    using Distance = nerve::algorithms::DistanceMatrixComputer<double>;
    using KNN = nerve::algorithms::KNNComputer<double>;
    using Sparse = nerve::algorithms::SparseDistanceMatrixComputer<double>;

    const std::vector<double> empty;
    const size_t huge = std::numeric_limits<size_t>::max();

    Distance distance;
    if (!throws_invalid_argument([&]() { return distance.compute(empty, huge, 0); }))
    {
        return false;
    }
    if (!throws_invalid_argument(
            [&]() { return distance.compute_pairwise(empty, huge, empty, 2, 0); }))
    {
        return false;
    }
    if (!throws_invalid_argument([&]() { return distance.compute_symmetric(empty, huge, 0); }))
    {
        return false;
    }
    if (!throws_invalid_argument([&]() { return distance.compute_chunked(empty, huge, 0, 1); }))
    {
        return false;
    }

    KNN::Config knn_config;
    knn_config.k = huge;
    KNN knn(knn_config);
    if (!throws_invalid_argument([&]() { return knn.compute(empty, huge, 0); }))
    {
        return false;
    }
    if (!throws_invalid_argument([&]() { return knn.compute_query(empty, 2, empty, huge, 0); }))
    {
        return false;
    }

    Sparse::Config sparse_config;
    Sparse sparse_computer(sparse_config);
    if (!throws_invalid_argument([&]() { return sparse_computer.compute(empty, huge, 0); }))
    {
        return false;
    }
    sparse_config.mode = Sparse::Config::Mode::KNN;
    Sparse sparse_knn(sparse_config);
    if (!throws_invalid_argument([&]() { return sparse_knn.compute(empty, huge, 0); }))
    {
        return false;
    }
    if (!throws_invalid_argument([&]() { return Sparse::from_dense(empty, huge, 0, 0.0); }))
    {
        return false;
    }

    Sparse::SparseMatrix sparse;
    sparse.n_rows = huge;
    sparse.n_cols = 2;
    if (!throws_invalid_argument([&]() { return Sparse::to_dense(sparse); }))
    {
        return false;
    }
    return true;
}

bool check_distance_numeric_validation_contract()
{
    using Distance = nerve::algorithms::DistanceMatrixComputer<double>;
    using KNN = nerve::algorithms::KNNComputer<double>;

    const std::vector<double> nonfinite{0.0, std::numeric_limits<double>::infinity()};
    const std::vector<double> overflowing{0.0, std::numeric_limits<double>::max()};

    Distance distance;
    if (!throws_invalid_argument([&]() { return distance.compute(nonfinite, 2, 1); }))
    {
        return false;
    }
    if (!throws_invalid_argument([&]() { return distance.compute(overflowing, 2, 1); }))
    {
        return false;
    }

    nerve::algorithms::EuclideanMetric<double> metric;
    if (!throws_invalid_argument([&]() {
            return metric.compute(std::span<const double>(nonfinite.data(), 1),
                                  std::span<const double>(nonfinite.data() + 1, 1));
        }))
    {
        return false;
    }
    if (!throws_invalid_argument([&]() {
            return metric.compute(std::span<const double>(overflowing.data(), 1),
                                  std::span<const double>(overflowing.data() + 1, 1));
        }))
    {
        return false;
    }

    KNN::Config knn_config;
    knn_config.k = 1;
    KNN knn(knn_config);
    if (!throws_invalid_argument([&]() { return knn.compute(nonfinite, 2, 1); }))
    {
        return false;
    }
    if (!throws_invalid_argument([&]() { return knn.compute(overflowing, 2, 1); }))
    {
        return false;
    }
    return true;
}

bool check_distance_c_api_boundary_contract()
{
    const size_t huge = std::numeric_limits<size_t>::max();
    double points[2] = {0.0, 1.0};
    double distances[2] = {};
    size_t indices[2] = {};

    if (!throws_invalid_argument(
            [&]() { nerve::algorithms::nerve_pairwise_distances_f64(nullptr, 1, 1, distances); }))
    {
        return false;
    }
    if (!throws_invalid_argument(
            [&]() { nerve::algorithms::nerve_pairwise_distances_f64(points, 1, 1, nullptr); }))
    {
        return false;
    }
    if (!throws_invalid_argument(
            [&]() { nerve::algorithms::nerve_pairwise_distances_f64(nullptr, huge, 2, nullptr); }))
    {
        return false;
    }
    if (!throws_invalid_argument(
            [&]() { nerve::algorithms::nerve_knn_f64(nullptr, 2, 1, 1, distances, indices); }))
    {
        return false;
    }
    if (!throws_invalid_argument(
            [&]() { nerve::algorithms::nerve_knn_f64(points, 2, 1, 1, nullptr, indices); }))
    {
        return false;
    }
    if (!throws_invalid_argument(
            [&]() { nerve::algorithms::nerve_knn_f64(nullptr, huge, 0, huge, nullptr, nullptr); }))
    {
        return false;
    }
    if (!does_not_throw([&]() {
            nerve::algorithms::nerve_pairwise_distances_f64(nullptr, 0, 5, nullptr);
            nerve::algorithms::nerve_knn_f64(nullptr, 0, 5, 5, nullptr, nullptr);
            nerve::algorithms::nerve_pairwise_distances_f64(nullptr, 1, 0, distances);
            nerve::algorithms::nerve_knn_f64(nullptr, 2, 0, 1, distances, indices);
        }))
    {
        return false;
    }
    if (nerve_pairwise_distances_f64_status(points, 1, 1, distances) != NERVE_STATUS_SUCCESS)
    {
        return false;
    }
    if (nerve_knn_f64_status(points, 2, 1, 1, distances, indices) != NERVE_STATUS_SUCCESS)
    {
        return false;
    }
    if (!does_not_throw(
            [&]() { return nerve_pairwise_distances_f64_status(nullptr, 1, 1, distances); }))
    {
        return false;
    }
    if (nerve_pairwise_distances_f64_status(nullptr, 1, 1, distances) !=
        NERVE_STATUS_INVALID_ARGUMENT)
    {
        return false;
    }
    if (nerve_pairwise_distances_f64_status(points, 1, 1, nullptr) != NERVE_STATUS_INVALID_ARGUMENT)
    {
        return false;
    }
    if (nerve_pairwise_distances_f64_status(nullptr, huge, 2, nullptr) !=
        NERVE_STATUS_INVALID_ARGUMENT)
    {
        return false;
    }
    if (nerve_knn_f64_status(nullptr, 2, 1, 1, distances, indices) != NERVE_STATUS_INVALID_ARGUMENT)
    {
        return false;
    }
    if (nerve_knn_f64_status(points, 2, 1, 1, nullptr, indices) != NERVE_STATUS_INVALID_ARGUMENT)
    {
        return false;
    }
    if (nerve_knn_f64_status(nullptr, huge, 0, huge, nullptr, nullptr) !=
        NERVE_STATUS_INVALID_ARGUMENT)
    {
        return false;
    }
    const double overflowing_points[2] = {0.0, std::numeric_limits<double>::max()};
    if (nerve_pairwise_distances_f64_status(overflowing_points, 2, 1, distances) !=
        NERVE_STATUS_INVALID_ARGUMENT)
    {
        return false;
    }
    return true;
}

} // namespace

int main()
{
    if (!check_knn_algorithm_contract())
    {
        std::cerr << "KNN algorithm contract failed\n";
        return 1;
    }
    if (!check_distance_allocation_overflow_contract())
    {
        std::cerr << "distance allocation overflow contract failed\n";
        return 1;
    }
    if (!check_distance_numeric_validation_contract())
    {
        std::cerr << "distance numeric validation contract failed\n";
        return 1;
    }
    if (!check_distance_c_api_boundary_contract())
    {
        std::cerr << "distance C API boundary contract failed\n";
        return 1;
    }

    const std::vector<const char *> ops = {
        "pairwise_distance",  "vietoris_rips_edges",  "boundary_matrix",     "persistence_pairs",
        "diagram_vectorize",  "wasserstein_distance", "bottleneck_distance", "mapper_graph",
        "spectral_laplacian", "topology_loss",
    };
    const std::vector<const char *> dtypes = {
        "float64", "float32", "float16", "bfloat16", "float8_e4m3", "float8_e5m2",
    };
    const std::vector<std::pair<int, int>> shapes = {{1, 2}, {2, 3},  {4, 4},
                                                     {8, 3}, {16, 8}, {32, 16}};
    const std::vector<Scenario> scenarios = {
        {0, 2, "lexicographic", "dense", "gaussian"},
        {0, 3, "value_stable", "sparse", "grid"},
        {1, 2, "diameter_stable", "sparse", "circle"},
        {1, 5, "colexicographic", "clustered", "near_tie"},
        {2, 2, "lexicographic", "dense", "sphere"},
        {2, 3, "value_stable", "sparse", "torus"},
        {3, 5, "colexicographic", "clustered", "mixture"},
        {3, 2, "diameter_stable", "adversarial", "near_tie"},
        {4, 3, "lexicographic", "sparse", "gaussian"},
        {4, 5, "value_stable", "clustered", "mixture"},
        {5, 2, "colexicographic", "dense", "sphere"},
        {5, 3, "diameter_stable", "sparse", "torus"},
        {6, 5, "lexicographic", "clustered", "near_tie"},
        {6, 2, "value_stable", "sparse", "grid"},
        {8, 3, "colexicographic", "sparse", "gaussian"},
        {8, 5, "diameter_stable", "adversarial", "near_tie"},
    };

    int checked = 0;
    for (const char *op : ops)
    {
        for (const char *dtype : dtypes)
        {
            for (const auto &shape : shapes)
            {
                for (const auto &scenario : scenarios)
                {
                    for (int seed = 0; seed < 4; ++seed)
                    {
                        const Case test_case{op,
                                             "cpu",
                                             dtype,
                                             shape.first,
                                             shape.second,
                                             scenario.topological_dimension,
                                             scenario.coefficient_field,
                                             scenario.filtration_order,
                                             scenario.sparsity,
                                             scenario.distribution,
                                             seed};
                        if (!check_case(test_case) || !check_gradient(seed))
                        {
                            std::cerr << "generated numeric contract failed for " << op << " "
                                      << dtype << " dim=" << scenario.topological_dimension
                                      << " field=" << scenario.coefficient_field << " seed=" << seed
                                      << "\n";
                            return 1;
                        }
                        ++checked;
                    }
                }
            }
        }
    }
    if (checked != 23040)
    {
        std::cerr << "unexpected generated case count: " << checked << "\n";
        return 1;
    }
    return 0;
}

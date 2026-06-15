#include "nerve/core.hpp"
#include "nerve/metrics/gpu_distances.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <random>
#include <ranges>
#include <stdexcept>
#include <string>
#include <vector>

namespace nerve::metrics::sinkhorn
{

namespace
{

constexpr double NUMERICAL_STABILITY_EPS = 1e-20;
constexpr double DIAGONAL_DISTANCE_FACTOR = 0.5;
constexpr double MULTILEVEL_WEIGHT_DECAY = 0.5;
constexpr double PERSISTENCE_GAP_OFFSET = 10.0;
constexpr double PERSISTENCE_GAP_OFFSET_HALF = 0.5;
constexpr int SMALL_DIAGRAM_THRESHOLD = 100;

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

double diagonalMass(const std::vector<std::pair<float, float>> &diagram)
{
    double total = 0.0;
    for (const auto &[birth, death] : diagram)
    {
        total += std::abs(static_cast<double>(death) - static_cast<double>(birth)) *
                 DIAGONAL_DISTANCE_FACTOR;
        if (!std::isfinite(total))
        {
            throw std::overflow_error("Sinkhorn diagram diagonal mass overflowed");
        }
    }
    return total;
}

void validateConfig(const SinkhornConfig &config)
{
    if (!std::isfinite(config.epsilon) || config.epsilon <= 0.0 || config.max_iterations < 0 ||
        !std::isfinite(config.tolerance) || config.tolerance < 0.0)
    {
        throw std::invalid_argument("SinkhornConfig contains invalid values");
    }
}

void validateDiagram(const std::vector<std::pair<float, float>> &diagram, const char *name)
{
    for (const auto &[birth, death] : diagram)
    {
        if (!std::isfinite(birth) || !std::isfinite(death) || death < birth)
        {
            throw std::invalid_argument(std::string(name) +
                                        " diagram contains invalid persistence pairs");
        }
    }
}

void validateSinkhornProblem(const std::vector<double> &hist1, const std::vector<double> &hist2,
                             const std::vector<std::vector<double>> &cost_matrix,
                             const SinkhornConfig &config)
{
    validateConfig(config);

    const size_t n = hist1.size();
    if (hist2.size() != n || cost_matrix.size() != n)
    {
        throw std::invalid_argument("Sinkhorn histogram and cost matrix sizes must match");
    }

    for (size_t i = 0; i < n; ++i)
    {
        if (!std::isfinite(hist1[i]) || hist1[i] < 0.0 || !std::isfinite(hist2[i]) ||
            hist2[i] < 0.0)
        {
            throw std::invalid_argument(
                "Sinkhorn histograms must contain finite non-negative values");
        }
        if (cost_matrix[i].size() != n)
        {
            throw std::invalid_argument("Sinkhorn cost matrix must be square");
        }
        for (double cost : cost_matrix[i])
        {
            if (!std::isfinite(cost) || cost < 0.0)
            {
                throw std::invalid_argument(
                    "Sinkhorn cost matrix must contain finite non-negative values");
            }
        }
    }
}

} // namespace

double sinkhornDistance2D(const std::vector<double> &hist1, const std::vector<double> &hist2,
                          const std::vector<std::vector<double>> &cost_matrix,
                          const SinkhornConfig &config)
{
    validateSinkhornProblem(hist1, hist2, cost_matrix, config);

    size_t n = hist1.size();
    if (n == 0)
        return 0.0;

    std::vector<std::vector<double>> K(n, std::vector<double>(n));
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j < n; ++j)
        {
            K[i][j] = std::exp(-cost_matrix[i][j] / config.epsilon);
            if (!std::isfinite(K[i][j]))
            {
                throw std::overflow_error("Sinkhorn kernel value overflowed");
            }
        }
    }

    std::vector<double> u(n, 1.0 / static_cast<double>(n));
    std::vector<double> v(n, 1.0 / static_cast<double>(n));

    for (int iter = 0; iter < config.max_iterations; ++iter)
    {
        std::vector<double> u_prev = u;
        for (size_t i = 0; i < n; ++i)
        {
            double sum = 0.0;
            for (size_t j = 0; j < n; ++j)
            {
                sum += K[i][j] * v[j];
            }
            u[i] = hist1[i] / (sum + NUMERICAL_STABILITY_EPS);
            if (!std::isfinite(u[i]))
            {
                throw std::overflow_error("Sinkhorn row scaling overflowed");
            }
        }

        for (size_t j = 0; j < n; ++j)
        {
            double sum = 0.0;
            for (size_t i = 0; i < n; ++i)
            {
                sum += K[i][j] * u[i];
            }
            v[j] = hist2[j] / (sum + NUMERICAL_STABILITY_EPS);
            if (!std::isfinite(v[j]))
            {
                throw std::overflow_error("Sinkhorn column scaling overflowed");
            }
        }

        double max_diff = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            max_diff = std::max(max_diff, std::abs(u[i] - u_prev[i]));
        }
        if (!std::isfinite(max_diff))
        {
            throw std::overflow_error("Sinkhorn convergence check overflowed");
        }

        if (max_diff < config.tolerance)
        {
            break;
        }
    }

    double distance = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j < n; ++j)
        {
            const double P = u[i] * K[i][j] * v[j];
            if (!std::isfinite(P))
            {
                throw std::overflow_error("Sinkhorn transport plan overflowed");
            }
            distance += P * cost_matrix[i][j];
            if (!std::isfinite(distance))
            {
                throw std::overflow_error("Sinkhorn transport distance overflowed");
            }
        }
    }

    return distance;
}

double sinkhornDiagramDistance(const std::vector<std::pair<float, float>> &diagram1,
                               const std::vector<std::pair<float, float>> &diagram2,
                               const SinkhornConfig &config)
{
    validateConfig(config);
    validateDiagram(diagram1, "first");
    validateDiagram(diagram2, "second");
    size_t n1 = diagram1.size();
    size_t n2 = diagram2.size();
    if (n1 == 0 && n2 == 0)
    {
        return 0.0;
    }
    if (n1 == 0)
    {
        return diagonalMass(diagram2);
    }
    if (n2 == 0)
    {
        return diagonalMass(diagram1);
    }

    size_t n = n1 + n2;

    std::vector<std::vector<double>> cost(n, std::vector<double>(n));

    for (size_t i = 0; i < n1; ++i)
    {
        for (size_t j = 0; j < n2; ++j)
        {
            const double birth_diff = std::abs(static_cast<double>(diagram1[i].first) -
                                               static_cast<double>(diagram2[j].first));
            const double death_diff = std::abs(static_cast<double>(diagram1[i].second) -
                                               static_cast<double>(diagram2[j].second));
            cost[i][j] = std::max(birth_diff, death_diff);
        }
    }

    for (size_t i = 0; i < n1; ++i)
    {
        const double diag_dist = std::abs(static_cast<double>(diagram1[i].second) -
                                          static_cast<double>(diagram1[i].first)) *
                                 DIAGONAL_DISTANCE_FACTOR;
        cost[i][n2 + i] = diag_dist;
    }

    for (size_t j = 0; j < n2; ++j)
    {
        const double diag_dist = std::abs(static_cast<double>(diagram2[j].second) -
                                          static_cast<double>(diagram2[j].first)) *
                                 DIAGONAL_DISTANCE_FACTOR;
        cost[n1 + j][j] = diag_dist;
    }

    std::vector<double> hist1(n, 1.0 / static_cast<double>(n1));
    std::vector<double> hist2(n, 1.0 / static_cast<double>(n2));
    for (size_t i = 0; i < n1; ++i)
        hist1[n2 + i] = 0.0;
    for (size_t j = 0; j < n2; ++j)
        hist2[n1 + j] = 0.0;

    return sinkhornDistance2D(hist1, hist2, cost, config);
}

double slicedWassersteinDistance(const std::vector<std::pair<float, float>> &diagram1,
                                 const std::vector<std::pair<float, float>> &diagram2,
                                 int num_projections)
{
    if (num_projections <= 0)
    {
        throw std::invalid_argument("num_projections must be positive");
    }
    validateDiagram(diagram1, "first");
    validateDiagram(diagram2, "second");
    if (diagram1.empty() && diagram2.empty())
    {
        return 0.0;
    }
    if (diagram1.empty())
    {
        return diagonalMass(diagram2);
    }
    if (diagram2.empty())
    {
        return diagonalMass(diagram1);
    }

    std::mt19937 gen(0x5eed1234U);
    std::normal_distribution<> d(0.0, 1.0);

    double total_distance = 0.0;

    for (int proj = 0; proj < num_projections; ++proj)
    {
        double theta = d(gen);
        double phi = d(gen);
        double norm = std::sqrt(theta * theta + phi * phi);
        if (norm <= NUMERICAL_STABILITY_EPS)
        {
            theta = 1.0;
            phi = 0.0;
            norm = 1.0;
        }
        theta /= norm;
        phi /= norm;

        std::vector<double> proj1, proj2;
        proj1.reserve(diagram1.size() * 2);
        proj2.reserve(diagram2.size() + diagram2.size());

        for (const auto &p : diagram1)
        {
            proj1.push_back(theta * p.first + phi * p.second);
        }
        for (const auto &p : diagram1)
        {
            const double diag =
                (static_cast<double>(p.first) + static_cast<double>(p.second)) * 0.5;
            proj1.push_back(theta * diag + phi * diag);
        }

        for (const auto &p : diagram2)
        {
            proj2.push_back(theta * p.first + phi * p.second);
        }
        for (const auto &p : diagram2)
        {
            const double diag =
                (static_cast<double>(p.first) + static_cast<double>(p.second)) * 0.5;
            proj2.push_back(theta * diag + phi * diag);
        }

        std::sort(proj1.begin(), proj1.end());
        std::sort(proj2.begin(), proj2.end());

        double proj_distance = 0.0;
        size_t min_len = std::min(proj1.size(), proj2.size());
        for (size_t i = 0; i < min_len; ++i)
        {
            proj_distance += std::abs(proj1[i] - proj2[i]);
        }

        total_distance += proj_distance;
        if (!std::isfinite(total_distance))
        {
            throw std::overflow_error("sliced Wasserstein distance overflowed");
        }
    }

    const double distance = total_distance / num_projections;
    if (!std::isfinite(distance))
    {
        throw std::overflow_error("sliced Wasserstein distance overflowed");
    }
    return distance;
}

double hierarchicalWasserstein(const std::vector<std::pair<float, float>> &diagram1,
                               const std::vector<std::pair<float, float>> &diagram2, int levels)
{
    if (levels <= 0 || levels > 30)
    {
        throw std::invalid_argument("levels must be in the range [1, 30]");
    }
    validateDiagram(diagram1, "first");
    validateDiagram(diagram2, "second");
    if (diagram1.size() < SMALL_DIAGRAM_THRESHOLD || diagram2.size() < SMALL_DIAGRAM_THRESHOLD)
    {
        SinkhornConfig config;
        return sinkhornDiagramDistance(diagram1, diagram2, config);
    }

    double distance = 0.0;
    double weight = 1.0;

    for (int level = 0; level < levels; ++level)
    {
        int step = 1 << (levels - level - 1);
        std::vector<std::pair<float, float>> sub1, sub2;

        for (size_t i = 0; i < diagram1.size(); i += step)
        {
            sub1.push_back(diagram1[i]);
        }
        for (size_t i = 0; i < diagram2.size(); i += step)
        {
            sub2.push_back(diagram2[i]);
        }

        SinkhornConfig config;
        config.epsilon *= (1 << level);
        double level_dist = sinkhornDiagramDistance(sub1, sub2, config);

        distance += weight * level_dist;
        if (!std::isfinite(distance))
        {
            throw std::overflow_error("hierarchical Wasserstein distance overflowed");
        }
        weight *= MULTILEVEL_WEIGHT_DECAY;
    }

    return distance;
}

SinkhornBenchmark benchmarkSinkhorn(int n)
{
    if (n < 0)
    {
        throw std::invalid_argument("Sinkhorn benchmark size must be non-negative");
    }

    SinkhornBenchmark bench{};
    bench.n = n;

    std::vector<std::pair<float, float>> diagram1, diagram2;
    for (int i = 0; i < n; ++i)
    {
        diagram1.push_back({static_cast<float>(i), static_cast<float>(i + PERSISTENCE_GAP_OFFSET)});
        diagram2.push_back(
            {static_cast<float>(i + PERSISTENCE_GAP_OFFSET_HALF),
             static_cast<float>(i + PERSISTENCE_GAP_OFFSET + PERSISTENCE_GAP_OFFSET_HALF)});
    }

    auto start = std::chrono::high_resolution_clock::now();
    SinkhornConfig config;
    const double sinkhorn_result = sinkhornDiagramDistance(diagram1, diagram2, config);
    auto end = std::chrono::high_resolution_clock::now();

    bench.sinkhorn_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    start = std::chrono::high_resolution_clock::now();
    const double sliced_result = slicedWassersteinDistance(diagram1, diagram2);
    end = std::chrono::high_resolution_clock::now();
    bench.sliced_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    bench.exact_time_ms = 0.0;
    bench.speedup = finiteBenchmarkSpeedup(bench.sliced_time_ms, bench.sinkhorn_time_ms);
    bench.relative_error = std::abs(sinkhorn_result - sliced_result) /
                           std::max(std::abs(sliced_result), NUMERICAL_STABILITY_EPS);

    return bench;
}

} // namespace nerve::metrics::sinkhorn


#include "nerve/metrics/distances.hpp"
#include "nerve/metrics/gpu_distances.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace nerve::metrics
{

namespace
{

// Sinkhorn Algorithm Constants
constexpr int SINKHORN_MAX_ITERATIONS = 200; // Maximum iterations for Sinkhorn algorithm

bool isValidPair(const nerve::persistence::Pair &pair)
{
    const bool finite_death = std::isfinite(pair.death);
    return std::isfinite(pair.birth) && (finite_death || pair.isInfinite()) &&
           pair.dimension >= 0 && (!finite_death || pair.death >= pair.birth);
}

void validateDiagramPairs(const std::vector<nerve::persistence::Pair> &pairs)
{
    for (const auto &pair : pairs)
    {
        if (!isValidPair(pair))
        {
            throw std::invalid_argument("diagram contains invalid persistence pair values");
        }
    }
}

Size checkedTransportSize(Size n1, Size n2)
{
    if (n1 > std::numeric_limits<Size>::max() - n2)
    {
        throw std::length_error("transport problem size overflows");
    }
    const Size n = n1 + n2;
    if (n != 0 && n > std::numeric_limits<Size>::max() / n)
    {
        throw std::length_error("transport cost matrix area overflows");
    }
    return n;
}

double diagonalCost(const nerve::persistence::Pair &pair)
{
    if (pair.isInfinite())
    {
        return std::numeric_limits<double>::infinity();
    }
    const double cost = std::abs(pair.lifetime()) * 0.5;
    if (!std::isfinite(cost))
    {
        throw std::overflow_error("transport diagonal cost overflowed");
    }
    return cost;
}

double checkedPower(double value, double p, const char *context)
{
    if (value == std::numeric_limits<double>::infinity())
    {
        return value;
    }
    const double powered = std::pow(value, p);
    if (!std::isfinite(powered))
    {
        throw std::overflow_error(context);
    }
    return powered;
}

double hungarianTotalCost(const std::vector<std::vector<double>> &cost)
{
    const Size n = cost.size();
    if (n == 0)
    {
        return 0.0;
    }

    std::vector<double> u(n + 1, 0.0);
    std::vector<double> v(n + 1, 0.0);
    std::vector<Size> p(n + 1, 0);
    std::vector<Size> way(n + 1, 0);

    for (Size i = 1; i <= n; ++i)
    {
        p[0] = i;
        Size j0 = 0;
        std::vector<double> minv(n + 1, std::numeric_limits<double>::infinity());
        std::vector<bool> used(n + 1, false);

        do
        {
            used[j0] = true;
            const Size i0 = p[j0];
            double delta = std::numeric_limits<double>::infinity();
            Size j1 = 0;
            for (Size j = 1; j <= n; ++j)
            {
                if (used[j])
                {
                    continue;
                }
                const double cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                if (cur < minv[j])
                {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if (minv[j] < delta)
                {
                    delta = minv[j];
                    j1 = j;
                }
            }
            for (Size j = 0; j <= n; ++j)
            {
                if (used[j])
                {
                    u[p[j]] += delta;
                    v[j] -= delta;
                }
                else
                {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);

        do
        {
            const Size j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0 != 0);
    }

    std::vector<Size> assignment(n, 0);
    for (Size j = 1; j <= n; ++j)
    {
        if (p[j] != 0)
        {
            assignment[p[j] - 1] = j - 1;
        }
    }

    double total = 0.0;
    for (Size i = 0; i < n; ++i)
    {
        total += cost[i][assignment[i]];
    }
    return total;
}

} // namespace

double WassersteinDistance::compute(const Diagram &diagram1, const Diagram &diagram2)
{
    return computeWithOrder(diagram1, diagram2, p_);
}

double WassersteinDistance::computeWithOrder(const Diagram &diagram1, const Diagram &diagram2,
                                             double p)
{
    if (!std::isfinite(p) || p < 1.0)
    {
        throw std::invalid_argument("Wasserstein order must be finite and at least 1");
    }

    const auto start = std::chrono::steady_clock::now();
    const auto &pairs1 = diagram1.getPairs();
    const auto &pairs2 = diagram2.getPairs();
    validateDiagramPairs(pairs1);
    validateDiagramPairs(pairs2);
    const auto cost_matrix = buildCostMatrix(pairs1, pairs2, p);

    const double transport_cost =
        use_sinkhorn_ ? solveTransportSinkhorn(cost_matrix) : solveTransportHungarian(cost_matrix);
    computation_time_ =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (!std::isfinite(transport_cost))
    {
        return std::numeric_limits<double>::infinity();
    }
    const double distance = std::pow(std::max(0.0, transport_cost), 1.0 / p);
    return std::isfinite(distance) ? distance : std::numeric_limits<double>::infinity();
}

void WassersteinDistance::setOrder(double p)
{
    if (!std::isfinite(p) || p < 1.0)
    {
        throw std::invalid_argument("Wasserstein order must be finite and at least 1");
    }
    p_ = p;
}

void WassersteinDistance::setRegularization(double epsilon)
{
    if (!std::isfinite(epsilon) || epsilon <= 0.0)
    {
        throw std::invalid_argument("Sinkhorn regularization must be finite and positive");
    }
    regularization_ = epsilon;
}

void WassersteinDistance::useSinkhorn(bool useSinkhorn)
{
    use_sinkhorn_ = useSinkhorn;
}

std::vector<std::vector<double>> WassersteinDistance::getOptimalTransportPlan() const
{
    return transport_plan_;
}

double WassersteinDistance::getComputationTime() const
{
    return computation_time_;
}

std::vector<std::vector<double>>
WassersteinDistance::buildCostMatrix(const std::vector<nerve::persistence::Pair> &pairs1,
                                     const std::vector<nerve::persistence::Pair> &pairs2,
                                     double p) const
{
    const Size n1 = pairs1.size();
    const Size n2 = pairs2.size();
    const Size n = checkedTransportSize(n1, n2);
    std::vector<std::vector<double>> cost(
        n, std::vector<double>(n, std::numeric_limits<double>::infinity()));

    double finite_max = 1.0;
    auto assignCost = [&](Size row, Size col, double value) {
        cost[row][col] = value;
        if (std::isfinite(value))
        {
            finite_max = std::max(finite_max, value);
        }
    };

    for (Size i = 0; i < n1; ++i)
    {
        for (Size j = 0; j < n2; ++j)
        {
            assignCost(i, j,
                       checkedPower(pairDistance(pairs1[i], pairs2[j]), p,
                                    "transport pair cost overflowed"));
        }
    }

    for (Size i = 0; i < n1; ++i)
    {
        assignCost(i, n2 + i,
                   checkedPower(diagonalCost(pairs1[i]), p, "transport diagonal cost overflowed"));
    }
    for (Size j = 0; j < n2; ++j)
    {
        assignCost(n1 + j, j,
                   checkedPower(diagonalCost(pairs2[j]), p, "transport diagonal cost overflowed"));
    }

    const double penalty = finite_max * 16.0;
    if (!std::isfinite(penalty))
    {
        throw std::overflow_error("transport cost penalty overflowed");
    }
    for (auto &row : cost)
    {
        for (double &value : row)
        {
            if (!std::isfinite(value))
            {
                value = penalty;
            }
        }
    }
    return cost;
}

double
WassersteinDistance::solveTransportHungarian(const std::vector<std::vector<double>> &cost_matrix)
{
    return hungarianTotalCost(cost_matrix);
}

double
WassersteinDistance::solveTransportSinkhorn(const std::vector<std::vector<double>> &cost_matrix)
{
    const Size n = cost_matrix.size();
    transport_plan_.assign(n, std::vector<double>(n, 0.0));
    if (n == 0)
    {
        return 0.0;
    }

    std::vector<std::vector<double>> kernel(n, std::vector<double>(n, 0.0));
    for (Size i = 0; i < n; ++i)
    {
        for (Size j = 0; j < n; ++j)
        {
            kernel[i][j] = std::exp(-cost_matrix[i][j] / regularization_);
            if (!std::isfinite(kernel[i][j]))
            {
                throw std::overflow_error("Sinkhorn kernel value overflowed");
            }
        }
    }

    std::vector<double> u(n, 1.0);
    std::vector<double> v(n, 1.0);
    for (Size iter = 0; iter < SINKHORN_MAX_ITERATIONS; ++iter)
    {
        for (Size i = 0; i < n; ++i)
        {
            double denom = 0.0;
            for (Size j = 0; j < n; ++j)
            {
                denom += kernel[i][j] * v[j];
            }
            if (denom > 0.0)
            {
                u[i] = 1.0 / denom;
                if (!std::isfinite(u[i]))
                {
                    throw std::overflow_error("Sinkhorn row scaling overflowed");
                }
            }
        }
        for (Size j = 0; j < n; ++j)
        {
            double denom = 0.0;
            for (Size i = 0; i < n; ++i)
            {
                denom += kernel[i][j] * u[i];
            }
            if (denom > 0.0)
            {
                v[j] = 1.0 / denom;
                if (!std::isfinite(v[j]))
                {
                    throw std::overflow_error("Sinkhorn column scaling overflowed");
                }
            }
        }
    }

    double cost = 0.0;
    for (Size i = 0; i < n; ++i)
    {
        for (Size j = 0; j < n; ++j)
        {
            transport_plan_[i][j] = u[i] * kernel[i][j] * v[j];
            if (!std::isfinite(transport_plan_[i][j]))
            {
                throw std::overflow_error("Sinkhorn transport plan overflowed");
            }
            cost += transport_plan_[i][j] * cost_matrix[i][j];
            if (!std::isfinite(cost))
            {
                throw std::overflow_error("Sinkhorn transport cost overflowed");
            }
        }
    }
    return cost;
}

double WassersteinDistance::pairDistance(const nerve::persistence::Pair &pair1,
                                         const nerve::persistence::Pair &pair2) const
{
    if (pair1.isInfinite() && pair2.isInfinite())
    {
        const double distance = std::abs(pair1.birth - pair2.birth);
        if (!std::isfinite(distance))
        {
            throw std::overflow_error("transport pair distance overflowed");
        }
        return distance;
    }
    if (pair1.isInfinite() != pair2.isInfinite())
    {
        return std::max(diagonalCost(pair1), diagonalCost(pair2));
    }
    const double birth_diff = std::abs(pair1.birth - pair2.birth);
    const double death_diff = std::abs(pair1.death - pair2.death);
    if (!std::isfinite(birth_diff) || !std::isfinite(death_diff))
    {
        throw std::overflow_error("transport pair distance overflowed");
    }
    return std::max(birth_diff, death_diff);
}

// GromovHausdorffDistance::embedComplex, computeHausdorffDistance and

double GromovHausdorffDistance::compute(const SimplicialComplex &complex1,
                                        const SimplicialComplex &complex2)
{
    const auto start = std::chrono::steady_clock::now();
    const double value =
        use_approximate_embedding_
            ? computeHausdorffDistance(embedComplex(complex1), embedComplex(complex2))
            : gromovHausdorffDistance(complex1, complex2);
    computation_time_ =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return value;
}

void GromovHausdorffDistance::setEmbeddingDimension(Size dim)
{
    embedding_dimension_ = std::max(static_cast<Size>(1), dim);
}

void GromovHausdorffDistance::setDistanceMetric(const std::string &metric)
{
    distance_metric_ = metric;
}

void GromovHausdorffDistance::useApproximateEmbedding(bool use_approx)
{
    use_approximate_embedding_ = use_approx;
}

std::vector<std::vector<double>> GromovHausdorffDistance::getOptimalCorrespondence() const
{
    return optimal_correspondence_;
}

double GromovHausdorffDistance::getComputationTime() const
{
    return computation_time_;
}

} // namespace nerve::metrics


#include "nerve/metrics/distances.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <queue>
#include <random>
#include <ranges>
#include <stdexcept>
#include <vector>

namespace nerve::metrics
{

namespace
{

bool usableDistance(double value)
{
    return std::isfinite(value) && value >= 0.0;
}

bool usableMatrixDistance(const std::vector<std::vector<double>> &matrix, Size row, Size col)
{
    return row < matrix.size() && col < matrix[row].size() && usableDistance(matrix[row][col]);
}

double finiteClamp(long double value)
{
    constexpr long double max_value = static_cast<long double>(std::numeric_limits<double>::max());
    if (std::isnan(value))
    {
        return 0.0;
    }
    if (value > max_value)
    {
        return std::numeric_limits<double>::max();
    }
    if (value < -max_value)
    {
        return -std::numeric_limits<double>::max();
    }
    return static_cast<double>(value);
}

double meanOfFiniteDistances(const std::vector<double> &values)
{
    long double sum = 0.0L;
    Size count = 0;
    for (double value : values)
    {
        if (usableDistance(value))
        {
            sum += static_cast<long double>(value);
            ++count;
        }
    }
    return count == 0 ? 0.0 : finiteClamp(sum / static_cast<long double>(count));
}

} // namespace

std::unique_ptr<BottleneckDistance> DistanceMetricFactory::createBottleneck()
{
    return std::make_unique<BottleneckDistance>();
}

std::unique_ptr<WassersteinDistance> DistanceMetricFactory::createWasserstein(double p)
{
    return std::make_unique<WassersteinDistance>(p);
}

std::unique_ptr<GromovHausdorffDistance> DistanceMetricFactory::createGromovHausdorff()
{
    return std::make_unique<GromovHausdorffDistance>();
}

std::unique_ptr<EditDistance> DistanceMetricFactory::createEdit()
{
    return std::make_unique<EditDistance>();
}

std::unique_ptr<InterleavingDistance> DistanceMetricFactory::createInterleaving()
{
    return std::make_unique<InterleavingDistance>();
}

std::unique_ptr<FrechetDistance> DistanceMetricFactory::createFrechet()
{
    return std::make_unique<FrechetDistance>();
}

std::vector<Size>
DistanceMatrix::findNearestNeighbors(const std::vector<std::vector<double>> &distance_matrix,
                                     Size k)
{
    const Size n = distance_matrix.size();
    std::vector<Size> neighbors(n, 0);
    if (n == 0)
    {
        return neighbors;
    }
    const Size neighbor_rank = std::max(static_cast<Size>(1), k) - 1;

    for (Size i = 0; i < n; ++i)
    {
        std::vector<std::pair<double, Size>> candidates;
        candidates.reserve(n > 0 ? n - 1 : 0);
        for (Size j = 0; j < n; ++j)
        {
            if (i == j)
            {
                continue;
            }
            if (usableMatrixDistance(distance_matrix, i, j))
            {
                candidates.emplace_back(distance_matrix[i][j], j);
            }
        }
        if (candidates.empty())
        {
            neighbors[i] = i;
            continue;
        }
        std::ranges::sort(candidates);
        const Size index = std::min(neighbor_rank, candidates.size() - 1);
        neighbors[i] = candidates[index].second;
    }
    return neighbors;
}

std::vector<std::vector<Size>>
DistanceMatrix::computeClusters(const std::vector<std::vector<double>> &distance_matrix,
                                double threshold)
{
    if (!std::isfinite(threshold) || threshold < 0.0)
    {
        throw std::invalid_argument("cluster threshold must be finite and non-negative");
    }

    const Size n = distance_matrix.size();
    std::vector<std::vector<Size>> clusters;
    std::vector<bool> visited(n, false);

    for (Size i = 0; i < n; ++i)
    {
        if (visited[i])
        {
            continue;
        }

        std::vector<Size> cluster;
        std::queue<Size> queue;
        visited[i] = true;
        queue.push(i);

        while (!queue.empty())
        {
            const Size node = queue.front();
            queue.pop();
            cluster.push_back(node);
            for (Size j = 0; j < n; ++j)
            {
                if (!visited[j] && usableMatrixDistance(distance_matrix, node, j) &&
                    distance_matrix[node][j] <= threshold)
                {
                    visited[j] = true;
                    queue.push(j);
                }
            }
        }
        clusters.push_back(std::move(cluster));
    }
    return clusters;
}

double DistanceStatistics::computeMean(const std::vector<std::vector<double>> &distance_matrix)
{
    long double sum = 0.0L;
    Size count = 0;
    for (Size i = 0; i < distance_matrix.size(); ++i)
    {
        for (Size j = i + 1; j < distance_matrix[i].size(); ++j)
        {
            if (usableDistance(distance_matrix[i][j]))
            {
                sum += static_cast<long double>(distance_matrix[i][j]);
                ++count;
            }
        }
    }
    return count == 0 ? 0.0 : finiteClamp(sum / static_cast<long double>(count));
}

double
DistanceStatistics::computeStdDeviation(const std::vector<std::vector<double>> &distance_matrix)
{
    const double mean = computeMean(distance_matrix);
    long double sq_sum = 0.0L;
    Size count = 0;
    for (Size i = 0; i < distance_matrix.size(); ++i)
    {
        for (Size j = i + 1; j < distance_matrix[i].size(); ++j)
        {
            if (usableDistance(distance_matrix[i][j]))
            {
                const long double diff = static_cast<long double>(distance_matrix[i][j]) -
                                         static_cast<long double>(mean);
                sq_sum += diff * diff;
                ++count;
            }
        }
    }
    return count == 0 ? 0.0 : finiteClamp(std::sqrt(sq_sum / static_cast<long double>(count)));
}

std::vector<double>
DistanceStatistics::computeRowMeans(const std::vector<std::vector<double>> &distance_matrix)
{
    std::vector<double> means;
    means.reserve(distance_matrix.size());
    for (const auto &row : distance_matrix)
    {
        means.push_back(meanOfFiniteDistances(row));
    }
    return means;
}

double DistanceStatistics::permutationTest(const std::vector<std::vector<double>> &distance_matrix1,
                                           const std::vector<std::vector<double>> &distance_matrix2,
                                           Size num_permutations)
{
    if (num_permutations == 0)
    {
        throw std::invalid_argument("num_permutations must be positive");
    }

    const double observed = std::abs(computeMean(distance_matrix1) - computeMean(distance_matrix2));
    std::vector<double> samples;
    samples.reserve(distance_matrix1.size() + distance_matrix2.size());

    for (const auto &row : distance_matrix1)
    {
        samples.push_back(meanOfFiniteDistances(row));
    }
    for (const auto &row : distance_matrix2)
    {
        samples.push_back(meanOfFiniteDistances(row));
    }

    std::mt19937_64 rng(0xC0FFEEULL);
    Size at_least_observed = 0;
    const Size split = distance_matrix1.size();

    for (Size iteration = 0; iteration < num_permutations; ++iteration)
    {
        std::shuffle(samples.begin(), samples.end(), rng);

        long double lhs = 0.0L;
        for (Size i = 0; i < split; ++i)
        {
            lhs += static_cast<long double>(samples[i]);
        }
        lhs /= static_cast<double>(split == 0 ? 1 : split);

        long double rhs = 0.0L;
        for (Size i = split; i < samples.size(); ++i)
        {
            rhs += static_cast<long double>(samples[i]);
        }
        const Size rhs_count = samples.size() - split;
        rhs /= static_cast<double>(rhs_count == 0 ? 1 : rhs_count);

        if (std::abs(static_cast<double>(lhs - rhs)) >= observed)
        {
            ++at_least_observed;
        }
    }

    return static_cast<double>(at_least_observed) / static_cast<double>(num_permutations);
}

double DistanceStatistics::silhouetteScore(const std::vector<std::vector<double>> &distance_matrix,
                                           const std::vector<Size> &cluster_labels)
{
    const Size n = distance_matrix.size();
    if (n == 0 || cluster_labels.size() != n)
    {
        return 0.0;
    }

    double total = 0.0;
    for (Size i = 0; i < n; ++i)
    {
        const Size own_label = cluster_labels[i];

        double intra_sum = 0.0;
        Size intra_count = 0;
        for (Size j = 0; j < n; ++j)
        {
            if (i != j && cluster_labels[j] == own_label &&
                usableMatrixDistance(distance_matrix, i, j))
            {
                intra_sum += distance_matrix[i][j];
                ++intra_count;
            }
        }
        const double a = intra_count == 0 ? 0.0 : intra_sum / static_cast<double>(intra_count);

        double b = std::numeric_limits<double>::infinity();
        for (Size label_idx = 0; label_idx < n; ++label_idx)
        {
            if (cluster_labels[label_idx] == own_label)
            {
                continue;
            }
            const Size other_label = cluster_labels[label_idx];
            double inter_sum = 0.0;
            Size inter_count = 0;
            for (Size j = 0; j < n; ++j)
            {
                if (cluster_labels[j] == other_label && usableMatrixDistance(distance_matrix, i, j))
                {
                    inter_sum += distance_matrix[i][j];
                    ++inter_count;
                }
            }
            if (inter_count > 0)
            {
                b = std::min(b, inter_sum / static_cast<double>(inter_count));
            }
        }

        if (std::isfinite(b))
        {
            const double denominator = std::max(a, b);
            if (denominator > 0.0)
            {
                const double score = (b - a) / denominator;
                if (std::isfinite(score))
                {
                    total += score;
                }
            }
        }
    }
    return total / static_cast<double>(n);
}

std::vector<std::vector<double>>
DistanceStatistics::multidimensionalScaling(const std::vector<std::vector<double>> &distance_matrix,
                                            Size target_dimension)
{
    const Size n = distance_matrix.size();
    if (n == 0 || target_dimension == 0)
    {
        return {};
    }

    std::vector<std::vector<double>> centered(n, std::vector<double>(n, 0.0));
    if (n > std::numeric_limits<Size>::max() / n)
    {
        throw std::length_error("distance matrix area overflows");
    }

    long double mean = 0.0L;
    Size count = 0;
    for (const auto &row : distance_matrix)
    {
        for (double value : row)
        {
            if (usableDistance(value))
            {
                const long double distance = static_cast<long double>(value);
                mean += distance * distance;
                ++count;
            }
        }
    }
    mean = count == 0 ? 0.0L : mean / static_cast<long double>(count);

    for (Size i = 0; i < n; ++i)
    {
        for (Size j = 0; j < n; ++j)
        {
            const double raw =
                j < distance_matrix[i].size() && usableDistance(distance_matrix[i][j])
                    ? distance_matrix[i][j]
                    : 0.0;
            const long double distance = static_cast<long double>(raw);
            centered[i][j] = finiteClamp(-0.5L * (distance * distance - mean));
        }
    }

    std::vector<std::vector<double>> embedding(n, std::vector<double>(target_dimension, 0.0));
    for (Size i = 0; i < n; ++i)
    {
        for (Size d = 0; d < std::min(target_dimension, n); ++d)
        {
            embedding[i][d] = centered[i][d];
        }
    }
    return embedding;
}

} // namespace nerve::metrics

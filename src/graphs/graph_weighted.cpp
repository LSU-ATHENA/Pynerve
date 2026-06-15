#include "nerve/graphs/graph.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <utility>

namespace nerve::graphs
{
namespace
{

constexpr double kJacobiTolerance = 1e-12;

std::vector<std::vector<double>> identityMatrix(Size n)
{
    std::vector<std::vector<double>> matrix(n, std::vector<double>(n, 0.0));
    for (Size i = 0; i < n; ++i)
    {
        matrix[i][i] = 1.0;
    }
    return matrix;
}

struct EigenDecomposition
{
    std::vector<double> values;
    std::vector<std::vector<double>> vectors;
};

EigenDecomposition decomposeSymmetric(std::vector<std::vector<double>> matrix)
{
    const Size n = matrix.size();
    EigenDecomposition result;
    if (n == 0)
    {
        return result;
    }

    auto eigenvectors_by_column = identityMatrix(n);
    const Size max_iterations = std::max<Size>(1, 64 * n * n);
    for (Size iter = 0; iter < max_iterations; ++iter)
    {
        Size p = 0;
        Size q = 1;
        double largest = 0.0;
        for (Size i = 0; i < n; ++i)
        {
            for (Size j = i + 1; j < n; ++j)
            {
                const double magnitude = std::abs(matrix[i][j]);
                if (magnitude > largest)
                {
                    largest = magnitude;
                    p = i;
                    q = j;
                }
            }
        }
        if (largest < kJacobiTolerance || n == 1)
        {
            break;
        }

        const double app = matrix[p][p];
        const double aqq = matrix[q][q];
        const double apq = matrix[p][q];
        const double angle = 0.5 * std::atan2(2.0 * apq, aqq - app);
        const double c = std::cos(angle);
        const double s = std::sin(angle);

        for (Size k = 0; k < n; ++k)
        {
            if (k == p || k == q)
            {
                continue;
            }
            const double mkp = matrix[k][p];
            const double mkq = matrix[k][q];
            matrix[k][p] = c * mkp - s * mkq;
            matrix[p][k] = matrix[k][p];
            matrix[k][q] = s * mkp + c * mkq;
            matrix[q][k] = matrix[k][q];
        }

        matrix[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        matrix[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        matrix[p][q] = 0.0;
        matrix[q][p] = 0.0;

        for (Size k = 0; k < n; ++k)
        {
            const double vkp = eigenvectors_by_column[k][p];
            const double vkq = eigenvectors_by_column[k][q];
            eigenvectors_by_column[k][p] = c * vkp - s * vkq;
            eigenvectors_by_column[k][q] = s * vkp + c * vkq;
        }
    }

    std::vector<std::pair<double, std::vector<double>>> pairs;
    pairs.reserve(n);
    for (Size i = 0; i < n; ++i)
    {
        std::vector<double> vector(n);
        for (Size row = 0; row < n; ++row)
        {
            vector[row] = eigenvectors_by_column[row][i];
        }
        pairs.emplace_back(matrix[i][i], std::move(vector));
    }
    std::ranges::sort(pairs, {}, &std::pair<double, std::vector<double>>::first);

    result.values.reserve(n);
    result.vectors.reserve(n);
    for (auto &[value, vector] : pairs)
    {
        result.values.push_back(value);
        result.vectors.push_back(std::move(vector));
    }
    return result;
}

std::vector<std::vector<double>>
spectralEmbedding(const std::vector<std::vector<double>> &laplacian, Size k)
{
    auto decomposition = decomposeSymmetric(laplacian);
    const Size n = laplacian.size();
    const Size width = std::min(k, decomposition.vectors.size());
    std::vector<std::vector<double>> points(n, std::vector<double>(width, 0.0));
    for (Size feature = 0; feature < width; ++feature)
    {
        for (Size vertex = 0; vertex < n; ++vertex)
        {
            points[vertex][feature] = decomposition.vectors[feature][vertex];
        }
    }
    return points;
}

double squaredDistance(const std::vector<double> &a, const std::vector<double> &b)
{
    const Size dim = std::min(a.size(), b.size());
    double total = 0.0;
    for (Size i = 0; i < dim; ++i)
    {
        const double diff = a[i] - b[i];
        total += diff * diff;
    }
    return total;
}

void validatePointRows(const std::vector<std::vector<double>> &points, Size expected_rows)
{
    if (points.size() != expected_rows)
    {
        throw std::invalid_argument("Points size mismatch");
    }
    if (points.empty())
    {
        return;
    }
    const Size dim = points.front().size();
    if (dim == 0)
    {
        throw std::invalid_argument("Point coordinates must be non-empty");
    }
    for (const auto &point : points)
    {
        if (point.size() != dim)
        {
            throw std::invalid_argument("Point dimensions must be consistent");
        }
        if (!std::ranges::all_of(point, [](double value) { return std::isfinite(value); }))
        {
            throw std::invalid_argument("Point coordinates must be finite");
        }
    }
}

std::vector<int> kMeansLabels(const std::vector<std::vector<double>> &points, Size k)
{
    const Size n = points.size();
    if (n == 0 || k == 0)
    {
        return {};
    }
    const Size clusters = std::min(k, n);
    std::vector<std::vector<double>> centers;
    centers.reserve(clusters);
    for (Size c = 0; c < clusters; ++c)
    {
        centers.push_back(points[(c * n) / clusters]);
    }

    std::vector<int> labels(n, 0);
    for (Size iter = 0; iter < 32; ++iter)
    {
        bool changed = false;
        for (Size i = 0; i < n; ++i)
        {
            double best_distance = std::numeric_limits<double>::infinity();
            int best_label = 0;
            for (Size c = 0; c < clusters; ++c)
            {
                const double distance = squaredDistance(points[i], centers[c]);
                if (distance < best_distance)
                {
                    best_distance = distance;
                    best_label = static_cast<int>(c);
                }
            }
            if (labels[i] != best_label)
            {
                labels[i] = best_label;
                changed = true;
            }
        }
        if (!changed && iter > 0)
        {
            break;
        }

        std::vector<std::vector<double>> next_centers(
            clusters, std::vector<double>(points.front().size(), 0.0));
        std::vector<Size> counts(clusters, 0);
        for (Size i = 0; i < n; ++i)
        {
            const Size label = static_cast<Size>(labels[i]);
            ++counts[label];
            for (Size d = 0; d < points[i].size(); ++d)
            {
                next_centers[label][d] += points[i][d];
            }
        }
        for (Size c = 0; c < clusters; ++c)
        {
            if (counts[c] == 0)
            {
                next_centers[c] = points[(c * n) / clusters];
                continue;
            }
            for (double &value : next_centers[c])
            {
                value /= static_cast<double>(counts[c]);
            }
        }
        centers = std::move(next_centers);
    }
    return labels;
}

} // namespace

WeightedGraph::WeightedGraph(Size numVertices)
    : Graph(numVertices)
    , weighted_laplacian_valid_(false)
{}

void WeightedGraph::addVertex()
{
    Graph::addVertex();
    invalidateWeightedLaplacian();
}

void WeightedGraph::addVertices(Size count)
{
    Graph::addVertices(count);
    if (count > 0)
    {
        invalidateWeightedLaplacian();
    }
}

void WeightedGraph::removeVertex(Index vertex)
{
    Graph::removeVertex(vertex);
    invalidateWeightedLaplacian();
}

void WeightedGraph::addEdge(Index u, Index v, double weight)
{
    Graph::addEdge(u, v, weight);
    invalidateWeightedLaplacian();
}

void WeightedGraph::removeEdge(Index u, Index v)
{
    Graph::removeEdge(u, v);
    invalidateWeightedLaplacian();
}

void WeightedGraph::setEdgeWeight(Index u, Index v, double weight)
{
    Graph::setEdgeWeight(u, v, weight);
    invalidateWeightedLaplacian();
}

void WeightedGraph::setAllWeights(double weight)
{
    for (const auto &e : getEdges())
    {
        Graph::setEdgeWeight(e.first, e.second, weight);
    }
    invalidateWeightedLaplacian();
}

void WeightedGraph::normalizeWeights()
{
    double max_w = 0.0;
    for (const auto &e : getEdges())
    {
        max_w = std::max(max_w, getEdgeWeight(e.first, e.second));
    }
    if (max_w > 0.0)
    {
        for (const auto &e : getEdges())
        {
            Graph::setEdgeWeight(e.first, e.second, getEdgeWeight(e.first, e.second) / max_w);
        }
    }
    invalidateWeightedLaplacian();
}

void WeightedGraph::applyDistanceWeights(const std::vector<std::vector<double>> &points)
{
    validatePointRows(points, numVertices());
    for (const auto &e : getEdges())
    {
        const auto &a = points[e.first];
        const auto &b = points[e.second];
        double d2 = 0.0;
        for (Size i = 0; i < a.size(); ++i)
        {
            const double diff = a[i] - b[i];
            d2 += diff * diff;
        }
        Graph::setEdgeWeight(e.first, e.second, std::sqrt(d2));
    }
    invalidateWeightedLaplacian();
}

std::vector<std::vector<double>> WeightedGraph::getWeightedLaplacian() const
{
    if (!weighted_laplacian_valid_)
    {
        const_cast<WeightedGraph *>(this)->computeWeightedLaplacian();
    }
    return weighted_laplacian_;
}

std::vector<double> WeightedGraph::computeEigenvalues(Size k) const
{
    auto decomposition = decomposeSymmetric(getWeightedLaplacian());
    if (k > 0 && k < decomposition.values.size())
    {
        decomposition.values.resize(k);
    }
    return decomposition.values;
}

std::vector<std::vector<double>> WeightedGraph::computeEigenvectors(Size k) const
{
    auto decomposition = decomposeSymmetric(getWeightedLaplacian());
    if (k > 0 && k < decomposition.vectors.size())
    {
        decomposition.vectors.resize(k);
    }
    return decomposition.vectors;
}

std::vector<int> WeightedGraph::spectralCluster(Size k) const
{
    return kMeansLabels(spectralEmbedding(getWeightedLaplacian(), k), k);
}

std::vector<int> WeightedGraph::normalizedSpectralCluster(Size k) const
{
    const Size n = numVertices();
    if (n == 0 || k == 0)
    {
        return {};
    }
    const auto adjacency = getAdjacencyMatrix();
    const auto degrees = getDegreeSequence();
    std::vector<std::vector<double>> normalized(n, std::vector<double>(n, 0.0));
    for (Size i = 0; i < n; ++i)
    {
        normalized[i][i] = degrees[i] > 0.0 ? 1.0 : 0.0;
        for (Size j = 0; j < n; ++j)
        {
            if (i == j || adjacency[i][j] <= 0.0 || degrees[i] <= 0.0 || degrees[j] <= 0.0)
            {
                continue;
            }
            normalized[i][j] = -adjacency[i][j] / std::sqrt(degrees[i] * degrees[j]);
        }
    }
    return kMeansLabels(spectralEmbedding(normalized, k), k);
}

void WeightedGraph::computeWeightedLaplacian()
{
    weighted_laplacian_ = getLaplacianMatrix();
    weighted_laplacian_valid_ = true;
}

void WeightedGraph::invalidateWeightedLaplacian()
{
    weighted_laplacian_valid_ = false;
}

} // namespace nerve::graphs

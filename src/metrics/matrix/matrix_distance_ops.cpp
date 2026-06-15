
#include "nerve/algebra/complex.hpp"
#include "nerve/metrics/distances.hpp"
#include "nerve/persistence/core/core_types.hpp"

#include <cmath>
#include <stdexcept>

namespace nerve::metrics
{
namespace
{

double computePointCloudDistance(const std::vector<std::vector<double>> &lhs,
                                 const std::vector<std::vector<double>> &rhs,
                                 const std::string &metric)
{
    double distance = 0.0;
    if (metric == "hausdorff")
    {
        distance = hausdorffDistance(lhs, rhs);
    }
    else if (metric == "chamfer")
    {
        distance = chamferDistance(lhs, rhs);
    }
    else if (metric == "earth_movers" || metric == "emd")
    {
        distance = earthMoversDistance(lhs, rhs);
    }
    else
    {
        throw std::invalid_argument("Unsupported point cloud distance metric");
    }
    if (!std::isfinite(distance))
    {
        throw std::overflow_error("point cloud distance is non-finite");
    }
    return distance;
}

} // namespace

double DistanceMetricFactory::computeDistance(MetricType type, const Diagram &diagram1,
                                              const Diagram &diagram2, double parameter)
{
    switch (type)
    {
        case MetricType::BOTTLENECK:
            return bottleneckDistance(diagram1, diagram2);
        case MetricType::WASSERSTEIN:
            return wassersteinDistance(diagram1, diagram2, parameter);
        case MetricType::FREDHET:
            return frechetDistance(diagram1, diagram2);
        default:
            throw std::invalid_argument("Unsupported metric type for persistence diagrams");
    }
}

double DistanceMetricFactory::computeDistance(MetricType type, const SimplicialComplex &complex1,
                                              const SimplicialComplex &complex2)
{
    switch (type)
    {
        case MetricType::GROMOV_HAUSDORFF:
            return gromovHausdorffDistance(complex1, complex2);
        case MetricType::EDIT:
            return editDistance(complex1, complex2);
        case MetricType::INTERLEAVING:
            return interleavingDistance(complex1, complex2);
        default:
            throw std::invalid_argument("Unsupported metric type for simplicial complexes");
    }
}

std::vector<std::vector<double>>
DistanceMatrix::computeDiagramDistanceMatrix(const std::vector<Diagram> &diagrams,
                                             DistanceMetricFactory::MetricType metric)
{
    const Size n = diagrams.size();
    std::vector<std::vector<double>> matrix(n, std::vector<double>(n, 0.0));
    for (Size i = 0; i < n; ++i)
    {
        for (Size j = i + 1; j < n; ++j)
        {
            const double distance =
                DistanceMetricFactory::computeDistance(metric, diagrams[i], diagrams[j]);
            matrix[i][j] = distance;
            matrix[j][i] = distance;
        }
    }
    return matrix;
}

std::vector<std::vector<double>>
DistanceMatrix::computeComplexDistanceMatrix(const std::vector<SimplicialComplex> &complexes,
                                             DistanceMetricFactory::MetricType metric)
{
    const Size n = complexes.size();
    std::vector<std::vector<double>> matrix(n, std::vector<double>(n, 0.0));
    for (Size i = 0; i < n; ++i)
    {
        for (Size j = i + 1; j < n; ++j)
        {
            const double distance =
                DistanceMetricFactory::computeDistance(metric, complexes[i], complexes[j]);
            matrix[i][j] = distance;
            matrix[j][i] = distance;
        }
    }
    return matrix;
}

std::vector<std::vector<double>> DistanceMatrix::computePointCloudDistanceMatrix(
    const std::vector<std::vector<std::vector<double>>> &point_clouds, const std::string &metric)
{
    const Size n = point_clouds.size();
    std::vector<std::vector<double>> matrix(n, std::vector<double>(n, 0.0));
    for (Size i = 0; i < n; ++i)
    {
        for (Size j = i + 1; j < n; ++j)
        {
            const double distance =
                computePointCloudDistance(point_clouds[i], point_clouds[j], metric);
            matrix[i][j] = distance;
            matrix[j][i] = distance;
        }
    }
    return matrix;
}

} // namespace nerve::metrics

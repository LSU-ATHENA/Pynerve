
#include "nerve/common/accelerated_types.hpp"
#include "nerve/persistence/kernels/detail/kernels_detail.hpp"
#include "nerve/persistence/kernels/ph4_ops.hpp"

#include <cmath>
#include <limits>
#include <vector>

namespace
{

bool check_ph6_high_dim_ops_basic()
{
    nerve::PointCloud cloud = {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};
    if (!nerve::persistence::hasValidPointCloud(cloud))
        return false;
    return true;
}

bool check_point_cloud_validation()
{
    nerve::PointCloud empty;
    if (!nerve::persistence::hasValidPointCloud(empty))
        return false;
    nerve::PointCloud bad = {{0.0, 0.0}, {1.0}};
    if (nerve::persistence::hasValidPointCloud(bad))
        return false;
    nerve::PointCloud nonfinite = {{0.0, 0.0}, {1.0, std::numeric_limits<double>::infinity()}};
    if (nerve::persistence::hasValidPointCloud(nonfinite))
        return false;
    return true;
}

bool check_euclidean_distance()
{
    std::vector<double> a = {0.0, 0.0};
    std::vector<double> b = {3.0, 4.0};
    double d = nerve::persistence::euclideanDistance(a, b);
    if (std::abs(d - 5.0) > 1e-12)
        return false;
    std::vector<double> c = {0.0};
    if (std::isfinite(nerve::persistence::euclideanDistance(a, c)))
        return false;
    return true;
}

bool check_ph6_construction()
{
    nerve::PersistenceBudget budget;
    budget.memory_limit_mb = 1024;
    budget.time_limit_ms = 10000;
    nerve::persistence::PH6HighDimensional ph6(budget);
    (void)ph6;
    return true;
}

bool check_overwrite_diagram_from_exact()
{
    nerve::persistence::ExactPersistenceResult exact;
    exact.pairs = {{0.0, 1.0, 0}, {0.5, std::numeric_limits<double>::infinity(), 1}};
    nerve::Diagram diagram;
    nerve::persistence::overwriteDiagramFromExact(exact, &diagram);
    if (diagram.size() != 2)
        return false;
    return true;
}

} // namespace

int main()
{
    if (!check_ph6_high_dim_ops_basic())
        return 1;
    if (!check_point_cloud_validation())
        return 1;
    if (!check_euclidean_distance())
        return 1;
    if (!check_ph6_construction())
        return 1;
    if (!check_overwrite_diagram_from_exact())
        return 1;
    return 0;
}

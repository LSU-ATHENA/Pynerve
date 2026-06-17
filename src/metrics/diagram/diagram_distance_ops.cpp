#include "nerve/metrics/distances.hpp"
#include "nerve/persistence/core/core_types.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#if defined(NERVE_HAS_CUDA)
#include "nerve/gpu/bottleneck_distance.cuh"
#include "nerve/gpu/kernel_launcher.hpp"
#include "nerve/gpu/wasserstein_distance.cuh"
#include "nerve/math/persistence_metrics/point2d.hpp"

#include <future>
#endif

namespace nerve::metrics
{

struct DiagramPoint
{
    double birth = 0.0;
    double death = 0.0;
};

bool isValidDiagram(const Diagram &diagram)
{
    for (const auto &pair : diagram.getPairs())
    {
        if (!std::isfinite(pair.birth))
        {
            return false;
        }
        const bool finite_death = std::isfinite(pair.death);
        if ((!finite_death && !pair.isInfinite()) || pair.dimension < 0 ||
            (finite_death && pair.death < pair.birth))
        {
            return false;
        }
    }
    return true;
}

inline bool hasSupportedMatchingSize(size_t n1, size_t n2)
{
    const size_t max_index = static_cast<size_t>(std::numeric_limits<int>::max());
    return n1 <= max_index && n2 <= max_index && n1 <= max_index - n2;
}

std::vector<DiagramPoint> diagramToPoints(const Diagram &diagram)
{
    std::vector<DiagramPoint> points;
    const auto &pairs = diagram.getPairs();
    points.reserve(pairs.size());
    for (const auto &pair : pairs)
    {
        if (pair.isInfinite())
        {
            points.push_back(DiagramPoint{pair.birth, pair.birth});
        }
        else
        {
            points.push_back(DiagramPoint{pair.birth, pair.death});
        }
    }
    return points;
}

double computeBottleneck(const std::vector<DiagramPoint> &points1,
                         const std::vector<DiagramPoint> &points2);
double computeWasserstein(const std::vector<DiagramPoint> &points1,
                          const std::vector<DiagramPoint> &points2, double p);

double bottleneckDistance(const Diagram &diagram1, const Diagram &diagram2)
{
    if (!isValidDiagram(diagram1) || !isValidDiagram(diagram2))
    {
        return std::numeric_limits<double>::infinity();
    }
#ifdef NERVE_HAS_CUDA
    const auto pts1 = diagramToPoints(diagram1);
    const auto pts2 = diagramToPoints(diagram2);
    if (hasSupportedMatchingSize(pts1.size(), pts2.size()))
    {
        std::vector<nerve::math::Point2D> p1, p2;
        p1.reserve(pts1.size());
        p2.reserve(pts2.size());
        for (auto &x : pts1)
            p1.emplace_back(x.birth, x.death);
        for (auto &x : pts2)
            p2.emplace_back(x.birth, x.death);
        std::promise<double> promise;
        auto future = promise.get_future();
        nerve::gpu::bottleneck::compute_bottleneck_distance_gpu(
            p1, p2, [&promise](double r) { promise.set_value(r); });
        return future.get();
    }
    return computeBottleneck(pts1, pts2);
#else
    return computeBottleneck(diagramToPoints(diagram1), diagramToPoints(diagram2));
#endif
}

double wassersteinDistance(const Diagram &diagram1, const Diagram &diagram2, double p)
{
    if (!std::isfinite(p) || p < 1.0 || !isValidDiagram(diagram1) || !isValidDiagram(diagram2))
    {
        return std::numeric_limits<double>::infinity();
    }
#ifdef NERVE_HAS_CUDA
    const auto pts1 = diagramToPoints(diagram1);
    const auto pts2 = diagramToPoints(diagram2);
    if (p == 2.0 && hasSupportedMatchingSize(pts1.size(), pts2.size()))
    {
        std::vector<nerve::math::Point2D> p1, p2;
        p1.reserve(pts1.size());
        p2.reserve(pts2.size());
        for (auto &x : pts1)
            p1.emplace_back(x.birth, x.death);
        for (auto &x : pts2)
            p2.emplace_back(x.birth, x.death);
        std::promise<double> promise;
        auto future = promise.get_future();
        nerve::gpu::wasserstein::compute_sinkhorn_distance_gpu(
            p1, p2, p, 0.1, 100, [&promise](double r) { promise.set_value(r); });
        return future.get();
    }
    return computeWasserstein(pts1, pts2, p);
#else
    return computeWasserstein(diagramToPoints(diagram1), diagramToPoints(diagram2), p);
#endif
}

double frechetDistance(const Diagram &diagram1, const Diagram &diagram2)
{
    return bottleneckDistance(diagram1, diagram2);
}

double gromovHausdorffDistance(const Diagram &diagram1, const Diagram &diagram2)
{
    return bottleneckDistance(diagram1, diagram2);
}

} // namespace nerve::metrics

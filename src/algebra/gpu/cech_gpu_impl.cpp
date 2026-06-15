
#ifdef __CUDACC__

#include "nerve/algebra/complex.hpp"
#include "nerve/errors/errors.hpp"
#include "nerve/gpu/compute_manager.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <ranges>
#include <string_view>
#include <vector>

namespace nerve
{
namespace gpu
{
namespace algebra
{
namespace detail
{
extern void launchCechBallIntersection(const double *d_points, int *d_intersection_graph,
                                       int n_points, int point_dim, double radius,
                                       cudaStream_t stream);

extern void launchCechSimplexValidation(const double *d_points, const int *d_simplex_vertices,
                                        const int *d_simplex_sizes, int *d_is_valid,
                                        double *d_filtration_values, int n_simplices, int max_dim,
                                        int point_dim, double radius, cudaStream_t stream);

extern void launchAlphaFiltration(const double *d_points, const int *d_simplex_vertices,
                                  const int *d_simplex_sizes, double *d_alpha_values,
                                  int n_simplices, int max_dim, int point_dim, cudaStream_t stream);

extern void launchMinimalEnclosingBall(const double *d_points, const int *d_simplex_vertices,
                                       const int *d_simplex_sizes, double *d_meb_radii,
                                       int n_simplices, int max_dim, int point_dim,
                                       cudaStream_t stream);

namespace
{

bool checkedProduct(std::size_t lhs, std::size_t rhs, std::size_t &out) noexcept
{
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs)
    {
        return false;
    }
    out = lhs * rhs;
    return true;
}

bool checkedByteCount(std::size_t count, std::size_t element_size, std::size_t &out) noexcept
{
    return checkedProduct(count, element_size, out);
}

bool checkedIntSize(std::size_t value, int &out) noexcept
{
    if (value > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        return false;
    }
    out = static_cast<int>(value);
    return true;
}

errors::ErrorResult<void> resourceLimit(std::string_view message)
{
    return errors::ErrorResult<void>::error(errors::ErrorCode::E41_RESOURCE_LIMIT, message);
}

errors::ErrorResult<void> invalidSimplices(std::string_view message)
{
    return errors::ErrorResult<void>::error(errors::ErrorCode::E88_INVALID_SIMPLICES, message);
}

double checkedPointDistance(const std::vector<std::vector<double>> &points, int lhs, int rhs)
{
    double distance_sq = 0.0;
    for (std::size_t d = 0; d < points[static_cast<std::size_t>(lhs)].size(); ++d)
    {
        const double diff =
            points[static_cast<std::size_t>(lhs)][d] - points[static_cast<std::size_t>(rhs)][d];
        const double contribution = diff * diff;
        const double next_distance_sq = distance_sq + contribution;
        if (!std::isfinite(diff) || !std::isfinite(contribution) ||
            !std::isfinite(next_distance_sq))
        {
            return std::numeric_limits<double>::infinity();
        }
        distance_sq = next_distance_sq;
    }
    return std::sqrt(distance_sq);
}

} // namespace

class GPUCechComplexEngine
{
public:
    struct CechSimplex
    {
        std::vector<int> vertices;
        double filtration_value;
        double alpha_value;
        int dimension;
        bool isValid;
    };

    static errors::ErrorResult<void>
    constructCechComplex(const std::vector<std::vector<double>> &points, double max_radius,
                         int max_dimension, std::vector<CechSimplex> &out_simplices)
    {
        int n_points = 0;
        if (!checkedIntSize(points.size(), n_points))
        {
            return resourceLimit("Cech GPU point count exceeds int range");
        }
        if (n_points == 0)
        {
            return errors::ErrorResult<void>::success();
        }
        if (!std::isfinite(max_radius) || max_radius < 0.0 || max_dimension < 0)
        {
            return invalidSimplices(
                "Cech GPU radius and dimension must be finite and non-negative");
        }
        const double diameter_limit = max_radius * 2.0;
        if (!std::isfinite(diameter_limit))
        {
            return resourceLimit("Cech GPU radius diameter exceeds supported range");
        }

        int point_dim = 0;
        if (!checkedIntSize(points[0].size(), point_dim))
        {
            return resourceLimit("Cech GPU point dimension exceeds int range");
        }
        if (point_dim == 0)
        {
            return errors::ErrorResult<void>::success();
        }
        for (const auto &pt : points)
        {
            if (pt.size() != static_cast<std::size_t>(point_dim))
            {
                return invalidSimplices("Cech GPU points must have a consistent dimension");
            }
        }
        std::size_t point_values = 0;
        std::size_t point_bytes = 0;
        if (!checkedProduct(static_cast<std::size_t>(n_points), static_cast<std::size_t>(point_dim),
                            point_values) ||
            !checkedByteCount(point_values, sizeof(double), point_bytes))
        {
            return resourceLimit("Cech GPU point allocation exceeds host limits");
        }
        std::vector<double> flat_points;
        flat_points.reserve(point_values);
        for (const auto &pt : points)
        {
            for (double coord : pt)
            {
                if (!std::isfinite(coord))
                {
                    return invalidSimplices("Cech GPU point coordinates must be finite");
                }
                flat_points.push_back(coord);
            }
        }
        double *d_points = nullptr;
        int *d_intersection_graph = nullptr;

        cudaError_t err;

        err = cudaMalloc(&d_points, point_bytes);
        if (err != cudaSuccess)
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E10_GPU_OOM);
        }

        err = cudaMemcpy(d_points, flat_points.data(), point_bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            cudaFree(d_points);
            return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }

        size_t graph_size = 0;
        size_t graph_bytes = 0;
        if (!checkedProduct(static_cast<std::size_t>(n_points), static_cast<std::size_t>(n_points),
                            graph_size) ||
            !checkedByteCount(graph_size, sizeof(int), graph_bytes))
        {
            cudaFree(d_points);
            return resourceLimit("Cech GPU intersection graph allocation exceeds host limits");
        }
        err = cudaMalloc(&d_intersection_graph, graph_bytes);
        if (err != cudaSuccess)
        {
            cudaFree(d_points);
            return errors::ErrorResult<void>::error(errors::ErrorCode::E10_GPU_OOM);
        }
        err = cudaMemset(d_intersection_graph, 0, graph_bytes);
        if (err != cudaSuccess)
        {
            cudaFree(d_points);
            cudaFree(d_intersection_graph);
            return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }
        launchCechBallIntersection(d_points, d_intersection_graph, n_points, point_dim, max_radius,
                                   nullptr);

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            cudaFree(d_points);
            cudaFree(d_intersection_graph);
            return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
        {
            cudaFree(d_points);
            cudaFree(d_intersection_graph);
            return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }
        std::vector<int> intersectionGraph(graph_size);
        err = cudaMemcpy(intersectionGraph.data(), d_intersection_graph, graph_bytes,
                         cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            cudaFree(d_points);
            cudaFree(d_intersection_graph);
            return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }
        cudaFree(d_intersection_graph);
        out_simplices.clear();
        for (int i = 0; i < n_points; ++i)
        {
            CechSimplex simplex;
            simplex.vertices = {i};
            simplex.filtration_value = 0.0;
            simplex.alpha_value = 0.0;
            simplex.dimension = 0;
            simplex.isValid = true;
            out_simplices.push_back(simplex);
        }
        std::vector<CechSimplex> edges;
        for (int i = 0; i < n_points; ++i)
        {
            for (int j = i + 1; j < n_points; ++j)
            {
                const std::size_t graph_index =
                    static_cast<std::size_t>(i) * static_cast<std::size_t>(n_points) +
                    static_cast<std::size_t>(j);
                if (intersectionGraph[graph_index] != 0)
                {
                    CechSimplex simplex;
                    simplex.vertices = {i, j};
                    const double dist = checkedPointDistance(points, i, j);
                    simplex.filtration_value = dist / 2.0;
                    simplex.alpha_value = dist / 2.0;
                    simplex.dimension = 1;
                    simplex.isValid = std::isfinite(simplex.filtration_value) &&
                                      simplex.filtration_value <= max_radius;

                    if (simplex.isValid)
                    {
                        edges.push_back(simplex);
                    }
                }
            }
        }
        out_simplices.insert(out_simplices.end(), edges.begin(), edges.end());
        if (max_dimension >= 2)
        {
            expandCechCliquesCpu(points, edges, intersectionGraph, n_points, max_dimension,
                                 max_radius, out_simplices);
        }
        cudaFree(d_points);
        std::ranges::sort(out_simplices, {}, [](const CechSimplex &s) {
            return std::pair(s.dimension, s.filtration_value);
        });

        return errors::ErrorResult<void>::success();
    }

    static errors::ErrorResult<void>
    constructAlphaComplex(const std::vector<std::vector<double>> &points, double max_alpha,
                          int max_dimension, std::vector<CechSimplex> &out_simplices)
    {
        auto result = constructCechComplex(points, max_alpha, max_dimension, out_simplices);

        if (result.isSuccess())
        {
        }

        return result;
    }

private:
    static void expandCechCliquesCpu(const std::vector<std::vector<double>> &points,
                                     const std::vector<CechSimplex> &edges,
                                     const std::vector<int> &intersectionGraph, int n_points,
                                     int max_dimension, double max_radius,
                                     std::vector<CechSimplex> &out_simplices)
    {
        std::vector<std::vector<int>> adjacencyList(n_points);
        for (const auto &edge : edges)
        {
            int v1 = edge.vertices[0];
            int v2 = edge.vertices[1];
            adjacencyList[v1].push_back(v2);
            adjacencyList[v2].push_back(v1);
        }
        for (auto &list : adjacencyList)
        {
            std::ranges::sort(list);
        }
        std::vector<CechSimplex> current_simplices = edges;

        for (int dim = 2; dim <= max_dimension && !current_simplices.empty(); ++dim)
        {
            std::vector<CechSimplex> new_simplices;

            for (const auto &simplex : current_simplices)
            {
                if (simplex.dimension != dim - 1)
                    continue;
                std::vector<int> common_neighbors = adjacencyList[simplex.vertices[0]];
                for (size_t i = 1; i < simplex.vertices.size(); ++i)
                {
                    std::vector<int> temp;
                    const auto &neighbors = adjacencyList[simplex.vertices[i]];
                    std::set_intersection(common_neighbors.begin(), common_neighbors.end(),
                                          neighbors.begin(), neighbors.end(),
                                          std::back_inserter(temp));
                    common_neighbors = std::move(temp);
                    if (common_neighbors.empty())
                        break;
                }
                for (int new_vertex : common_neighbors)
                {
                    if (new_vertex <= simplex.vertices.back())
                        continue;

                    CechSimplex new_simplex;
                    new_simplex.vertices = simplex.vertices;
                    new_simplex.vertices.push_back(new_vertex);
                    new_simplex.dimension = dim;

                    int n_vertices = new_simplex.vertices.size();
                    double meb_radius = 0.0;

                    if (n_vertices == 2)
                    {
                        meb_radius = checkedPointDistance(points, new_simplex.vertices[0],
                                                          new_simplex.vertices[1]) *
                                     0.5;
                    }
                    else if (n_vertices == 3)
                    {
                        int v0 = new_simplex.vertices[0];
                        int v1 = new_simplex.vertices[1];
                        int v2 = new_simplex.vertices[2];
                        const double a = checkedPointDistance(points, v1, v2);
                        const double b = checkedPointDistance(points, v0, v2);
                        const double c = checkedPointDistance(points, v0, v1);
                        double s = (a + b + c) * 0.5;
                        double area = std::sqrt(std::max(0.0, s * (s - a) * (s - b) * (s - c)));

                        if (area > 1e-10)
                        {
                            const double circumR = (a * b * c) / (4.0 * area);
                            meb_radius = std::isfinite(circumR)
                                             ? circumR
                                             : std::numeric_limits<double>::infinity();
                        }
                        else
                        {
                            meb_radius = std::max(a, std::max(b, c)) * 0.5;
                        }
                    }
                    else
                    {
                        std::vector<double> dists;
                        for (size_t i = 0; i < new_simplex.vertices.size(); ++i)
                        {
                            for (size_t j = i + 1; j < new_simplex.vertices.size(); ++j)
                            {
                                dists.push_back(checkedPointDistance(
                                    points, new_simplex.vertices[i], new_simplex.vertices[j]));
                            }
                        }
                        double max_dist = 0.0;
                        for (double d : dists)
                        {
                            max_dist = std::max(max_dist, d);
                        }
                        meb_radius = max_dist * 0.5;
                        if (n_vertices > 3)
                        {
                            for (size_t i = 0; i < new_simplex.vertices.size(); ++i)
                            {
                                for (size_t j = i + 1; j < new_simplex.vertices.size(); ++j)
                                {
                                    for (size_t k = j + 1; k < new_simplex.vertices.size(); ++k)
                                    {
                                        int vi = new_simplex.vertices[i];
                                        int vj = new_simplex.vertices[j];
                                        int vk = new_simplex.vertices[k];

                                        const double a = checkedPointDistance(points, vj, vk);
                                        const double b = checkedPointDistance(points, vi, vk);
                                        const double c = checkedPointDistance(points, vi, vj);

                                        double s = (a + b + c) * 0.5;
                                        double area = std::sqrt(
                                            std::max(0.0, s * (s - a) * (s - b) * (s - c)));

                                        if (area > 1e-10)
                                        {
                                            const double circumR = (a * b * c) / (4.0 * area);
                                            meb_radius =
                                                std::isfinite(circumR)
                                                    ? std::max(meb_radius, circumR)
                                                    : std::numeric_limits<double>::infinity();
                                        }
                                    }
                                }
                            }
                        }
                    }

                    new_simplex.filtration_value = meb_radius;
                    new_simplex.alpha_value = meb_radius;
                    new_simplex.isValid = std::isfinite(new_simplex.filtration_value) &&
                                          new_simplex.filtration_value <= max_radius;

                    if (new_simplex.isValid)
                    {
                        new_simplices.push_back(new_simplex);
                    }
                }
            }

            out_simplices.insert(out_simplices.end(), new_simplices.begin(), new_simplices.end());
            current_simplices = std::move(new_simplices);
        }
    }
};

} // namespace detail
} // namespace algebra
} // namespace gpu
} // namespace nerve

#endif // __CUDACC__

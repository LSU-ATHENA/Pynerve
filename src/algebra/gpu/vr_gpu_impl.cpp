#ifdef __CUDACC__

#include "detail/vr_gpu_cuda_launchers.inl"
#include "detail/vr_gpu_helpers.inl"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <vector>

namespace nerve::gpu::algebra::detail
{

void expandCliquesCpu(const std::vector<VRSimplex> &edges, const std::vector<int> &adjacencyMatrix,
                      int n_points, int max_dimension, double max_radius,
                      const std::vector<std::vector<double>> &points,
                      std::vector<VRSimplex> &out_simplices);

static void expandCliquesGpu(const std::vector<VRSimplex> &edges,
                             const std::vector<int> &adjacencyMatrix, int n_points,
                             int max_dimension, double max_radius,
                             const std::vector<std::vector<double>> &points,
                             std::vector<VRSimplex> &out_simplices);

class GPUVRComplexEngine
{
public:
    static errors::ErrorResult<void>
    constructVrComplex(const std::vector<std::vector<double>> &points, double max_radius,
                       int max_dimension, std::vector<VRSimplex> &out_simplices)
    {
        int n_points = 0;
        if (!checkedIntSize(points.size(), n_points))
        {
            return resourceLimit("VR GPU point count exceeds int range");
        }
        if (n_points == 0)
        {
            return errors::ErrorResult<void>::success();
        }
        if (!std::isfinite(max_radius) || max_radius < 0.0 || max_dimension < 0)
        {
            return invalidSimplices("VR GPU radius and dimension must be finite and non-negative");
        }
        const double max_radius_sq = max_radius * max_radius;
        if (!std::isfinite(max_radius_sq))
        {
            return resourceLimit("VR GPU radius square exceeds supported range");
        }

        int point_dim = 0;
        if (!checkedIntSize(points[0].size(), point_dim))
        {
            return resourceLimit("VR GPU point dimension exceeds int range");
        }
        if (point_dim == 0)
        {
            return errors::ErrorResult<void>::success();
        }
        for (const auto &pt : points)
        {
            if (pt.size() != static_cast<std::size_t>(point_dim))
            {
                return invalidSimplices("VR GPU points must have a consistent dimension");
            }
        }

        std::size_t point_values = 0;
        std::size_t point_bytes = 0;
        if (!checkedProduct(static_cast<std::size_t>(n_points), static_cast<std::size_t>(point_dim),
                            point_values) ||
            !checkedByteCount(point_values, sizeof(double), point_bytes))
        {
            return resourceLimit("VR GPU point allocation exceeds host limits");
        }

        std::vector<double> flat_points;
        flat_points.reserve(point_values);
        for (const auto &pt : points)
        {
            for (double coord : pt)
            {
                if (!std::isfinite(coord))
                {
                    return invalidSimplices("VR GPU point coordinates must be finite");
                }
                flat_points.push_back(coord);
            }
        }

        double *d_points = nullptr;
        int *d_adjacency = nullptr;

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

        size_t adjacency_size = 0;
        size_t adjacency_bytes = 0;
        if (!checkedProduct(static_cast<std::size_t>(n_points), static_cast<std::size_t>(n_points),
                            adjacency_size) ||
            !checkedByteCount(adjacency_size, sizeof(int), adjacency_bytes))
        {
            cudaFree(d_points);
            return resourceLimit("VR GPU adjacency allocation exceeds host limits");
        }
        err = cudaMalloc(&d_adjacency, adjacency_bytes);
        if (err != cudaSuccess)
        {
            cudaFree(d_points);
            return errors::ErrorResult<void>::error(errors::ErrorCode::E10_GPU_OOM);
        }

        err = cudaMemset(d_adjacency, 0, adjacency_bytes);
        if (err != cudaSuccess)
        {
            cudaFree(d_points);
            cudaFree(d_adjacency);
            return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }

        int *d_edge_count = nullptr;
        err = cudaMalloc(&d_edge_count, sizeof(int));
        if (err == cudaSuccess)
        {
            cudaMemset(d_edge_count, 0, sizeof(int));

            launchOptimizedEdgeDetection(d_points, d_adjacency, d_edge_count, n_points, point_dim,
                                         max_radius, nullptr);

            err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                launchVrEdgeDetection(d_points, d_adjacency, n_points, point_dim, max_radius_sq,
                                      nullptr);
            }

            cudaFree(d_edge_count);
        }
        else
        {
            launchVrEdgeDetection(d_points, d_adjacency, n_points, point_dim, max_radius_sq,
                                  nullptr);
        }

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            cudaFree(d_points);
            cudaFree(d_adjacency);
            return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
        {
            cudaFree(d_points);
            cudaFree(d_adjacency);
            return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }

        std::vector<int> adjacencyMatrix(adjacency_size);
        err = cudaMemcpy(adjacencyMatrix.data(), d_adjacency, adjacency_bytes,
                         cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            cudaFree(d_points);
            cudaFree(d_adjacency);
            return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }

        out_simplices.clear();

        for (int i = 0; i < n_points; ++i)
        {
            VRSimplex simplex;
            simplex.vertices = {i};
            simplex.filtration_value = 0.0;
            simplex.dimension = 0;
            out_simplices.push_back(simplex);
        }

        std::vector<VRSimplex> edges;
        for (int i = 0; i < n_points; ++i)
        {
            for (int j = i + 1; j < n_points; ++j)
            {
                const std::size_t adjacency_index =
                    static_cast<std::size_t>(i) * static_cast<std::size_t>(n_points) +
                    static_cast<std::size_t>(j);
                if (adjacencyMatrix[adjacency_index] != 0)
                {
                    VRSimplex simplex;
                    simplex.vertices = {i, j};

                    simplex.filtration_value = checkedPointDistance(points, i, j);
                    simplex.dimension = 1;
                    edges.push_back(simplex);
                }
            }
        }
        out_simplices.insert(out_simplices.end(), edges.begin(), edges.end());

        if (max_dimension >= 2)
        {
            if (edges.size() > GPU_THRESHOLD && n_points <= 4096)
            {
                try
                {
                    expandCliquesGpu(edges, adjacencyMatrix, n_points, max_dimension, max_radius,
                                     points, out_simplices);
                }
                catch (const std::exception &)
                {
                    cudaFree(d_points);
                    cudaFree(d_adjacency);
                    return resourceLimit("VR GPU clique expansion exceeded host limits");
                }
            }
            else
            {
                expandCliquesCpu(edges, adjacencyMatrix, n_points, max_dimension, max_radius,
                                 points, out_simplices);
            }
        }

        cudaFree(d_points);
        cudaFree(d_adjacency);

        std::ranges::sort(out_simplices, {}, [](const VRSimplex &s) {
            return std::pair(s.dimension, s.filtration_value);
        });

        return errors::ErrorResult<void>::success();
    }
};

} // namespace nerve::gpu::algebra::detail

#endif // __CUDACC__

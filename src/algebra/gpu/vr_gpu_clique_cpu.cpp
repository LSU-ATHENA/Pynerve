#ifdef __CUDACC__

#include "detail/vr_gpu_helpers.inl"

#include <algorithm>
#include <iterator>
#include <ranges>
#include <vector>

namespace nerve::gpu::algebra::detail
{

void expandCliquesCpu(const std::vector<VRSimplex> &edges, const std::vector<int> &adjacencyMatrix,
                      int n_points, int max_dimension, double max_radius,
                      const std::vector<std::vector<double>> &points,
                      std::vector<VRSimplex> &out_simplices)
{
    std::vector<std::vector<int>> adjacencyList(n_points);
    for (const auto &edge : edges)
    {
        int v1 = edge.vertices[0];
        int v2 = edge.vertices[1];
        adjacencyList[v1].push_back(v2);
        adjacencyList[v2].push_back(v1);
    }
    for (auto &neighbors : adjacencyList)
    {
        std::ranges::sort(neighbors);
    }

    std::vector<VRSimplex> current_simplices = edges;

    for (int dim = 2; dim <= max_dimension && !current_simplices.empty(); ++dim)
    {
        std::vector<VRSimplex> new_simplices;

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
                                      neighbors.begin(), neighbors.end(), std::back_inserter(temp));
                common_neighbors = std::move(temp);
                if (common_neighbors.empty())
                    break;
            }

            for (int new_vertex : common_neighbors)
            {
                if (new_vertex <= simplex.vertices.back())
                    continue;

                VRSimplex new_simplex;
                new_simplex.vertices = simplex.vertices;
                new_simplex.vertices.push_back(new_vertex);
                new_simplex.dimension = dim;

                new_simplex.filtration_value = simplex.filtration_value;
                for (int v : simplex.vertices)
                {
                    double dist = checkedPointDistance(points, v, new_vertex);
                    if (dist > new_simplex.filtration_value)
                    {
                        new_simplex.filtration_value = dist;
                    }
                }

                if (new_simplex.filtration_value <= max_radius)
                {
                    new_simplices.push_back(new_simplex);
                }
            }
        }

        out_simplices.insert(out_simplices.end(), new_simplices.begin(), new_simplices.end());
        current_simplices = std::move(new_simplices);
    }
}

} // namespace nerve::gpu::algebra::detail

#endif // __CUDACC__

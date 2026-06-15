#include <cuda_runtime.h>

namespace nerve::gpu::detail
{

void launchDistanceMatrixAsync(const double *d_points, double *d_distance_matrix, int n_points,
                               int point_dim, double max_radius, cudaStream_t stream);

void launchEdgeDetectionAsync(const double *d_points, int *d_adjacency, int *d_edge_count,
                              int n_points, int point_dim, double max_radius, cudaStream_t stream);

void launchSimplexFiltrationAsync(const double *d_distance_matrix, const int *d_simplex_vertices,
                                  const int *d_simplex_sizes, double *d_filtration_values,
                                  int n_simplices, int max_dim, int n_points, cudaStream_t stream);

} // namespace nerve::gpu::detail

extern "C"
{
    void launchOptimizedDistanceMatrix(const double *d_points, double *d_distance_matrix,
                                       int n_points, int point_dim, double max_radius,
                                       cudaStream_t stream)
    {
        nerve::gpu::detail::launchDistanceMatrixAsync(d_points, d_distance_matrix, n_points,
                                                      point_dim, max_radius, stream);
    }

    void launchOptimizedEdgeDetection(const double *d_points, int *d_adjacency, int *d_edge_count,
                                      int n_points, int point_dim, double max_radius,
                                      cudaStream_t stream)
    {
        nerve::gpu::detail::launchEdgeDetectionAsync(d_points, d_adjacency, d_edge_count, n_points,
                                                     point_dim, max_radius, stream);
    }

    void launchOptimizedSimplexFiltration(const double *d_distance_matrix,
                                          const int *d_simplex_vertices, const int *d_simplex_sizes,
                                          double *d_filtration_values, int n_simplices, int max_dim,
                                          int n_points, cudaStream_t stream)
    {
        nerve::gpu::detail::launchSimplexFiltrationAsync(d_distance_matrix, d_simplex_vertices,
                                                         d_simplex_sizes, d_filtration_values,
                                                         n_simplices, max_dim, n_points, stream);
    }
}

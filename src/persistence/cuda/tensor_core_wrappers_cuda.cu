#include "nerve/persistence/cuda/cuda_tensor_core.hpp"

extern "C"
{
    void launchTensorCoreDistanceMatrix(const double *d_points, float *d_distance_matrix,
                                        int n_points, int point_dim, double max_radius,
                                        cudaStream_t stream)
    {
        nerve::gpu::tensorcore::launchTensorCoreDistanceMatrix(
            d_points, d_distance_matrix, n_points, point_dim, max_radius, stream);
    }

    int areTensorCoresAvailable()
    {
        return nerve::gpu::tensorcore::areTensorCoresAvailable() ? 1 : 0;
    }
}

#ifdef NERVE_HAS_CUDA

#include "detail/vr_medium_hybrid_helpers.inl"
#include "nerve/gpu/distance_fasted.cuh"
#include "nerve/gpu/distance_tedjoin.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <vector>

namespace nerve::persistence
{
namespace
{

bool computeDistanceMatrixGPU(const std::vector<double> &points, size_t point_dim,
                              size_t num_points, std::vector<double> &flat_matrix)
{
    size_t matrix_size = 0;
    if (!checkedSquareCount(num_points, matrix_size) ||
        matrix_size > std::vector<double>().max_size())
    {
        return false;
    }
    if (num_points > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        point_dim > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        return false;
    }

    flat_matrix.resize(matrix_size, 0.0);

    double *d_points = nullptr;
    double *d_distances = nullptr;
    cudaError_t err;
    const auto n_int = static_cast<int>(num_points);
    const auto dim_int = static_cast<int>(point_dim);
    const auto stride_int = static_cast<int>(num_points);

    err = cudaMalloc(&d_points, matrix_size * sizeof(double));
    if (err != cudaSuccess)
    {
        flat_matrix.clear();
        return false;
    }
    err = cudaMalloc(&d_distances, matrix_size * sizeof(double));
    if (err != cudaSuccess)
    {
        cudaFree(d_points);
        flat_matrix.clear();
        return false;
    }

    err = cudaMemcpy(d_points, points.data(), num_points * point_dim * sizeof(double),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        cudaFree(d_points);
        cudaFree(d_distances);
        flat_matrix.clear();
        return false;
    }

    err = nerve::gpu::tedjoin::launchFp64TensorDistance(d_points, n_int, dim_int, d_distances,
                                                        stride_int);
    if (err != cudaSuccess)
    {
        cudaFree(d_points);
        cudaFree(d_distances);
        flat_matrix.clear();
        return false;
    }

    err = cudaMemcpy(flat_matrix.data(), d_distances, matrix_size * sizeof(double),
                     cudaMemcpyDeviceToHost);
    cudaFree(d_points);
    cudaFree(d_distances);
    if (err != cudaSuccess)
    {
        flat_matrix.clear();
        return false;
    }

    return true;
}

} // namespace
} // namespace nerve::persistence

#endif // NERVE_HAS_CUDA

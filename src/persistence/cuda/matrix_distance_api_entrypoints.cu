#include "nerve/persistence/cuda/cuda_distance_matrix.hpp"

#include <cmath>

namespace nerve::persistence::accelerated
{

errors::ErrorResult<void> computeDistanceMatrixGpu(core::BufferView<const double> points,
                                                   core::BufferView<double> &distances,
                                                   Size point_dim, double max_radius)
{
    auto matrix_result = CUDADistanceMatrix::create();
    if (matrix_result.isError())
    {
        return errors::ErrorResult<void>::error(matrix_result.errorCode());
    }
    return matrix_result.value()->compute(points, distances, point_dim, max_radius);
}

errors::ErrorResult<void> computeDistanceMatrixBatchGpu(
    const std::vector<const double *> &points_batch, const std::vector<double *> &distances_batch,
    const std::vector<Size> &n_points_batch, Size point_dim, double max_radius, Size batch_size)
{
    if (point_dim == 0 || !std::isfinite(max_radius) || !(max_radius > 0.0))
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E51_PH_INPUT,
                                                "Invalid batch distance parameters");
    }
    if (points_batch.size() != batch_size || distances_batch.size() != batch_size ||
        n_points_batch.size() != batch_size)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E51_PH_INPUT,
                                                "Batch size mismatch");
    }

    auto matrix_result = CUDADistanceMatrix::create();
    if (matrix_result.isError())
    {
        return errors::ErrorResult<void>::error(matrix_result.errorCode());
    }

    for (Size i = 0; i < batch_size; ++i)
    {
        if (points_batch[i] == nullptr || distances_batch[i] == nullptr)
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E51_PH_INPUT,
                                                    "Batch point/distance pointer is null");
        }
        core::BufferView<const double> points(points_batch[i], n_points_batch[i] * point_dim);
        core::BufferView<double> distances(distances_batch[i],
                                           n_points_batch[i] * n_points_batch[i]);

        auto result = matrix_result.value()->compute(points, distances, point_dim, max_radius);
        if (result.isError())
        {
            return result;
        }
    }

    return errors::ErrorResult<void>::ok();
}

} // namespace nerve::persistence::accelerated
